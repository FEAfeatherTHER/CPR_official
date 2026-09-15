# Copyright (c) 2023 Amphion.
# Adapted from P-MUSE/Amphion under the MIT license for this project.
"""Minimal DiT layers retained from the P-MUSE implementation."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=value.device, dtype=torch.float32)
            * (-math.log(10_000) / (half - 1))
        )
        angles = scale * value.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1).to(value.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency = SinusPositionEmbedding(frequency_dim)
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequency(timestep)
        return self.mlp(frequencies.to(self.mlp[0].weight.dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim * 6)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        values = self.linear(F.silu(embedding)).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = values
        x = self.norm(x) * (1 + scale_attn[:, None]) + shift_attn[:, None]
        return x, gate_attn, shift_ff, scale_ff, gate_ff


class AdaLayerNormConv(nn.Module):
    """AdaLN modulation for an optional convolutional residual branch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim * 3)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        shift, scale, gate = self.linear(F.silu(embedding)).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x, gate


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(embedding)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class ConvBlock(nn.Module):
    """FireRedTTS3-style temporal convolution for a ``[B,T,C]`` stream."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            if mask.ndim != 2 or mask.shape != x.shape[:2]:
                raise ValueError("ConvBlock mask must have shape [B,T]")
            x = x * mask[..., None].to(x.dtype)
        x = self.layers[0](x.transpose(1, 2))
        x = self.layers[1](x)
        if mask is not None:
            x = x * mask[:, None].to(x.dtype)
        x = self.layers[2](x).transpose(1, 2)
        if mask is not None:
            x = x * mask[..., None].to(x.dtype)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def apply_rotary(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inverse = 1.0 / (
        10_000 ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    )
    if positions.ndim == 1:
        positions = positions[None]
    angles = positions.float()[..., None] * inverse
    cos = angles.cos()[:, None]
    sin = angles.sin()[:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        qk_norm: str | None,
        rope_heads: int | None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.rope_heads = heads if rope_heads is None else rope_heads
        inner = heads * dim_head
        self.qkv = nn.Linear(dim, inner * 3)
        self.output = nn.Linear(inner, dim)
        self.dropout = dropout
        if qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        elif qk_norm is None:
            self.q_norm = self.k_norm = nn.Identity()
        else:
            raise ValueError(f"unsupported qk_norm: {qk_norm}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, frames, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, frames, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, frames, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, frames, self.heads, self.dim_head).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_heads:
            q = torch.cat((apply_rotary(q[:, :self.rope_heads], positions), q[:, self.rope_heads:]), dim=1)
            k = torch.cat((apply_rotary(k[:, :self.rope_heads], positions), k[:, self.rope_heads:]), dim=1)
        attention_mask = None
        if mask is not None:
            if mask.ndim == 2:
                attention_mask = mask[:, None, None, :].expand(batch, 1, frames, frames)
            elif mask.ndim == 3:
                attention_mask = mask[:, None]
            else:
                raise ValueError("DiT mask must have shape [B,N] or [B,N,N]")
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(output.transpose(1, 2).reshape(batch, frames, -1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        inner = int(dim * multiplier)
        self.layers = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
