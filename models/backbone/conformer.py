"""Dependency-free Conformer blocks built on the project's attention infra."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.layers.modules import SelfAttention


class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(ff_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.silu(self.input(x))))


class ConformerConvolution(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer")
        self.pointwise_in = nn.Conv1d(dim, dim * 2, 1)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask[..., None].to(x.dtype)
        x = F.glu(self.pointwise_in(x.transpose(1, 2)), dim=1)
        x = x * mask[:, None].to(x.dtype)
        x = self.depthwise(x).transpose(1, 2)
        x = F.silu(self.norm(x)).transpose(1, 2)
        x = self.pointwise_out(x).transpose(1, 2)
        return x * mask[..., None].to(x.dtype)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        ff_dim: int,
        conv_kernel_size: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        dim_head = dim // heads
        if dim_head % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1 = ConformerFeedForward(dim, ff_dim, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = SelfAttention(
            dim,
            heads,
            dim_head,
            attention_dropout,
            "rms_norm",
            None,
        )
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = ConformerConvolution(dim, conv_kernel_size)
        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2 = ConformerFeedForward(dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_value = mask[..., None].to(x.dtype)
        x = x + 0.5 * self.dropout(self.ff1(self.ff1_norm(x)))
        x = x * mask_value
        x = x + self.dropout(
            self.attention(
                self.attention_norm(x),
                positions=positions,
                mask=mask,
            )
        )
        x = x * mask_value
        x = x + self.dropout(self.conv(self.conv_norm(x), mask))
        x = x * mask_value
        x = x + 0.5 * self.dropout(self.ff2(self.ff2_norm(x)))
        return self.final_norm(x) * mask_value


class ConformerEncoder(nn.Module):
    """Full-context, non-causal Conformer encoder."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        ff_dim: int,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=dim,
                    heads=heads,
                    ff_dim=ff_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("ConformerEncoder input must have shape [B,T,C]")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.ndim != 2 or mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [B,T]")
        positions = torch.arange(
            x.shape[1],
            device=x.device,
            dtype=torch.float32,
        )[None].expand(x.shape[0], -1)
        x = x * mask[..., None].to(x.dtype)
        for block in self.blocks:
            x = block(x, positions=positions, mask=mask)
        return x * mask[..., None].to(x.dtype)
