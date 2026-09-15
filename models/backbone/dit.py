# Copyright (c) 2023 Amphion.
# Adapted from P-MUSE/Amphion under the MIT license for this project.
"""P-MUSE-style diffusion Transformer."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.layers.modules import (
    AdaLayerNorm,
    AdaLayerNormConv,
    AdaLayerNormFinal,
    ConvBlock,
    FeedForward,
    SelfAttention,
    TimestepEmbedding,
)


class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float,
        dropout: float,
        qk_norm: str | None,
        pe_attn_head: int | None,
        enable_conv: bool = False,
    ) -> None:
        super().__init__()
        self.attention_norm = AdaLayerNorm(dim)
        self.attention = SelfAttention(dim, heads, dim_head, dropout, qk_norm, pe_attn_head)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, ff_mult, dropout)
        self.enable_conv = bool(enable_conv)
        if self.enable_conv:
            self.conv_norm = AdaLayerNormConv(dim)
            self.conv = ConvBlock(dim)

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        norm, gate_attn, shift_ff, scale_ff, gate_ff = self.attention_norm(x, embedding)
        x = x + gate_attn[:, None] * self.attention(norm, positions=positions, mask=mask)
        if self.enable_conv:
            if mask is None:
                token_mask = None
            elif mask.ndim == 2:
                token_mask = mask
            elif mask.ndim == 3:
                token_mask = mask.any(dim=-1)
            else:
                raise ValueError("DiT mask must have shape [B,N] or [B,N,N]")
            norm, gate_conv = self.conv_norm(x, embedding)
            x = x + gate_conv[:, None] * self.conv(norm, mask=token_mask)
        norm = self.ff_norm(x) * (1 + scale_ff[:, None]) + shift_ff[:, None]
        return x + gate_ff[:, None] * self.ff(norm)


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 128,
        dropout: float = 0.1,
        ff_mult: float = 4,
        latent_dim: int = 128,
        qk_norm: str | None = "rms_norm",
        pe_attn_head: int | None = None,
        checkpoint_activations: bool = False,
        enable_conv: bool = False,
        **_unused,
    ) -> None:
        super().__init__()
        if heads * dim_head != dim:
            raise ValueError("heads * dim_head must equal dim")
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.ff_dim = int(dim * ff_mult)
        self.time_embedding = TimestepEmbedding(dim)
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                enable_conv=enable_conv,
            )
            for _ in range(depth)
        ])
        self.output_norm = AdaLayerNormFinal(dim)
        self.output_projection = nn.Linear(dim, latent_dim)
        self.checkpoint_activations = checkpoint_activations
        self._initialize_adaln()

    def _initialize_adaln(self) -> None:
        for block in self.blocks:
            nn.init.zeros_(block.attention_norm.linear.weight)
            nn.init.zeros_(block.attention_norm.linear.bias)
            if block.enable_conv:
                nn.init.zeros_(block.conv_norm.linear.weight)
                nn.init.zeros_(block.conv_norm.linear.bias)
        nn.init.zeros_(self.output_norm.linear.weight)
        nn.init.zeros_(self.output_norm.linear.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_unused,
    ) -> torch.Tensor:
        batch, frames, _ = x.shape
        if t.ndim == 0:
            t = t.expand(batch)
        embedding = self.time_embedding(t)
        if label is not None:
            if label.shape != embedding.shape:
                raise ValueError("label embedding must match timestep embedding")
            embedding = embedding + label
        if position_ids is None:
            position_ids = torch.arange(frames, device=x.device, dtype=torch.float32)[None].expand(batch, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids[None].expand(batch, -1)
        for block in self.blocks:
            if self.checkpoint_activations and self.training:
                x = checkpoint(block, x, embedding, position_ids, mask, use_reentrant=False)
            else:
                x = block(x, embedding, position_ids, mask)
        return self.output_projection(self.output_norm(x, embedding))
