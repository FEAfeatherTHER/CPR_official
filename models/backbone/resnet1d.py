"""Small temporal ResNet bottleneck used by MIDI encoders."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResNet1DDownsample(nn.Module):
    """A 2x temporal bottleneck with an explicit residual shortcut."""

    def __init__(
        self,
        *,
        dim: int,
        bottleneck_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("dim and bottleneck_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.input_norm = nn.LayerNorm(dim)
        self.main_reduce = nn.Conv1d(dim, bottleneck_dim, 1, bias=False)
        self.main_temporal = nn.Conv1d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)
        self.main_expand = nn.Conv1d(bottleneck_dim, dim, 1, bias=False)
        self.shortcut = nn.Conv1d(dim, dim, 1, stride=2, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("ResNet1DDownsample input must have shape [B,T,C]")
        if x.shape[1] % 2:
            raise ValueError("ResNet1DDownsample requires an even time length")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:2] or mask.ndim != 2:
            raise ValueError("mask must have shape [B,T]")
        x = x * mask[..., None].to(x.dtype)
        main = self.input_norm(x).transpose(1, 2)
        main = F.silu(self.main_reduce(main))
        main = self.main_temporal(main).transpose(1, 2)
        main = self.bottleneck_norm(main)
        main = self.main_expand(F.silu(main).transpose(1, 2))
        shortcut = self.shortcut(x.transpose(1, 2))
        output = (main + shortcut).transpose(1, 2)
        output_mask = mask[:, ::2]
        return output * output_mask[..., None].to(output.dtype)
