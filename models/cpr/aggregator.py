"""Full-attention Mel patch encoder with a learned CLS token."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MelPatchAggregator(nn.Module):
    def __init__(
        self,
        *,
        mel_dim: int = 128,
        hidden_size: int = 1024,
        depth: int = 4,
        heads: int = 8,
        ff_dim: int = 4096,
        patch_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.patch_size = patch_size
        self.input_projection = nn.Linear(mel_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.position = nn.Parameter(torch.empty(1, patch_size + 1, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        mel: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [B,T,mel_dim]")
        batch, frames, channels = mel.shape
        if frame_mask is None:
            frame_mask = torch.ones((batch, frames), dtype=torch.bool, device=mel.device)
        if frame_mask.shape != (batch, frames):
            raise ValueError("frame_mask must have shape [B,T]")
        patches = math.ceil(frames / self.patch_size)
        padded_frames = patches * self.patch_size
        if padded_frames != frames:
            mel = F.pad(mel, (0, 0, 0, padded_frames - frames))
            frame_mask = F.pad(frame_mask, (0, padded_frames - frames), value=False)
        mel = mel.reshape(batch * patches, self.patch_size, channels)
        mask = frame_mask.reshape(batch * patches, self.patch_size)
        tokens = self.input_projection(mel)
        cls = self.cls_token.expand(batch * patches, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.position
        key_padding = torch.cat(
            (torch.zeros((batch * patches, 1), dtype=torch.bool, device=mel.device), ~mask),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding)
        patch_mask = mask.any(dim=1).reshape(batch, patches)
        output = encoded[:, 0].reshape(batch, patches, -1)
        return output.masked_fill(~patch_mask[..., None], 0.0), patch_mask
