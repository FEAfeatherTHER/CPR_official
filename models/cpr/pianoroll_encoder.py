"""P-MUSE velocity/onset gated pianoroll encoder."""

from __future__ import annotations

import torch
from torch import nn


class PianorollEncoder(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int | None = None) -> None:
        super().__init__()
        intermediate_size = intermediate_size or hidden_size
        self.velocity_projection = nn.Sequential(
            nn.Linear(128, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, intermediate_size),
        )
        self.onset_projection = nn.Sequential(
            nn.Linear(128, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, intermediate_size * 3),
        )
        self.output_projection = nn.Linear(intermediate_size, hidden_size)

    def forward(self, pianoroll: torch.Tensor) -> torch.Tensor:
        if pianoroll.ndim != 4 or pianoroll.shape[1] != 2 or pianoroll.shape[-1] != 128:
            raise ValueError("pianoroll must have shape [B,2,T,128]")
        velocity, onset = pianoroll[:, 0], pianoroll[:, 1]
        gate, scale, shift = self.onset_projection(onset).chunk(3, dim=-1)
        velocity = self.velocity_projection(velocity)
        return self.output_projection(velocity + gate * (scale * velocity + shift))
