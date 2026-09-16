"""Dual-rate ternary pianoroll encoder."""

from __future__ import annotations

import torch
from torch import nn

from models.backbone.conformer import ConformerEncoder
from models.backbone.resnet1d import ResNet1DDownsample


class TernaryPianorollEmbedding(nn.Module):
    def __init__(
        self,
        *,
        pitch_count: int = 88,
        embedding_dim: int = 6,
        velocity_vocab_size: int = 127,
        velocity_on_sustain: bool = True,
    ) -> None:
        super().__init__()
        if pitch_count <= 0 or embedding_dim <= 0 or velocity_vocab_size <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.pitch_count = int(pitch_count)
        self.embedding_dim = int(embedding_dim)
        self.velocity_vocab_size = int(velocity_vocab_size)
        self.velocity_on_sustain = bool(velocity_on_sustain)
        self.onset_pitch = nn.Parameter(torch.empty(pitch_count, embedding_dim))
        self.sustain_pitch = nn.Parameter(torch.empty(pitch_count, embedding_dim))
        self.velocity = nn.Embedding(velocity_vocab_size, embedding_dim)
        nn.init.normal_(self.onset_pitch, std=0.02)
        nn.init.normal_(self.sustain_pitch, std=0.02)
        nn.init.normal_(self.velocity.weight, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.pitch_count * self.embedding_dim

    def forward(self, roll: torch.Tensor) -> torch.Tensor:
        if roll.ndim != 4 or roll.shape[1] != 2 or roll.shape[-1] != self.pitch_count:
            raise ValueError(
                f"pianoroll must have shape [B,2,T,{self.pitch_count}]"
            )
        state = roll[:, 0].long()
        velocity = roll[:, 1].long()
        torch._assert_async(
            ((state >= 0) & (state <= 2)).all(),
            "ternary state values must be in 0..2",
        )
        torch._assert_async(
            ((velocity >= 0) & (velocity <= self.velocity_vocab_size)).all(),
            "velocity values are outside the configured vocabulary",
        )
        onset = state == 1
        sustain = state == 2
        features = (
            onset[..., None].to(self.onset_pitch.dtype) * self.onset_pitch[None, None]
            + sustain[..., None].to(self.sustain_pitch.dtype)
            * self.sustain_pitch[None, None]
        )
        velocity_index = velocity.clamp(min=1, max=self.velocity_vocab_size) - 1
        velocity_features = self.velocity(velocity_index)
        velocity_state = (onset | sustain) if self.velocity_on_sustain else onset
        velocity_mask = velocity_state & (velocity > 0)
        features = features + velocity_features * velocity_mask[..., None].to(
            velocity_features.dtype
        )
        return features.flatten(start_dim=-2)


class PianorollEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        pitch_min: int = 21,
        pitch_count: int = 88,
        embedding_dim: int = 6,
        velocity_vocab_size: int = 127,
        velocity_on_sustain: bool = True,
        bottleneck_dim: int = 132,
        resnet_kernel_size: int = 3,
        conformer_depth: int = 2,
        conformer_heads: int = 8,
        conformer_ff_dim: int = 2112,
        conformer_conv_kernel_size: int = 15,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pitch_min < 0 or pitch_min + pitch_count > 128:
            raise ValueError("pitch range must be inside MIDI 0..127")
        self.pitch_min = int(pitch_min)
        self.embedding = TernaryPianorollEmbedding(
            pitch_count=pitch_count,
            embedding_dim=embedding_dim,
            velocity_vocab_size=velocity_vocab_size,
            velocity_on_sustain=velocity_on_sustain,
        )
        dim = self.embedding.output_dim
        self.high_downsample = ResNet1DDownsample(
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            kernel_size=resnet_kernel_size,
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.conformer = ConformerEncoder(
            dim=dim,
            depth=conformer_depth,
            heads=conformer_heads,
            ff_dim=conformer_ff_dim,
            conv_kernel_size=conformer_conv_kernel_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.output_projection = nn.Linear(dim, hidden_size)

    def forward(
        self,
        pianoroll_50: torch.Tensor,
        pianoroll_25: torch.Tensor,
        *,
        midi_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if pianoroll_50.shape[0] != pianoroll_25.shape[0]:
            raise ValueError("50-fps and 25-fps batch sizes must match")
        if pianoroll_50.shape[2] != 2 * pianoroll_25.shape[2]:
            raise ValueError("50-fps time length must be twice the 25-fps length")
        if midi_lengths.ndim != 1 or midi_lengths.shape[0] != pianoroll_25.shape[0]:
            raise ValueError("midi_lengths must have shape [B]")
        torch._assert_async(
            ((midi_lengths > 0) & (midi_lengths <= pianoroll_25.shape[2])).all(),
            "midi_lengths are outside the 25-fps tensor",
        )
        low_mask = (
            torch.arange(pianoroll_25.shape[2], device=pianoroll_25.device)[None]
            < midi_lengths[:, None]
        )
        high_mask = (
            torch.arange(pianoroll_50.shape[2], device=pianoroll_50.device)[None]
            < (midi_lengths * 2)[:, None]
        )
        high = self.embedding(pianoroll_50)
        low = self.embedding(pianoroll_25)
        high = self.high_downsample(high, mask=high_mask)
        fused = self.fusion_norm(high + low)
        fused = fused * low_mask[..., None].to(fused.dtype)
        output = self.output_projection(self.conformer(fused, mask=low_mask))
        return output * low_mask[..., None].to(output.dtype)
