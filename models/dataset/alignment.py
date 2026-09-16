from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import random

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalAlignment:
    """Exact time-grid ratios derived from model configuration."""

    mel_fps: int
    midi_fps: int
    patch_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("mel_fps", self.mel_fps),
            ("midi_fps", self.midi_fps),
            ("patch_size", self.patch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def patch_fps(self) -> float:
        return self.mel_fps / self.patch_size

    @property
    def midi_rope_scale(self) -> float:
        return self.patch_fps / self.midi_fps

    @property
    def patches_per_block(self) -> int:
        midi_frames_per_patch = Fraction(
            self.midi_fps * self.patch_size,
            self.mel_fps,
        )
        return midi_frames_per_patch.denominator

    @property
    def mel_frames_per_block(self) -> int:
        return self.patch_size * self.patches_per_block

    @property
    def midi_frames_per_block(self) -> int:
        frames = Fraction(
            self.midi_fps * self.mel_frames_per_block,
            self.mel_fps,
        )
        if frames.denominator != 1:
            raise RuntimeError("alignment block does not contain whole MIDI frames")
        return frames.numerator

    @property
    def block_duration(self) -> float:
        return self.mel_frames_per_block / self.mel_fps

    def lengths_from_mel_frames(self, valid_mel_frames: int) -> "AlignmentLengths":
        return AlignmentLengths.from_mel_frames(
            valid_mel_frames,
            alignment=self,
        )


DEFAULT_ALIGNMENT = TemporalAlignment(mel_fps=50, midi_fps=25, patch_size=5)
MEL_FPS = DEFAULT_ALIGNMENT.mel_fps
MIDI_FPS = DEFAULT_ALIGNMENT.midi_fps
PATCH_SIZE = DEFAULT_ALIGNMENT.patch_size
MEL_FRAMES_PER_BLOCK = DEFAULT_ALIGNMENT.mel_frames_per_block
PATCHES_PER_BLOCK = DEFAULT_ALIGNMENT.patches_per_block
MIDI_FRAMES_PER_BLOCK = DEFAULT_ALIGNMENT.midi_frames_per_block


@dataclass(frozen=True)
class AlignmentLengths:
    valid_mel_frames: int
    blocks: int
    padded_mel_frames: int
    patch_frames: int
    midi_frames: int
    valid_patch_frames: int

    @classmethod
    def from_mel_frames(
        cls,
        valid_mel_frames: int,
        *,
        alignment: TemporalAlignment = DEFAULT_ALIGNMENT,
    ) -> "AlignmentLengths":
        if valid_mel_frames <= 0:
            raise ValueError("valid_mel_frames must be positive")
        blocks = math.ceil(valid_mel_frames / alignment.mel_frames_per_block)
        return cls(
            valid_mel_frames=valid_mel_frames,
            blocks=blocks,
            padded_mel_frames=blocks * alignment.mel_frames_per_block,
            patch_frames=blocks * alignment.patches_per_block,
            midi_frames=blocks * alignment.midi_frames_per_block,
            valid_patch_frames=math.ceil(valid_mel_frames / alignment.patch_size),
        )


def lengths_from_duration(
    duration: float,
    *,
    alignment: TemporalAlignment = DEFAULT_ALIGNMENT,
) -> AlignmentLengths:
    if duration <= 0:
        raise ValueError("duration must be positive")
    return AlignmentLengths.from_mel_frames(
        math.ceil(duration * alignment.mel_fps),
        alignment=alignment,
    )


def mel_frames_from_audio_samples(audio_samples: int, *, hop_length: int = 480) -> int:
    """Frame count produced by the project's padded, center=False STFT."""
    if audio_samples < hop_length:
        raise ValueError("audio must contain at least one Mel hop")
    return audio_samples // hop_length


def lengths_from_audio_samples(
    audio_samples: int,
    *,
    hop_length: int = 480,
    alignment: TemporalAlignment = DEFAULT_ALIGNMENT,
) -> AlignmentLengths:
    return AlignmentLengths.from_mel_frames(
        mel_frames_from_audio_samples(audio_samples, hop_length=hop_length),
        alignment=alignment,
    )


def lengths_from_last_note(
    last_note_off: float,
    *,
    max_release_duration: float = 3.0,
    target_duration: float | None = None,
    alignment: TemporalAlignment = DEFAULT_ALIGNMENT,
) -> AlignmentLengths:
    if last_note_off < 0:
        raise ValueError("last_note_off must be non-negative")
    if max_release_duration < 0:
        raise ValueError("max_release_duration must be non-negative")
    if target_duration is not None:
        if target_duration < last_note_off:
            raise ValueError("target_duration cannot be shorter than the last note_off")
        duration = target_duration
    else:
        duration = last_note_off + max_release_duration
    if duration <= 0:
        raise ValueError("target duration must be positive")
    return lengths_from_duration(duration, alignment=alignment)


def sample_prompt_blocks(
    total_blocks: int,
    *,
    rng: random.Random | None = None,
    minimum_ratio: float = 0.2,
    maximum_ratio: float = 0.8,
) -> int:
    if total_blocks < 2:
        raise ValueError("at least two alignment blocks are required")
    if not 0 < minimum_ratio <= maximum_ratio < 1:
        raise ValueError("prompt ratios must satisfy 0 < min <= max < 1")
    rng = rng or random
    ratio = rng.uniform(minimum_ratio, maximum_ratio)
    return min(max(int(round(ratio * total_blocks)), 1), total_blocks - 1)


def pad_pianoroll(roll: torch.Tensor, target_frames: int) -> torch.Tensor:
    if roll.ndim < 2:
        raise ValueError("pianoroll must have a time dimension")
    time_dim = -2
    current = roll.shape[time_dim]
    if current > target_frames:
        index = [slice(None)] * roll.ndim
        index[time_dim] = slice(0, target_frames)
        return roll[tuple(index)]
    if current == target_frames:
        return roll
    # torch.nn.functional.pad lists the last dimension first.
    return F.pad(roll, (0, 0, 0, target_frames - current))


def length_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError("lengths must be a 1-D tensor")
    max_length = int(lengths.max()) if max_length is None else int(max_length)
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]
