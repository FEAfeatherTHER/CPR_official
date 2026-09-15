"""Frozen LAION CLAP audio encoder loaded as an external asset."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def slice_audio_batch(
    audio: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    max_samples: int,
) -> list[torch.Tensor]:
    if audio.ndim != 2:
        raise ValueError("audio must have shape [B,T]")
    if lengths is None:
        lengths = torch.full(
            (audio.shape[0],), audio.shape[1], dtype=torch.long, device=audio.device
        )
    if lengths.shape != (audio.shape[0],):
        raise ValueError("lengths must have shape [B]")
    return [
        audio[index, :min(max(int(lengths[index]), 1), max_samples)]
        for index in range(audio.shape[0])
    ]


def crop_audio_clips(
    clips: list[torch.Tensor],
    *,
    max_samples: int,
    mode: str,
) -> list[torch.Tensor]:
    if mode not in {"random", "first", "last"}:
        raise ValueError("CLAP crop mode must be random, first, or last")
    output = []
    for clip in clips:
        if clip.numel() <= max_samples:
            output.append(clip)
            continue
        if mode == "first":
            start = 0
        elif mode == "last":
            start = clip.numel() - max_samples
        else:
            start = int(torch.randint(0, clip.numel() - max_samples + 1, (1,), device=clip.device))
        output.append(clip[start:start + max_samples])
    return output


class FrozenCLAPEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        max_duration: float = 30.0,
        embedding_duration: float = 10.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CLAP checkpoint not found: {checkpoint_path}")
        import laion_clap

        self.device_name = torch.device(device)
        self.max_duration = max_duration
        self.embedding_duration = embedding_duration
        self.max_samples = round(embedding_duration * 48_000)
        self._clap = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base",
            device=self.device_name,
        )
        self._clap.load_ckpt(str(checkpoint_path), verbose=False)
        self._clap.to(self.device_name)
        self._clap.eval()
        for parameter in self._clap.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        audio: torch.Tensor,
        *,
        sample_rate: int = 24_000,
        lengths: torch.Tensor | None = None,
        crop_mode: str = "random",
    ) -> torch.Tensor:
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if audio.ndim != 2:
            raise ValueError("CLAP audio must have shape [B,T] or [B,C,T]")
        audio = audio.to(self.device_name, dtype=torch.float32)
        clips = slice_audio_batch(
            audio,
            lengths,
            max_samples=round(self.max_duration * sample_rate),
        )
        clips = crop_audio_clips(
            clips,
            max_samples=round(self.embedding_duration * sample_rate),
            mode=crop_mode,
        )
        if sample_rate != 48_000:
            # Keep torchaudio lazy: importing this external backend should not
            # probe CUDA when a CLI is only displaying its help text.
            import torchaudio

            clips = [torchaudio.functional.resample(clip, sample_rate, 48_000) for clip in clips]
        clips = [clip[:self.max_samples] for clip in clips]
        return self._clap.get_audio_embedding_from_data(clips, use_tensor=True).float()
