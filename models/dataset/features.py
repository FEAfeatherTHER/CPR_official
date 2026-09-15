"""Audio, Mel, and MIDI features shared by training and inference."""

from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F


class NormalizedMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24_000,
        n_fft: int = 1920,
        hop_length: int = 480,
        win_length: int = 1920,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float = 12_000.0,
        mean: float = -4.92,
        variance: float = 8.14,
    ) -> None:
        super().__init__()
        mel = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(win_length))
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mean = mean
        self.std = math.sqrt(variance)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.mean(dim=1)
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [B,T] or [B,C,T]")
        pad = (self.n_fft - self.hop_length) // 2
        waveform = F.pad(waveform.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs().clamp_min(1e-5)
        mel = torch.matmul(self.mel_basis, spectrum).clamp_min(1e-5).log()
        return ((mel - self.mean) / self.std).transpose(1, 2)


def load_audio_segment(
    path: str | Path,
    start_sec: float,
    duration_sec: float,
    *,
    sample_rate: int = 24_000,
) -> np.ndarray:
    info = sf.info(str(path))
    start = max(0, round(start_sec * info.samplerate))
    frames = max(1, round(duration_sec * info.samplerate))
    audio, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if info.samplerate != sample_rate:
        audio = librosa.resample(audio, orig_sr=info.samplerate, target_sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def midi_to_pianoroll(
    midi_path: str | Path,
    *,
    start_sec: float,
    duration_sec: float,
    frames: int,
    fps: int = 25,
) -> np.ndarray:
    try:
        from symusic import Score
    except ImportError as exc:
        raise ImportError("symusic is required for MIDI loading") from exc
    score = Score(str(midi_path), ttype="second")
    end_sec = start_sec + duration_sec
    velocity = np.zeros((frames, 128), dtype=np.float32)
    onset = np.zeros((frames, 128), dtype=np.float32)
    for track in score.tracks:
        for note in track.notes:
            note_start = float(note.start)
            note_end = float(note.end)
            if note_end <= start_sec or note_start >= end_sec or not 0 <= note.pitch < 128:
                continue
            relative_start = max(note_start, start_sec) - start_sec
            relative_end = min(note_end, end_sec) - start_sec
            first = min(max(int(relative_start * fps), 0), frames - 1)
            last = min(max(math.ceil(relative_end * fps), first + 1), frames)
            velocity[first:last, note.pitch] = np.maximum(
                velocity[first:last, note.pitch], float(note.velocity) / 127.0
            )
            if note_start >= start_sec:
                onset[first, note.pitch] = 1.0
    return np.stack((velocity, onset), axis=0)


def _score_to_ternary_pianoroll(
    score,
    *,
    start_sec: float,
    duration_sec: float,
    frames: int,
    fps: int,
    pitch_min: int,
    pitch_count: int,
    velocity_on_sustain: bool = False,
) -> np.ndarray:
    if start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if frames <= 0 or fps <= 0:
        raise ValueError("frames and fps must be positive")
    if pitch_count <= 0 or pitch_min < 0 or pitch_min + pitch_count > 128:
        raise ValueError("pitch range must be inside MIDI 0..127")

    end_sec = start_sec + duration_sec
    state = np.zeros((frames, pitch_count), dtype=np.uint8)
    velocity = np.zeros((frames, pitch_count), dtype=np.uint8)
    sustain_velocity = (
        np.zeros((frames, pitch_count), dtype=np.uint8)
        if velocity_on_sustain
        else None
    )
    for track in score.tracks:
        for note in track.notes:
            note_start = float(note.start)
            note_end = float(note.end)
            pitch_index = int(note.pitch) - pitch_min
            if (
                note_end <= start_sec
                or note_start >= end_sec
                or not 0 <= pitch_index < pitch_count
            ):
                continue
            relative_start = max(note_start, start_sec) - start_sec
            relative_end = min(note_end, end_sec) - start_sec
            first = min(max(int(relative_start * fps), 0), frames - 1)
            last = min(max(math.ceil(relative_end * fps), first + 1), frames)
            note_velocity = min(max(int(note.velocity), 1), 127)
            if sustain_velocity is not None:
                sustain_velocity[first:last, pitch_index] = np.maximum(
                    sustain_velocity[first:last, pitch_index],
                    np.uint8(note_velocity),
                )
            has_onset = note_start >= start_sec
            sustain_start = first + 1 if has_onset else first
            if sustain_start < last:
                current = state[sustain_start:last, pitch_index]
                state[sustain_start:last, pitch_index] = np.where(
                    current == 1,
                    current,
                    np.uint8(2),
                )
            if has_onset:
                state[first, pitch_index] = 1
                velocity[first, pitch_index] = max(
                    int(velocity[first, pitch_index]),
                    note_velocity,
                )
    if sustain_velocity is not None:
        velocity = np.where(
            state == 2,
            sustain_velocity,
            velocity,
        ).astype(np.uint8, copy=False)
    return np.stack((state, velocity), axis=0)


def midi_to_dual_ternary_pianoroll(
    midi_path: str | Path,
    *,
    start_sec: float,
    duration_sec: float,
    high_frames: int,
    low_frames: int,
    high_fps: int = 50,
    low_fps: int = 25,
    pitch_min: int = 21,
    pitch_count: int = 88,
    velocity_on_sustain: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned 50/25-fps ternary state and velocity rolls."""

    if high_fps != 2 * low_fps:
        raise ValueError("high_fps must be twice low_fps")
    if high_frames != 2 * low_frames:
        raise ValueError("high_frames must be twice low_frames")
    try:
        from symusic import Score
    except ImportError as exc:
        raise ImportError("symusic is required for MIDI loading") from exc
    score = Score(str(midi_path), ttype="second")
    common = dict(
        score=score,
        start_sec=start_sec,
        duration_sec=duration_sec,
        pitch_min=pitch_min,
        pitch_count=pitch_count,
        velocity_on_sustain=velocity_on_sustain,
    )
    high = _score_to_ternary_pianoroll(
        **common,
        frames=high_frames,
        fps=high_fps,
    )
    low = _score_to_ternary_pianoroll(
        **common,
        frames=low_frames,
        fps=low_fps,
    )
    return high, low


def last_note_off(midi_path: str | Path) -> float:
    try:
        from symusic import Score
    except ImportError as exc:
        raise ImportError("symusic is required for MIDI loading") from exc
    score = Score(str(midi_path), ttype="second")
    return max((float(note.end) for track in score.tracks for note in track.notes), default=0.0)
