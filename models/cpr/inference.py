"""Standalone Composer–Performer audio inference pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the CPR_official project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inference_options(
    *, steps: int, cfg_scale: float, max_release_duration: float,
    target_duration: float | None,
) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")
    if not isinstance(cfg_scale, (int, float)) or not math.isfinite(cfg_scale):
        raise ValueError("cfg-scale must be finite")
    if cfg_scale < 0:
        raise ValueError("cfg-scale must be non-negative")
    if (not isinstance(max_release_duration, (int, float))
            or not math.isfinite(max_release_duration) or max_release_duration < 0):
        raise ValueError("max-release-duration must be finite and non-negative")
    if target_duration is not None and (
        not isinstance(target_duration, (int, float))
        or not math.isfinite(target_duration) or target_duration <= 0
    ):
        raise ValueError("target-duration must be finite and positive")


def validate_output_path(output_path: str | Path, *input_paths: str | Path) -> Path:
    output = resolve_project_path(output_path).resolve()
    for path in input_paths:
        if output == resolve_project_path(path).resolve():
            raise ValueError(f"output must not overwrite an input or checkpoint: {output}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    return output


def validate_input_paths(*input_paths: str | Path) -> tuple[Path, ...]:
    resolved = tuple(resolve_project_path(path) for path in input_paths)
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"input is empty: {path}")
    return resolved


def validate_inference_lengths(
    prompt, target, *, max_prompt_duration: float,
    max_target_duration: float, max_composer_tokens: int,
) -> None:
    prompt_duration = prompt.valid_mel_frames / 50.0
    target_duration = target.valid_mel_frames / 50.0
    if prompt_duration > max_prompt_duration:
        raise ValueError(
            f"prompt duration {prompt_duration:.2f}s exceeds {max_prompt_duration:.2f}s"
        )
    if target_duration > max_target_duration:
        raise ValueError(
            f"target duration {target_duration:.2f}s exceeds {max_target_duration:.2f}s"
        )
    final_tokens = 1 + 7 * (prompt.blocks + target.blocks)
    if final_tokens > max_composer_tokens:
        raise ValueError(
            f"Composer token count {final_tokens} exceeds limit {max_composer_tokens}"
        )


def _load_prompt_audio(path: Path, sample_rate: int):
    import librosa
    import soundfile as sf
    import torch

    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
    if audio.size == 0:
        raise ValueError("prompt audio is empty")
    return torch.from_numpy(audio).float()[None]


def _load_model(config: dict, checkpoint_path: Path, device):
    import torch
    from models.utils.checkpoint import load_composer_performer_checkpoint
    from models.cpr.factory import build_composer_performer, validate_checkpoint_model_config

    saved_config_path = checkpoint_path.with_name("composer_performer_config.json")
    if not saved_config_path.is_file():
        raise FileNotFoundError(f"saved CP architecture not found: {saved_config_path}")
    saved = json.loads(saved_config_path.read_text(encoding="utf-8"))
    if "qwen_config" not in saved:
        raise ValueError("saved CP architecture does not contain qwen_config")
    validate_checkpoint_model_config(config, saved)
    model = build_composer_performer(config, saved_qwen_config=saved["qwen_config"])
    load_composer_performer_checkpoint(model, checkpoint_path)
    dtype_name = config["inference"].get("model_dtype", "bfloat16")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported inference model_dtype: {dtype_name}")
    if device.type == "cpu":
        dtype = torch.float32
    return model.eval().to(device=device, dtype=dtype)


def run_inference(
    *, config: dict, checkpoint_path: str | Path,
    prompt_audio_path: str | Path, prompt_midi_path: str | Path,
    target_midi_path: str | Path, output_path: str | Path,
    device: str = "cuda", seed: int = 114, target_duration: float | None = None,
    max_release_duration: float | None = None, steps: int | None = None,
    schedule: str | None = None, cfg_scale: float | None = None,
) -> Path:
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    from models.dataset.alignment import AlignmentLengths, lengths_from_last_note
    from models.utils.audio import peak_normalize
    from models.codec.clap import FrozenCLAPEncoder
    from models.dataset.features import NormalizedMelSpectrogram, last_note_off, midi_to_pianoroll
    from models.base.ode_sampler import validate_ode_schedule
    from models.codec.vocos import Vocos, load_vocos_checkpoint

    inference = config["inference"]
    steps = inference["ode_steps"] if steps is None else steps
    cfg_scale = inference["cfg_scale"] if cfg_scale is None else cfg_scale
    max_release_duration = (
        inference["max_release_duration"] if max_release_duration is None
        else max_release_duration
    )
    validate_inference_options(
        steps=steps, cfg_scale=cfg_scale,
        max_release_duration=max_release_duration,
        target_duration=target_duration,
    )
    schedule = validate_ode_schedule(
        inference["schedule"] if schedule is None else schedule
    )
    (checkpoint_path, prompt_audio_path, prompt_midi_path,
     target_midi_path) = validate_input_paths(
        checkpoint_path, prompt_audio_path, prompt_midi_path, target_midi_path,
    )
    output_path = validate_output_path(
        output_path, checkpoint_path, prompt_audio_path,
        prompt_midi_path, target_midi_path,
    )
    sample_rate = int(config["data"]["sample_rate"])
    if sample_rate != 24000:
        raise ValueError("CP output sample_rate must be 24000")
    torch_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    generator = torch.Generator(device=torch_device).manual_seed(seed)

    with torch.inference_mode():
        waveform = _load_prompt_audio(prompt_audio_path, sample_rate).to(torch_device)
        mel_extractor = NormalizedMelSpectrogram().eval().to(torch_device)
        prompt_valid_mel = mel_extractor(waveform)
        prompt_lengths = AlignmentLengths.from_mel_frames(prompt_valid_mel.shape[1])
        prompt_mel = F.pad(
            prompt_valid_mel,
            (0, 0, 0, prompt_lengths.padded_mel_frames - prompt_valid_mel.shape[1]),
        )
        target_note_off = last_note_off(target_midi_path)
        target_lengths = lengths_from_last_note(
            target_note_off, max_release_duration=max_release_duration,
            target_duration=target_duration,
        )
        validate_inference_lengths(
            prompt_lengths, target_lengths,
            max_prompt_duration=inference["max_prompt_duration"],
            max_target_duration=inference["max_target_duration"],
            max_composer_tokens=inference["max_composer_tokens"],
        )
        prompt_roll = midi_to_pianoroll(
            prompt_midi_path, start_sec=0.0,
            duration_sec=prompt_lengths.blocks * 0.2,
            frames=prompt_lengths.midi_frames,
        )
        target_roll = midi_to_pianoroll(
            target_midi_path, start_sec=0.0,
            duration_sec=target_lengths.blocks * 0.2,
            frames=target_lengths.midi_frames,
        )
        pianoroll = torch.from_numpy(np.concatenate((prompt_roll, target_roll), axis=1))
        pianoroll = pianoroll.float()[None].to(torch_device)
        model = _load_model(config, checkpoint_path, torch_device)
        clap_encoder = FrozenCLAPEncoder(
            resolve_project_path(config["external"]["clap_checkpoint"]),
            device=torch_device,
        ).eval()
        clap_embedding = clap_encoder(waveform, sample_rate=sample_rate, crop_mode="last")
        model_dtype = next(model.parameters()).dtype
        generated_mel = model.generate_target_mel(
            pianoroll=pianoroll.to(model_dtype),
            prompt_mel=prompt_mel.to(model_dtype),
            prompt_mel_length=prompt_valid_mel.shape[1],
            target_lengths=target_lengths,
            clap_embedding=clap_embedding.to(model_dtype),
            steps=steps, schedule=schedule, cfg_scale=cfg_scale,
            generator=generator,
        )
        vocos = Vocos()
        load_vocos_checkpoint(
            vocos, resolve_project_path(config["external"]["vocos_checkpoint"]),
        )
        vocos.eval().to(torch_device)
        output_audio = vocos(generated_mel.float().transpose(1, 2))[0, 0].float()
        output_audio = peak_normalize(output_audio, 0.98).cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        sf.write(stream, output_audio, sample_rate, format="WAV", subtype="PCM_16")
    return output_path
