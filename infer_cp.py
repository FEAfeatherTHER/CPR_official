#!/usr/bin/env python3
"""Generate mono 24 kHz piano audio from prompt audio/MIDI and target MIDI."""

from __future__ import annotations

import argparse
import math


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def _finite_positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("steps must be positive")
    return number


def _finite_cfg(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("cfg-scale must be finite")
    if number < 0:
        raise argparse.ArgumentTypeError("cfg-scale must be non-negative")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/composer_performer.json")
    parser.add_argument("--checkpoint", default="checkpoints/composer_performer.safetensors")
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-midi", required=True)
    parser.add_argument("--target-midi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--target-duration", type=_finite_positive)
    parser.add_argument("--max-release-duration", type=_finite_nonnegative)
    parser.add_argument("--steps", type=_positive_int)
    parser.add_argument("--schedule", choices=("uniform", "cosine"))
    parser.add_argument("--cfg-scale", type=_finite_cfg)
    args = parser.parse_args()

    # This module is intentionally imported only after argument parsing so --help
    # needs neither torch, audio libraries, model assets, nor CUDA.
    from models.cpr.inference import (load_config, run_inference, validate_input_paths,
                               validate_output_path)

    try:
        validate_output_path(
            args.output, args.config, args.checkpoint, args.prompt_audio,
            args.prompt_midi, args.target_midi,
        )
        validate_input_paths(
            args.config, args.checkpoint, args.prompt_audio,
            args.prompt_midi, args.target_midi,
        )
        config = load_config(args.config)
        output = run_inference(
            config=config, checkpoint_path=args.checkpoint,
            prompt_audio_path=args.prompt_audio,
            prompt_midi_path=args.prompt_midi,
            target_midi_path=args.target_midi, output_path=args.output,
            device=args.device, seed=args.seed,
            target_duration=args.target_duration,
            max_release_duration=args.max_release_duration,
            steps=args.steps, schedule=args.schedule, cfg_scale=args.cfg_scale,
        )
        print(f"Saved: {output}")
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
