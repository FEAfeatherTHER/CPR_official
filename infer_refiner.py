"""Refine one 24 kHz audio file into a 48 kHz audio file."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/refiner.json")
    parser.add_argument("--checkpoint", type=Path, help="Override the Refiner checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="24 kHz mono or stereo WAV")
    parser.add_argument("--output", type=Path, required=True, help="New 48 kHz mono WAV")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    args = parser.parse_args()
    try:
        with args.config.open() as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ValueError("Refiner configuration must be a JSON object")
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = Path(config["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
        from models.codec.refiner import run_refiner
        output = run_refiner(
            input_path=args.input, output_path=args.output, checkpoint_path=checkpoint,
            device=args.device, cutoff=config["cutoff_hz"], transition_bins=config["transition_bins"],
            peak=config["peak"],
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Refiner error: {error}\n")
    print(f"Saved 48 kHz Refiner audio: {output}")


if __name__ == "__main__":
    main()
