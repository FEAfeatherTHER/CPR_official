#!/usr/bin/env python3
"""Export standalone inference assets from trusted training checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch


CP_WEIGHTS_NAME = "composer_performer.safetensors"
CP_CONFIG_NAME = "composer_performer_config.json"


def _refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing destination: {existing[0]}")


def _cpu_tensor_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model state must map string keys to tensors")
        output[key] = value.detach().cpu().contiguous()
    return output


def _inference_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "qwen_config" not in config:
        raise ValueError("source checkpoint config does not contain qwen_config")
    if "model" not in config:
        raise ValueError("source checkpoint config does not contain model")
    fields = ("mel_dim", "clap_dim", "history_patches",
              "performer_clap_conditioning", "composer_rope", "aggregator", "dit")
    model = config["model"]
    missing = [field for field in fields if field not in model]
    if missing:
        raise ValueError(f"source model config is missing inference fields: {missing}")
    inference_model = {field: deepcopy(model[field]) for field in fields}
    qwen_config = deepcopy(config["qwen_config"])
    qwen_config.pop("_name_or_path", None)
    qwen_config.pop("name_or_path", None)
    output = {"model": inference_model, "qwen_config": qwen_config}
    if "data" in config:
        data_fields = ("sample_rate", "mel_fps", "midi_fps")
        output["data"] = {
            field: deepcopy(config["data"][field]) for field in data_fields
            if field in config["data"]
        }
    if "inference" in config:
        output["inference"] = deepcopy(config["inference"])
    return output


def export_composer_performer_state(
    source: Mapping[str, Any], output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    weights_path = output_dir / CP_WEIGHTS_NAME
    config_path = output_dir / CP_CONFIG_NAME
    _refuse_existing(weights_path, config_path)
    if "model" not in source or "config" not in source:
        raise ValueError("CP source must contain model and config")
    state = _cpu_tensor_state(source["model"])
    filtered = {key: value for key, value in state.items()
                if not key.startswith("repa_head.")}
    dropped = set(state) - set(filtered)
    if any(not key.startswith("repa_head.") for key in dropped):
        raise RuntimeError("export attempted to drop a non-REPA model key")
    if not filtered:
        raise ValueError("CP source contains no inference model tensors")
    config = _inference_config(source["config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(filtered, str(weights_path))
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return weights_path, config_path


def copy_asset(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"asset source not found: {source}")
    _refuse_existing(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _load_trusted(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {path}")
    # The CLI contract explicitly accepts trusted legacy training checkpoints.
    # Loading once avoids retaining a second multi-gigabyte partial deserialization
    # through an exception traceback when weights_only rejects legacy metadata.
    return torch.load(path, map_location="cpu", weights_only=False)


def export_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    destinations = (
        output_dir / CP_WEIGHTS_NAME, output_dir / CP_CONFIG_NAME,
        output_dir / "vocos.safetensors",
        output_dir / "refiner.bin",
    )
    _refuse_existing(*destinations)
    export_composer_performer_state(_load_trusted(Path(args.composer_performer_source)),
                                    output_dir)
    copy_asset(args.vocos_source, output_dir / "vocos.safetensors")
    copy_asset(args.refiner_source, output_dir / "refiner.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composer-performer-source", required=True)
    parser.add_argument("--vocos-source", required=True)
    parser.add_argument("--refiner-source", required=True)
    parser.add_argument("--output-dir", default="checkpoints")
    export_all(parser.parse_args())


if __name__ == "__main__":
    main()
