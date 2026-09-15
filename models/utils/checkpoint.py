"""Strict safetensors loading for Composer–Performer weights."""

from __future__ import annotations

from pathlib import Path

from torch import nn


def validate_checkpoint_schema(model: nn.Module, checkpoint_path: str | Path) -> None:
    from safetensors import safe_open

    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as stream:
        actual = {key: tuple(stream.get_slice(key).get_shape()) for key in stream.keys()}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong_shape = sorted(
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    )
    if missing or unexpected or wrong_shape:
        raise ValueError(
            "checkpoint schema mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"wrong_shape={wrong_shape[:5]}"
        )


def load_composer_performer_checkpoint(
    model: nn.Module, checkpoint_path: str | Path,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Composer-Performer checkpoint not found: {checkpoint_path}")
    validate_checkpoint_schema(model, checkpoint_path)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    return model
