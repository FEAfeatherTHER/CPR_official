"""Construct the inference-only Composer–Performer from saved architecture."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any

from models.cpr.aggregator import MelPatchAggregator
from models.cpr.composer import Composer
from models.backbone.dit import DiT
from models.cpr.model import ComposerPerformer
from models.cpr.performer import Performer
from models.cpr.pianoroll_encoder import PianorollEncoder
from models.backbone.qwen_composer import ContinuousQwenComposer


_ARCHITECTURE_FIELDS = (
    "mel_dim", "clap_dim", "history_patches", "performer_clap_conditioning",
    "composer_rope", "aggregator", "dit",
)


def validate_composer_rope_config(config: dict[str, Any]) -> dict[str, float]:
    try:
        rope = config["model"]["composer_rope"]
    except KeyError as error:
        raise ValueError("model.composer_rope is required") from error
    if not isinstance(rope, dict) or rope.get("type") != "time_modality_2d":
        raise ValueError("model.composer_rope.type must be time_modality_2d")
    if rope.get("layout") != "interleaved":
        raise ValueError("model.composer_rope.layout must be interleaved")
    expected = {"midi_time_scale": 0.4, "midi_modality_id": 0.0,
                "audio_modality_id": 1.0, "clap_modality_id": 1.0}
    values = {}
    for field, expected_value in expected.items():
        value = rope.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"model.composer_rope.{field} must be numeric")
        if float(value) != expected_value:
            raise ValueError(f"model.composer_rope.{field} must equal {expected_value:g}")
        values[field] = float(value)
    return values


def validate_checkpoint_model_config(current: dict[str, Any], saved: dict[str, Any]) -> None:
    try:
        current_model, saved_model = current["model"], saved["model"]
    except KeyError as error:
        raise ValueError("checkpoint config is missing the model section") from error
    for field in _ARCHITECTURE_FIELDS:
        if field not in current_model or field not in saved_model:
            raise ValueError(f"model.{field} is required in runtime and checkpoint configs")
        if current_model[field] != saved_model[field]:
            raise ValueError(
                f"checkpoint model.{field} mismatch: current={current_model[field]!r}, "
                f"saved={saved_model[field]!r}"
            )


def build_composer_performer(
    config: dict[str, Any], *, saved_qwen_config: dict[str, Any],
) -> ComposerPerformer:
    model_config = config["model"]
    rope = validate_composer_rope_config(config)
    history = model_config.get("history_patches")
    if isinstance(history, bool) or not isinstance(history, Integral) or not 1 <= history <= 5:
        raise ValueError("history_patches must be an integer between 1 and 5")
    if model_config.get("performer_clap_conditioning") is not True:
        raise ValueError("model.performer_clap_conditioning must be true")
    qwen = ContinuousQwenComposer.from_config(saved_qwen_config)
    hidden = qwen.config.hidden_size
    aggregator_config = deepcopy(model_config["aggregator"])
    if aggregator_config["hidden_size"] != hidden:
        raise ValueError("Aggregator hidden_size must equal Qwen hidden_size")
    aggregator = MelPatchAggregator(**aggregator_config)
    composer = Composer(
        qwen, aggregator, PianorollEncoder(hidden_size=hidden),
        clap_dim=model_config["clap_dim"], **rope,
    )
    dit_config = deepcopy(model_config["dit"])
    performer = Performer(
        dit=DiT(**dit_config), mel_dim=model_config["mel_dim"], composer_dim=hidden,
        hidden_size=dit_config["dim"], clap_dim=model_config["clap_dim"],
        clap_conditioning=True, patch_size=aggregator_config["patch_size"],
        history_patches=int(history),
    )
    return ComposerPerformer(composer, performer)
