"""CLAP-conditioned Composer input construction and Qwen orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.dataset.alignment import length_mask
from models.cpr.aggregator import MelPatchAggregator
from models.cpr.pianoroll_encoder import PianorollEncoder
from models.backbone.qwen_composer import ContinuousQwenComposer


def build_composer_attention_mask(
    midi_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Build the four-region mask for ``[CLAP, MIDI, prompt, target]``."""
    if not (midi_lengths.shape == prompt_lengths.shape == target_lengths.shape):
        raise ValueError("all length tensors must have the same shape")
    batch = midi_lengths.numel()
    max_midi = int(midi_lengths.max())
    max_prompt = int(prompt_lengths.max())
    max_target = int(target_lengths.max())
    prompt_start = 1 + max_midi
    target_start = prompt_start + max_prompt
    total = target_start + max_target
    output = torch.zeros((batch, total, total), dtype=torch.bool, device=midi_lengths.device)
    output[:, 0, 0] = True
    for index in range(batch):
        midi = int(midi_lengths[index])
        prompt = int(prompt_lengths[index])
        target = int(target_lengths[index])
        midi_end = 1 + midi
        prompt_end = prompt_start + prompt
        target_end = target_start + target
        output[index, 1:midi_end, 1:midi_end] = True
        output[index, prompt_start:prompt_end, 0] = True
        output[index, prompt_start:prompt_end, 1:midi_end] = True
        output[index, prompt_start:prompt_end, prompt_start:prompt_end] = True
        output[index, target_start:target_end, 0] = True
        output[index, target_start:target_end, 1:midi_end] = True
        output[index, target_start:target_end, prompt_start:prompt_end] = True
        output[index, target_start:target_end, target_start:target_end] = torch.tril(
            torch.ones((target, target), dtype=torch.bool, device=output.device)
        )
    return output


def build_position_ids(
    midi_frames: int,
    prompt_patches: int,
    target_patches: int,
    *,
    midi_time_scale: float,
    midi_modality_id: float,
    audio_modality_id: float,
    clap_modality_id: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    clap_time = torch.zeros(1, dtype=torch.float32, device=device)
    midi_time = (
        torch.arange(midi_frames, dtype=torch.float32, device=device)
        * midi_time_scale
    )
    prompt_time = torch.arange(
        prompt_patches, dtype=torch.float32, device=device
    )
    target_time = torch.arange(
        prompt_patches,
        prompt_patches + target_patches,
        dtype=torch.float32,
        device=device,
    )
    time = torch.cat((clap_time, midi_time, prompt_time, target_time))
    modality = torch.cat((
        torch.full((1,), clap_modality_id, dtype=torch.float32, device=device),
        torch.full(
            (midi_frames,),
            midi_modality_id,
            dtype=torch.float32,
            device=device,
        ),
        torch.full(
            (prompt_patches + target_patches,),
            audio_modality_id,
            dtype=torch.float32,
            device=device,
        ),
    ))
    return torch.stack((time, modality), dim=0)


@dataclass
class ComposerCache:
    cache: Any
    next_cache_position: int
    next_patch_position: float
    prompt_hidden: torch.Tensor


class Composer(nn.Module):
    def __init__(
        self,
        backbone: ContinuousQwenComposer,
        aggregator: MelPatchAggregator,
        pianoroll_encoder: PianorollEncoder,
        *,
        clap_dim: int = 512,
        midi_time_scale: float = 0.4,
        midi_modality_id: float = 0.0,
        audio_modality_id: float = 1.0,
        clap_modality_id: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.pianoroll_encoder = pianoroll_encoder
        hidden_size = backbone.config.hidden_size
        self.clap_token_projection = nn.Linear(clap_dim, hidden_size)
        self.midi_time_scale = float(midi_time_scale)
        self.midi_modality_id = float(midi_modality_id)
        self.audio_modality_id = float(audio_modality_id)
        self.clap_modality_id = float(clap_modality_id)
        self.target_bos = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.target_bos, std=0.02)

    def prefill(
        self,
        *,
        pianoroll: torch.Tensor,
        prompt_mel: torch.Tensor,
        prompt_mel_length: int,
        clap_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, ComposerCache]:
        if pianoroll.shape[0] != 1 or prompt_mel.shape[0] != 1:
            raise ValueError("autoregressive inference currently supports batch size 1")
        midi = self.pianoroll_encoder(pianoroll)
        prompt_frame_mask = length_mask(
            torch.tensor([prompt_mel_length], device=prompt_mel.device), prompt_mel.shape[1]
        )
        prompt, valid_prompt = self.aggregator(prompt_mel, prompt_frame_mask)
        prompt_count = prompt.shape[1]
        valid_prompt_count = int(valid_prompt.sum())
        if valid_prompt_count < 1:
            raise ValueError("prompt must contain at least one valid patch")
        clap = self.clap_token_projection(clap_embedding)[:, None]
        bos = self.target_bos[None, None].expand(1, 1, -1)
        inputs = torch.cat((clap, midi, prompt, bos), dim=1)
        midi_lengths = torch.tensor([midi.shape[1]], device=inputs.device)
        attention = build_composer_attention_mask(
            midi_lengths,
            torch.tensor([prompt_count], device=inputs.device),
            torch.tensor([1], device=inputs.device),
        )
        positions = build_position_ids(
            midi.shape[1],
            prompt_count,
            1,
            midi_time_scale=self.midi_time_scale,
            midi_modality_id=self.midi_modality_id,
            audio_modality_id=self.audio_modality_id,
            clap_modality_id=self.clap_modality_id,
            device=inputs.device,
        )[:, None, :]
        hidden, cache = self.backbone.prefill(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
        )
        prompt_start = 1 + midi.shape[1]
        prompt_hidden = hidden[:, prompt_start:prompt_start + valid_prompt_count]
        return hidden[:, -1:], ComposerCache(
            cache=cache,
            next_cache_position=inputs.shape[1],
            next_patch_position=float(prompt_count + 1),
            prompt_hidden=prompt_hidden,
        )

    def decode_patch(
        self,
        patch_token: torch.Tensor,
        state: ComposerCache,
    ) -> tuple[torch.Tensor, ComposerCache]:
        if patch_token.ndim == 2:
            patch_token = patch_token[:, None]
        total_keys = state.next_cache_position + 1
        attention = torch.ones((1, 1, total_keys), dtype=torch.bool, device=patch_token.device)
        position = torch.tensor(
            [
                [[state.next_patch_position]],
                [[self.audio_modality_id]],
            ],
            dtype=torch.float32,
            device=patch_token.device,
        )
        hidden, cache = self.backbone.decode_step(
            input_embed=patch_token,
            attention_mask=attention,
            position_id=position,
            cache=state.cache,
            cache_position=torch.tensor([state.next_cache_position], device=patch_token.device),
        )
        return hidden, ComposerCache(
            cache=cache,
            next_cache_position=state.next_cache_position + 1,
            next_patch_position=state.next_patch_position + 1.0,
            prompt_hidden=state.prompt_hidden,
        )
