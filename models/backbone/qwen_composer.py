# Qwen3 dependency: Copyright 2025 The Qwen team, Alibaba Group and the
# HuggingFace Inc. team. All rights reserved. Upstream license: Apache-2.0.
# Source: https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py
# CPR adds continuous audio/MIDI conditioning and time/modality rotary positions.
"""Qwen3 Composer with interleaved time/modality rotary positions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model


class TimeModalityRotaryEmbedding(nn.Module):
    """Split Qwen rotary pairs equally between time and modality axes."""

    def __init__(self, base_rotary: nn.Module) -> None:
        super().__init__()
        inv_freq = base_rotary.inv_freq.detach().clone()
        if inv_freq.numel() % 2:
            raise ValueError(
                "Qwen RoPE head dimension must be divisible by 4 for an equal "
                "time/modality split"
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = base_rotary.attention_scaling

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 3 or position_ids.shape[0] != 2:
            raise ValueError(
                "time/modality position_ids must have shape [2,B,S]"
            )
        if position_ids.shape[1:] != x.shape[:2]:
            raise ValueError(
                "time/modality position_ids batch and sequence dimensions "
                "must match inputs_embeds"
            )

        coordinates = position_ids.to(device=x.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        all_freqs = coordinates[..., None] * inv_freq[None, None, None, :]
        pair_indices = torch.arange(inv_freq.numel(), device=x.device)
        modality_pairs = (pair_indices % 2).bool()[None, None, :]
        freqs = torch.where(modality_pairs, all_freqs[1], all_freqs[0])
        embedding = torch.cat((freqs, freqs), dim=-1)

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            cos = embedding.cos() * self.attention_scaling
            sin = embedding.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ContinuousQwenComposer(nn.Module):
    """Qwen3 trunk receiving continuous embeddings and dual-axis positions."""

    def __init__(self, model: Qwen3Model) -> None:
        super().__init__()
        model.rotary_emb = TimeModalityRotaryEmbedding(model.rotary_emb)
        self.model = model
        self.model.embed_tokens = None

    @classmethod
    def from_config(
        cls,
        config: Qwen3Config | dict,
    ) -> "ContinuousQwenComposer":
        if isinstance(config, dict):
            config = Qwen3Config.from_dict(config)
        config._attn_implementation = "sdpa"
        return cls(Qwen3Model(config))

    @property
    def config(self) -> Qwen3Config:
        return self.model.config

    @staticmethod
    def _attention_dict(
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if mask.ndim != 3:
            raise ValueError("attention_mask must have shape [B,Q,K]")
        additive = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        additive.masked_fill_(~mask.bool(), torch.finfo(dtype).min)
        return {"full_attention": additive[:, None, :, :]}

    def _run(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
    ):
        if position_ids.ndim != 3 or position_ids.shape[0] != 2:
            raise ValueError("position_ids must have shape [2,B,S]")
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=self._attention_dict(
                attention_mask, inputs_embeds.dtype
            ),
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            return_dict=True,
        )

    def prefill(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        output = self._run(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            cache_position=torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ),
        )
        return output.last_hidden_state, output.past_key_values

    def decode_step(
        self,
        *,
        input_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        position_id: torch.Tensor,
        cache,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        if input_embed.shape[1] != 1:
            raise ValueError("decode_step accepts exactly one patch token")
        output = self._run(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
            position_ids=position_id,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        return output.last_hidden_state, output.past_key_values
