"""Inference-only Composer–Performer runtime."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.dataset.alignment import AlignmentLengths
from models.cpr.composer import Composer
from models.base.ode_sampler import OdeSchedule, cfg_combine, euler_sample_patch
from models.cpr.performer import Performer


def select_prompt_history(
    prompt_mel: torch.Tensor,
    prompt_hidden: torch.Tensor,
    *,
    valid_frames: int,
    history_patches: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the last valid prompt patches and preserve a partial-tail mask."""
    if prompt_mel.ndim != 3 or prompt_mel.shape[0] != 1:
        raise ValueError("prompt_mel must have shape [1,T,C]")
    if prompt_hidden.ndim != 3 or prompt_hidden.shape[0] != 1:
        raise ValueError("prompt_hidden must have shape [1,K,D]")
    if not 1 <= int(history_patches) <= 5:
        raise ValueError("history_patches must be between 1 and 5")
    if not 0 < valid_frames <= prompt_mel.shape[1]:
        raise ValueError("valid prompt frames must be inside the prompt tensor")
    valid_patches = (int(valid_frames) + patch_size - 1) // patch_size
    if valid_patches < history_patches or prompt_hidden.shape[1] < history_patches:
        raise ValueError(
            f"history_patches={history_patches} requires {history_patches} prompt patches, "
            f"but only {min(valid_patches, prompt_hidden.shape[1])} are available"
        )
    start_patch = valid_patches - history_patches
    start_frame = start_patch * patch_size
    end_frame = valid_patches * patch_size
    if end_frame > prompt_mel.shape[1]:
        raise ValueError("prompt_mel must be padded to a complete patch")
    history_mel = prompt_mel[:, start_frame:end_frame]
    history_hidden = prompt_hidden[:, -history_patches:]
    absolute_frames = torch.arange(start_frame, end_frame, device=prompt_mel.device)
    history_mask = (absolute_frames < valid_frames)[None]
    return history_mel, history_hidden, history_mask


class ComposerPerformer(nn.Module):
    """Generate normalized target Mel frames from prompt context and target MIDI."""

    def __init__(
        self,
        composer: Composer,
        performer: Performer,
    ) -> None:
        super().__init__()
        self.composer = composer
        self.performer = performer

    @property
    def history_patches(self) -> int:
        return self.performer.history_patches

    @torch.inference_mode()
    def generate_target_mel(
        self,
        *,
        pianoroll: torch.Tensor,
        prompt_mel: torch.Tensor,
        prompt_mel_length: int,
        target_lengths: AlignmentLengths,
        clap_embedding: torch.Tensor,
        steps: int = 25,
        schedule: OdeSchedule = "uniform",
        cfg_scale: float = 2.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if pianoroll.shape[0] != 1:
            raise ValueError("inference supports batch size 1")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be positive")
        if not isinstance(cfg_scale, (int, float)) or not math.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite")
        if cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")
        current_hidden, composer_state = self.composer.prefill(
            pianoroll=pianoroll,
            prompt_mel=prompt_mel,
            prompt_mel_length=prompt_mel_length,
            clap_embedding=clap_embedding,
        )
        patch_size = self.performer.patch_size
        history, history_hidden, history_mask = select_prompt_history(
            prompt_mel,
            composer_state.prompt_hidden,
            valid_frames=prompt_mel_length,
            history_patches=self.history_patches,
            patch_size=patch_size,
        )
        noises = torch.randn(
            (1, target_lengths.patch_frames, patch_size, history.shape[-1]),
            device=history.device,
            dtype=history.dtype,
            generator=generator,
        )
        generated: list[torch.Tensor] = []
        for patch_index in range(target_lengths.patch_frames):
            hidden_states = torch.cat((history_hidden, current_hidden), dim=1)
            current_noise = noises[:, patch_index]

            def velocity(
                clean_history: torch.Tensor,
                current: torch.Tensor,
                time: torch.Tensor,
            ) -> torch.Tensor:
                branch_mask = torch.cat(
                    (history_mask, torch.ones((1, patch_size), dtype=torch.bool,
                                              device=history.device)), dim=1,
                ).repeat(2, 1)
                performer_inputs = dict(
                    clean_history=clean_history.repeat(2, 1, 1),
                    noisy_current=current.repeat(2, 1, 1),
                    hidden_states=hidden_states.repeat(2, 1, 1),
                    t=time.repeat(2),
                    condition_drop=torch.tensor([False, True], dtype=torch.bool,
                                                device=history.device),
                    clap_drop=torch.tensor([False, True], dtype=torch.bool,
                                           device=history.device),
                    mask=branch_mask,
                )
                if self.performer.clap_conditioning:
                    performer_inputs["clap"] = clap_embedding.repeat(2, 1)
                prediction = self.performer(**performer_inputs)[
                    :, self.history_patches * patch_size:
                ]
                conditional, unconditional = prediction.chunk(2, dim=0)
                return cfg_combine(unconditional, conditional, cfg_scale)

            current = euler_sample_patch(
                velocity, history, current_noise, steps=steps, schedule=schedule,
            )
            generated.append(current)
            if patch_index + 1 < target_lengths.patch_frames:
                patch_token, _ = self.composer.aggregator(
                    current,
                    torch.ones((1, patch_size), dtype=torch.bool, device=current.device),
                )
                next_hidden, composer_state = self.composer.decode_patch(
                    patch_token[:, 0], composer_state,
                )
                history = torch.cat((history[:, patch_size:], current), dim=1)
                history_mask = torch.cat(
                    (history_mask[:, patch_size:],
                     torch.ones((1, patch_size), dtype=torch.bool, device=history.device)),
                    dim=1,
                )
                history_hidden = torch.cat(
                    (history_hidden[:, 1:], current_hidden), dim=1,
                )
                current_hidden = next_hidden
        mel = torch.cat(generated, dim=1)
        return mel[:, :target_lengths.valid_mel_frames]
