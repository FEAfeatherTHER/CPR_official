"""Clean-history local conditional-flow performer for Piano Renderer v1.1.8."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn

from models.backbone.dit import DiT


def _validate_drop_mask(mask: torch.Tensor, batch: int, device, name: str) -> None:
    if mask.shape != (batch,) or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape [B]")
    if mask.device != torch.device(device):
        raise ValueError(f"{name} must be on the acoustic tensor device")


def _validate_history_patches(history_patches: int) -> int:
    if isinstance(history_patches, bool) or not isinstance(history_patches, Integral):
        raise ValueError("history_patches must be an integer between 1 and 5")
    history_patches = int(history_patches)
    if not 1 <= history_patches <= 5:
        raise ValueError("history_patches must be between 1 and 5")
    return history_patches


def expand_patch_hidden(hidden_states: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """Copy each patch state to its Mel frames without temporal interpolation."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B,K,D]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    return hidden_states.repeat_interleave(int(patch_size), dim=1)


def build_conditioned_input(
    clean_history: torch.Tensor,
    noisy_current: torch.Tensor,
    hidden_states: torch.Tensor,
    clap: torch.Tensor | None,
    *,
    clap_conditioning: bool,
    zero_clap: torch.Tensor | None,
    condition_drop: torch.Tensor | None,
    patch_size: int,
    clap_drop: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate acoustic and Composer conditions, with optional direct CLAP."""
    if clean_history.ndim != 3 or noisy_current.ndim != 3:
        raise ValueError("acoustic tensors must have shape [B,T,C]")
    if clean_history.shape[0] != noisy_current.shape[0]:
        raise ValueError("history and current batch sizes must match")
    if clean_history.shape[2] != noisy_current.shape[2]:
        raise ValueError("history and current Mel dimensions must match")
    if noisy_current.shape[1] != patch_size or clean_history.shape[1] % patch_size:
        raise ValueError("current must be one patch and history must contain complete patches")
    history_patches = clean_history.shape[1] // patch_size
    expected_hidden = history_patches + 1
    if hidden_states.shape[:2] != (clean_history.shape[0], expected_hidden):
        raise ValueError("hidden_states must contain history plus current patch states")
    if not isinstance(clap_conditioning, bool):
        raise ValueError("clap_conditioning must be a boolean")
    if clap_conditioning:
        if clap is None or clap.ndim != 2 or clap.shape[0] != clean_history.shape[0]:
            raise ValueError("clap must have shape [B,C] when CLAP is enabled")
        if zero_clap is None or zero_clap.shape != clap.shape[1:]:
            raise ValueError("zero_clap must have shape [C] when CLAP is enabled")
    elif clap is not None or zero_clap is not None:
        raise ValueError("clap must be omitted when Performer CLAP is disabled")
    batch = clean_history.shape[0]
    if condition_drop is None:
        condition_drop = torch.zeros(batch, dtype=torch.bool, device=clean_history.device)
    _validate_drop_mask(condition_drop, batch, clean_history.device, "condition_drop")
    if clap_drop is None:
        clap_drop = condition_drop
    _validate_drop_mask(clap_drop, batch, clean_history.device, "clap_drop")
    if torch.any(condition_drop & ~clap_drop):
        raise ValueError("dropping hidden states requires dropping direct CLAP")

    acoustic = torch.cat((clean_history, noisy_current), dim=1)
    hidden = expand_patch_hidden(hidden_states, patch_size=patch_size)
    hidden = hidden.masked_fill(condition_drop[:, None, None], 0.0)
    if not clap_conditioning:
        return torch.cat((acoustic, hidden), dim=-1)
    assert clap is not None
    assert zero_clap is not None
    clap = torch.where(clap_drop[:, None], zero_clap[None].to(clap.dtype), clap)
    clap = clap[:, None].expand(-1, acoustic.shape[1], -1)
    return torch.cat((acoustic, hidden, clap), dim=-1)


class Performer(nn.Module):
    def __init__(
        self,
        *,
        dit: DiT,
        mel_dim: int = 128,
        composer_dim: int = 1024,
        hidden_size: int = 1024,
        clap_dim: int = 512,
        clap_conditioning: bool = True,
        patch_size: int = 5,
        history_patches: int = 2,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.mel_dim = int(mel_dim)
        self.patch_size = int(patch_size)
        self.history_patches = _validate_history_patches(history_patches)
        if not isinstance(clap_conditioning, bool):
            raise ValueError("clap_conditioning must be a boolean")
        self.clap_conditioning = clap_conditioning
        self.input_projection = nn.Linear(
            mel_dim + composer_dim + (clap_dim if clap_conditioning else 0),
            hidden_size,
            bias=False,
        )
        if clap_conditioning:
            self.zero_clap = nn.Parameter(torch.zeros(clap_dim))
        else:
            self.register_parameter("zero_clap", None)

    def forward(
        self,
        *,
        clean_history: torch.Tensor,
        noisy_current: torch.Tensor,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        clap: torch.Tensor | None = None,
        condition_drop: torch.Tensor | None = None,
        clap_drop: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clean_history.shape[1] != self.history_patches * self.patch_size:
            raise ValueError("clean_history length does not match history_patches")
        conditioned = build_conditioned_input(
            clean_history,
            noisy_current,
            hidden_states,
            clap,
            clap_conditioning=self.clap_conditioning,
            zero_clap=self.zero_clap,
            condition_drop=condition_drop,
            clap_drop=clap_drop,
            patch_size=self.patch_size,
        )
        return self.dit(self.input_projection(conditioned), t, mask=mask)
