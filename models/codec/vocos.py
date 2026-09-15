# Copyright (c) 2023 Amphion.
# Adapted from P-MUSE/Amphion under the MIT license for this project.
"""The P-MUSE 24 kHz Vocos decoder and checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same") -> None:
        super().__init__()
        if padding not in {"center", "same"}:
            raise ValueError("padding must be 'center' or 'same'")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spectrum,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        pad = (self.win_length - self.hop_length) // 2
        _, _, frames = spectrum.shape
        inverse = torch.fft.irfft(spectrum, self.n_fft, dim=1, norm="backward")
        inverse = inverse * self.window[None, :, None]
        output_size = (frames - 1) * self.hop_length + self.win_length
        audio = F.fold(
            inverse,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        window_square = self.window.square().expand(1, frames, -1).transpose(1, 2)
        envelope = F.fold(
            window_square,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        return audio / envelope.clamp_min(1e-11)


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str) -> None:
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, n_fft, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude, phase = self.out(x).transpose(1, 2).chunk(2, dim=1)
        magnitude = magnitude.exp().clamp_max(1e2)
        return self.istft(magnitude * (phase.cos() + 1j * phase.sin()))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor, cond_embedding_id=None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        return residual + (self.gamma * x).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class Vocos(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 128,
        dim: int = 1024,
        intermediate_dim: int = 4096,
        num_layers: int = 30,
        n_fft: int = 1920,
        hop_size: int = 480,
        padding: str = "same",
    ) -> None:
        super().__init__()
        self.backbone = VocosBackbone(input_channels, dim, intermediate_dim, num_layers)
        self.head = ISTFTHead(dim, n_fft, hop_size, padding)

    def forward(self, normalized_mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(normalized_mel))[:, None]


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Vocos checkpoint not found: {path}")
    candidates = sorted(path.rglob("model.safetensors")) + sorted(path.rglob("pytorch_model.bin"))
    if not candidates:
        raise FileNotFoundError(f"Vocos checkpoint not found below: {path}")
    if len(candidates) > 1:
        raise ValueError(f"multiple Vocos checkpoints found below {path}; pass one file explicitly")
    return candidates[0]


def load_vocos_checkpoint(model: Vocos, path: str | Path) -> Vocos:
    checkpoint = _resolve_checkpoint(Path(path))
    validate_vocos_checkpoint_schema(model, checkpoint)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def validate_vocos_checkpoint_schema(model: Vocos, path: str | Path) -> None:
    checkpoint = _resolve_checkpoint(Path(path))
    if checkpoint.suffix != ".safetensors":
        return
    from safetensors import safe_open

    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as stream:
        actual = {
            key: tuple(stream.get_slice(key).get_shape())
            for key in stream.keys()
        }
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong_shape = sorted(
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    )
    if missing or unexpected or wrong_shape:
        raise ValueError(
            "Vocos checkpoint schema mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, wrong_shape={wrong_shape[:5]}"
        )
