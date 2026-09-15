# Adapted from LavaSR by Yatharth Sharma (Apache-2.0; see licenses/Apache-2.0.txt).
# Source: https://github.com/ysharma3501/LavaSR/tree/33ac040892519c1bb4aed7eb32e79af51cc29e2a
# Vocos components: Copyright (c) 2023 Charactr Inc. (MIT; license text in LICENSE).
# CPR changes: SFT configuration, standalone inference, and inverse-STFT handling.
"""Standalone (inference-only) LavaSR SFT bandwidth-extension model.

Self-contained: depends ONLY on torch / torchaudio / numpy / librosa (no
AnyTrainer repo imports). Reproduces exactly the trained ``VocosBWE`` used for
the cascade Stage-2 SFT run, so the shipped checkpoint loads with strict=True.

Signal path (forward):
  wav @ input_sr(48k) --resample--> 24k
    --> MelVQGAN mel (torch.stft + librosa slaney filterbank + log-clip 1e-5)
    --> (mel - mel_mean) / sqrt(mel_var)                      # 50 Hz, 128-dim
    --> ConvTranspose1d x2 (50 Hz -> 100 Hz)
    --> VocosBackbone (ConvNeXt)
    --> ISTFTHead (n_fft=1920, hop=480, "same")               # -> wav @ 48k
"""
import math

import torch
import torch.nn as nn
import torchaudio
from librosa.filters import mel as librosa_mel_fn


# --- SFT architecture constants (must match the trained checkpoint) ----------
INPUT_SAMPLE_RATE = 48000
MEL_SAMPLE_RATE = 24000
FRAME_UPSAMPLE = 2
MEL_N_FFT = 1920
MEL_HOP = 480
MEL_WIN = 1920
N_MELS = 128
MEL_FMIN = 0
MEL_FMAX = 12000
MEL_MEAN = -4.92
MEL_VAR = 8.14
BB_DIM = 512
BB_INTERMEDIATE = 1536
BB_LAYERS = 8
HEAD_N_FFT = 1920
HEAD_HOP = 480


def safe_log(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clip(x, min=clip_val))


# =============================================================================
# Mel front-end (flow_sr / Stage-1-faithful MelVQGAN)
# =============================================================================
class MelSpectrogram(nn.Module):
    """torch.stft + librosa slaney filterbank + log-clip(1e-5). Matches the
    repo ``models.codec.melvqgan.melspec.MelSpectrogram`` forward exactly."""

    def __init__(self, n_fft, num_mels, sampling_rate, hop_size, win_size,
                 fmin, fmax, center=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("hann_window", torch.hann_window(win_size))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode="reflect",
        ).squeeze(1)
        spec = torch.stft(
            y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=self.hann_window, center=self.center, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        return safe_log(spec)


class MelVQGANFeatures(nn.Module):
    """MelVQGAN mel + per-corpus mean/var normalization."""

    def __init__(self, sample_rate=MEL_SAMPLE_RATE, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
                 win_length=MEL_WIN, n_mels=N_MELS, f_min=MEL_FMIN, f_max=MEL_FMAX,
                 mel_mean=MEL_MEAN, mel_var=MEL_VAR):
        super().__init__()
        self.mel = MelSpectrogram(
            n_fft=n_fft, num_mels=n_mels, sampling_rate=sample_rate,
            hop_size=hop_length, win_size=win_length, fmin=f_min, fmax=f_max,
        )
        self.mel_mean = float(mel_mean)
        self.mel_var = float(mel_var)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel(audio)
        return (mel - self.mel_mean) / math.sqrt(self.mel_var)


# =============================================================================
# ISTFT head (from repo vocos.py)
# =============================================================================
class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length,
                               self.window, center=True)
        pad = (self.win_length - self.hop_length) // 2
        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]
        return y / window_envelope.clamp_min(1e-11)


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        S = mag * (torch.cos(p) + 1j * torch.sin(p))
        return self.istft(S)


# =============================================================================
# ConvNeXt backbone (from repo vocos.py, non-adanorm path)
# =============================================================================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, num_layers: int):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = 1 / num_layers
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(dim=dim, intermediate_dim=intermediate_dim,
                          layer_scale_init_value=layer_scale_init_value)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


# =============================================================================
# Full generator
# =============================================================================
class VocosBWE(nn.Module):
    """Inference-only LavaSR SFT generator (fixed to the SFT architecture)."""

    def __init__(self):
        super().__init__()
        self.input_sample_rate = INPUT_SAMPLE_RATE
        self.mel_sample_rate = MEL_SAMPLE_RATE
        self.frame_upsample = FRAME_UPSAMPLE

        self.feature_extractor = MelVQGANFeatures()
        self.mel_upsampler = nn.ConvTranspose1d(
            N_MELS, N_MELS, kernel_size=2 * FRAME_UPSAMPLE,
            stride=FRAME_UPSAMPLE, padding=FRAME_UPSAMPLE // 2,
        )
        self.backbone = VocosBackbone(
            input_channels=N_MELS, dim=BB_DIM,
            intermediate_dim=BB_INTERMEDIATE, num_layers=BB_LAYERS,
        )
        self.head = ISTFTHead(dim=BB_DIM, n_fft=HEAD_N_FFT, hop_length=HEAD_HOP, padding="same")

    def forward_mel(self, mel: torch.Tensor) -> torch.Tensor:
        mel = self.mel_upsampler(mel)
        x = self.backbone(mel)
        return self.head(x)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) at ``input_sample_rate`` (48 kHz) -> (B, T) at 48 kHz."""
        if self.input_sample_rate != self.mel_sample_rate:
            wav = torchaudio.functional.resample(wav, self.input_sample_rate, self.mel_sample_rate)
        feats = self.feature_extractor(wav)
        return self.forward_mel(feats)

    @torch.no_grad()
    def load_checkpoint(self, path: str, strict: bool = True):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        return missing, unexpected


# =============================================================================
# Linkwitz-Riley merge (low band from input, highs from model)
# =============================================================================
class FastLRMerge:
    """Merge low freq from ``b`` (input) and high freq from ``a`` (model)."""

    def __init__(self, sample_rate=48000, cutoff=4000, transition_bins=256, device="cpu"):
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.transition_bins = transition_bins
        self.device = device
        self.mask_cache = {}
        x = torch.linspace(-1, 1, steps=transition_bins, device=device)
        t = (x + 1) / 2
        self.fade_template = (3 * t ** 2 - 2 * t ** 3).to(torch.complex64)

    def _get_mask(self, n_bins, ndim):
        key = (n_bins, ndim)
        if key in self.mask_cache:
            return self.mask_cache[key]
        cutoff_bin = int((self.cutoff / (self.sample_rate / 2)) * n_bins)
        mask = torch.ones(n_bins, device=self.device, dtype=torch.complex64)
        half = self.transition_bins // 2
        start = max(0, cutoff_bin - half)
        end = min(n_bins, cutoff_bin + half)
        fade = self.fade_template[: end - start]
        mask[:start] = 0
        mask[start:end] = fade
        mask[end:] = 1
        for _ in range(ndim - 1):
            mask = mask.unsqueeze(0)
        self.mask_cache[key] = mask
        return mask

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        spec1 = torch.fft.rfft(a, dim=-1)
        spec2 = torch.fft.rfft(b, dim=-1)
        mask = self._get_mask(spec1.size(-1), spec1.ndim)
        spec2 = spec2 + (spec1 - spec2) * mask
        return torch.fft.irfft(spec2, n=a.size(-1), dim=-1)
