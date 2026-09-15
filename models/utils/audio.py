"""Audio tensor utilities."""

import math
from numbers import Real

import torch


_FLOAT8_DTYPES = frozenset(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if (dtype := getattr(torch, name, None)) is not None
)


def peak_normalize(audio: torch.Tensor, target_peak: float = 0.98) -> torch.Tensor:
    """Attenuate audio so its global peak does not exceed ``target_peak``.

    Audio already at or below the target is returned unchanged.  A single gain
    is applied across all samples and channels, and no clipping is performed.
    """
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")
    if not audio.is_floating_point():
        raise TypeError("audio must have a floating-point dtype")
    if audio.dtype in _FLOAT8_DTYPES:
        raise TypeError("float8 audio is not supported")
    if audio.numel() == 0:
        raise ValueError("audio must be non-empty")
    peak = audio.abs().amax()
    if not bool(torch.isfinite(peak)):
        raise ValueError("audio must contain only finite values")
    if isinstance(target_peak, bool) or not isinstance(target_peak, Real):
        raise TypeError("target_peak must be a real number")
    target_peak = float(target_peak)
    if not math.isfinite(target_peak) or not 0 < target_peak < 1:
        raise ValueError("target_peak must satisfy 0 < target_peak < 1")

    normalization_threshold = torch.tensor(
        target_peak,
        dtype=audio.dtype,
        device="cpu",
    )
    if float(normalization_threshold) > target_peak:
        normalization_threshold = torch.nextafter(
            normalization_threshold,
            torch.tensor(float("-inf"), dtype=audio.dtype, device="cpu"),
        )
    if float(normalization_threshold) == 0.0:
        safe_target = normalization_threshold
    else:
        safe_target = torch.nextafter(
            normalization_threshold,
            torch.tensor(float("-inf"), dtype=audio.dtype, device="cpu"),
        )
    normalization_threshold = normalization_threshold.to(device=audio.device)
    safe_target = safe_target.to(device=audio.device)

    attenuation_gain = safe_target.to(torch.float64) / peak.to(torch.float64)
    gain = torch.where(
        peak > normalization_threshold,
        attenuation_gain,
        torch.ones_like(attenuation_gain),
    )
    dtype_gain = gain.to(dtype=audio.dtype)
    needs_gain_margin = (peak > normalization_threshold) & (
        peak * dtype_gain > safe_target
    )
    gain = torch.where(
        needs_gain_margin,
        torch.nextafter(
            dtype_gain,
            torch.tensor(float("-inf"), dtype=audio.dtype, device=audio.device),
        ).to(torch.float64),
        gain,
    )
    return audio * gain
