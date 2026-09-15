"""Independent Refiner: a 24 kHz waveform in, a 48 kHz waveform out."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio

from models.codec.refiner_model import FastLRMerge, VocosBWE
from models.utils.audio import peak_normalize


class Refiner(nn.Module):
    """LavaSR SFT model with reference-preserving low-frequency fusion."""

    def __init__(self, model: VocosBWE, *, device="cpu", cutoff=11025, transition_bins=4096):
        super().__init__()
        self.device = torch.device(device)
        if not 0 < cutoff < 24000 or not isinstance(transition_bins, int) or transition_bins < 2:
            raise ValueError("Invalid Refiner low-frequency fusion configuration")
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.merge = FastLRMerge(sample_rate=48000, cutoff=cutoff, transition_bins=transition_bins, device=self.device)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", cutoff=11025, transition_bins=4096):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Refiner checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = VocosBWE()
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"Refiner checkpoint mismatch: {error}") from error
        return cls(model, device=device, cutoff=cutoff, transition_bins=transition_bins)

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return [B, 2*T] audio without changing the original duration."""
        if audio.ndim == 1:
            audio = audio[None]
        if audio.ndim != 2 or min(audio.shape) == 0:
            raise ValueError("Input audio must have nonempty shape [T] or [B,T] at 24 kHz")
        if not torch.isfinite(audio).all():
            raise ValueError("Input audio contains non-finite values")
        audio = audio.to(device=self.device, dtype=torch.float32)
        target_frames = 2 * audio.shape[-1]
        padded_frames = max(960, ((audio.shape[-1] + 479) // 480) * 480)
        padded = F.pad(audio, (0, padded_frames - audio.shape[-1]))
        reference = torchaudio.functional.resample(padded, 24000, 48000)
        generated = self.model(reference)
        if generated.shape[-1] < reference.shape[-1]:
            raise RuntimeError("Refiner produced fewer frames than its padded input")
        output = self.merge(generated[..., :reference.shape[-1]], reference)[..., :target_frames]
        if not torch.isfinite(output).all():
            raise RuntimeError("Refiner output contains non-finite values")
        return output


def load_input(path) -> torch.Tensor:
    """Read 24 kHz audio, averaging channels without resampling."""
    values, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != 24000:
        raise ValueError(f"Refiner expects 24000 Hz (24 kHz) input, received {sample_rate} Hz")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Input audio must be nonempty and finite")
    return torch.from_numpy(values.mean(axis=1).copy())[None]


def normalize_audio(audio: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
    """Limit peaks without amplifying quiet input, matching CP output handling."""
    return peak_normalize(audio, peak)


def run_refiner(*, input_path, output_path, checkpoint_path, device="cuda", cutoff=11025, transition_bins=4096, peak=0.98) -> Path:
    input_path, output_path, checkpoint_path = map(Path, (input_path, output_path, checkpoint_path))
    if output_path.resolve() in {input_path.resolve(), checkpoint_path.resolve()}:
        raise ValueError("Output must be different from input and checkpoint paths")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}. Choose a new output path.")
    audio = load_input(input_path)
    model = Refiner.from_checkpoint(checkpoint_path, device=device, cutoff=cutoff, transition_bins=transition_bins)
    output = normalize_audio(model(audio), peak=peak)[0].cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing file even if it appeared during inference.
    with output_path.open("xb") as stream:
        sf.write(stream, output, 48000, format="WAV", subtype="PCM_24")
    return output_path
