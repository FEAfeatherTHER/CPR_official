"""Behavioral tests for the independent 24 kHz to 48 kHz Refiner."""

import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import tempfile
import unittest
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / "models" / "codec" / "refiner.py").is_file(), "Refiner API is not implemented"
    return importlib.import_module("models.codec.refiner")


class RefinerTests(unittest.TestCase):
    def make_refiner(self):
        module = api()
        torch.set_num_threads(2)
        torch.manual_seed(17)
        return module.Refiner(module.VocosBWE().eval(), device="cpu")

    def test_refiner_preserves_duration_and_finite_samples(self):
        refiner = self.make_refiner()
        for length in [1, 719, 960, 1201, 2401]:
            with self.subTest(length=length):
                output = refiner(torch.linspace(-0.1, 0.1, length)[None])
                self.assertEqual(output.shape, (1, length * 2))
                self.assertTrue(torch.isfinite(output).all())

    def test_refiner_matches_explicit_reference_signal_path(self):
        refiner = self.make_refiner()
        audio = torch.sin(torch.arange(2401) * 0.08)[None] * 0.1
        reference = torchaudio.functional.resample(F.pad(audio, (0, 479)), 24000, 48000)
        with torch.inference_mode():
            predicted = refiner.model(reference)
            expected = refiner.merge(predicted[..., :5760], reference)[..., :4802]
        torch.testing.assert_close(refiner(audio), expected, rtol=0, atol=0)

    def test_refiner_batch_matches_individual_clips(self):
        refiner = self.make_refiner()
        clips = torch.stack((torch.sin(torch.arange(1201) * 0.08), torch.cos(torch.arange(1201) * 0.13))) * 0.1
        expected = torch.cat([refiner(clip) for clip in clips])
        torch.testing.assert_close(refiner(clips), expected, rtol=1e-5, atol=2e-6)

    def test_refiner_rejects_invalid_waveform(self):
        refiner = self.make_refiner()
        for bad in [torch.empty(1, 0), torch.zeros(1, 1, 100), torch.tensor([[float("nan")]])]:
            with self.assertRaises(ValueError):
                refiner(bad)

    def test_read_input_averages_stereo_and_rejects_wrong_rate(self):
        module = api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, np.array([[0.25, 0.75], [-0.5, 0.5]], dtype=np.float32), 24000, subtype="FLOAT")
            torch.testing.assert_close(module.load_input(path), torch.tensor([[0.5, 0.0]]))
            sf.write(path, np.zeros(100), 16000)
            with self.assertRaisesRegex(ValueError, "24000|24 kHz"):
                module.load_input(path)

    def test_peak_normalization_is_safe_for_silence(self):
        module = api()
        torch.testing.assert_close(module.normalize_audio(torch.zeros(1, 20)), torch.zeros(1, 20))
        quiet = torch.tensor([[0.1, -0.5]])
        torch.testing.assert_close(module.normalize_audio(quiet), quiet, rtol=0, atol=0)
        loud = module.normalize_audio(torch.tensor([[0.5, -2.0]]))
        self.assertLessEqual(float(loud.abs().max()), 0.98)
        torch.testing.assert_close(loud, torch.tensor([[0.245, -0.98]]))

    def test_refiner_loads_raw_weights_strictly_and_writes_pcm24(self):
        module = api()
        refiner = self.make_refiner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight_path = root / "refiner.bin"
            torch.save(refiner.model.state_dict(), weight_path)
            audio_path, output = root / "input.wav", root / "output.wav"
            sf.write(audio_path, np.zeros(1000), 24000)
            module.run_refiner(input_path=audio_path, output_path=output, checkpoint_path=weight_path, device="cpu")
            info = sf.info(output)
            self.assertEqual((info.frames, info.samplerate, info.channels, info.subtype), (2000, 48000, 1, "PCM_24"))
            with self.assertRaises(FileExistsError):
                module.run_refiner(input_path=audio_path, output_path=output, checkpoint_path=weight_path, device="cpu")
            with self.assertRaisesRegex(ValueError, "input|different|overwrite"):
                module.run_refiner(input_path=audio_path, output_path=audio_path, checkpoint_path=weight_path, device="cpu")
            state = dict(refiner.model.state_dict())
            state.pop(next(iter(state)))
            torch.save(state, weight_path)
            with self.assertRaisesRegex((ValueError, RuntimeError), "missing|Missing|mismatch"):
                module.Refiner.from_checkpoint(weight_path, device="cpu")

    def test_cli_help_does_not_load_weights(self):
        result = subprocess.run([sys.executable, str(ROOT / "infer_refiner.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_cli_reports_invalid_config_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.json"
            config.write_text("[]")
            result = subprocess.run(
                [sys.executable, str(ROOT / "infer_refiner.py"), "--config", str(config),
                 "--input", "unused.wav", "--output", "unused-output.wav"],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refiner error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_reports_missing_config_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "infer_refiner.py"), "--config", str(Path(directory) / "missing.json"),
                 "--input", "unused.wav", "--output", "unused-output.wav"],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refiner error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
