"""Contracts for safe inference-asset export and strict CP loading."""

import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import torch

from models.utils.checkpoint import load_composer_performer_checkpoint
from scripts.export_weights import export_composer_performer_state


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = torch.nn.Parameter(torch.zeros(2))


class ExportTests(unittest.TestCase):
    @staticmethod
    def inference_model_config():
        return {
            "mel_dim": 4, "clap_dim": 8, "history_patches": 2,
            "performer_clap_conditioning": True,
            "composer_rope": {"type": "time_modality_2d"},
            "aggregator": {"patch_size": 5}, "dit": {"depth": 1},
        }

    def test_cp_export_omits_only_repa_and_preserves_values_and_config(self):
        state = {
            "composer.keep": torch.tensor([1.25, -2.5]),
            "performer.zero_clap": torch.tensor([3.0]),
            "repa_head.output_projection.weight": torch.ones(1, 1),
        }
        saved = {"model": self.inference_model_config(),
                 "qwen_config": {"hidden_size": 8, "_name_or_path": "/private/qwen"},
                 "data": {"sample_rate": 24000, "source_jsonl": "/private/train.jsonl"}}
        with tempfile.TemporaryDirectory() as directory:
            weights, config = export_composer_performer_state(
                {"model": state, "config": saved}, Path(directory)
            )
            from safetensors.torch import load_file
            actual = load_file(str(weights))
            self.assertEqual(set(actual), {"composer.keep", "performer.zero_clap"})
            torch.testing.assert_close(actual["composer.keep"], state["composer.keep"])
            exported = json.loads(config.read_text())
            self.assertEqual(exported["model"], saved["model"])
            self.assertEqual(exported["qwen_config"], {"hidden_size": 8})
            self.assertEqual(exported["data"], {"sample_rate": 24000})

    def test_strict_loader_rejects_missing_or_unexpected_keys(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_file({"wrong": torch.ones(2)}, str(path))
            with self.assertRaisesRegex(ValueError, "checkpoint schema mismatch"):
                load_composer_performer_checkpoint(TinyModule(), path)

    def test_exported_state_strict_loads_and_retains_parameter(self):
        source = {
            "model": {"keep": torch.tensor([6.0, -7.0]),
                      "repa_head.unused": torch.ones(1)},
            "config": {"model": self.inference_model_config(),
                       "qwen_config": {"hidden_size": 8}},
        }
        with tempfile.TemporaryDirectory() as directory:
            weights, _ = export_composer_performer_state(source, Path(directory))
            model = load_composer_performer_checkpoint(TinyModule(), weights)
            torch.testing.assert_close(model.keep, torch.tensor([6.0, -7.0]))

    def test_export_cli_does_not_require_create_or_touch_clap(self):
        script = Path(__file__).resolve().parents[1] / "scripts/export_weights.py"
        for existing_clap in (False, True):
            with self.subTest(existing_clap=existing_clap), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.pt"
                torch.save({"model": {"keep": torch.ones(2)},
                            "config": {"model": self.inference_model_config(),
                                       "qwen_config": {}}}, source)
                vocos, refiner = root / "vocos.source", root / "refiner.source"
                vocos.write_bytes(b"vocos fixture")
                refiner.write_bytes(b"refiner fixture")
                output = root / "checkpoints"
                output.mkdir()
                clap = output / "clap.pt"
                if existing_clap:
                    clap.write_bytes(b"separately downloaded CLAP")
                result = subprocess.run(
                    [sys.executable, str(script), "--composer-performer-source", str(source),
                     "--vocos-source", str(vocos), "--refiner-source", str(refiner),
                     "--output-dir", str(output)],
                    cwd=root, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = {"composer_performer.safetensors", "composer_performer_config.json",
                            "vocos.safetensors", "refiner.bin"}
                if existing_clap:
                    expected.add("clap.pt")
                    self.assertEqual(clap.read_bytes(), b"separately downloaded CLAP")
                self.assertEqual({path.name for path in output.iterdir()}, expected)
                self.assertEqual((output / "vocos.safetensors").read_bytes(), vocos.read_bytes())
                self.assertEqual((output / "refiner.bin").read_bytes(), refiner.read_bytes())

    def test_export_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "composer_performer.safetensors").touch()
            with self.assertRaises(FileExistsError):
                export_composer_performer_state(
                    {"model": {"keep": torch.ones(1)},
                     "config": {"model": self.inference_model_config(),
                                "qwen_config": {}}}, root,
                )


if __name__ == "__main__":
    unittest.main()
