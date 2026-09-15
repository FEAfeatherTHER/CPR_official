"""CLI validation that does not load model assets."""

import subprocess
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ComposerPerformerCliTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "infer_cp.py"), *arguments],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_help_does_not_import_costly_audio_or_model_dependencies(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--seed", result.stdout)
        self.assertIn("--max-release-duration", result.stdout)

    def test_validation_rejects_invalid_numbers_before_files_are_loaded(self):
        common = ["--config", "missing.json", "--checkpoint", "missing.safetensors",
                  "--prompt-audio", "a.wav", "--prompt-midi", "a.mid",
                  "--target-midi", "b.mid", "--output", "out.wav"]
        for extra, message in [(["--steps", "0"], "steps must be positive"),
                               (["--max-release-duration", "nan"], "finite and non-negative"),
                               (["--cfg-scale", "inf"], "cfg-scale must be finite")]:
            with self.subTest(extra=extra):
                result = self.run_cli(*(common + extra))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_output_cannot_exist_or_alias_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("c.json", "w.safetensors", "p.wav", "p.mid", "t.mid")]
            for path in paths:
                path.touch()
            common = ["--config", str(paths[0]), "--checkpoint", str(paths[1]),
                      "--prompt-audio", str(paths[2]), "--prompt-midi", str(paths[3]),
                      "--target-midi", str(paths[4])]
            result = self.run_cli(*(common + ["--output", str(paths[2])]))
            self.assertIn("must not overwrite an input", result.stderr)
            existing = root / "existing.wav"
            existing.touch()
            result = self.run_cli(*(common + ["--output", str(existing)]))
            self.assertIn("output already exists", result.stderr)

    def test_empty_user_input_is_rejected_before_model_loading(self):
        from models.cpr.inference import validate_input_paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.mid"
            empty.touch()
            with self.assertRaisesRegex(ValueError, "input is empty"):
                validate_input_paths(empty)


if __name__ == "__main__":
    unittest.main()
