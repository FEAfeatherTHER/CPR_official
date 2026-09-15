"""CPU contracts for the independent Composer–Performer runtime."""

import unittest
from typing import get_type_hints

import torch
from transformers import Qwen3Config

from models.dataset.alignment import AlignmentLengths, lengths_from_last_note
from models.cpr.composer import build_position_ids
from models.cpr.model import ComposerPerformer, select_prompt_history
from models.cpr.factory import build_composer_performer
from models.cpr.performer import build_conditioned_input
from models.backbone.qwen_composer import ContinuousQwenComposer


def tiny_config():
    return {
        "data": {"sample_rate": 24000, "mel_fps": 50, "midi_fps": 25},
        "model": {
            "mel_dim": 4, "clap_dim": 8, "history_patches": 2,
            "performer_clap_conditioning": True,
            "composer_rope": {
                "type": "time_modality_2d", "layout": "interleaved",
                "midi_time_scale": 0.4, "midi_modality_id": 0,
                "audio_modality_id": 1, "clap_modality_id": 1,
            },
            "aggregator": {"mel_dim": 4, "hidden_size": 32, "depth": 1,
                           "heads": 4, "ff_dim": 64, "patch_size": 5, "dropout": 0},
            "dit": {"dim": 32, "depth": 1, "heads": 4, "dim_head": 8,
                    "dropout": 0, "ff_mult": 2, "latent_dim": 4,
                    "qk_norm": "rms_norm", "pe_attn_head": None,
                    "checkpoint_activations": False, "enable_conv": True},
        },
    }


def tiny_qwen_config():
    return Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128,
    ).to_dict()


class ComposerPerformerCoreTests(unittest.TestCase):
    def test_qwen_public_cache_method_annotations_resolve(self):
        self.assertIn("return", get_type_hints(ContinuousQwenComposer.prefill))
        self.assertIn("return", get_type_hints(ContinuousQwenComposer.decode_step))

    def test_release_alignment_preserves_valid_tail(self):
        lengths = lengths_from_last_note(0.51, max_release_duration=3.0)
        self.assertEqual((lengths.valid_mel_frames, lengths.blocks,
                          lengths.patch_frames, lengths.midi_frames),
                         (176, 18, 36, 90))
        with self.assertRaises(ValueError):
            lengths_from_last_note(0.51, target_duration=0.5)

    def test_partial_prompt_history_mask_is_retained(self):
        mel = torch.arange(20).float().reshape(1, 20, 1)
        hidden = torch.arange(4).float().reshape(1, 4, 1)
        history, states, mask = select_prompt_history(
            mel, hidden, valid_frames=17, history_patches=2, patch_size=5,
        )
        self.assertEqual(history.flatten().tolist(), list(range(10, 20)))
        self.assertEqual(states.flatten().tolist(), [2.0, 3.0])
        self.assertEqual(mask.flatten().tolist(), [True] * 7 + [False] * 3)

    def test_cfg_unconditional_branch_retains_clean_acoustic_history(self):
        history = torch.tensor([[[2.0], [3.0], [4.0], [5.0], [6.0]]])
        current = torch.tensor([[[7.0]]])
        hidden = torch.ones((1, 6, 2))
        conditioned = build_conditioned_input(
            history, current, hidden, torch.tensor([[9.0]]),
            clap_conditioning=True, zero_clap=torch.tensor([-1.0]),
            condition_drop=torch.tensor([True]), clap_drop=torch.tensor([True]),
            patch_size=1,
        )
        torch.testing.assert_close(conditioned[0, :, 0],
                                   torch.tensor([2., 3., 4., 5., 6., 7.]))
        torch.testing.assert_close(conditioned[0, :, 1:3], torch.zeros((6, 2)))
        torch.testing.assert_close(conditioned[0, :, 3], -torch.ones(6))

    def test_interleaved_positions_use_midi_scaled_time_and_audio_axis(self):
        positions = build_position_ids(
            3, 2, 2, midi_time_scale=0.4, midi_modality_id=0,
            audio_modality_id=1, clap_modality_id=1,
        )
        torch.testing.assert_close(
            positions,
            torch.tensor([[0, 0, 0.4, 0.8, 0, 1, 2, 3],
                          [1, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32),
            rtol=0, atol=0,
        )

    def test_tiny_qwen_cache_and_seeded_target_noise_are_deterministic(self):
        torch.manual_seed(7)
        model = build_composer_performer(tiny_config(), saved_qwen_config=tiny_qwen_config())
        self.assertIsInstance(model, ComposerPerformer)
        model.eval()
        roll = torch.zeros((1, 2, 15, 128))
        prompt = torch.zeros((1, 20, 4))
        clap = torch.zeros((1, 8))
        lengths = AlignmentLengths.from_mel_frames(10)
        with torch.inference_mode():
            first = model.generate_target_mel(
                pianoroll=roll, prompt_mel=prompt, prompt_mel_length=17,
                target_lengths=lengths, clap_embedding=clap, steps=2,
                schedule="cosine", cfg_scale=1,
                generator=torch.Generator().manual_seed(114),
            )
            second = model.generate_target_mel(
                pianoroll=roll, prompt_mel=prompt, prompt_mel_length=17,
                target_lengths=lengths, clap_embedding=clap, steps=2,
                schedule="cosine", cfg_scale=1,
                generator=torch.Generator().manual_seed(114),
            )
        self.assertEqual(tuple(first.shape), (1, 10, 4))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
