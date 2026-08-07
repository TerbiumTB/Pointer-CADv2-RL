import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.nn as nn
import yaml

from dpo_train import resolve_replay_lengths, set_deterministic_likelihood_mode


class FakeCheckpointLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = True
        self.dropout = nn.Dropout(0.5)


class DPOTrainingModeTest(unittest.TestCase):
    def test_checkpoint_layer_is_active_while_stochastic_children_stay_eval(self):
        layer = FakeCheckpointLayer()
        model = nn.Sequential(layer)
        model.train()

        set_deterministic_likelihood_mode(
            model, gradient_checkpointing=True
        )

        self.assertTrue(layer.training)
        self.assertFalse(layer.dropout.training)

    def test_replay_lengths_are_checked_against_rollout_runtime(self):
        with TemporaryDirectory() as directory:
            Path(directory, "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "generation": {
                            "max_input_length": 3072,
                            "max_generation_steps": 1024,
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                resolve_replay_lengths(
                    {
                        "max_input_length": 3072,
                        "max_replay_length": 4096,
                    },
                    directory,
                ),
                (3072, 4096),
            )
            with self.assertRaisesRegex(ValueError, "need at least 4096"):
                resolve_replay_lengths(
                    {
                        "max_input_length": 3072,
                        "max_replay_length": 3072,
                    },
                    directory,
                )


if __name__ == "__main__":
    unittest.main()
