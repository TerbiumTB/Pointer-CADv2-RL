import tempfile
import unittest
from pathlib import Path

from rl.data import load_scores
from rl.reference_scores import (
    behavior_checkpoint_hash,
    behavior_score_record,
    materialize_behavior_scores,
    validate_behavior_temperatures,
)
from rl.rollout_store import RolloutStore
from rl.schemas import StepRecord, TrajectoryRecord


CHECKPOINT_HASH = "a" * 64


def trajectory(num_steps=2):
    return TrajectoryRecord(
        trajectory_id="trajectory-0",
        task_id="task-0",
        rollout_index=0,
        seed=0,
        termination_reason="model_end",
        num_generated_steps=num_steps,
        valid=True,
        final_cad_path=None,
        final_mesh_path=None,
        execution_error=None,
        metrics_json="{}",
        generation_time_seconds=0.0,
        execution_time_seconds=0.0,
        metrics_version="test",
    )


def step(index, offset=0.0):
    return StepRecord(
        trajectory_id="trajectory-0",
        step_index=index,
        state_before_id=f"state-{index}",
        state_before_graph_path=f"data/states/state-{index}.bin",
        state_after_id=f"state-{index + 1}",
        plan_text="plan",
        plan_token_ids=[10, 11],
        parameter_map_json='{"angle":[],"length":[]}',
        labels=[1, 2],
        parameters=[-4, -4],
        pointers=[-4, -4],
        behavior_plan_logps=[-1.0 - offset, -2.0],
        behavior_label_logps=[-3.0, -4.0],
        behavior_parameter_logps=[0.0, -5.0],
        behavior_pointer_logps=[-6.0, 0.0],
        execution_valid=True,
        execution_error=None,
    )


def rollout_config(checkpoint_hash=CHECKPOINT_HASH, temperature=1.0):
    return {
        "generation": {
            "temperatures": {
                "plan": temperature,
                "label": temperature,
                "parameter": temperature,
                "pointer": temperature,
            }
        },
        "runtime": {"checkpoint_sha256": checkpoint_hash},
    }


class ReferenceScoreTest(unittest.TestCase):
    def test_behavior_score_sums_every_step_and_channel(self):
        record = behavior_score_record(
            trajectory(), [step(1, offset=0.5), step(0)], CHECKPOINT_HASH
        )

        self.assertEqual(record.plan_logp, -6.5)
        self.assertEqual(record.label_logp, -14.0)
        self.assertEqual(record.parameter_logp, -10.0)
        self.assertEqual(record.pointer_logp, -12.0)
        self.assertEqual(record.total_logp, -42.5)

    def test_checkpoint_hash_defaults_to_rollout_runtime_and_must_match(self):
        config = rollout_config()
        self.assertEqual(behavior_checkpoint_hash(config), CHECKPOINT_HASH)
        with self.assertRaisesRegex(ValueError, "does not match"):
            behavior_checkpoint_hash(config, "b" * 64)

    def test_non_unit_rollout_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperatures are 1.0"):
            validate_behavior_temperatures(
                rollout_config(temperature=0.7)
            )

    def test_materialization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rollout"
            store = RolloutStore.create(root, rollout_config())
            store.append(
                trajectories=[trajectory()],
                steps=[step(0), step(1, offset=0.5)],
            )
            config = {"rollout_root": str(root), "write_shard_size": 1}

            first = materialize_behavior_scores(config)
            second = materialize_behavior_scores(config)
            scores = load_scores(str(root))

            self.assertEqual(first.written, 1)
            self.assertEqual(first.skipped_existing, 0)
            self.assertEqual(second.written, 0)
            self.assertEqual(second.skipped_existing, 1)
            self.assertEqual(len(scores), 1)
            self.assertEqual(scores[0].checkpoint_hash, CHECKPOINT_HASH)


if __name__ == "__main__":
    unittest.main()
