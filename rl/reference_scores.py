import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rl.data import load_scores, load_steps, load_trajectories, steps_by_trajectory
from rl.rollout_store import RolloutStore
from rl.schemas import ScoreRecord, StepRecord, TrajectoryRecord
from rl.storage import read_yaml


CHANNEL_FIELDS = {
    "plan": "behavior_plan_logps",
    "label": "behavior_label_logps",
    "parameter": "behavior_parameter_logps",
    "pointer": "behavior_pointer_logps",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ScoreBuildSummary:
    checkpoint_hash: str
    trajectories: int
    written: int
    skipped_existing: int
    skipped_empty: int


def behavior_checkpoint_hash(
    rollout_config: Dict, configured_hash: Optional[str] = None
) -> str:
    runtime_hash = rollout_config.get("runtime", {}).get("checkpoint_sha256")
    if configured_hash is not None:
        configured_hash = str(configured_hash).strip().lower()
    if runtime_hash is not None:
        runtime_hash = str(runtime_hash).strip().lower()
    if configured_hash and runtime_hash and configured_hash != runtime_hash:
        raise ValueError(
            "Configured checkpoint_hash does not match the checkpoint SHA256 "
            f"persisted by the rollout run: {configured_hash} != {runtime_hash}."
        )
    checkpoint_hash = configured_hash or runtime_hash
    if checkpoint_hash is None:
        raise ValueError(
            "The rollout config has no runtime.checkpoint_sha256; provide the "
            "exact rollout checkpoint SHA256 as checkpoint_hash."
        )
    if SHA256_PATTERN.fullmatch(checkpoint_hash) is None:
        raise ValueError(
            "checkpoint_hash must be the full lowercase 64-character SHA256."
        )
    return checkpoint_hash


def validate_behavior_temperatures(rollout_config: Dict) -> None:
    temperatures = rollout_config.get("generation", {}).get("temperatures", {})
    invalid = {}
    for channel in CHANNEL_FIELDS:
        value = float(temperatures.get(channel, 1.0))
        if value != 1.0:
            invalid[channel] = value
    if invalid:
        raise ValueError(
            "Behavior log-probabilities can be used as cached reference scores "
            "only when all rollout temperatures are 1.0; found "
            f"{invalid}. Replay the trajectories under the reference checkpoint "
            "instead."
        )


def behavior_score_record(
    trajectory: TrajectoryRecord,
    steps: Sequence[StepRecord],
    checkpoint_hash: str,
) -> ScoreRecord:
    ordered_steps = sorted(steps, key=lambda record: record.step_index)
    expected_indices = list(range(trajectory.num_generated_steps))
    actual_indices = [step.step_index for step in ordered_steps]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Trajectory {trajectory.trajectory_id!r} declares "
            f"{trajectory.num_generated_steps} steps but stores indices "
            f"{actual_indices}."
        )
    for step in ordered_steps:
        if step.trajectory_id != trajectory.trajectory_id:
            raise ValueError(
                f"Step {step.step_index} belongs to {step.trajectory_id!r}, not "
                f"{trajectory.trajectory_id!r}."
            )

    totals = {}
    for channel, field_name in CHANNEL_FIELDS.items():
        values = [
            float(value)
            for step in ordered_steps
            for value in getattr(step, field_name)
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Trajectory {trajectory.trajectory_id!r} contains non-finite "
                f"{channel} behavior log-probabilities."
            )
        totals[channel] = math.fsum(values)

    return ScoreRecord(
        trajectory_id=trajectory.trajectory_id,
        checkpoint_hash=checkpoint_hash,
        plan_logp=totals["plan"],
        label_logp=totals["label"],
        parameter_logp=totals["parameter"],
        pointer_logp=totals["pointer"],
        total_logp=math.fsum(totals.values()),
    )


def _existing_score_keys(scores: Sequence[ScoreRecord]) -> set[Tuple[str, str]]:
    result = set()
    for score in scores:
        key = (score.trajectory_id, score.checkpoint_hash)
        if key in result:
            raise ValueError(
                "Duplicate cached score for trajectory "
                f"{score.trajectory_id!r} and checkpoint "
                f"{score.checkpoint_hash!r}."
            )
        result.add(key)
    return result


def materialize_behavior_scores(config: Dict) -> ScoreBuildSummary:
    rollout_root = Path(config["rollout_root"])
    rollout_config = read_yaml(rollout_root / "config.yaml")
    validate_behavior_temperatures(rollout_config)
    checkpoint_hash = behavior_checkpoint_hash(
        rollout_config, config.get("checkpoint_hash")
    )
    write_shard_size = int(config.get("write_shard_size", 4096))
    if write_shard_size <= 0:
        raise ValueError("write_shard_size must be positive.")

    trajectories = load_trajectories(str(rollout_root))
    trajectory_steps = steps_by_trajectory(load_steps(str(rollout_root)))
    existing_keys = _existing_score_keys(load_scores(str(rollout_root)))
    store = RolloutStore(rollout_root)

    pending: List[ScoreRecord] = []
    written = 0
    skipped_existing = 0
    skipped_empty = 0
    for trajectory in trajectories:
        if trajectory.num_generated_steps == 0:
            skipped_empty += 1
            continue
        key = (trajectory.trajectory_id, checkpoint_hash)
        if key in existing_keys:
            skipped_existing += 1
            continue
        try:
            steps = trajectory_steps[trajectory.trajectory_id]
        except KeyError as exc:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_id!r} has no stored steps."
            ) from exc
        pending.append(
            behavior_score_record(trajectory, steps, checkpoint_hash)
        )
        existing_keys.add(key)
        if len(pending) >= write_shard_size:
            store.append(trajectories=(), steps=(), scores=pending)
            written += len(pending)
            pending.clear()

    if pending:
        store.append(trajectories=(), steps=(), scores=pending)
        written += len(pending)

    return ScoreBuildSummary(
        checkpoint_hash=checkpoint_hash,
        trajectories=len(trajectories),
        written=written,
        skipped_existing=skipped_existing,
        skipped_empty=skipped_empty,
    )
