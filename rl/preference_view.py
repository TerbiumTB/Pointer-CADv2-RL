from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, List, Optional, Sequence

from rl.schemas import PreferenceRecord, TrajectoryRecord
from rl.storage import write_parquet, write_yaml


RewardFunction = Callable[[TrajectoryRecord], float]


def _pair_id(
    task_id: str,
    preferred_trajectory_id: str,
    rejected_trajectory_id: str,
    reward_version: str,
) -> str:
    value = "|".join(
        (
            task_id,
            preferred_trajectory_id,
            rejected_trajectory_id,
            reward_version,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def build_preference_records(
    trajectories: Sequence[TrajectoryRecord],
    reward_function: RewardFunction,
    reward_version: str,
    min_reward_margin: float = 0.0,
    max_pairs_per_task: Optional[int] = None,
    seed: int = 0,
    pairing_strategy: str = "all_pairs",
) -> List[PreferenceRecord]:
    """Rank trajectories per task without copying trajectory payloads."""
    if min_reward_margin < 0:
        raise ValueError("min_reward_margin must be non-negative.")
    if max_pairs_per_task is not None and max_pairs_per_task <= 0:
        raise ValueError("max_pairs_per_task must be positive when provided.")
    if pairing_strategy != "all_pairs":
        raise ValueError("Only the deterministic all_pairs strategy is supported.")

    grouped: DefaultDict[str, List[TrajectoryRecord]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.task_id].append(trajectory)

    preferences: List[PreferenceRecord] = []
    for task_id, task_trajectories in sorted(grouped.items()):
        task_pairs = []
        for first, second in itertools.combinations(task_trajectories, 2):
            first_reward = float(reward_function(first))
            second_reward = float(reward_function(second))
            if not math.isfinite(first_reward) or not math.isfinite(second_reward):
                continue
            margin = abs(first_reward - second_reward)
            if margin <= min_reward_margin:
                continue
            if first_reward > second_reward:
                preferred, rejected = first, second
                preferred_reward, rejected_reward = first_reward, second_reward
            else:
                preferred, rejected = second, first
                preferred_reward, rejected_reward = second_reward, first_reward
            task_pairs.append(
                PreferenceRecord(
                    pair_id=_pair_id(
                        task_id,
                        preferred.trajectory_id,
                        rejected.trajectory_id,
                        reward_version,
                    ),
                    task_id=task_id,
                    preferred_trajectory_id=preferred.trajectory_id,
                    rejected_trajectory_id=rejected.trajectory_id,
                    preferred_reward=preferred_reward,
                    rejected_reward=rejected_reward,
                    reward_margin=preferred_reward - rejected_reward,
                    reward_version=reward_version,
                    pairing_strategy=pairing_strategy,
                )
            )

        if max_pairs_per_task is not None and len(task_pairs) > max_pairs_per_task:
            task_seed = int.from_bytes(
                hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big"
            )
            task_pairs = random.Random(seed ^ task_seed).sample(
                task_pairs, max_pairs_per_task
            )
        preferences.extend(sorted(task_pairs, key=lambda pair: pair.pair_id))
    return preferences


def write_preference_view(
    output_root: Path,
    records_by_split: Dict[str, Sequence[PreferenceRecord]],
    config: Dict,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        write_parquet(
            output_root / f"{split}.parquet",
            records_by_split.get(split, []),
            PreferenceRecord,
        )
    write_yaml(output_root / "config.yaml", config)
