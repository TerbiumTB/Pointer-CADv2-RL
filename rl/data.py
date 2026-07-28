"""Read-only accessors for episode, rollout and preference datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

from rl.schemas import (
    EpisodeRecord,
    PreferenceRecord,
    ScoreRecord,
    StepRecord,
    TrajectoryRecord,
)
from rl.storage import read_parquet, read_parquet_dataset


def load_episode_index(root: str, split: str) -> List[EpisodeRecord]:
    return read_parquet(Path(root) / f"{split}.parquet", EpisodeRecord)


def load_trajectories(rollout_root: str) -> List[TrajectoryRecord]:
    return read_parquet_dataset(
        Path(rollout_root) / "trajectories", TrajectoryRecord
    )


def load_steps(rollout_root: str) -> List[StepRecord]:
    return read_parquet_dataset(Path(rollout_root) / "steps", StepRecord)


def load_scores(rollout_root: str) -> List[ScoreRecord]:
    return read_parquet_dataset(Path(rollout_root) / "scores", ScoreRecord)


def load_preference_view(root: str, split: str) -> List[PreferenceRecord]:
    return read_parquet(Path(root) / f"{split}.parquet", PreferenceRecord)


def steps_by_trajectory(
    steps: List[StepRecord],
) -> Dict[str, List[StepRecord]]:
    grouped: DefaultDict[str, List[StepRecord]] = defaultdict(list)
    for step in steps:
        grouped[step.trajectory_id].append(step)
    return {
        trajectory_id: sorted(records, key=lambda record: record.step_index)
        for trajectory_id, records in grouped.items()
    }


def trajectories_by_id(
    trajectories: List[TrajectoryRecord],
) -> Dict[str, TrajectoryRecord]:
    result: Dict[str, TrajectoryRecord] = {}
    for trajectory in trajectories:
        if trajectory.trajectory_id in result:
            raise ValueError(
                f"Duplicate trajectory_id: {trajectory.trajectory_id}"
            )
        result[trajectory.trajectory_id] = trajectory
    return result


def episodes_by_id(episodes: List[EpisodeRecord]) -> Dict[str, EpisodeRecord]:
    result: Dict[str, EpisodeRecord] = {}
    for episode in episodes:
        if episode.task_id in result:
            raise ValueError(f"Duplicate task_id: {episode.task_id}")
        result[episode.task_id] = episode
    return result
