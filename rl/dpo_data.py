from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from torch.utils.data import Dataset

from rl.data import (
    episodes_by_id,
    load_episode_index,
    load_preference_view,
    load_scores,
    load_steps,
    load_trajectories,
    steps_by_trajectory,
    trajectories_by_id,
)
from rl.schemas import EpisodeRecord, PreferenceRecord, StepRecord


@dataclass(frozen=True)
class DPOCompletion:
    trajectory_id: str
    task_id: str
    prompt: str
    steps: List[StepRecord]
    reference_logp: Optional[float] = None


@dataclass(frozen=True)
class DPOPair:
    pair_id: str
    preferred: DPOCompletion
    rejected: DPOCompletion
    reward_margin: float


def collate_dpo_pairs(examples: Sequence[DPOPair]) -> List[DPOPair]:
    # Variable-length episodes remain Python records. The scorer flattens their
    # steps into one PointerCAD forward batch.
    return list(examples)


class PointerCADDataset(Dataset):
    def __init__(
        self,
        episodes_root: str,
        rollout_root: str,
        preferences_root: str,
        split: str,
        reference_checkpoint_hash: Optional[str] = None,
    ):
        self.preferences = load_preference_view(preferences_root, split)
        episodes = episodes_by_id(load_episode_index(episodes_root, split))
        trajectories = trajectories_by_id(load_trajectories(rollout_root))
        trajectory_steps = steps_by_trajectory(load_steps(rollout_root))

        score_by_trajectory: Dict[str, float] = {}
        if reference_checkpoint_hash is not None:
            for score in load_scores(rollout_root):
                if score.checkpoint_hash != reference_checkpoint_hash:
                    continue
                if score.trajectory_id in score_by_trajectory:
                    raise ValueError(
                        "Duplicate cached score for trajectory "
                        f"{score.trajectory_id!r} and checkpoint "
                        f"{reference_checkpoint_hash!r}."
                    )
                score_by_trajectory[score.trajectory_id] = score.total_logp

        self.completions: Dict[str, DPOCompletion] = {}
        referenced_trajectory_ids = {
            trajectory_id
            for preference in self.preferences
            for trajectory_id in (
                preference.preferred_trajectory_id,
                preference.rejected_trajectory_id,
            )
        }
        for trajectory_id in referenced_trajectory_ids:
            try:
                trajectory = trajectories[trajectory_id]
            except KeyError as exc:
                raise ValueError(
                    f"Preference references missing trajectory {trajectory_id!r}."
                ) from exc
            try:
                episode = episodes[trajectory.task_id]
            except KeyError as exc:
                raise ValueError(
                    f"Trajectory {trajectory_id!r} references task "
                    f"{trajectory.task_id!r} outside split {split!r}."
                ) from exc
            steps = trajectory_steps.get(trajectory_id, [])
            if not steps:
                raise ValueError(
                    f"Trajectory {trajectory_id!r} has no stored steps."
                )
            if len(steps) != trajectory.num_generated_steps:
                raise ValueError(
                    f"Trajectory {trajectory_id!r} declares "
                    f"{trajectory.num_generated_steps} steps but stores "
                    f"{len(steps)}."
                )
            reference_logp = None
            if reference_checkpoint_hash is not None:
                try:
                    reference_logp = score_by_trajectory[trajectory_id]
                except KeyError as exc:
                    raise ValueError(
                        f"No cached reference score for trajectory "
                        f"{trajectory_id!r} and checkpoint "
                        f"{reference_checkpoint_hash!r}."
                    ) from exc
            self.completions[trajectory_id] = DPOCompletion(
                trajectory_id=trajectory_id,
                task_id=trajectory.task_id,
                prompt=episode.prompt,
                steps=steps,
                reference_logp=reference_logp,
            )

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, index: int) -> DPOPair:
        preference: PreferenceRecord = self.preferences[index]
        preferred = self.completions[preference.preferred_trajectory_id]
        rejected = self.completions[preference.rejected_trajectory_id]
        if preferred.task_id != preference.task_id:
            raise ValueError("Preferred trajectory belongs to a different task.")
        if rejected.task_id != preference.task_id:
            raise ValueError("Rejected trajectory belongs to a different task.")
        return DPOPair(
            pair_id=preference.pair_id,
            preferred=preferred,
            rejected=rejected,
            reward_margin=preference.reward_margin,
        )
