import argparse
from pathlib import Path
from typing import Dict, List

import yaml

from rl.data import load_episode_index, load_trajectories
from rl.preference_data import trajectory_reward
from rl.preference_view import build_preference_records, write_preference_view
from rl.schemas import PreferenceRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="./config/rl_preferences.yaml",
        help="Preference view YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    task_splits = {}
    for split in ("train", "validation"):
        for episode in load_episode_index(config["episodes_root"], split):
            if episode.task_id in task_splits:
                raise ValueError(f"Duplicate task_id: {episode.task_id}")
            task_splits[episode.task_id] = split

    trajectories_by_split = {"train": [], "validation": []}
    for trajectory in load_trajectories(config["rollout_root"]):
        if trajectory.num_generated_steps == 0:
            continue
        split = task_splits.get(trajectory.task_id)
        if split is not None:
            trajectories_by_split[split].append(trajectory)

    reward_config = config["reward"]
    pairing = config["pairing"]
    preferences_by_split: Dict[str, List[PreferenceRecord]] = {}
    for split in ("train", "validation"):
        preferences_by_split[split] = build_preference_records(
            trajectories=trajectories_by_split[split],
            reward_function=lambda trajectory: trajectory_reward(
                trajectory, reward_config
            ),
            reward_version=reward_config["version"],
            min_reward_margin=float(pairing.get("min_reward_margin", 0.0)),
            max_pairs_per_task=pairing.get("max_pairs_per_task"),
            seed=int(pairing.get("seed", 0)),
            pairing_strategy=pairing.get("strategy", "all_pairs"),
        )

    write_preference_view(
        output_root=Path(config["output_root"]),
        records_by_split=preferences_by_split,
        config=config,
    )


if __name__ == "__main__":
    main()
