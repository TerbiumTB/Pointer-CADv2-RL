"""Reward composition used when materializing preferred/rejected views."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from rl.preference_view import build_preference_records, write_preference_view
from rl.schemas import TrajectoryRecord


def calculate_reward(
    valid: bool,
    metrics: Dict[str, Any],
    reward_config: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compose a reward from stored raw metrics without rerunning CAD execution."""
    if not valid:
        invalid_penalty = float(reward_config["invalid_penalty"])
        return invalid_penalty, {"invalid": invalid_penalty}

    components: Dict[str, float] = {
        "valid": float(reward_config.get("valid_bonus", 0.0))
    }
    weights = reward_config.get("weights", {})
    directions = reward_config.get("directions", {})
    for metric_name, raw_weight in weights.items():
        value = metrics.get(metric_name)
        if value is None:
            continue
        weight = float(raw_weight)
        direction = float(directions.get(metric_name, 1.0))
        components[metric_name] = weight * direction * float(value)
    return sum(components.values()), components


def trajectory_reward(
    trajectory: TrajectoryRecord, reward_config: Dict[str, Any]
) -> float:
    return calculate_reward(
        valid=trajectory.valid,
        metrics=trajectory.metrics,
        reward_config=reward_config,
    )[0]


__all__ = [
    "build_preference_records",
    "calculate_reward",
    "trajectory_reward",
    "write_preference_view",
]
