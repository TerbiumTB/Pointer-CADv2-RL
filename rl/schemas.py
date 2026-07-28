from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


VALID_SPLITS = {"train", "validation", "test"}


def canonical_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_json_object(value: str) -> Dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _require_identifier(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty, trimmed string.")


def _require_relative_path(value: Optional[str], field_name: str) -> None:
    if value is None:
        return
    if not value or value.startswith("/") or value.startswith(".."):
        raise ValueError(f"{field_name} must be a relative path, got {value!r}.")


class RecordMixin:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeRecord(RecordMixin):
    """One full-model RL task derived from the source step-level dataset."""

    task_id: str
    split: str
    source_dataset: str
    chunk: str
    model_id: str
    prompt_variant: str
    prompt: str
    target_cad_path: str
    source_step_ids: List[str]
    num_steps: int
    operation_types: List[str]
    preprocessing_version: str
    initial_cad_path: Optional[str] = None
    initial_graph_path: Optional[str] = None

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split {self.split!r}.")
        if self.num_steps != len(self.source_step_ids) or self.num_steps <= 0:
            raise ValueError("num_steps must equal the non-empty source_step_ids length.")
        if self.operation_types and len(self.operation_types) != self.num_steps:
            raise ValueError("operation_types must be empty or contain one item per step.")
        _require_relative_path(self.target_cad_path, "target_cad_path")
        _require_relative_path(self.initial_cad_path, "initial_cad_path")
        _require_relative_path(self.initial_graph_path, "initial_graph_path")


@dataclass(frozen=True)
class TrajectoryRecord(RecordMixin):
    """Summary of one complete policy attempt for an episode."""

    trajectory_id: str
    task_id: str
    rollout_index: int
    seed: int
    termination_reason: str
    num_generated_steps: int
    valid: bool
    final_cad_path: Optional[str]
    final_mesh_path: Optional[str]
    execution_error: Optional[str]
    metrics_json: str
    generation_time_seconds: float
    execution_time_seconds: float
    metrics_version: str

    def __post_init__(self) -> None:
        _require_identifier(self.trajectory_id, "trajectory_id")
        _require_identifier(self.task_id, "task_id")
        if self.rollout_index < 0 or self.num_generated_steps < 0:
            raise ValueError("rollout_index and num_generated_steps must be non-negative.")
        _require_relative_path(self.final_cad_path, "final_cad_path")
        _require_relative_path(self.final_mesh_path, "final_mesh_path")
        parse_json_object(self.metrics_json)

    @property
    def metrics(self) -> Dict[str, Any]:
        return parse_json_object(self.metrics_json)


@dataclass(frozen=True)
class StepRecord(RecordMixin):
    """One environment transition inside a generated trajectory."""

    trajectory_id: str
    step_index: int
    state_before_id: str
    state_after_id: Optional[str]
    plan_text: str
    plan_token_ids: List[int]
    parameter_map_json: str
    labels: List[int]
    parameters: List[int]
    pointers: List[int]
    behavior_plan_logps: List[float]
    behavior_label_logps: List[float]
    behavior_parameter_logps: List[float]
    behavior_pointer_logps: List[float]
    execution_valid: bool
    execution_error: Optional[str]

    def __post_init__(self) -> None:
        _require_identifier(self.trajectory_id, "trajectory_id")
        _require_identifier(self.state_before_id, "state_before_id")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative.")
        if not (len(self.labels) == len(self.parameters) == len(self.pointers)):
            raise ValueError("labels, parameters and pointers must have equal length.")
        parse_json_object(self.parameter_map_json)

    @property
    def parameter_map(self) -> Dict[str, Any]:
        return parse_json_object(self.parameter_map_json)


@dataclass(frozen=True)
class ScoreRecord(RecordMixin):
    """Cached trajectory log-probability under a particular checkpoint."""

    trajectory_id: str
    checkpoint_hash: str
    plan_logp: float
    label_logp: float
    parameter_logp: float
    pointer_logp: float
    total_logp: float

    def __post_init__(self) -> None:
        _require_identifier(self.trajectory_id, "trajectory_id")
        _require_identifier(self.checkpoint_hash, "checkpoint_hash")


@dataclass(frozen=True)
class PreferenceRecord(RecordMixin):
    """A lightweight preferred/rejected view over two stored trajectories."""

    pair_id: str
    task_id: str
    preferred_trajectory_id: str
    rejected_trajectory_id: str
    preferred_reward: float
    rejected_reward: float
    reward_margin: float
    reward_version: str
    pairing_strategy: str

    def __post_init__(self) -> None:
        _require_identifier(self.pair_id, "pair_id")
        _require_identifier(self.task_id, "task_id")
        if self.preferred_trajectory_id == self.rejected_trajectory_id:
            raise ValueError("A preference pair must contain two different trajectories.")
        if self.preferred_reward <= self.rejected_reward:
            raise ValueError("preferred_reward must be greater than rejected_reward.")
        expected_margin = self.preferred_reward - self.rejected_reward
        if abs(self.reward_margin - expected_margin) > 1e-8:
            raise ValueError("reward_margin does not match preferred/rejected rewards.")
