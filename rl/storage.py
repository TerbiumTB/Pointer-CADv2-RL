import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Type, TypeVar

import yaml

from rl.schemas import (
    EpisodeRecord,
    PreferenceRecord,
    ScoreRecord,
    StepRecord,
    TrajectoryRecord,
)


RecordT = TypeVar("RecordT")


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "RL dataset persistence requires pyarrow. Install it in the training "
            "environment before running dataset builders."
        ) from exc
    return pa, pq


def parquet_schema(record_type: Type[Any]):
    pa, _ = _pyarrow()
    schemas = {
        EpisodeRecord: pa.schema(
            [
                ("task_id", pa.string()),
                ("split", pa.string()),
                ("source_dataset", pa.string()),
                ("chunk", pa.string()),
                ("model_id", pa.string()),
                ("prompt_variant", pa.string()),
                ("prompt", pa.string()),
                ("target_cad_path", pa.string()),
                ("source_step_ids", pa.list_(pa.string())),
                ("num_steps", pa.int32()),
                ("operation_types", pa.list_(pa.string())),
                ("preprocessing_version", pa.string()),
                ("initial_cad_path", pa.string()),
                ("initial_graph_path", pa.string()),
            ]
        ),
        TrajectoryRecord: pa.schema(
            [
                ("trajectory_id", pa.string()),
                ("task_id", pa.string()),
                ("rollout_index", pa.int32()),
                ("seed", pa.int64()),
                ("termination_reason", pa.string()),
                ("num_generated_steps", pa.int32()),
                ("valid", pa.bool_()),
                ("final_cad_path", pa.string()),
                ("final_mesh_path", pa.string()),
                ("execution_error", pa.string()),
                ("metrics_json", pa.string()),
                ("generation_time_seconds", pa.float64()),
                ("execution_time_seconds", pa.float64()),
                ("metrics_version", pa.string()),
            ]
        ),
        StepRecord: pa.schema(
            [
                ("trajectory_id", pa.string()),
                ("step_index", pa.int32()),
                ("state_before_id", pa.string()),
                ("state_before_graph_path", pa.string()),
                ("state_after_id", pa.string()),
                ("plan_text", pa.string()),
                ("plan_token_ids", pa.list_(pa.int64())),
                ("parameter_map_json", pa.string()),
                ("labels", pa.list_(pa.int64())),
                ("parameters", pa.list_(pa.int64())),
                ("pointers", pa.list_(pa.int64())),
                ("behavior_plan_logps", pa.list_(pa.float64())),
                ("behavior_label_logps", pa.list_(pa.float64())),
                ("behavior_parameter_logps", pa.list_(pa.float64())),
                ("behavior_pointer_logps", pa.list_(pa.float64())),
                ("execution_valid", pa.bool_()),
                ("execution_error", pa.string()),
            ]
        ),
        ScoreRecord: pa.schema(
            [
                ("trajectory_id", pa.string()),
                ("checkpoint_hash", pa.string()),
                ("plan_logp", pa.float64()),
                ("label_logp", pa.float64()),
                ("parameter_logp", pa.float64()),
                ("pointer_logp", pa.float64()),
                ("total_logp", pa.float64()),
            ]
        ),
        PreferenceRecord: pa.schema(
            [
                ("pair_id", pa.string()),
                ("task_id", pa.string()),
                ("preferred_trajectory_id", pa.string()),
                ("rejected_trajectory_id", pa.string()),
                ("preferred_reward", pa.float64()),
                ("rejected_reward", pa.float64()),
                ("reward_margin", pa.float64()),
                ("reward_version", pa.string()),
                ("pairing_strategy", pa.string()),
            ]
        ),
    }
    try:
        return schemas[record_type]
    except KeyError as exc:
        raise TypeError(f"Unsupported record type: {record_type!r}") from exc


def write_parquet(
    path: Path, records: Sequence[Any], record_type: Type[Any]
) -> None:
    """Atomically write records, including a correctly typed empty table."""
    pa, pq = _pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = parquet_schema(record_type)
    rows = [record.to_dict() for record in records]
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_parquet(path: Path, record_type: Type[RecordT]) -> List[RecordT]:
    _, pq = _pyarrow()
    table = pq.read_table(Path(path), schema=parquet_schema(record_type))
    return [record_type(**row) for row in table.to_pylist()]


def read_parquet_dataset(directory: Path, record_type: Type[RecordT]) -> List[RecordT]:
    records: List[RecordT] = []
    for path in sorted(Path(directory).glob("part-*.parquet")):
        records.extend(read_parquet(path, record_type))
    return records


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            yaml.safe_dump(value, file, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}.")
    return value


def next_part_path(directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("part-*.parquet"))
    if not existing:
        index = 0
    else:
        index = max(int(path.stem.split("-")[-1]) for path in existing) + 1
    return directory / f"part-{index:06d}.parquet"
