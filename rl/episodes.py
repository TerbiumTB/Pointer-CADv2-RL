import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
from tqdm import tqdm

from rl.schemas import EpisodeRecord, VALID_SPLITS
from rl.storage import write_parquet, write_yaml


def parse_step_id(step_id: str) -> Tuple[str, str, str]:
    parts = step_id.split("_")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Expected '<chunk>_<model_id>_<part_id>', got {step_id!r}."
        )
    return parts[0], parts[1], parts[2]


def _part_sort_key(part_id: str):
    try:
        return 0, int(part_id)
    except ValueError:
        return 1, part_id


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _target_path(model_dir: Path, model_id: str, last_part_id: str) -> Path:
    step_target = model_dir / "json" / f"{model_id}_{last_part_id}.json"
    model_target = model_dir / f"{model_id}.json"
    if step_target.exists():
        return step_target
    if model_target.exists():
        return model_target
    raise FileNotFoundError(
        f"No final CAD JSON found at {step_target} or {model_target}."
    )


def _operation_type(model_dir: Path, model_id: str, part_id: str) -> str:
    step_json = model_dir / "json" / f"{model_id}_{part_id}.json"
    if not step_json.exists():
        return "unknown"
    sequence = _read_json(step_json).get("sequence", [])
    if not isinstance(sequence, list) or not sequence:
        return "unknown"
    last_operation = sequence[-1]
    if not isinstance(last_operation, dict):
        return "unknown"
    return str(last_operation.get("type", "unknown"))


def _source_relative(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def build_episode_records(
    source_root: Path,
    split_path: Path,
    source_dataset: str,
    prompt_variants: Sequence[str],
    preprocessing_version: str,
) -> Dict[str, List[EpisodeRecord]]:
    """Create one episode per (source CAD model, prompt variant).

    The source split remains authoritative. This function only groups its
    step-level identifiers by model and validates that a model never crosses
    split boundaries.
    """
    source_root = Path(source_root)
    split_data = _read_json(Path(split_path))
    grouped: Dict[str, DefaultDict[Tuple[str, str], List[str]]] = {
        split: defaultdict(list) for split in VALID_SPLITS
    }
    owner_split: Dict[Tuple[str, str], str] = {}

    for split in ("train", "validation", "test"):
        step_ids = split_data.get(split, [])
        if not isinstance(step_ids, list):
            raise ValueError(f"Split {split!r} must be a list.")
        for step_id in tqdm(step_ids):
            chunk, model_id, part_id = parse_step_id(str(step_id))
            model_key = (chunk, model_id)
            previous_split = owner_split.setdefault(model_key, split)
            if previous_split != split:
                raise ValueError(
                    f"Model {chunk}/{model_id} appears in both "
                    f"{previous_split!r} and {split!r}."
                )
            grouped[split][model_key].append(part_id)

    result: Dict[str, List[EpisodeRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split in ("train", "validation", "test"):
        for (chunk, model_id), unsorted_part_ids in tqdm(sorted(grouped[split].items())):
            part_ids = sorted(set(unsorted_part_ids), key=_part_sort_key)
            if len(part_ids) != len(unsorted_part_ids):
                raise ValueError(
                    f"Duplicate step identifiers found for {chunk}/{model_id}."
                )

            model_dir = source_root / chunk / model_id
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Missing model directory: {model_dir}")
            target_path = _target_path(model_dir, model_id, part_ids[-1])
            operation_types = [
                _operation_type(model_dir, model_id, part_id)
                for part_id in part_ids
            ]

            for prompt_variant in prompt_variants:
                prompt_path = model_dir / f"prompt_{prompt_variant}.txt"
                if not prompt_path.exists():
                    raise FileNotFoundError(f"Missing prompt: {prompt_path}")
                prompt = prompt_path.read_text(encoding="utf-8")
                task_id = f"{chunk}_{model_id}_{prompt_variant}"
                result[split].append(
                    EpisodeRecord(
                        task_id=task_id,
                        split=split,
                        source_dataset=source_dataset,
                        chunk=chunk,
                        model_id=model_id,
                        prompt_variant=prompt_variant,
                        prompt=prompt,
                        target_cad_path=_source_relative(target_path, source_root),
                        source_step_ids=part_ids,
                        num_steps=len(part_ids),
                        operation_types=operation_types,
                        preprocessing_version=preprocessing_version,
                    )
                )
    return result


def write_episode_index(
    output_root: Path,
    records_by_split: Dict[str, List[EpisodeRecord]],
    config: Dict,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_parquet(
            output_root / f"{split}.parquet",
            records_by_split.get(split, []),
            EpisodeRecord,
        )
    write_yaml(output_root / "config.yaml", config)
