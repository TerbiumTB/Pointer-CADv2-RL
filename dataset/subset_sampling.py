"""Sampling and validation helpers for PointerCAD dataset subsets.

This module intentionally depends only on the Python standard library.  It can
therefore inspect and sample a dataset before the PyTorch/DGL/CAD environment is
installed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

from tqdm.auto import tqdm


SPLITS = ("train", "validation", "test")
SPECIAL_OPERATIONS = ("fillet", "chamfer")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    split: str
    chunk: str
    model_id: str
    step_ids: Tuple[str, ...]
    part_ids: Tuple[str, ...]
    operation_types: Tuple[str, ...]

    @property
    def num_steps(self) -> int:
        return len(self.step_ids)

    @property
    def is_multistep(self) -> bool:
        return self.num_steps >= 2

    @property
    def has_fillet(self) -> bool:
        return any("fillet" in value.lower() for value in self.operation_types)

    @property
    def has_chamfer(self) -> bool:
        return any("chamfer" in value.lower() for value in self.operation_types)

    @property
    def is_plain(self) -> bool:
        return (
            bool(self.operation_types)
            and not self.has_fillet
            and not self.has_chamfer
        )


def parse_step_id(step_id: str) -> Tuple[str, str, str]:
    parts = step_id.split("_")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Expected '<chunk>_<model_id>_<part_id>', got {step_id!r}."
        )
    return parts[0], parts[1], parts[2]


def part_sort_key(part_id: str) -> Tuple[int, object]:
    try:
        return 0, int(part_id)
    except ValueError:
        return 1, part_id


def read_split_file(split_path: Path) -> Dict[str, List[str]]:
    logger.info("Reading source split: %s", split_path)
    with Path(split_path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Split file must contain a JSON object: {split_path}")

    # Some older PointerCAD comments say "val", while the current trainer uses
    # "validation". Accept the former at input and always emit the latter.
    if "val" in raw and "validation" not in raw:
        raw["validation"] = raw["val"]

    result: Dict[str, List[str]] = {}
    seen: Set[str] = set()
    model_owners: Dict[Tuple[str, str], str] = {}
    for split in SPLITS:
        values = raw.get(split, [])
        if not isinstance(values, list):
            raise ValueError(f"Split {split!r} must be a list.")
        result[split] = []
        for raw_step_id in tqdm(
            values,
            desc=f"Read {split} split",
            unit="step",
        ):
            step_id = str(raw_step_id)
            chunk, model_id, _ = parse_step_id(step_id)
            if step_id in seen:
                raise ValueError(f"Duplicate step ID in split file: {step_id}")
            seen.add(step_id)
            model_key = (chunk, model_id)
            previous = model_owners.setdefault(model_key, split)
            if previous != split:
                raise ValueError(
                    f"Model {chunk}/{model_id} crosses splits "
                    f"{previous!r} and {split!r}."
                )
            result[split].append(step_id)
    logger.info(
        "Loaded %d step records: train=%d, validation=%d, test=%d",
        sum(len(result[split]) for split in SPLITS),
        len(result["train"]),
        len(result["validation"]),
        len(result["test"]),
    )
    return result


def _read_json_object(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _operation_types_from_json(path: Path) -> List[str]:
    value = _read_json_object(path)
    sequence = value.get("sequence", [])
    if not isinstance(sequence, list):
        return []
    result = []
    for operation in sequence:
        if isinstance(operation, dict):
            operation_type = operation.get("type")
            if operation_type is not None:
                result.append(str(operation_type))
    return result


def inspect_operation_types(
    dataset_dir: Path,
    chunk: str,
    model_id: str,
    part_ids: Sequence[str],
) -> Tuple[str, ...]:
    """Read operation types, preferring the final progressive CAD JSON.

    A plan-text fallback is used only for identifying fillet/chamfer when the
    source JSON does not expose operation types.
    """
    model_dir = Path(dataset_dir) / chunk / model_id
    last_part_id = sorted(part_ids, key=part_sort_key)[-1]
    candidates = (
        model_dir / "json" / f"{model_id}_{last_part_id}.json",
        model_dir / f"{model_id}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            operation_types = _operation_types_from_json(candidate)
            if operation_types:
                return tuple(operation_types)

    fallback: List[str] = []
    for part_id in sorted(part_ids, key=part_sort_key):
        plan_path = model_dir / "plan" / f"{model_id}_{part_id}.txt"
        if not plan_path.is_file():
            continue
        plan = plan_path.read_text(encoding="utf-8", errors="replace").lower()
        if "fillet" in plan:
            fallback.append("FilletFeature")
        elif "chamfer" in plan:
            fallback.append("ChamferFeature")
        else:
            fallback.append("unknown")
    return tuple(fallback or ["unknown"])


def build_model_index(
    dataset_dir: Path,
    split_data: Mapping[str, Sequence[str]],
    inspect_operations: bool = True,
) -> List[ModelInfo]:
    logger.info(
        "Building model index%s",
        " and inspecting operation types" if inspect_operations else "",
    )
    grouped: MutableMapping[
        Tuple[str, str, str], List[Tuple[str, str]]
    ] = defaultdict(list)
    for split in SPLITS:
        step_ids = split_data.get(split, [])
        for step_id in tqdm(
            step_ids,
            desc=f"Group {split} steps",
            unit="step",
        ):
            chunk, model_id, part_id = parse_step_id(step_id)
            grouped[(split, chunk, model_id)].append((part_id, step_id))

    result = []
    grouped_items = sorted(grouped.items())
    for (split, chunk, model_id), parts_and_ids in tqdm(
        grouped_items,
        desc=(
            "Inspect CAD models"
            if inspect_operations
            else "Build CAD model index"
        ),
        unit="model",
    ):
        ordered = sorted(parts_and_ids, key=lambda value: part_sort_key(value[0]))
        part_ids = tuple(value[0] for value in ordered)
        step_ids = tuple(value[1] for value in ordered)
        operation_types = (
            inspect_operation_types(dataset_dir, chunk, model_id, part_ids)
            if inspect_operations
            else ()
        )
        result.append(
            ModelInfo(
                split=split,
                chunk=chunk,
                model_id=model_id,
                step_ids=step_ids,
                part_ids=part_ids,
                operation_types=operation_types,
            )
        )
    counts = model_counts(result)
    logger.info(
        "Indexed %(models)d models, %(step_records)d step records, "
        "%(chunks)d chunks",
        counts,
    )
    return result


def model_counts(models: Iterable[ModelInfo]) -> Dict[str, int]:
    values = list(models)
    return {
        "models": len(values),
        "step_records": sum(model.num_steps for model in values),
        "chunks": len({model.chunk for model in values}),
        "multistep_models": sum(model.is_multistep for model in values),
        "fillet_models": sum(model.has_fillet for model in values),
        "chamfer_models": sum(model.has_chamfer for model in values),
        "plain_models": sum(model.is_plain for model in values),
    }


def _requirement_value(requirements: Mapping[str, object], name: str) -> int:
    value = int(requirements.get(name, 0))
    if value < 0:
        raise ValueError(f"Requirement {name!r} cannot be negative.")
    return value


def _requirements_met(
    models: Sequence[ModelInfo], requirements: Mapping[str, object]
) -> bool:
    counts = model_counts(models)
    return all(
        counts[name] >= _requirement_value(requirements, f"min_{name}")
        for name in (
            "multistep_models",
            "fillet_models",
            "chamfer_models",
            "plain_models",
        )
    )


def _sample_smoke_split_models(
    models: Sequence[ModelInfo],
    config: Mapping[str, object],
    rng: random.Random,
    dataset_dir: Path,
) -> List[ModelInfo]:
    num_chunks = int(config["num_chunks"])
    models_per_chunk = int(config["models_per_chunk"])
    split = str(config.get("split", "train"))
    requirements = config.get("requirements", {})
    if not isinstance(requirements, dict):
        raise ValueError("smoke requirements must be a mapping.")
    if num_chunks < 2:
        raise ValueError("Smoke subset must use at least two chunks.")
    if models_per_chunk < 2:
        raise ValueError("Smoke subset must have at least two models per chunk.")

    by_chunk: Dict[str, List[ModelInfo]] = defaultdict(list)
    for model in models:
        if model.split == split:
            by_chunk[model.chunk].append(model)
    eligible_chunks = sorted(
        chunk
        for chunk, chunk_models in by_chunk.items()
        if len(chunk_models) >= models_per_chunk
    )
    if len(eligible_chunks) < num_chunks:
        raise ValueError(
            f"Only {len(eligible_chunks)} chunks have at least "
            f"{models_per_chunk} {split} models; need {num_chunks}."
        )

    attempts = int(config.get("max_sampling_attempts", 2000))
    logger.info(
        "Sampling smoke subset: split=%s, chunks=%d, models_per_chunk=%d, "
        "eligible_chunks=%d",
        split,
        num_chunks,
        models_per_chunk,
        len(eligible_chunks),
    )
    operation_cache: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    best_counts: Dict[str, int] = {}
    best_score = -1
    attempt_progress = tqdm(
        range(1, attempts + 1),
        desc="Sample smoke candidates",
        unit="attempt",
    )
    for attempt in attempt_progress:
        chosen_chunks = rng.sample(eligible_chunks, num_chunks)
        selected = [
            model
            for chunk in chosen_chunks
            for model in rng.sample(by_chunk[chunk], models_per_chunk)
        ]

        # Multi-step status is already known from the split and costs no file
        # reads. Reject unsuitable candidates before opening their CAD JSONs.
        if sum(model.is_multistep for model in selected) < _requirement_value(
            requirements, "min_multistep_models"
        ):
            continue

        inspected = []
        for model in selected:
            model_key = (model.chunk, model.model_id)
            operation_types = operation_cache.get(model_key)
            if operation_types is None:
                operation_types = inspect_operation_types(
                    dataset_dir,
                    model.chunk,
                    model.model_id,
                    model.part_ids,
                )
                operation_cache[model_key] = operation_types
            inspected.append(replace(model, operation_types=operation_types))

        counts = model_counts(inspected)
        score = sum(
            min(
                counts[name],
                _requirement_value(requirements, f"min_{name}"),
            )
            for name in (
                "multistep_models",
                "fillet_models",
                "chamfer_models",
                "plain_models",
            )
        )
        if score > best_score:
            best_score = score
            best_counts = counts
        attempt_progress.set_postfix(
            fillet=counts["fillet_models"],
            chamfer=counts["chamfer_models"],
            multistep=counts["multistep_models"],
            plain=counts["plain_models"],
            cached=len(operation_cache),
            refresh=False,
        )

        if _requirements_met(inspected, requirements):
            result = sorted(
                inspected,
                key=lambda item: (item.chunk, item.model_id, item.split),
            )
            # The iterator increments after the loop body. Account for the
            # successful final attempt before returning early.
            attempt_progress.update(1)
            attempt_progress.close()
            logger.info(
                "Smoke subset selected on attempt %d after inspecting %d unique "
                "models: %s",
                attempt,
                len(operation_cache),
                model_counts(result),
            )
            return result

    attempt_progress.close()
    raise ValueError(
        "Could not satisfy smoke sampling constraints after "
        f"{attempts} attempts and {len(operation_cache)} inspected models. "
        f"Best sampled candidate counts: {best_counts}"
    )


def sample_smoke_models(
    models: Sequence[ModelInfo],
    config: Mapping[str, object],
    rng: random.Random,
    dataset_dir: Path,
) -> List[ModelInfo]:
    """Sample independent train and validation model sets for RL smoke tests."""
    split_configs = config.get("splits")
    if not isinstance(split_configs, dict):
        raise ValueError("smoke.splits must be a mapping.")

    required_splits = {"train", "validation"}
    missing = sorted(required_splits - set(split_configs))
    unknown = sorted(set(split_configs) - set(SPLITS))
    if missing:
        raise ValueError(
            f"smoke.splits is missing required splits: {', '.join(missing)}"
        )
    if unknown:
        raise ValueError(
            f"smoke.splits contains unsupported splits: {', '.join(unknown)}"
        )

    selected: List[ModelInfo] = []
    for split in SPLITS:
        if split not in split_configs:
            continue
        split_config = split_configs[split]
        if not isinstance(split_config, dict):
            raise ValueError(f"smoke.splits.{split} must be a mapping.")
        sampling_config = dict(split_config)
        sampling_config["split"] = split
        sampling_config.setdefault(
            "max_sampling_attempts",
            int(config.get("max_sampling_attempts", 2000)),
        )
        selected.extend(
            _sample_smoke_split_models(
                models,
                sampling_config,
                rng,
                dataset_dir=dataset_dir,
            )
        )
    return sorted(
        selected,
        key=lambda item: (item.split, item.chunk, item.model_id),
    )


def _largest_remainder_targets(
    total: int, fractions: Mapping[str, object]
) -> Dict[str, int]:
    normalized = {split: float(fractions.get(split, 0.0)) for split in SPLITS}
    fraction_sum = sum(normalized.values())
    if total <= 0 or fraction_sum <= 0:
        raise ValueError("target_step_records and split fractions must be positive.")
    normalized = {key: value / fraction_sum for key, value in normalized.items()}
    raw = {key: total * value for key, value in normalized.items()}
    targets = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(SPLITS, key=lambda key: raw[key] - targets[key], reverse=True)
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def _sample_models_to_step_target(
    models: Sequence[ModelInfo], target: int, rng: random.Random
) -> List[ModelInfo]:
    if target <= 0:
        return []
    candidates = list(models)
    rng.shuffle(candidates)
    selected: List[ModelInfo] = []
    selected_steps = 0
    remaining: List[ModelInfo] = []
    for model in candidates:
        if selected_steps + model.num_steps <= target:
            selected.append(model)
            selected_steps += model.num_steps
            if selected_steps == target:
                return selected
        else:
            remaining.append(model)

    if selected_steps < target and remaining:
        best = min(
            remaining,
            key=lambda model: (
                abs(target - (selected_steps + model.num_steps)),
                model.num_steps,
            ),
        )
        if abs(target - (selected_steps + best.num_steps)) < target - selected_steps:
            selected.append(best)
    return selected


def sample_target_models(
    models: Sequence[ModelInfo],
    config: Mapping[str, object],
    rng: random.Random,
) -> Tuple[List[ModelInfo], Dict[str, int]]:
    explicit_targets = config.get("target_step_records_by_split")
    if explicit_targets is not None:
        if not isinstance(explicit_targets, dict):
            raise ValueError("target_step_records_by_split must be a mapping.")
        targets = {split: int(explicit_targets.get(split, 0)) for split in SPLITS}
    else:
        total = int(config["target_step_records"])
        fractions = config.get(
            "split_fractions",
            {"train": 0.9, "validation": 0.05, "test": 0.05},
        )
        if not isinstance(fractions, dict):
            raise ValueError("split_fractions must be a mapping.")
        targets = _largest_remainder_targets(total, fractions)

    selected: List[ModelInfo] = []
    logger.info("Sampling target step counts by split: %s", targets)
    for split in SPLITS:
        split_models = [model for model in models if model.split == split]
        available = sum(model.num_steps for model in split_models)
        if targets[split] > available:
            raise ValueError(
                f"Requested {targets[split]} {split} records, but only "
                f"{available} are available."
            )
        selected.extend(
            _sample_models_to_step_target(split_models, targets[split], rng)
        )
    logger.info("Target-record subset selected: %s", model_counts(selected))
    return selected, targets


def selected_split_data(
    source_split: Mapping[str, Sequence[str]], models: Sequence[ModelInfo]
) -> Dict[str, List[str]]:
    selected_ids = {
        step_id for model in models for step_id in model.step_ids
    }
    return {
        split: [
            step_id
            for step_id in source_split.get(split, [])
            if step_id in selected_ids
        ]
        for split in SPLITS
    }


def _copy_model(source: Path, destination: Path, strategy: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if strategy == "copy":
        shutil.copytree(source, destination, symlinks=False)
    elif strategy == "hardlink":
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            copy_function=os.link,
        )
    elif strategy == "symlink":
        destination.symlink_to(source.absolute(), target_is_directory=True)
    else:
        raise ValueError(
            "materialization must be one of: copy, hardlink, symlink, none."
        )


def materialize_subset(
    source_dataset_dir: Path,
    output_root: Path,
    models: Sequence[ModelInfo],
    split_data: Mapping[str, Sequence[str]],
    manifest: Mapping[str, object],
    strategy: str,
    overwrite: bool = False,
) -> None:
    source_dataset_dir = Path(source_dataset_dir).absolute()
    output_root = Path(output_root).absolute()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Pass --overwrite to replace it."
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        dataset_output = staging / "dataset"
        if strategy != "none":
            logger.info(
                "Materializing %d models with strategy=%s into %s",
                len(models),
                strategy,
                output_root,
            )
            dataset_output.mkdir()
            ordered_models = sorted(
                models, key=lambda item: (item.chunk, item.model_id)
            )
            for model in tqdm(
                ordered_models,
                desc=f"Materialize ({strategy})",
                unit="model",
            ):
                source = source_dataset_dir / model.chunk / model.model_id
                if not source.is_dir():
                    raise FileNotFoundError(f"Missing model directory: {source}")
                destination = dataset_output / model.chunk / model.model_id
                _copy_model(source, destination, strategy)
        else:
            logger.info(
                "Materialization disabled; writing split and manifest only"
            )

        with (staging / "train_val_test.json").open("w", encoding="utf-8") as file:
            json.dump(split_data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        with (staging / "subset_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
            file.write("\n")

        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
        logger.info("Subset written successfully: %s", output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def expected_step_paths(
    model_dir: Path,
    model_id: str,
    part_id: str,
    prompt_variants: Sequence[str],
) -> List[Path]:
    paths = [
        model_dir / f"prompt_{variant}.txt" for variant in prompt_variants
    ]
    paths.extend(
        [
            model_dir / "plan" / f"{model_id}_{part_id}.txt",
            model_dir / "parameter" / f"{model_id}_{part_id}.pkl",
            model_dir / "vector" / f"{model_id}_{part_id}.pkl",
            model_dir / "graph" / f"{model_id}_{part_id}.bin",
        ]
    )
    step_json = model_dir / "json" / f"{model_id}_{part_id}.json"
    model_json = model_dir / f"{model_id}.json"
    paths.append(step_json if step_json.exists() else model_json)
    return paths


def validate_subset(
    dataset_dir: Path,
    split_path: Path,
    requirements: Mapping[str, object],
    prompt_variants: Sequence[str],
    check_files: bool = True,
) -> Dict[str, object]:
    logger.info("Validating subset dataset=%s split=%s", dataset_dir, split_path)
    split_data = read_split_file(split_path)
    models = build_model_index(dataset_dir, split_data, inspect_operations=True)
    counts = model_counts(models)
    errors: List[str] = []

    minimums = {
        "chunks": int(requirements.get("min_chunks", 0)),
        "models": int(requirements.get("min_models", 0)),
        "step_records": int(requirements.get("min_step_records", 0)),
        "multistep_models": int(requirements.get("min_multistep_models", 0)),
        "fillet_models": int(requirements.get("min_fillet_models", 0)),
        "chamfer_models": int(requirements.get("min_chamfer_models", 0)),
        "plain_models": int(requirements.get("min_plain_models", 0)),
    }
    for name, minimum in minimums.items():
        if counts[name] < minimum:
            errors.append(f"{name}: found {counts[name]}, expected at least {minimum}")

    maximum_records = requirements.get("max_step_records")
    if maximum_records is not None and counts["step_records"] > int(maximum_records):
        errors.append(
            f"step_records: found {counts['step_records']}, "
            f"expected at most {int(maximum_records)}"
        )

    minimum_models_per_chunk = int(requirements.get("min_models_per_chunk", 0))
    per_chunk_models = Counter(model.chunk for model in models)
    for chunk, count in sorted(per_chunk_models.items()):
        if count < minimum_models_per_chunk:
            errors.append(
                f"chunk {chunk}: found {count} models, expected at least "
                f"{minimum_models_per_chunk}"
            )

    missing_files: List[str] = []
    if check_files:
        logger.info("Checking required files for %d models", len(models))
        for model in tqdm(
            models,
            desc="Validate model files",
            unit="model",
        ):
            model_dir = Path(dataset_dir) / model.chunk / model.model_id
            for part_id in model.part_ids:
                for path in expected_step_paths(
                    model_dir, model.model_id, part_id, prompt_variants
                ):
                    if not path.is_file():
                        missing_files.append(str(path))
        if missing_files:
            errors.append(f"{len(missing_files)} required files are missing")

    by_split = {
        split: model_counts([model for model in models if model.split == split])
        for split in SPLITS
    }
    models_per_split_chunk = {
        split: dict(
            sorted(
                Counter(
                    model.chunk for model in models if model.split == split
                ).items()
            )
        )
        for split in SPLITS
    }
    operation_histogram = Counter(
        operation_type
        for model in models
        for operation_type in model.operation_types
    )
    logger.info(
        "Validation %s: %s",
        "passed" if not errors else "failed",
        counts,
    )
    return {
        "ok": not errors,
        "errors": errors,
        "counts": counts,
        "by_split": by_split,
        "models_per_split_chunk": models_per_split_chunk,
        "models_per_chunk": dict(sorted(per_chunk_models.items())),
        "operation_types": dict(sorted(operation_histogram.items())),
        "missing_files": missing_files,
    }
