#!/usr/bin/env python3
"""Validate PointerCAD subset structure and sampling constraints."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from dataset.subset_sampling import validate_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subset_root",
        help="Directory containing train_val_test.json and subset_manifest.json.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Override dataset directory (required for materialization=none).",
    )
    parser.add_argument(
        "--report",
        help="Optional path for the machine-readable JSON report.",
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Check split/statistical constraints without checking every file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def _requirements(manifest: Mapping[str, object]) -> Mapping[str, object]:
    sampling = manifest.get("sampling", {})
    if not isinstance(sampling, dict):
        return {}
    if sampling.get("mode") == "smoke":
        return {}

    result = {}
    total_target = sum(
        int(value)
        for value in manifest.get("target_step_records_by_split", {}).values()
    )
    tolerance = int(sampling.get("record_tolerance", 100))
    result["min_step_records"] = max(0, total_target - tolerance)
    result["max_step_records"] = total_target + tolerance
    validation_requirements = sampling.get("requirements", {})
    if isinstance(validation_requirements, dict):
        result.update(validation_requirements)
    return result


def _smoke_split_errors(
    report: Mapping[str, object], sampling: Mapping[str, object]
) -> list[str]:
    split_configs = sampling.get("splits")
    if not isinstance(split_configs, dict):
        return ["smoke.splits must be a mapping"]

    errors = []
    required = {"train", "validation"}
    missing = sorted(required - set(split_configs))
    if missing:
        errors.append(
            f"smoke.splits is missing required splits: {', '.join(missing)}"
        )

    by_split = report["by_split"]
    by_split_chunk = report["models_per_split_chunk"]
    requirement_names = (
        "multistep_models",
        "fillet_models",
        "chamfer_models",
        "plain_models",
    )
    for split, split_config in split_configs.items():
        if split not in ("train", "validation", "test"):
            errors.append(f"unsupported smoke split: {split}")
            continue
        if not isinstance(split_config, dict):
            errors.append(f"smoke.splits.{split} must be a mapping")
            continue
        expected_chunks = int(split_config["num_chunks"])
        expected_models_per_chunk = int(split_config["models_per_chunk"])
        split_counts = by_split[split]
        chunk_counts = by_split_chunk[split]
        if split_counts["chunks"] != expected_chunks:
            errors.append(
                f"{split} chunks: found {split_counts['chunks']}, "
                f"expected {expected_chunks}"
            )
        for chunk, count in chunk_counts.items():
            if count != expected_models_per_chunk:
                errors.append(
                    f"{split} chunk {chunk}: found {count} models, "
                    f"expected {expected_models_per_chunk}"
                )
        expected_models = expected_chunks * expected_models_per_chunk
        if split_counts["models"] != expected_models:
            errors.append(
                f"{split} models: found {split_counts['models']}, "
                f"expected {expected_models}"
            )
        requirements = split_config.get("requirements", {})
        if not isinstance(requirements, dict):
            errors.append(f"smoke.splits.{split}.requirements must be a mapping")
            continue
        for name in requirement_names:
            minimum = int(requirements.get(f"min_{name}", 0))
            if split_counts[name] < minimum:
                errors.append(
                    f"{split} {name}: found {split_counts[name]}, "
                    f"expected at least {minimum}"
                )
    return errors


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    subset_root = Path(args.subset_root).absolute()
    manifest_path = subset_root / "subset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    materialization = str(manifest.get("materialization", "copy"))
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).absolute()
    elif materialization == "none":
        dataset_dir = Path(str(manifest["source_dataset_dir"]))
    else:
        dataset_dir = subset_root / "dataset"

    report = validate_subset(
        dataset_dir=dataset_dir,
        split_path=subset_root / "train_val_test.json",
        requirements=_requirements(manifest),
        prompt_variants=manifest.get("prompt_variants", ["abs", "exp"]),
        check_files=not args.skip_file_check,
    )
    sampling = manifest.get("sampling", {})
    if isinstance(sampling, dict) and sampling.get("mode") == "smoke":
        report["errors"].extend(_smoke_split_errors(report, sampling))
        report["ok"] = not report["errors"]
    targets = manifest.get("target_step_records_by_split", {})
    if isinstance(targets, dict) and targets:
        tolerance = int(manifest.get("sampling", {}).get("record_tolerance", 100))
        for split, raw_target in targets.items():
            target = int(raw_target)
            actual = int(report["by_split"][split]["step_records"])
            if abs(actual - target) > tolerance:
                report["errors"].append(
                    f"{split} step_records: found {actual}, target {target} "
                    f"± {tolerance}"
                )
        report["ok"] = not report["errors"]
    report["profile"] = manifest.get("profile")
    report["dataset_dir"] = str(dataset_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
            file.write("\n")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
