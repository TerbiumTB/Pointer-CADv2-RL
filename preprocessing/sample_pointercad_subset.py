#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Mapping

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from dataset.subset_sampling import (
    SPLITS,
    build_model_index,
    materialize_subset,
    model_counts,
    read_split_file,
    sample_smoke_models,
    sample_target_models,
    selected_split_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="./config/dataset_subsets.yaml",
        help="Subset sampling YAML configuration.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        help="Profile to build; repeat to build several. Default: all profiles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing profile output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and report models without writing or copying anything.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return value


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = _load_config(Path(args.config))
    source = config["source"]
    profiles = config["profiles"]
    if not isinstance(source, dict) or not isinstance(profiles, dict):
        raise ValueError("source and profiles must be mappings.")

    requested_profiles = args.profile or list(profiles)
    unknown = sorted(set(requested_profiles) - set(profiles))
    if unknown:
        raise ValueError(f"Unknown profiles: {', '.join(unknown)}")

    dataset_dir = Path(str(source["dataset_dir"])).absolute()
    split_path = Path(str(source["split_filepath"])).absolute()
    split_data = read_split_file(split_path)

    # Building the global index only groups cheap step IDs. Smoke operation
    # constraints are checked lazily on each randomly sampled candidate.
    models = build_model_index(
        dataset_dir, split_data, inspect_operations=False
    )

    for profile_name in requested_profiles:
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {profile_name!r} must be a mapping.")
        seed = int(profile.get("seed", 42))
        rng = random.Random(seed)
        mode = str(profile["mode"])
        targets: Dict[str, int] = {}
        if mode == "smoke":
            selected = sample_smoke_models(
                models,
                profile,
                rng,
                dataset_dir=dataset_dir,
            )
        elif mode == "target_records":
            selected, targets = sample_target_models(models, profile, rng)
        else:
            raise ValueError(
                f"Unsupported mode {mode!r}; use smoke or target_records."
            )

        subset_split = selected_split_data(split_data, selected)
        counts = model_counts(selected)
        by_split = {
            split: model_counts(
                [model for model in selected if model.split == split]
            )
            for split in SPLITS
        }
        manifest = {
            "format_version": "pointercad-subset-v1",
            "profile": profile_name,
            "seed": seed,
            "source_dataset_dir": str(dataset_dir),
            "source_split_filepath": str(split_path),
            "materialization": str(profile.get("materialization", "copy")),
            "operations_inspected_during_sampling": mode == "smoke",
            "prompt_variants": list(
                source.get("prompt_variants", ["abs", "exp"])
            ),
            "sampling": profile,
            "target_step_records_by_split": targets,
            "counts": counts,
            "by_split": by_split,
        }
        print(json.dumps(manifest, indent=2, ensure_ascii=False))

        if not args.dry_run:
            materialize_subset(
                source_dataset_dir=dataset_dir,
                output_root=Path(str(profile["output_dir"])),
                models=selected,
                split_data=subset_split,
                manifest=manifest,
                strategy=str(profile.get("materialization", "copy")),
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
