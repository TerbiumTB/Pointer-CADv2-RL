"""Materialize cached reference scores from matching rollout behavior logps."""

import argparse
from pathlib import Path

import yaml
from loguru import logger

from rl.reference_scores import materialize_behavior_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="./config/rl_reference_scores.yaml",
        help="Cached reference score YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML object in {args.config}.")

    summary = materialize_behavior_scores(config)
    logger.info(
        "Cached reference scores: checkpoint={} trajectories={} written={} "
        "skipped_existing={} skipped_empty={}",
        summary.checkpoint_hash,
        summary.trajectories,
        summary.written,
        summary.skipped_existing,
        summary.skipped_empty,
    )


if __name__ == "__main__":
    main()
