import argparse
from pathlib import Path

import yaml
from loguru import logger

from rl.rollout_generator import generate_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and execute full-episode PointerCAD rollouts."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./config/rl_rollouts.yaml",
        help="Rollout generation YAML configuration.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=(
            "Delete the existing <output_root>/<run_id> directory and generate "
            "the rollout run from scratch. By default, existing trajectories "
            "are skipped and only missing trajectories are generated."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override generation.batch_size for this run.",
    )
    parser.add_argument(
        "--cpu-workers-per-gpu",
        type=int,
        default=None,
        help="Override generation.cpu_workers_per_gpu for this run.",
    )
    parser.add_argument(
        "--batch-wait-seconds",
        type=float,
        default=None,
        help="Override generation.batch_wait_seconds for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML object in {args.config}.")
    generation = config.setdefault("generation", {})
    overrides = {
        "batch_size": args.batch_size,
        "cpu_workers_per_gpu": args.cpu_workers_per_gpu,
        "batch_wait_seconds": args.batch_wait_seconds,
    }
    for name, value in overrides.items():
        if value is not None:
            generation[name] = value
    output = generate_rollouts(config, force=args.force)
    logger.info("Rollout generation completed: {}", output)


if __name__ == "__main__":
    main()
