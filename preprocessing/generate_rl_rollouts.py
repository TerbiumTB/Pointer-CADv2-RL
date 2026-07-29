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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML object in {args.config}.")
    output = generate_rollouts(config)
    logger.info("Rollout generation completed: {}", output)


if __name__ == "__main__":
    main()
