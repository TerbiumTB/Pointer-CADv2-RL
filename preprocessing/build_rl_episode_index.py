import argparse
from pathlib import Path

import yaml

from rl.episodes import build_episode_records, write_episode_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="./config/rl_dataset.yaml",
        help="RL dataset YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    source = config["source"]
    output_root = Path(config["output_root"]) / "episodes"
    records = build_episode_records(
        source_root=Path(source["dataset_dir"]),
        split_path=Path(source["split_filepath"]),
        source_dataset=source["name"],
        prompt_variants=source["prompt_variants"],
        preprocessing_version=config["preprocessing_version"],
    )
    write_episode_index(output_root, records, config)


if __name__ == "__main__":
    main()
