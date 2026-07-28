from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rl.schemas import ScoreRecord, StepRecord, TrajectoryRecord
from rl.storage import next_part_path, write_parquet, write_yaml


class RolloutStore:
    """A rollout run containing summaries, steps, scores and heavy data files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.trajectories_dir = self.root / "trajectories"
        self.steps_dir = self.root / "steps"
        self.scores_dir = self.root / "scores"
        self.data_dir = self.root / "data"
        self.cad_data_dir = self.data_dir / "cad"
        self.mesh_data_dir = self.data_dir / "mesh"
        self.state_data_dir = self.data_dir / "states"

    @classmethod
    def create(
        cls, root: Path, config: Dict[str, Any], exist_ok: bool = False
    ) -> "RolloutStore":
        store = cls(root)
        if store.root.exists() and any(store.root.iterdir()) and not exist_ok:
            raise FileExistsError(f"Rollout store is not empty: {store.root}")
        for directory in (
            store.trajectories_dir,
            store.steps_dir,
            store.scores_dir,
            store.cad_data_dir,
            store.mesh_data_dir,
            store.state_data_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        write_yaml(store.root / "config.yaml", config)
        return store

    def append(
        self,
        trajectories: Sequence[TrajectoryRecord],
        steps: Sequence[StepRecord],
        scores: Sequence[ScoreRecord] = (),
    ) -> None:
        """Write one immutable Parquet shard per supplied record group."""
        if trajectories:
            write_parquet(
                next_part_path(self.trajectories_dir),
                trajectories,
                TrajectoryRecord,
            )
        if steps:
            write_parquet(next_part_path(self.steps_dir), steps, StepRecord)
        if scores:
            write_parquet(next_part_path(self.scores_dir), scores, ScoreRecord)

    def add_data_file(
        self,
        source: Path,
        category: str,
        relative_name: Optional[str] = None,
    ) -> str:
        """Copy a generated heavy object into data/ and return its store path."""
        category_dirs = {
            "cad": self.cad_data_dir,
            "mesh": self.mesh_data_dir,
            "states": self.state_data_dir,
        }
        try:
            destination_root = category_dirs[category]
        except KeyError as exc:
            raise ValueError(
                f"Unknown data category {category!r}; expected cad, mesh or states."
            ) from exc

        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        relative_name = relative_name or source.name
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("relative_name must stay inside the selected data directory.")
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
        return destination.relative_to(self.root).as_posix()
