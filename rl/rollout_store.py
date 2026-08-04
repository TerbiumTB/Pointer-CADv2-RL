import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rl.schemas import ScoreRecord, StepRecord, TrajectoryRecord
from rl.storage import next_part_path, read_yaml, write_parquet, write_yaml


class RolloutStore:
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
    def remove_existing(cls, root: Path, expected_parent: Path) -> None:
        """Remove one explicitly named rollout run without following symlinks."""
        root = Path(root)
        expected_parent = Path(expected_parent)
        root_absolute = root.absolute()
        parent_absolute = expected_parent.absolute()
        if root_absolute.parent != parent_absolute:
            raise ValueError(
                f"Refusing to remove rollout store outside {expected_parent}: {root}"
            )
        if not root_absolute.name or root_absolute.name in {".", ".."}:
            raise ValueError(f"Unsafe rollout store path: {root}")
        if root.is_symlink():
            raise ValueError(f"Refusing to remove symlinked rollout store: {root}")
        if not root.exists():
            return
        if not root.is_dir():
            raise NotADirectoryError(root)
        shutil.rmtree(root)

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
        config_path = store.root / "config.yaml"
        if config_path.exists():
            existing_config = read_yaml(config_path)
            if existing_config != config:
                raise ValueError(
                    "Cannot continue a rollout store with a different runtime "
                    f"config: {store.root}. Use a new run_id or recreate the "
                    "run explicitly with --force."
                )
        else:
            write_yaml(config_path, config)
        return store

    def append(
        self,
        trajectories: Sequence[TrajectoryRecord],
        steps: Sequence[StepRecord],
        scores: Sequence[ScoreRecord] = (),
    ) -> None:
        if not trajectories:
            if steps:
                raise ValueError("Cannot append steps without trajectories.")
            if scores:
                write_parquet(
                    next_part_path(self.scores_dir), scores, ScoreRecord
                )
            return

        trajectory_path = next_part_path(self.trajectories_dir)
        shard_name = trajectory_path.name
        if steps:
            write_parquet(
                self.steps_dir / shard_name, steps, StepRecord
            )
        if scores:
            write_parquet(
                self.scores_dir / shard_name, scores, ScoreRecord
            )
        write_parquet(
            trajectory_path,
            trajectories,
            TrajectoryRecord,
        )

    def add_data_file(
        self,
        source: Path,
        category: str,
        relative_name: Optional[str] = None,
    ) -> str:
        destination = self.new_data_path(
            category, relative_name or Path(source).name
        )
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        return destination.relative_to(self.root).as_posix()

    def new_data_path(self, category: str, relative_name: str) -> Path:
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

        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("relative_name must stay inside the selected data directory.")
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        return destination

    def relative_path(self, path: Path) -> str:
        path = Path(path).absolute()
        root = self.root.absolute()
        try:
            return path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path is outside rollout store: {path}") from exc

    def discard_uncommitted_data(self, trajectory_id: str) -> None:
        if not trajectory_id or Path(trajectory_id).name != trajectory_id:
            raise ValueError(f"Unsafe trajectory ID: {trajectory_id!r}")
        state_directory = self.state_data_dir / trajectory_id
        if state_directory.exists():
            shutil.rmtree(state_directory)
        for path in (
            self.cad_data_dir / f"{trajectory_id}.step",
            self.mesh_data_dir / f"{trajectory_id}.stl",
        ):
            if path.exists():
                path.unlink()
