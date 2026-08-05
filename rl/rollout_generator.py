import copy
import gc
import hashlib
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import dgl
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm

from misc import create_mesh
from models.pointercad import PointerCAD
from models.processor import Text2CADProcessor
from rl.cad_environment import (
    CADEnvironment,
    evaluate_models,
    load_target_model,
    save_brep_graph,
)
from rl.data import load_episode_index, load_trajectories
from rl.prompts import prompt_message
from rl.rollout_store import RolloutStore
from rl.schemas import (
    EpisodeRecord,
    StepRecord,
    TrajectoryRecord,
    canonical_json,
)

PARALLEL_GENERATOR_VERSION = "rollout-generator-v4-vectorized-predict"
SYSTEMIC_GENERATION_ERROR_LIMIT = 8


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        if hasattr(os, "posix_fadvise") and hasattr(
            os, "POSIX_FADV_DONTNEED"
        ):
            try:
                os.posix_fadvise(
                    file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                )
            except OSError:
                pass
    return digest.hexdigest()


def load_checkpoint(model, checkpoint_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    logger.info(
        "Loading checkpoint {} ({:.2f} GiB) with CPU mmap",
        checkpoint_path,
        checkpoint_path.stat().st_size / (1024**3),
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning(
            "Missing checkpoint key groups: {}",
            sorted(
                set(key.split(".")[0] for key in incompatible.missing_keys)
            ),
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected checkpoint keys: {}", incompatible.unexpected_keys
        )
    del state_dict
    del checkpoint
    gc.collect()


def torch_dtype(name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model dtype {name!r}; expected {sorted(mapping)}."
        ) from exc


def trajectory_seed(base_seed: int, task_id: str, rollout_index: int) -> int:
    task_hash = int(
        hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12], 16
    )
    return (base_seed + task_hash + rollout_index) % (2**63 - 1)


def trajectory_id(
    run_id: str,
    task_id: str,
    rollout_index: int,
    seed: int,
) -> str:
    identity = f"{run_id}|{task_id}|{rollout_index}|{seed}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{task_id}__r{rollout_index:03d}__{suffix}"


def set_sampling_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@contextmanager
def suppress_native_stdout(enabled: bool) -> Iterator[None]:
    """Silence verbose native-library stdout while preserving Python logs."""
    if not enabled:
        yield
        return

    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_output:
            os.dup2(null_output.fileno(), 1)
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def parameter_map_to_lists(parameter_map: Dict[str, Any]) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}
    for name in ("length", "angle"):
        value = parameter_map.get(name)
        if value is None:
            result[name] = []
        elif torch.is_tensor(value):
            result[name] = [
                float(item) for item in value.detach().cpu().tolist()
            ]
        else:
            result[name] = [float(item) for item in value]
    return result


class _LazyMesh:
    """Build a mesh at most once and retain either its value or its failure."""

    _UNSET = object()

    def __init__(self, builder: Callable[[], Any]):
        self._builder = builder
        self._value = self._UNSET
        self._error: Optional[Exception] = None
        self.build_time_seconds = 0.0

    def get(self):
        if self._error is not None:
            raise self._error
        if self._value is self._UNSET:
            started_at = time.monotonic()
            try:
                self._value = self._builder()
            except Exception as exc:
                self._error = exc
                raise
            finally:
                self.build_time_seconds += time.monotonic() - started_at
        return self._value


def _graph_to_payload(graph: dgl.DGLGraph) -> Dict[str, Any]:
    source, destination = graph.edges()
    return {
        "source": source.detach().cpu().numpy(),
        "destination": destination.detach().cpu().numpy(),
        "num_nodes": graph.num_nodes(),
        "node_data": {
            name: value.detach().cpu().numpy()
            for name, value in graph.ndata.items()
        },
        "edge_data": {
            name: value.detach().cpu().numpy()
            for name, value in graph.edata.items()
        },
    }


def _graph_from_payload(payload: Dict[str, Any]) -> dgl.DGLGraph:
    graph = dgl.graph(
        (
            torch.from_numpy(payload["source"]),
            torch.from_numpy(payload["destination"]),
        ),
        num_nodes=int(payload["num_nodes"]),
    )
    for name, value in payload["node_data"].items():
        graph.ndata[name] = torch.from_numpy(value)
    for name, value in payload["edge_data"].items():
        graph.edata[name] = torch.from_numpy(value)
    return graph


def _save_final_data_impl(
    store: RolloutStore,
    execution: Dict[str, Any],
    outputs: Dict[str, Any],
    trajectory_id_value: str,
    environment: CADEnvironment,
    prediction_mesh_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[
    Optional[str],
    Optional[str],
    Dict[str, str],
    Dict[str, float],
]:
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {
        "step_export_time_seconds": 0.0,
        "mesh_export_time_seconds": 0.0,
    }
    cad_path = None
    mesh_path = None
    if not environment.model.seq:
        return cad_path, mesh_path, errors, timings

    if bool(outputs.get("save_step", True)):
        destination = store.new_data_path(
            "cad", f"{trajectory_id_value}.step"
        )
        started_at = time.monotonic()
        try:
            with suppress_native_stdout(
                bool(outputs.get("suppress_step_export_output", True))
            ):
                exported = environment.model.export_model(
                    str(destination),
                    timeout=float(
                        execution.get("build_timeout_seconds", 300.0)
                    ),
                )
            if exported is False:
                raise RuntimeError("STEP writer reported an export failure.")
            if not destination.is_file():
                raise RuntimeError("STEP writer did not create an output file.")
            cad_path = store.relative_path(destination)
        except Exception as exc:
            errors["step"] = f"{type(exc).__name__}: {exc}"
            if destination.exists():
                destination.unlink()
        finally:
            timings["step_export_time_seconds"] += (
                time.monotonic() - started_at
            )

    if bool(outputs.get("save_mesh", True)):
        destination = store.new_data_path(
            "mesh", f"{trajectory_id_value}.stl"
        )
        try:
            mesh = (
                prediction_mesh_factory()
                if prediction_mesh_factory is not None
                else create_mesh(environment.model)
            )
            started_at = time.monotonic()
            try:
                mesh.export(str(destination))
            finally:
                timings["mesh_export_time_seconds"] += (
                    time.monotonic() - started_at
                )
            if not destination.is_file():
                raise RuntimeError("Mesh exporter did not create an output file.")
            mesh_path = store.relative_path(destination)
        except Exception as exc:
            errors["mesh"] = f"{type(exc).__name__}: {exc}"
            if destination.exists():
                destination.unlink()
    return cad_path, mesh_path, errors, timings


def exception_summary(exc: Exception) -> str:
    """Return a useful one-line error even for message-less assertions."""
    message = str(exc).strip()
    summary = type(exc).__name__
    if message:
        summary = f"{summary}: {message}"
    extracted = traceback.extract_tb(exc.__traceback__)
    if extracted:
        frame = extracted[-1]
        location = f"{Path(frame.filename).name}:{frame.lineno}"
        summary = f"{summary} ({location})"
    return summary


def decode_step_generation(
    tokenizer,
    generated_ids: torch.Tensor,
    parameter_map: Dict[str, Any],
    labels: torch.Tensor,
    parameters: torch.Tensor,
    pointers: torch.Tensor,
    behavior_log_probs: Dict[str, List[float]],
):
    label_values = [int(value) for value in labels.detach().cpu().tolist()]
    parameter_values = [
        int(value) for value in parameters.detach().cpu().tolist()
    ]
    pointer_values = [
        int(value) for value in pointers.detach().cpu().tolist()
    ]
    if not label_values:
        raise ValueError("Generation finished without a CAD action.")
    if not (
        len(label_values) == len(parameter_values) == len(pointer_values)
    ):
        raise ValueError("Generated structured CAD channels have different lengths.")

    token_values = [
        int(value) for value in generated_ids.detach().cpu().tolist()
    ]
    cad_start_id = tokenizer.convert_tokens_to_ids("<|cad_start|>")
    cad_pad_id = tokenizer.convert_tokens_to_ids("<|cad_pad|>")
    padded_lm_token_ids = [
        token_id for token_id in token_values if token_id != cad_pad_id
    ]
    behavior = {
        name: [float(value) for value in behavior_log_probs.get(name, [])]
        for name in ("plan", "label", "parameter", "pointer")
    }
    if not (
        len(behavior["label"])
        == len(behavior["parameter"])
        == len(behavior["pointer"])
        == len(label_values)
    ):
        raise ValueError(
            "Behavior structured log-probabilities do not align with actions."
        )
    sampled_lm_token_count = len(behavior["plan"])
    if len(padded_lm_token_ids) < sampled_lm_token_count:
        raise ValueError(
            "Behavior LM log-probabilities do not align with generated LM tokens."
        )
    trailing_token_ids = padded_lm_token_ids[sampled_lm_token_count:]
    allowed_padding_ids = {151643}
    tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
    if tokenizer_pad_id is not None:
        allowed_padding_ids.add(int(tokenizer_pad_id))
    if any(
        token_id not in allowed_padding_ids
        for token_id in trailing_token_ids
    ):
        raise ValueError(
            "Generated LM tokens contain a non-padding suffix without "
            "behavior log-probabilities."
        )
    lm_token_ids = padded_lm_token_ids[:sampled_lm_token_count]
    try:
        cad_start_position = lm_token_ids.index(cad_start_id)
    except ValueError as exc:
        raise ValueError(
            "Generation produced CAD actions without a <|cad_start|> token."
        ) from exc
    plan_text = tokenizer.decode(lm_token_ids[:cad_start_position])
    return (
        plan_text,
        lm_token_ids,
        parameter_map_to_lists(parameter_map),
        label_values,
        parameter_values,
        pointer_values,
        behavior,
    )


@dataclass(frozen=True)
class _RolloutJob:
    episode: EpisodeRecord
    rollout_index: int
    seed: int
    trajectory_id: str


def _rollout_error_result(
    job: _RolloutJob,
    error: str,
    metrics_version: str,
    elapsed_seconds: float,
) -> Tuple[TrajectoryRecord, List[StepRecord]]:
    return (
        TrajectoryRecord(
            trajectory_id=job.trajectory_id,
            task_id=job.episode.task_id,
            rollout_index=job.rollout_index,
            seed=job.seed,
            termination_reason="rollout_error",
            num_generated_steps=0,
            valid=False,
            final_cad_path=None,
            final_mesh_path=None,
            execution_error=error,
            metrics_json=canonical_json(
                {
                    "termination_reason": "rollout_error",
                    "rollout_error": error,
                    "timings": {"rollout_time_seconds": elapsed_seconds},
                    "generation_counts": {
                        "plan_tokens": 0,
                        "cad_actions": 0,
                    },
                }
            ),
            generation_time_seconds=elapsed_seconds,
            execution_time_seconds=0.0,
            metrics_version=metrics_version,
        ),
        [],
    )


class _CPUWorkerTrajectory:
    """Own one CAD environment entirely inside an isolated CPU process."""

    def __init__(
        self,
        job: _RolloutJob,
        config: Dict[str, Any],
        rollout_root: Path,
        source_dataset_root: Path,
    ):
        self.job = job
        self.config = config
        self.generation = config["generation"]
        self.execution = config["execution"]
        self.metrics_config = config["metrics"]
        self.outputs = config["outputs"]
        self.store = RolloutStore(Path(rollout_root))
        self.source_dataset_root = Path(source_dataset_root)
        self.store.discard_uncommitted_data(job.trajectory_id)
        set_sampling_seed(job.seed, torch.device("cpu"))
        self.rollout_started_at = time.monotonic()
        self.environment = CADEnvironment(
            strict=bool(self.execution.get("strict", True)),
            surf_u_samples=int(self.execution.get("surf_u_samples", 32)),
            surf_v_samples=int(self.execution.get("surf_v_samples", 32)),
            curv_u_samples=int(self.execution.get("curv_u_samples", 32)),
            build_timeout_seconds=float(
                self.execution.get("build_timeout_seconds", 300.0)
            ),
        )
        self.steps: List[StepRecord] = []
        self.step_index = 0
        self.generation_seconds = 0.0
        self.execution_seconds = 0.0
        self.termination_reason = "max_episode_steps"
        self.trajectory_error: Optional[str] = None
        self.step_errors: List[str] = []
        self.gpu_batch_sizes: List[int] = []
        self.current_state_relative_path: Optional[str] = None
        self.detailed_timings: Dict[str, float] = {
            "target_load_time_seconds": 0.0,
            "state_graph_time_seconds": 0.0,
            "state_save_time_seconds": 0.0,
            "prompt_render_time_seconds": 0.0,
            "input_preparation_time_seconds": 0.0,
            "model_generation_time_seconds": 0.0,
            "generation_decode_time_seconds": 0.0,
            "step_export_time_seconds": 0.0,
            "mesh_export_time_seconds": 0.0,
        }
        self.target_model = None
        self.target_model_error = None
        started_at = time.monotonic()
        try:
            self.target_model = load_target_model(
                self.source_dataset_root,
                job.episode.target_cad_path,
            )
        except Exception as exc:
            self.target_model_error = exception_summary(exc)
        finally:
            self.detailed_timings["target_load_time_seconds"] += (
                time.monotonic() - started_at
            )

    def prepare_state(self) -> dgl.DGLGraph:
        state_path = self.store.new_data_path(
            "states",
            f"{self.job.trajectory_id}/step-{self.step_index:04d}.bin",
        )
        started_at = time.monotonic()
        try:
            graph = self.environment.state_graph()
        finally:
            self.detailed_timings["state_graph_time_seconds"] += (
                time.monotonic() - started_at
            )
        started_at = time.monotonic()
        try:
            save_brep_graph(graph, state_path)
            self.current_state_relative_path = self.store.relative_path(
                state_path
            )
        finally:
            self.detailed_timings["state_save_time_seconds"] += (
                time.monotonic() - started_at
            )
        return graph

    def apply_action(
        self,
        decoded,
        action_timings: Dict[str, float],
        generation_seconds: float,
        gpu_batch_size: int,
    ) -> bool:
        for name, value in action_timings.items():
            self.detailed_timings[name] += float(value)
        self.generation_seconds += float(generation_seconds)
        self.gpu_batch_sizes.append(int(gpu_batch_size))
        (
            plan_text,
            plan_token_ids,
            parameter_map,
            labels,
            parameters,
            pointers,
            behavior,
        ) = decoded

        execution_valid = False
        execution_error = None
        state_after_id = None
        model_end = False
        execution_started_at = time.monotonic()
        try:
            model_end, _ = self.environment.step(
                labels=labels,
                parameters=parameters,
                pointers=pointers,
                parameter_map=parameter_map,
            )
            execution_valid = True
            state_after_id = (
                f"{self.job.trajectory_id}:state:{self.step_index + 1:04d}"
            )
        except Exception as exc:
            execution_error = exception_summary(exc)
            self.step_errors.append(execution_error)
        finally:
            self.execution_seconds += (
                time.monotonic() - execution_started_at
            )

        if self.current_state_relative_path is None:
            raise RuntimeError("CPU worker has no prepared state for its action.")
        self.steps.append(
            StepRecord(
                trajectory_id=self.job.trajectory_id,
                step_index=self.step_index,
                state_before_id=(
                    f"{self.job.trajectory_id}:state:{self.step_index:04d}"
                ),
                state_before_graph_path=self.current_state_relative_path,
                state_after_id=state_after_id,
                plan_text=plan_text,
                plan_token_ids=plan_token_ids,
                parameter_map_json=canonical_json(parameter_map),
                labels=labels,
                parameters=parameters,
                pointers=pointers,
                behavior_plan_logps=behavior["plan"],
                behavior_label_logps=behavior["label"],
                behavior_parameter_logps=behavior["parameter"],
                behavior_pointer_logps=behavior["pointer"],
                execution_valid=execution_valid,
                execution_error=execution_error,
            )
        )
        self.current_state_relative_path = None
        self.step_index += 1

        if not execution_valid:
            self.termination_reason = "execution_error"
            self.trajectory_error = execution_error
            return True
        if model_end:
            self.termination_reason = "model_end"
            return True
        if self.step_index >= int(
            self.generation.get("max_episode_steps", 64)
        ):
            self.termination_reason = "max_episode_steps"
            return True
        return False

    def fail(self, termination_reason: str, error: str) -> None:
        self.termination_reason = termination_reason
        self.trajectory_error = error

    def finalize(self) -> Tuple[TrajectoryRecord, List[StepRecord]]:
        valid = self.termination_reason == "model_end"
        metrics: Dict[str, Any] = {
            "termination_reason": self.termination_reason,
            "step_execution_errors": self.step_errors,
        }
        if self.target_model_error is not None:
            metrics["target_model_error"] = self.target_model_error

        final_cad_path = None
        final_mesh_path = None
        prediction_mesh: Optional[_LazyMesh] = None
        target_mesh: Optional[_LazyMesh] = None
        if self.environment.model.seq:
            prediction_mesh = _LazyMesh(
                lambda: create_mesh(self.environment.model)
            )
            if self.target_model is not None:
                target_mesh = _LazyMesh(lambda: create_mesh(self.target_model))
                try:
                    evaluated = evaluate_models(
                        prediction=self.environment.model,
                        target=self.target_model,
                        enabled_metrics=self.metrics_config.get(
                            "enabled",
                            [
                                "iou",
                                "chamfer_distance",
                                "accuracy",
                                "f1",
                                "watertight",
                            ],
                        ),
                        chamfer_points=int(
                            self.metrics_config.get("chamfer_points", 8192)
                        ),
                        prediction_mesh_factory=prediction_mesh.get,
                        target_mesh_factory=target_mesh.get,
                    )
                    metrics.update(evaluated)
                except Exception as exc:
                    metrics["evaluation_error"] = exception_summary(exc)

            (
                final_cad_path,
                final_mesh_path,
                output_errors,
                output_timings,
            ) = _save_final_data_impl(
                store=self.store,
                execution=self.execution,
                outputs=self.outputs,
                trajectory_id_value=self.job.trajectory_id,
                environment=self.environment,
                prediction_mesh_factory=prediction_mesh.get,
            )
            self.detailed_timings.update(output_timings)
            metrics["output_errors"] = output_errors

        self.detailed_timings["generation_time_seconds"] = (
            self.generation_seconds
        )
        self.detailed_timings["execution_time_seconds"] = (
            self.execution_seconds
        )
        self.detailed_timings["prediction_mesh_build_time_seconds"] = (
            prediction_mesh.build_time_seconds
            if prediction_mesh is not None
            else 0.0
        )
        self.detailed_timings["target_mesh_build_time_seconds"] = (
            target_mesh.build_time_seconds if target_mesh is not None else 0.0
        )
        self.detailed_timings["rollout_time_seconds"] = (
            time.monotonic() - self.rollout_started_at
        )
        metrics["timings"] = self.detailed_timings
        metrics["generation_counts"] = {
            "plan_tokens": sum(
                len(step.plan_token_ids) for step in self.steps
            ),
            "cad_actions": sum(len(step.labels) for step in self.steps),
            "gpu_batches": len(self.gpu_batch_sizes),
            "mean_gpu_batch_size": (
                sum(self.gpu_batch_sizes) / len(self.gpu_batch_sizes)
                if self.gpu_batch_sizes
                else 0.0
            ),
            "min_gpu_batch_size": (
                min(self.gpu_batch_sizes) if self.gpu_batch_sizes else 0
            ),
            "max_gpu_batch_size": (
                max(self.gpu_batch_sizes) if self.gpu_batch_sizes else 0
            ),
        }
        return (
            TrajectoryRecord(
                trajectory_id=self.job.trajectory_id,
                task_id=self.job.episode.task_id,
                rollout_index=self.job.rollout_index,
                seed=self.job.seed,
                termination_reason=self.termination_reason,
                num_generated_steps=len(self.steps),
                valid=valid,
                final_cad_path=final_cad_path,
                final_mesh_path=final_mesh_path,
                execution_error=self.trajectory_error,
                metrics_json=canonical_json(metrics),
                generation_time_seconds=self.generation_seconds,
                execution_time_seconds=self.execution_seconds,
                metrics_version=str(self.metrics_config["version"]),
            ),
            self.steps,
        )


def _cpu_rollout_worker(
    worker_id: int,
    input_queue,
    output_queue,
    config: Dict[str, Any],
    rollout_root: str,
    source_dataset_root: str,
) -> None:
    torch.set_num_threads(
        int(config["execution"].get("cpu_threads_per_worker", 1))
    )
    runner: Optional[_CPUWorkerTrajectory] = None
    while True:
        message = input_queue.get()
        kind = message[0]
        if kind == "stop":
            return
        try:
            if kind == "start":
                job = message[1]
                runner = _CPUWorkerTrajectory(
                    job=job,
                    config=config,
                    rollout_root=Path(rollout_root),
                    source_dataset_root=Path(source_dataset_root),
                )
                try:
                    graph = runner.prepare_state()
                except Exception as exc:
                    runner.fail("state_graph_error", exception_summary(exc))
                    trajectory, steps = runner.finalize()
                    output_queue.put(("done", worker_id, trajectory, steps))
                    runner = None
                else:
                    output_queue.put(
                        (
                            "ready",
                            worker_id,
                            job.trajectory_id,
                            _graph_to_payload(graph),
                        )
                    )
            elif kind == "action":
                if runner is None:
                    raise RuntimeError("CPU worker received action without a job.")
                (
                    _,
                    decoded,
                    action_timings,
                    generation_seconds,
                    gpu_batch_size,
                ) = message
                if runner.apply_action(
                    decoded,
                    action_timings,
                    generation_seconds,
                    gpu_batch_size,
                ):
                    trajectory, steps = runner.finalize()
                    output_queue.put(
                        ("done", worker_id, trajectory, steps)
                    )
                    runner = None
                else:
                    try:
                        graph = runner.prepare_state()
                    except Exception as exc:
                        runner.fail(
                            "state_graph_error", exception_summary(exc)
                        )
                        trajectory, steps = runner.finalize()
                        output_queue.put(
                            ("done", worker_id, trajectory, steps)
                        )
                        runner = None
                    else:
                        output_queue.put(
                            (
                                "ready",
                                worker_id,
                                runner.job.trajectory_id,
                                _graph_to_payload(graph),
                            )
                        )
            elif kind == "fail":
                if runner is None:
                    raise RuntimeError("CPU worker received failure without a job.")
                _, termination_reason, error = message
                runner.fail(termination_reason, error)
                trajectory, steps = runner.finalize()
                output_queue.put(("done", worker_id, trajectory, steps))
                runner = None
            else:
                raise ValueError(f"Unknown CPU worker message: {kind!r}")
        except Exception as exc:
            error = exception_summary(exc)
            if runner is not None:
                runner.fail("rollout_error", error)
                try:
                    trajectory, steps = runner.finalize()
                except Exception as finalization_exc:
                    error = exception_summary(finalization_exc)
                    trajectory, steps = _rollout_error_result(
                        runner.job,
                        error,
                        str(config["metrics"]["version"]),
                        time.monotonic() - runner.rollout_started_at,
                    )
                output_queue.put(("done", worker_id, trajectory, steps))
                runner = None
            else:
                output_queue.put(("worker_error", worker_id, error))


class RolloutGenerator:
    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        store: RolloutStore,
        source_dataset_root: Path,
        config: Dict[str, Any],
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.store = store
        self.source_dataset_root = Path(source_dataset_root)
        self.generation = config["generation"]
        self._prompt_cache: OrderedDict[str, str] = OrderedDict()
        self._prompt_cache_size = max(
            64, int(self.generation.get("cpu_workers_per_gpu", 1)) * 2
        )

    def _render_prompt(self, prompt: str) -> str:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached
        messages = prompt_message(prompt)
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError(
                "Expected apply_chat_template to return one rendered string, "
                f"got {type(rendered).__name__}."
            )
        self._prompt_cache[prompt] = rendered
        if len(self._prompt_cache) > self._prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        return rendered

    def _generate_actions(
        self,
        prompts: Sequence[str],
        graphs: Sequence[dgl.DGLGraph],
        sampling_generators: Sequence[Optional[torch.Generator]],
    ) -> Tuple[
        List[Optional[Any]],
        List[Optional[str]],
        Dict[str, float],
    ]:
        if not prompts or len(prompts) != len(graphs):
            raise ValueError("Batched generation requires aligned prompts/graphs.")
        if len(sampling_generators) != len(prompts):
            raise ValueError(
                "Batched generation requires one RNG generator per prompt."
            )
        timings: Dict[str, float] = {}
        started_at = time.monotonic()
        text = [self._render_prompt(prompt) for prompt in prompts]
        timings["prompt_render_time_seconds"] = (
            time.monotonic() - started_at
        )

        started_at = time.monotonic()
        inputs = self.processor(
            text=text,
            breps=dgl.batch(list(graphs)),
            max_length=int(self.generation.get("max_input_length", 3072)),
        ).to(self.device)
        timings["input_preparation_time_seconds"] = (
            time.monotonic() - started_at
        )
        temperatures = self.generation["temperatures"]
        started_at = time.monotonic()
        outputs = self.model.predict(
            tokenizer=self.processor.tokenizer,
            max_steps=int(
                self.generation.get("max_generation_steps", 1024)
            ),
            mode="sample",
            temperature_lm=float(temperatures.get("plan", 1.0)),
            temperature_label=float(temperatures.get("label", 1.0)),
            temperature_parameter=float(
                temperatures.get("parameter", 1.0)
            ),
            temperature_pointer=float(temperatures.get("pointer", 1.0)),
            return_log_probs=True,
            sampling_generators=sampling_generators,
            **inputs,
        )
        timings["model_generation_time_seconds"] = (
            time.monotonic() - started_at
        )
        (
            generated_ids,
            parameter_maps,
            labels,
            parameters,
            pointers,
            behavior_log_probs,
        ) = outputs
        started_at = time.monotonic()
        decoded: List[Optional[Any]] = []
        decode_errors: List[Optional[str]] = []
        for index in range(len(prompts)):
            try:
                action = decode_step_generation(
                    tokenizer=self.processor.tokenizer,
                    generated_ids=generated_ids[index],
                    parameter_map=parameter_maps[index],
                    labels=labels[index],
                    parameters=parameters[index],
                    pointers=pointers[index],
                    behavior_log_probs=behavior_log_probs[index],
                )
            except ValueError as exc:
                if str(exc) != "Generation finished without a CAD action.":
                    raise
                decoded.append(None)
                decode_errors.append(exception_summary(exc))
            else:
                decoded.append(action)
                decode_errors.append(None)
        timings["generation_decode_time_seconds"] = (
            time.monotonic() - started_at
        )
        return decoded, decode_errors, timings


def _limit_episodes(
    episodes: Sequence[EpisodeRecord],
    split: str,
    generation_config: Dict[str, Any],
) -> List[EpisodeRecord]:
    selected_ids = generation_config.get("task_ids")
    if selected_ids:
        selected: Set[str] = {str(value) for value in selected_ids}
        episodes = [
            episode for episode in episodes if episode.task_id in selected
        ]
    limits = generation_config.get("max_episodes_per_split")
    if isinstance(limits, dict):
        limit = limits.get(split)
    else:
        limit = limits
    if limit is not None:
        episodes = list(episodes)[: int(limit)]
    return list(episodes)


def _run_batched_jobs(
    generator: RolloutGenerator,
    jobs: Sequence[_RolloutJob],
    config: Dict[str, Any],
    on_result: Callable[[TrajectoryRecord, List[StepRecord]], None],
) -> None:
    if not jobs:
        return
    generation_config = config["generation"]
    batch_size = int(generation_config.get("batch_size", 1))
    effective_batch_size = batch_size
    oom_batch_limit = batch_size
    requested_cpu_workers = int(
        generation_config.get("cpu_workers_per_gpu", batch_size)
    )
    if hasattr(os, "sched_getaffinity"):
        available_cpu_count = len(os.sched_getaffinity(0))
    else:
        available_cpu_count = os.cpu_count() or 1
    local_world_size = max(1, int(os.getenv("LOCAL_WORLD_SIZE", "1")))
    cpu_workers_per_rank = max(1, available_cpu_count // local_world_size)
    cpu_worker_count = min(
        requested_cpu_workers,
        cpu_workers_per_rank,
        len(jobs),
    )
    batch_wait_seconds = float(
        generation_config.get("batch_wait_seconds", 0.02)
    )
    if batch_size <= 0 or cpu_worker_count <= 0:
        raise ValueError("batch_size and cpu_workers_per_gpu must be positive.")
    if cpu_worker_count < requested_cpu_workers:
        logger.warning(
            "Capping CPU workers from {} to {} for this rank based on {} "
            "available CPUs and LOCAL_WORLD_SIZE={}.",
            requested_cpu_workers,
            cpu_worker_count,
            available_cpu_count,
            local_world_size,
        )
    if cpu_worker_count < batch_size:
        logger.warning(
            "cpu_workers_per_gpu={} is smaller than batch_size={}; GPU batches "
            "cannot become full.",
            cpu_worker_count,
            batch_size,
        )

    context = mp.get_context("spawn")
    output_queue = context.Queue(maxsize=max(4, cpu_worker_count * 2))
    input_queues = [context.Queue(maxsize=2) for _ in range(cpu_worker_count)]
    processes: List[mp.Process] = []
    worker_jobs: Dict[int, _RolloutJob] = {}
    sampling_generators: Dict[str, torch.Generator] = {}
    ready: Deque[Tuple[int, str, Dict[str, Any]]] = deque()
    next_job_index = 0
    completed = 0
    last_generation_error: Optional[str] = None
    consecutive_generation_errors = 0

    def start_process(worker_id: int) -> mp.Process:
        process = context.Process(
            target=_cpu_rollout_worker,
            args=(
                worker_id,
                input_queues[worker_id],
                output_queue,
                config,
                str(generator.store.root),
                str(generator.source_dataset_root),
            ),
            name=f"pointercad-cpu-{worker_id}",
        )
        process.start()
        return process

    def assign_job(worker_id: int) -> bool:
        nonlocal next_job_index
        if next_job_index >= len(jobs):
            worker_jobs.pop(worker_id, None)
            return False
        job = jobs[next_job_index]
        next_job_index += 1
        worker_jobs[worker_id] = job
        rng = torch.Generator(device=generator.device)
        rng.manual_seed(job.seed)
        sampling_generators[job.trajectory_id] = rng
        input_queues[worker_id].put(("start", job))
        return True

    def handle_message(message) -> None:
        nonlocal completed
        kind = message[0]
        if kind == "ready":
            _, worker_id, trajectory_id_value, graph_payload = message
            job = worker_jobs.get(worker_id)
            if job is None or job.trajectory_id != trajectory_id_value:
                raise RuntimeError(
                    "CPU worker returned a state for an unexpected trajectory."
                )
            ready.append((worker_id, trajectory_id_value, graph_payload))
            return
        if kind == "done":
            _, worker_id, trajectory, steps = message
            job = worker_jobs.get(worker_id)
            if job is None or job.trajectory_id != trajectory.trajectory_id:
                raise RuntimeError(
                    "CPU worker completed an unexpected trajectory."
                )
            on_result(trajectory, steps)
            sampling_generators.pop(job.trajectory_id, None)
            completed += 1
            assign_job(worker_id)
            return
        if kind == "worker_error":
            _, worker_id, error = message
            raise RuntimeError(f"CPU worker {worker_id} failed: {error}")
        raise ValueError(f"Unknown CPU worker output: {kind!r}")

    def recover_dead_workers() -> None:
        nonlocal completed
        for worker_id, process in enumerate(processes):
            if process.is_alive() or process.exitcode is None:
                continue
            job = worker_jobs.pop(worker_id, None)
            ready_items = [item for item in ready if item[0] != worker_id]
            ready.clear()
            ready.extend(ready_items)
            if job is not None:
                error = (
                    f"CPU worker {worker_id} exited with code "
                    f"{process.exitcode}."
                )
                logger.error("{} Trajectory {} failed.", error, job.trajectory_id)
                generator.store.discard_uncommitted_data(job.trajectory_id)
                trajectory, steps = _rollout_error_result(
                    job,
                    error,
                    str(config["metrics"]["version"]),
                    0.0,
                )
                on_result(trajectory, steps)
                sampling_generators.pop(job.trajectory_id, None)
                completed += 1
            if next_job_index < len(jobs):
                input_queues[worker_id].close()
                input_queues[worker_id] = context.Queue(maxsize=2)
                processes[worker_id] = start_process(worker_id)
                assign_job(worker_id)

    try:
        processes.extend(
            start_process(worker_id)
            for worker_id in range(cpu_worker_count)
        )
        for worker_id in range(cpu_worker_count):
            assign_job(worker_id)

        while completed < len(jobs):
            if not ready:
                try:
                    handle_message(output_queue.get(timeout=1.0))
                except queue.Empty:
                    recover_dead_workers()
                    continue

            deadline = time.monotonic() + batch_wait_seconds
            while len(ready) < effective_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    handle_message(output_queue.get(timeout=remaining))
                except queue.Empty:
                    break

            batch = [
                ready.popleft()
                for _ in range(min(effective_batch_size, len(ready)))
            ]
            if not batch:
                continue
            batch_jobs = [worker_jobs[worker_id] for worker_id, _, _ in batch]
            batch_generators = [
                sampling_generators[trajectory_id_value]
                for _, trajectory_id_value, _ in batch
            ]
            generator_states = [rng.get_state() for rng in batch_generators]
            started_at = time.monotonic()
            try:
                (
                    decoded,
                    decode_errors,
                    action_timings,
                ) = generator._generate_actions(
                    prompts=[job.episode.prompt for job in batch_jobs],
                    graphs=[
                        _graph_from_payload(graph_payload)
                        for _, _, graph_payload in batch
                    ],
                    sampling_generators=batch_generators,
                )
                generation_seconds = time.monotonic() - started_at
                successful_decodes = sum(
                    error is None for error in decode_errors
                )
                last_generation_error = None
                consecutive_generation_errors = 0
                if successful_decodes < len(batch):
                    error_counts = Counter(
                        error for error in decode_errors if error is not None
                    )
                    logger.warning(
                        "Decoded {}/{} batch items; failing only {} invalid "
                        "item(s) without repeating GPU generation: {}",
                        successful_decodes,
                        len(batch),
                        len(batch) - successful_decodes,
                        dict(error_counts),
                    )
                for batch_index, (worker_id, _, _) in enumerate(batch):
                    decode_error = decode_errors[batch_index]
                    if decode_error is None:
                        input_queues[worker_id].put(
                            (
                                "action",
                                decoded[batch_index],
                                action_timings,
                                generation_seconds,
                                len(batch),
                            )
                        )
                    else:
                        input_queues[worker_id].put(
                            ("fail", "generation_error", decode_error)
                        )
                effective_batch_size = min(
                    oom_batch_limit,
                    max(effective_batch_size, len(batch) * 2),
                )
            except Exception as exc:
                is_cuda_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(exc).lower()
                    and generator.device.type == "cuda"
                )
                if len(batch) > 1:
                    for rng, state in zip(batch_generators, generator_states):
                        rng.set_state(state)
                    effective_batch_size = max(1, len(batch) // 2)
                    if is_cuda_oom:
                        oom_batch_limit = min(
                            oom_batch_limit, effective_batch_size
                        )
                    for item in reversed(batch):
                        ready.appendleft(item)
                    if is_cuda_oom:
                        torch.cuda.empty_cache()
                        logger.warning(
                            "CUDA OOM at batch {}; retrying with adaptive "
                            "batch {}.",
                            len(batch),
                            effective_batch_size,
                        )
                    else:
                        batch_error = exception_summary(exc)
                        logger.warning(
                            "Batched generation failed at batch {}; retrying "
                            "smaller batches to isolate the failing item: {}",
                            len(batch),
                            batch_error,
                        )
                    continue
                error = exception_summary(exc)
                if error == last_generation_error:
                    consecutive_generation_errors += 1
                else:
                    last_generation_error = error
                    consecutive_generation_errors = 1
                if (
                    consecutive_generation_errors
                    >= SYSTEMIC_GENERATION_ERROR_LIMIT
                ):
                    raise RuntimeError(
                        "Aborting rollout generation after "
                        f"{consecutive_generation_errors} consecutive "
                        f"identical generation errors: {error}"
                    ) from exc
                for worker_id, _, _ in batch:
                    input_queues[worker_id].put(
                        ("fail", "generation_error", error)
                    )
            recover_dead_workers()
    finally:
        for worker_id, process in enumerate(processes):
            if process.is_alive():
                try:
                    input_queues[worker_id].put_nowait(("stop",))
                except queue.Full:
                    pass
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for input_queue in input_queues:
            input_queue.close()
        output_queue.close()


def _task_rank(task_id: str, world_size: int) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % world_size


def _generate_parallel_splits(
    generator: RolloutGenerator,
    store: RolloutStore,
    config: Dict[str, Any],
    existing_trajectories: Dict[str, TrajectoryRecord],
    rank: int,
    world_size: int,
) -> None:
    generation_config = config["generation"]
    trajectories_per_episode = int(
        generation_config["trajectories_per_episode"]
    )
    base_seed = int(generation_config.get("base_seed", 0))
    write_shard_size = int(generation_config.get("write_shard_size", 64))
    run_id = str(config["run_id"])
    splits = [
        str(value)
        for value in config.get("splits", ["train", "validation"])
    ]
    existing_ids = set(existing_trajectories)
    pending_trajectories: List[TrajectoryRecord] = []
    pending_steps: List[StepRecord] = []
    pending_ids: Set[str] = set()
    run_stats: Counter[str] = Counter()
    run_terminations: Counter[str] = Counter()

    def flush_pending() -> None:
        if not pending_trajectories:
            return
        store.append(pending_trajectories, pending_steps)
        existing_trajectories.update(
            {
                trajectory.trajectory_id: trajectory
                for trajectory in pending_trajectories
            }
        )
        existing_ids.update(pending_ids)
        pending_trajectories.clear()
        pending_steps.clear()
        pending_ids.clear()

    with torch.inference_mode():
        for split in splits:
            all_episodes = _limit_episodes(
                load_episode_index(config["episodes_root"], split),
                split,
                generation_config,
            )
            episodes = [
                episode
                for episode in all_episodes
                if _task_rank(episode.task_id, world_size) == rank
            ]
            split_stats: Counter[str] = Counter()
            split_terminations: Counter[str] = Counter()
            task_stats: Dict[str, Counter[str]] = {
                episode.task_id: Counter() for episode in episodes
            }
            task_terminations: Dict[str, Counter[str]] = {
                episode.task_id: Counter() for episode in episodes
            }
            jobs: List[_RolloutJob] = []
            progress = tqdm(
                total=len(episodes) * trajectories_per_episode,
                desc=f"rollouts:{split}:rank{rank}",
                unit="rollout",
                dynamic_ncols=True,
                disable=(rank != 0),
            )

            def update_progress(task_id: str) -> None:
                progress.update(1)
                progress.set_postfix(
                    rank=rank,
                    task=task_id,
                    valid=(
                        f"{split_stats['valid']}/{split_stats['rollouts']}"
                    ),
                    batch=generation_config.get("batch_size", 1),
                )

            def account_result(
                trajectory: TrajectoryRecord,
                steps: List[StepRecord],
                generated: bool,
            ) -> None:
                task_id = trajectory.task_id
                stats = task_stats[task_id]
                terminations = task_terminations[task_id]
                _update_rollout_stats(stats, trajectory)
                _update_rollout_stats(split_stats, trajectory)
                _update_rollout_stats(run_stats, trajectory)
                if generated:
                    stats["generated"] += 1
                    split_stats["generated"] += 1
                    run_stats["generated"] += 1
                else:
                    stats["skipped_existing"] += 1
                    split_stats["skipped_existing"] += 1
                    run_stats["skipped_existing"] += 1
                terminations[trajectory.termination_reason] += 1
                split_terminations[trajectory.termination_reason] += 1
                run_terminations[trajectory.termination_reason] += 1
                if generated:
                    if trajectory.trajectory_id in pending_ids:
                        raise ValueError(
                            "Duplicate pending trajectory ID: "
                            f"{trajectory.trajectory_id}"
                        )
                    pending_trajectories.append(trajectory)
                    pending_steps.extend(steps)
                    pending_ids.add(trajectory.trajectory_id)
                    elapsed = float(
                        trajectory.metrics.get("timings", {}).get(
                            "rollout_time_seconds", 0.0
                        )
                    )
                    _log_rollout_result(
                        trajectory,
                        trajectory.rollout_index + 1,
                        trajectories_per_episode,
                        elapsed,
                    )
                    if len(pending_trajectories) >= write_shard_size:
                        flush_pending()
                update_progress(task_id)
                if stats["rollouts"] == trajectories_per_episode:
                    logger.info(
                        "[rank {}] Task {} complete: valid(model_end)={}/{} "
                        "generated={} skipped_existing={} terminations={}",
                        rank,
                        task_id,
                        stats["valid"],
                        stats["rollouts"],
                        stats["generated"],
                        stats["skipped_existing"],
                        _format_termination_counts(terminations),
                    )

            for episode in episodes:
                for rollout_index in range(trajectories_per_episode):
                    seed = trajectory_seed(
                        base_seed, episode.task_id, rollout_index
                    )
                    identifier = trajectory_id(
                        run_id,
                        episode.task_id,
                        rollout_index,
                        seed,
                    )
                    if identifier in existing_ids:
                        account_result(
                            existing_trajectories[identifier], [], False
                        )
                        continue
                    jobs.append(
                        _RolloutJob(
                            episode=episode,
                            rollout_index=rollout_index,
                            seed=seed,
                            trajectory_id=identifier,
                        )
                    )

            logger.info(
                "[rank {}/{}] Split {}: tasks={} pending_rollouts={} "
                "batch_size={} cpu_workers={}",
                rank,
                world_size,
                split,
                len(episodes),
                len(jobs),
                generation_config.get("batch_size", 1),
                generation_config.get("cpu_workers_per_gpu", 1),
            )
            _run_batched_jobs(
                generator=generator,
                jobs=jobs,
                config=config,
                on_result=lambda trajectory, steps: account_result(
                    trajectory, steps, True
                ),
            )
            flush_pending()
            progress.close()
            logger.info(
                "[rank {}] Split {} complete: tasks={} rollouts={} "
                "valid(model_end)={}/{} generated={} skipped_existing={} "
                "terminations={}",
                rank,
                split,
                len(episodes),
                split_stats["rollouts"],
                split_stats["valid"],
                split_stats["rollouts"],
                split_stats["generated"],
                split_stats["skipped_existing"],
                _format_termination_counts(split_terminations),
            )
        flush_pending()

    logger.info(
        "[rank {}/{}] Rollout run {} complete: rollouts={} "
        "valid(model_end)={}/{} generated={} skipped_existing={} "
        "terminations={}",
        rank,
        world_size,
        run_id,
        run_stats["rollouts"],
        run_stats["valid"],
        run_stats["rollouts"],
        run_stats["generated"],
        run_stats["skipped_existing"],
        _format_termination_counts(run_terminations),
    )


def _update_rollout_stats(
    stats: Counter[str], trajectory: TrajectoryRecord
) -> None:
    stats["rollouts"] += 1
    stats["valid"] += int(trajectory.valid)
    stats["cad_saved"] += int(trajectory.final_cad_path is not None)
    stats["mesh_saved"] += int(trajectory.final_mesh_path is not None)


def _format_termination_counts(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    return ",".join(
        f"{reason}:{count}" for reason, count in sorted(counts.items())
    )


def _metrics_status(trajectory: TrajectoryRecord) -> str:
    metrics = trajectory.metrics
    if "target_model_error" in metrics:
        return "target_error"
    if "evaluation_error" in metrics:
        return "error"
    metric_errors = metrics.get("metric_errors")
    if isinstance(metric_errors, dict) and metric_errors:
        return "partial"
    if "operation_count" in metrics:
        return "ok"
    return "not_run"


def _trajectory_problem_summary(trajectory: TrajectoryRecord) -> str:
    problems = []
    if trajectory.execution_error:
        problems.append(f"error={trajectory.execution_error}")
    metrics = trajectory.metrics
    evaluation_error = metrics.get("evaluation_error")
    if evaluation_error:
        problems.append(f"evaluation_error={evaluation_error}")
    target_model_error = metrics.get("target_model_error")
    if target_model_error:
        problems.append(f"target_model_error={target_model_error}")
    metric_errors = metrics.get("metric_errors")
    if isinstance(metric_errors, dict) and metric_errors:
        problems.append(f"metric_errors={','.join(sorted(metric_errors))}")
    output_errors = metrics.get("output_errors")
    if isinstance(output_errors, dict):
        for output_name, error in sorted(output_errors.items()):
            problems.append(f"{output_name}_export_error={error}")
    return f" {'; '.join(problems)}" if problems else ""


def _log_rollout_result(
    trajectory: TrajectoryRecord,
    rollout_position: int,
    trajectories_per_episode: int,
    elapsed_seconds: float,
) -> None:
    log = logger.info if trajectory.valid else logger.warning
    metrics = trajectory.metrics
    timings = metrics.get("timings", {})
    counts = metrics.get("generation_counts", {})
    log(
        "[rank {}] Rollout {}/{} for task {} ({}): valid={} termination={} steps={} "
        "plan_tokens={} cad_actions={} mean_batch={} step_export={} mesh_export={} "
        "metrics={} time[generation={:.2f}s execution={:.2f}s "
        "state={:.2f}s metrics={:.2f}s mesh_build_subset={:.2f}s "
        "export={:.2f}s total={:.2f}s]{}",
        os.getenv("RANK", "0"),
        rollout_position,
        trajectories_per_episode,
        trajectory.task_id,
        trajectory.trajectory_id,
        trajectory.valid,
        trajectory.termination_reason,
        trajectory.num_generated_steps,
        counts.get("plan_tokens", "unknown"),
        counts.get("cad_actions", "unknown"),
        counts.get("mean_gpu_batch_size", 1.0),
        "saved" if trajectory.final_cad_path is not None else "missing",
        "saved" if trajectory.final_mesh_path is not None else "missing",
        _metrics_status(trajectory),
        float(timings.get("generation_time_seconds", 0.0)),
        float(timings.get("execution_time_seconds", 0.0)),
        float(timings.get("state_graph_time_seconds", 0.0))
        + float(timings.get("state_save_time_seconds", 0.0)),
        float(metrics.get("metrics_time_seconds", 0.0)),
        float(timings.get("prediction_mesh_build_time_seconds", 0.0))
        + float(timings.get("target_mesh_build_time_seconds", 0.0)),
        float(timings.get("step_export_time_seconds", 0.0))
        + float(timings.get("mesh_export_time_seconds", 0.0)),
        elapsed_seconds,
        _trajectory_problem_summary(trajectory),
    )


def build_runtime_config(
    config: Dict[str, Any], checkpoint_path: Path
) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    result["runtime"] = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return result


def validate_rollout_config(config: Dict[str, Any]) -> None:
    for key in (
        "run_id",
        "generator_version",
        "episodes_root",
        "source_dataset_root",
        "output_root",
        "model",
        "generation",
        "execution",
        "metrics",
        "outputs",
    ):
        if key not in config:
            raise ValueError(f"Missing rollout config field: {key}")

    run_id = str(config["run_id"])
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
    ):
        raise ValueError(f"run_id must be a safe directory name: {run_id!r}")
    if str(config["generator_version"]) != PARALLEL_GENERATOR_VERSION:
        raise ValueError(
            "This entry point only supports "
            f"generator_version={PARALLEL_GENERATOR_VERSION!r}. Update the "
            "config; use a new run_id or --force for an older rollout store."
        )
    for key in ("episodes_root", "source_dataset_root"):
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(config[key])
    torch_dtype(str(config["model"].get("dtype", "bfloat16")))

    generation = config["generation"]
    integer_fields = (
        "trajectories_per_episode",
        "max_episode_steps",
        "max_generation_steps",
        "max_input_length",
        "write_shard_size",
    )
    for key in integer_fields:
        if int(generation.get(key, 0)) <= 0:
            raise ValueError(f"generation.{key} must be positive.")
    for key in ("batch_size", "cpu_workers_per_gpu"):
        if int(generation.get(key, 1)) <= 0:
            raise ValueError(f"generation.{key} must be positive.")
    if float(generation.get("batch_wait_seconds", 0.02)) < 0:
        raise ValueError("generation.batch_wait_seconds cannot be negative.")
    if int(config["execution"].get("cpu_threads_per_worker", 1)) <= 0:
        raise ValueError("execution.cpu_threads_per_worker must be positive.")
    if int(generation["trajectories_per_episode"]) < 2:
        raise ValueError(
            "DPO rollout generation requires at least two trajectories per episode."
        )
    for name, value in generation["temperatures"].items():
        if name not in {"plan", "label", "parameter", "pointer"}:
            raise ValueError(f"Unknown sampling temperature: {name}")
        if float(value) <= 0:
            raise ValueError(f"Sampling temperature {name} must be positive.")
    task_ids = generation.get("task_ids")
    if task_ids is not None and not isinstance(task_ids, list):
        raise ValueError("generation.task_ids must be a list or null.")
    limits = generation.get("max_episodes_per_split")
    if limits is not None:
        values = limits.values() if isinstance(limits, dict) else [limits]
        if any(int(value) < 0 for value in values if value is not None):
            raise ValueError(
                "generation.max_episodes_per_split cannot be negative."
            )

    splits = config.get("splits", ["train", "validation"])
    if not splits or any(
        str(split) not in {"train", "validation", "test"}
        for split in splits
    ):
        raise ValueError("splits must contain train, validation or test.")
    if not str(config["metrics"].get("version", "")).strip():
        raise ValueError("metrics.version must be non-empty.")
    enabled_metrics = set(config["metrics"].get("enabled", []))
    unknown_metrics = enabled_metrics - {
        "iou",
        "chamfer_distance",
        "accuracy",
        "f1",
        "watertight",
    }
    if unknown_metrics:
        raise ValueError(
            f"Unsupported rollout metrics: {sorted(unknown_metrics)}"
        )
    for key in (
        "build_timeout_seconds",
        "surf_u_samples",
        "surf_v_samples",
        "curv_u_samples",
    ):
        if float(config["execution"].get(key, 0)) <= 0:
            raise ValueError(f"execution.{key} must be positive.")


def generate_rollouts(config: Dict[str, Any], force: bool = False) -> Path:
    validate_rollout_config(config)
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid distributed rank/world size: rank={rank}, world={world_size}."
        )
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    model_config = config["model"]
    checkpoint_path = Path(model_config["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if world_size > 1:
        runtime_values = [
            build_runtime_config(config, checkpoint_path)
            if rank == 0
            else None
        ]
        dist.broadcast_object_list(runtime_values, src=0)
        runtime_config = runtime_values[0]
    else:
        runtime_config = build_runtime_config(config, checkpoint_path)

    run_id = str(config["run_id"])
    output_root = Path(config["output_root"])
    rollout_root = output_root / run_id
    if force and rank == 0:
        logger.warning("Force enabled: removing rollout run {}", rollout_root)
        RolloutStore.remove_existing(rollout_root, expected_parent=output_root)
    if rank == 0:
        store = RolloutStore.create(
            rollout_root,
            runtime_config,
            exist_ok=True,
            writer_id=f"r{rank:04d}",
        )
    if world_size > 1:
        dist.barrier()
        if rank != 0:
            store = RolloutStore.create(
                rollout_root,
                runtime_config,
                exist_ok=True,
                writer_id=f"r{rank:04d}",
            )
    existing_trajectories = {
        trajectory.trajectory_id: trajectory
        for trajectory in load_trajectories(str(rollout_root))
    }
    if world_size > 1:
        dist.barrier()

    requested_device = torch.device(model_config.get("device", "cuda"))
    if requested_device.type == "cuda":
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count == 0:
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.device_count() returned 0."
            )
        local_rank = int(os.getenv("LOCAL_RANK", 0)) % cuda_device_count
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(
            "[rank {}/{}] Using CUDA device {} of {}: {}",
            rank,
            world_size,
            local_rank,
            cuda_device_count,
            torch.cuda.get_device_properties(device),
        )
    else:
        device = requested_device
    dtype = torch_dtype(model_config.get("dtype", "bfloat16"))
    model = PointerCAD(
        qwen_model=model_config["base_model"],
        dtype=dtype,
    )
    load_checkpoint(model, checkpoint_path)
    model.to(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Rollout model weights loaded on {}", device)
    model.eval()
    processor = Text2CADProcessor.from_pretrained(
        pretrained_model_name_or_path=model_config["base_model"],
        padding_side="left",
    )
    generator = RolloutGenerator(
        model=model,
        processor=processor,
        device=device,
        store=store,
        source_dataset_root=Path(config["source_dataset_root"]),
        config=config,
    )

    _generate_parallel_splits(
        generator=generator,
        store=store,
        config=config,
        existing_trajectories=existing_trajectories,
        rank=rank,
        world_size=world_size,
    )
    if world_size > 1:
        dist.barrier()
    return rollout_root
