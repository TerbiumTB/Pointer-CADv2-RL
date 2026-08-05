import copy
import gc
import hashlib
import os
import random
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
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
    lm_token_ids = [
        token_id for token_id in token_values if token_id != cad_pad_id
    ]
    try:
        cad_start_position = lm_token_ids.index(cad_start_id)
    except ValueError as exc:
        raise ValueError(
            "Generation produced CAD actions without a <|cad_start|> token."
        ) from exc
    plan_text = tokenizer.decode(lm_token_ids[:cad_start_position])
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
    if len(behavior["plan"]) != len(lm_token_ids):
        raise ValueError(
            "Behavior LM log-probabilities do not align with generated LM tokens."
        )
    return (
        plan_text,
        lm_token_ids,
        parameter_map_to_lists(parameter_map),
        label_values,
        parameter_values,
        pointer_values,
        behavior,
    )


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
        self.config = config
        self.generation = config["generation"]
        self.execution = config["execution"]
        self.metrics_config = config["metrics"]
        self.outputs = config["outputs"]
        self._cached_prompt: Optional[str] = None
        self._cached_prompt_text: Optional[str] = None
        self._target_mesh_task_id: Optional[str] = None
        self._target_mesh: Optional[_LazyMesh] = None

    def _environment(self) -> CADEnvironment:
        return CADEnvironment(
            strict=bool(self.execution.get("strict", True)),
            surf_u_samples=int(self.execution.get("surf_u_samples", 32)),
            surf_v_samples=int(self.execution.get("surf_v_samples", 32)),
            curv_u_samples=int(self.execution.get("curv_u_samples", 32)),
            build_timeout_seconds=float(
                self.execution.get("build_timeout_seconds", 300.0)
            ),
        )

    def _render_prompt(self, prompt: str) -> str:
        if prompt != self._cached_prompt:
            messages = [prompt_message(prompt)]
            self._cached_prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            self._cached_prompt = prompt
        if self._cached_prompt_text is None:
            raise RuntimeError("Prompt rendering cache was not initialized.")
        return self._cached_prompt_text

    def _target_mesh_for_task(
        self, task_id: str, target_model
    ) -> Optional[_LazyMesh]:
        if task_id != self._target_mesh_task_id:
            self._target_mesh_task_id = task_id
            self._target_mesh = (
                _LazyMesh(lambda: create_mesh(target_model))
                if target_model is not None
                else None
            )
        return self._target_mesh

    def _generate_action(self, prompt: str, graph: dgl.DGLGraph):
        timings: Dict[str, float] = {}
        started_at = time.monotonic()
        text = self._render_prompt(prompt)
        timings["prompt_render_time_seconds"] = (
            time.monotonic() - started_at
        )

        started_at = time.monotonic()
        inputs = self.processor(
            text=text,
            breps=dgl.batch([graph]),
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
        decoded = decode_step_generation(
            tokenizer=self.processor.tokenizer,
            generated_ids=generated_ids[0],
            parameter_map=parameter_maps[0],
            labels=labels[0],
            parameters=parameters[0],
            pointers=pointers[0],
            behavior_log_probs=behavior_log_probs[0],
        )
        timings["generation_decode_time_seconds"] = (
            time.monotonic() - started_at
        )
        return decoded, timings

    def _save_final_data(
        self,
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

        if bool(self.outputs.get("save_step", True)):
            destination = self.store.new_data_path(
                "cad", f"{trajectory_id_value}.step"
            )
            started_at = time.monotonic()
            try:
                with suppress_native_stdout(
                    bool(
                        self.outputs.get(
                            "suppress_step_export_output", True
                        )
                    )
                ):
                    exported = environment.model.export_model(
                        str(destination),
                        timeout=float(
                            self.execution.get(
                                "build_timeout_seconds", 300.0
                            )
                        ),
                    )
                if exported is False:
                    raise RuntimeError("STEP writer reported an export failure.")
                if not destination.is_file():
                    raise RuntimeError("STEP writer did not create an output file.")
                cad_path = self.store.relative_path(destination)
            except Exception as exc:
                errors["step"] = f"{type(exc).__name__}: {exc}"
                if destination.exists():
                    destination.unlink()
            finally:
                timings["step_export_time_seconds"] += (
                    time.monotonic() - started_at
                )

        if bool(self.outputs.get("save_mesh", True)):
            destination = self.store.new_data_path(
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
                mesh_path = self.store.relative_path(destination)
            except Exception as exc:
                errors["mesh"] = f"{type(exc).__name__}: {exc}"
                if destination.exists():
                    destination.unlink()
        return cad_path, mesh_path, errors, timings

    def generate(
        self,
        episode: EpisodeRecord,
        rollout_index: int,
        seed: int,
        trajectory_id_value: str,
        target_model=None,
        target_model_error: Optional[str] = None,
    ) -> Tuple[TrajectoryRecord, List[StepRecord]]:
        rollout_started_at = time.monotonic()
        set_sampling_seed(seed, self.device)
        environment = self._environment()
        target_mesh = self._target_mesh_for_task(
            episode.task_id, target_model
        )
        target_mesh_time_before = (
            target_mesh.build_time_seconds if target_mesh is not None else 0.0
        )
        steps: List[StepRecord] = []
        generation_seconds = 0.0
        execution_seconds = 0.0
        detailed_timings: Dict[str, float] = {
            "state_graph_time_seconds": 0.0,
            "state_save_time_seconds": 0.0,
            "prompt_render_time_seconds": 0.0,
            "input_preparation_time_seconds": 0.0,
            "model_generation_time_seconds": 0.0,
            "generation_decode_time_seconds": 0.0,
        }
        termination_reason = "max_episode_steps"
        trajectory_error = None
        step_errors: List[str] = []
        max_episode_steps = int(
            self.generation.get("max_episode_steps", 64)
        )

        for step_index in range(max_episode_steps):
            state_before_id = (
                f"{trajectory_id_value}:state:{step_index:04d}"
            )
            try:
                started_at = time.monotonic()
                try:
                    graph = environment.state_graph()
                finally:
                    detailed_timings["state_graph_time_seconds"] += (
                        time.monotonic() - started_at
                    )
                state_path = self.store.new_data_path(
                    "states",
                    f"{trajectory_id_value}/step-{step_index:04d}.bin",
                )
                started_at = time.monotonic()
                try:
                    save_brep_graph(graph, state_path)
                    state_relative_path = self.store.relative_path(state_path)
                finally:
                    detailed_timings["state_save_time_seconds"] += (
                        time.monotonic() - started_at
                    )
            except Exception as exc:
                termination_reason = "state_graph_error"
                trajectory_error = exception_summary(exc)
                break

            generation_started_at = time.monotonic()
            try:
                decoded, action_timings = self._generate_action(
                    episode.prompt, graph
                )
                for name, value in action_timings.items():
                    detailed_timings[name] += value
                (
                    plan_text,
                    plan_token_ids,
                    parameter_map,
                    labels,
                    parameters,
                    pointers,
                    behavior,
                ) = decoded
            except Exception as exc:
                termination_reason = "generation_error"
                trajectory_error = exception_summary(exc)
                break
            finally:
                generation_seconds += (
                    time.monotonic() - generation_started_at
                )

            execution_valid = False
            execution_error = None
            state_after_id = None
            model_end = False
            execution_started_at = time.monotonic()
            try:
                model_end, _ = environment.step(
                    labels=labels,
                    parameters=parameters,
                    pointers=pointers,
                    parameter_map=parameter_map,
                )
                execution_valid = True
                state_after_id = (
                    f"{trajectory_id_value}:state:{step_index + 1:04d}"
                )
            except Exception as exc:
                execution_error = exception_summary(exc)
                step_errors.append(execution_error)
            finally:
                execution_seconds += (
                    time.monotonic() - execution_started_at
                )

            steps.append(
                StepRecord(
                    trajectory_id=trajectory_id_value,
                    step_index=step_index,
                    state_before_id=state_before_id,
                    state_before_graph_path=state_relative_path,
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

            if not execution_valid:
                termination_reason = "execution_error"
                trajectory_error = execution_error
                break
            if model_end:
                termination_reason = "model_end"
                break

        valid = termination_reason == "model_end"
        metrics: Dict[str, Any] = {
            "termination_reason": termination_reason,
            "step_execution_errors": step_errors,
        }
        if target_model_error is not None:
            metrics["target_model_error"] = target_model_error
        final_cad_path = None
        final_mesh_path = None
        prediction_mesh: Optional[_LazyMesh] = None
        output_timings: Dict[str, float] = {
            "step_export_time_seconds": 0.0,
            "mesh_export_time_seconds": 0.0,
        }
        if environment.model.seq:
            prediction_mesh = _LazyMesh(
                lambda: create_mesh(environment.model)
            )
            if target_model is not None:
                try:
                    evaluated = evaluate_models(
                        prediction=environment.model,
                        target=target_model,
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
                        target_mesh_factory=(
                            target_mesh.get
                            if target_mesh is not None
                            else None
                        ),
                    )
                    metrics.update(evaluated)
                except Exception as exc:
                    metrics["evaluation_error"] = exception_summary(exc)

            (
                final_cad_path,
                final_mesh_path,
                output_errors,
                output_timings,
            ) = self._save_final_data(
                trajectory_id_value,
                environment,
                prediction_mesh_factory=prediction_mesh.get,
            )
            metrics["output_errors"] = output_errors

        detailed_timings.update(output_timings)
        detailed_timings["generation_time_seconds"] = generation_seconds
        detailed_timings["execution_time_seconds"] = execution_seconds
        detailed_timings["prediction_mesh_build_time_seconds"] = (
            prediction_mesh.build_time_seconds
            if prediction_mesh is not None
            else 0.0
        )
        detailed_timings["target_mesh_build_time_seconds"] = (
            target_mesh.build_time_seconds - target_mesh_time_before
            if target_mesh is not None
            else 0.0
        )
        detailed_timings["rollout_time_seconds"] = (
            time.monotonic() - rollout_started_at
        )
        metrics["timings"] = detailed_timings
        metrics["generation_counts"] = {
            "plan_tokens": sum(len(step.plan_token_ids) for step in steps),
            "cad_actions": sum(len(step.labels) for step in steps),
        }

        record = TrajectoryRecord(
            trajectory_id=trajectory_id_value,
            task_id=episode.task_id,
            rollout_index=rollout_index,
            seed=seed,
            termination_reason=termination_reason,
            num_generated_steps=len(steps),
            valid=valid,
            final_cad_path=final_cad_path,
            final_mesh_path=final_mesh_path,
            execution_error=trajectory_error,
            metrics_json=canonical_json(metrics),
            generation_time_seconds=generation_seconds,
            execution_time_seconds=execution_seconds,
            metrics_version=str(self.metrics_config["version"]),
        )
        return record, steps


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
        "Rollout {}/{} for task {} ({}): valid={} termination={} steps={} "
        "plan_tokens={} cad_actions={} step_export={} mesh_export={} "
        "metrics={} time[generation={:.2f}s execution={:.2f}s "
        "state={:.2f}s metrics={:.2f}s mesh_build_subset={:.2f}s "
        "export={:.2f}s total={:.2f}s]{}",
        rollout_position,
        trajectories_per_episode,
        trajectory.task_id,
        trajectory.trajectory_id,
        trajectory.valid,
        trajectory.termination_reason,
        trajectory.num_generated_steps,
        counts.get("plan_tokens", "unknown"),
        counts.get("cad_actions", "unknown"),
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
    result["generation"].pop("resume", None)
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
    if not str(config["generator_version"]).strip():
        raise ValueError("generator_version must be non-empty.")
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
    model_config = config["model"]
    checkpoint_path = Path(model_config["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    runtime_config = build_runtime_config(config, checkpoint_path)

    run_id = str(config["run_id"])
    output_root = Path(config["output_root"])
    rollout_root = output_root / run_id
    if "resume" in config["generation"]:
        logger.warning(
            "generation.resume is deprecated and ignored; rollout generation "
            "now skips existing trajectories by default. Use --force/-f to "
            "recreate the whole run."
        )
    if force:
        logger.warning("Force enabled: removing rollout run {}", rollout_root)
        RolloutStore.remove_existing(rollout_root, expected_parent=output_root)
    store = RolloutStore.create(rollout_root, runtime_config, exist_ok=True)
    existing_trajectories = {
        trajectory.trajectory_id: trajectory
        for trajectory in load_trajectories(str(rollout_root))
    }
    existing_ids = set(existing_trajectories)

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
            "Using CUDA device {} of {}: {}",
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

    generation_config = config["generation"]
    trajectories_per_episode = int(
        generation_config["trajectories_per_episode"]
    )
    base_seed = int(generation_config.get("base_seed", 0))
    write_shard_size = int(generation_config.get("write_shard_size", 64))
    if write_shard_size <= 0:
        raise ValueError("generation.write_shard_size must be positive.")
    splits = [str(value) for value in config.get("splits", ["train", "validation"])]
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
            episodes = _limit_episodes(
                load_episode_index(config["episodes_root"], split),
                split,
                generation_config,
            )
            split_stats: Counter[str] = Counter()
            split_terminations: Counter[str] = Counter()
            progress = tqdm(
                total=len(episodes) * trajectories_per_episode,
                desc=f"rollouts:{split}",
                unit="rollout",
                dynamic_ncols=True,
            )
            for episode_position, episode in enumerate(episodes, start=1):
                episode_started_at = time.monotonic()
                episode_stats: Counter[str] = Counter()
                episode_terminations: Counter[str] = Counter()
                target_model = None
                target_model_error = None
                try:
                    target_model = load_target_model(
                        Path(config["source_dataset_root"]),
                        episode.target_cad_path,
                    )
                except AssertionError as exc:
                    target_model_error = exception_summary(exc)
                    logger.opt(exception=exc).error(
                        "Cannot load target CAD for task {}; rollouts will "
                        "continue without geometry metrics: {}",
                        episode.task_id,
                        target_model_error,
                    )
                logger.info(
                    "Task {}/{} {}: generating {} rollouts; target_operations={}",
                    episode_position,
                    len(episodes),
                    episode.task_id,
                    trajectories_per_episode,
                    (
                        len(target_model.seq)
                        if target_model is not None
                        else "unavailable"
                    ),
                )
                for rollout_index in range(trajectories_per_episode):
                    rollout_position = rollout_index + 1
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
                        trajectory = existing_trajectories[identifier]
                        _update_rollout_stats(episode_stats, trajectory)
                        _update_rollout_stats(split_stats, trajectory)
                        _update_rollout_stats(run_stats, trajectory)
                        episode_stats["skipped_existing"] += 1
                        split_stats["skipped_existing"] += 1
                        run_stats["skipped_existing"] += 1
                        episode_terminations[trajectory.termination_reason] += 1
                        split_terminations[trajectory.termination_reason] += 1
                        run_terminations[trajectory.termination_reason] += 1
                        progress.update(1)
                        progress.set_postfix(
                            episode=f"{episode_position}/{len(episodes)}",
                            task=episode.task_id,
                            rollout=(
                                f"{rollout_position}/"
                                f"{trajectories_per_episode}"
                            ),
                            task_valid=(
                                f"{episode_stats['valid']}/"
                                f"{episode_stats['rollouts']}"
                            ),
                            total_valid=(
                                f"{split_stats['valid']}/"
                                f"{split_stats['rollouts']}"
                            ),
                        )
                        continue
                    if identifier in pending_ids:
                        raise ValueError(
                            f"Duplicate pending trajectory ID: {identifier}"
                        )
                    store.discard_uncommitted_data(identifier)
                    rollout_started_at = time.monotonic()
                    try:
                        trajectory, steps = generator.generate(
                            episode=episode,
                            rollout_index=rollout_index,
                            seed=seed,
                            trajectory_id_value=identifier,
                            target_model=target_model,
                            target_model_error=target_model_error,
                        )
                    except Exception as exc:
                        # Individual rollouts are fault-isolation boundaries. A
                        # malformed generated operation must not terminate the
                        # complete dataset run even if it escapes a narrower
                        # generation/execution/metric handler above.
                        error = exception_summary(exc)
                        logger.opt(exception=exc).error(
                            "Unexpected failure in rollout {} for task {}: {}",
                            identifier,
                            episode.task_id,
                            error,
                        )
                        store.discard_uncommitted_data(identifier)
                        trajectory = TrajectoryRecord(
                            trajectory_id=identifier,
                            task_id=episode.task_id,
                            rollout_index=rollout_index,
                            seed=seed,
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
                                }
                            ),
                            generation_time_seconds=(
                                time.monotonic() - rollout_started_at
                            ),
                            execution_time_seconds=0.0,
                            metrics_version=str(config["metrics"]["version"]),
                        )
                        steps = []
                    pending_trajectories.append(trajectory)
                    pending_steps.extend(steps)
                    pending_ids.add(identifier)
                    _update_rollout_stats(episode_stats, trajectory)
                    _update_rollout_stats(split_stats, trajectory)
                    _update_rollout_stats(run_stats, trajectory)
                    episode_stats["generated"] += 1
                    split_stats["generated"] += 1
                    run_stats["generated"] += 1
                    episode_terminations[trajectory.termination_reason] += 1
                    split_terminations[trajectory.termination_reason] += 1
                    run_terminations[trajectory.termination_reason] += 1
                    _log_rollout_result(
                        trajectory,
                        rollout_position,
                        trajectories_per_episode,
                        time.monotonic() - rollout_started_at,
                    )
                    if len(pending_trajectories) >= write_shard_size:
                        flush_pending()
                    progress.update(1)
                    progress.set_postfix(
                        episode=f"{episode_position}/{len(episodes)}",
                        task=episode.task_id,
                        rollout=(
                            f"{rollout_position}/{trajectories_per_episode}"
                        ),
                        task_valid=(
                            f"{episode_stats['valid']}/"
                            f"{episode_stats['rollouts']}"
                        ),
                        total_valid=(
                            f"{split_stats['valid']}/"
                            f"{split_stats['rollouts']}"
                        ),
                    )
                logger.info(
                    "Task {}/{} {} complete: valid(model_end)={}/{} "
                    "step_export={}/{} mesh_export={}/{} generated={} "
                    "skipped_existing={} terminations={} elapsed={:.1f}s",
                    episode_position,
                    len(episodes),
                    episode.task_id,
                    episode_stats["valid"],
                    episode_stats["rollouts"],
                    episode_stats["cad_saved"],
                    episode_stats["rollouts"],
                    episode_stats["mesh_saved"],
                    episode_stats["rollouts"],
                    episode_stats["generated"],
                    episode_stats["skipped_existing"],
                    _format_termination_counts(episode_terminations),
                    time.monotonic() - episode_started_at,
                )
            progress.close()
            flush_pending()
            logger.info(
                "Split {} complete: tasks={} rollouts={} valid(model_end)={}/{} "
                "step_export={}/{} mesh_export={}/{} generated={} "
                "skipped_existing={} "
                "terminations={}",
                split,
                len(episodes),
                split_stats["rollouts"],
                split_stats["valid"],
                split_stats["rollouts"],
                split_stats["cad_saved"],
                split_stats["rollouts"],
                split_stats["mesh_saved"],
                split_stats["rollouts"],
                split_stats["generated"],
                split_stats["skipped_existing"],
                _format_termination_counts(split_terminations),
            )
        flush_pending()
    logger.info(
        "Rollout run {} complete: rollouts={} valid(model_end)={}/{} "
        "step_export={}/{} mesh_export={}/{} generated={} skipped_existing={} "
        "terminations={}",
        run_id,
        run_stats["rollouts"],
        run_stats["valid"],
        run_stats["rollouts"],
        run_stats["cad_saved"],
        run_stats["rollouts"],
        run_stats["mesh_saved"],
        run_stats["rollouts"],
        run_stats["generated"],
        run_stats["skipped_existing"],
        _format_termination_counts(run_terminations),
    )
    return rollout_root
