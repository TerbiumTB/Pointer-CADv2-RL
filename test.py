#!/usr/bin/env python3
"""Deterministic full-episode PointerCAD evaluation.

One torchrun rank owns each GPU.  Isolated spawn workers own OpenCascade CAD
environments and overlap B-Rep construction, execution, metrics, and exports
with batched GPU generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
import torch.distributed as dist
import yaml
from loguru import logger
from tqdm import tqdm

from misc import create_mesh
from models.pointercad import PointerCAD
from models.processor import Text2CADProcessor
from rl.cad_environment import CADEnvironment, evaluate_models, load_target_model
from rl.data import load_episode_index
from rl.prompts import prompt_message
from rl.rollout_generator import (
    _LazyMesh,
    _graph_from_payload,
    _graph_to_payload,
    exception_summary,
    file_sha256,
    load_checkpoint,
    parameter_map_to_lists,
    suppress_native_stdout,
    torch_dtype,
)


EVALUATOR_VERSION = "pointercad-evaluator-v2-cpu-worker-pipeline"
SYSTEMIC_GENERATION_ERROR_LIMIT = 8
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    split: str
    chunk: str
    model_id: str
    prompt_variant: str
    prompt: str
    target_cad_path: str


@dataclass(frozen=True)
class GeneratedAction:
    parameter_map: Dict[str, List[float]]
    labels: List[int]
    parameters: List[int]
    pointers: List[int]
    plan_token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config_path", default="./config/test.yaml"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Exact shared output directory for this evaluation run.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation config must be a mapping: {path}")
    return value


def _part_sort_key(part_id: str) -> Tuple[int, object]:
    try:
        return 0, int(part_id)
    except ValueError:
        return 1, part_id


def _load_tasks_from_source(config: Mapping[str, Any]) -> List[EvaluationTask]:
    dataset = config["dataset"]
    source_root = Path(str(dataset["dataset_dir"])).absolute()
    split_path = Path(str(dataset["split_filepath"])).absolute()
    split = str(dataset.get("split", "test"))
    prompt_variant = str(dataset.get("prompt_variant", "exp"))
    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)
    step_ids = split_data.get(split, [])
    if not isinstance(step_ids, list):
        raise ValueError(f"Split {split!r} must be a list in {split_path}.")

    final_parts: Dict[Tuple[str, str], str] = {}
    for raw_step_id in step_ids:
        parts = str(raw_step_id).split("_")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                "Expected '<chunk>_<model_id>_<part_id>', "
                f"got {raw_step_id!r}."
            )
        chunk, model_id, part_id = parts
        key = (chunk, model_id)
        previous = final_parts.get(key)
        if previous is None or _part_sort_key(part_id) > _part_sort_key(previous):
            final_parts[key] = part_id

    tasks: List[EvaluationTask] = []
    for (chunk, model_id), part_id in sorted(final_parts.items()):
        model_dir = source_root / chunk / model_id
        prompt_path = model_dir / f"prompt_{prompt_variant}.txt"
        step_target = model_dir / "json" / f"{model_id}_{part_id}.json"
        model_target = model_dir / f"{model_id}.json"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing evaluation prompt: {prompt_path}")
        if step_target.is_file():
            target_path = step_target
        elif model_target.is_file():
            target_path = model_target
        else:
            raise FileNotFoundError(
                f"Missing final target CAD at {step_target} or {model_target}."
            )
        tasks.append(
            EvaluationTask(
                task_id=f"{chunk}_{model_id}_{prompt_variant}",
                split=split,
                chunk=chunk,
                model_id=model_id,
                prompt_variant=prompt_variant,
                prompt=prompt_path.read_text(encoding="utf-8"),
                target_cad_path=target_path.relative_to(source_root).as_posix(),
            )
        )
    return tasks


def load_tasks(config: Mapping[str, Any]) -> List[EvaluationTask]:
    dataset = config["dataset"]
    split = str(dataset.get("split", "test"))
    prompt_variant = str(dataset.get("prompt_variant", "exp"))
    episodes_root = dataset.get("episodes_root")
    if episodes_root and (Path(str(episodes_root)) / f"{split}.parquet").is_file():
        episodes = load_episode_index(str(episodes_root), split)
        tasks = [
            EvaluationTask(
                task_id=episode.task_id,
                split=episode.split,
                chunk=episode.chunk,
                model_id=episode.model_id,
                prompt_variant=episode.prompt_variant,
                prompt=episode.prompt,
                target_cad_path=episode.target_cad_path,
            )
            for episode in episodes
            if episode.prompt_variant == prompt_variant
        ]
    else:
        if episodes_root:
            logger.warning(
                "Episode index not found at {}; building lightweight tasks from "
                "the authoritative split instead.",
                Path(str(episodes_root)) / f"{split}.parquet",
            )
        tasks = _load_tasks_from_source(config)

    max_models = config.get("generation", {}).get("max_models")
    if max_models is not None:
        tasks = tasks[: int(max_models)]
    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation tasks contain duplicate task_id values.")
    if not tasks:
        raise ValueError(
            f"No {split}/{prompt_variant} evaluation tasks were selected."
        )
    return tasks


def validate_config(config: Mapping[str, Any]) -> None:
    for section in (
        "model",
        "dataset",
        "generation",
        "execution",
        "metrics",
        "outputs",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config section {section!r} must be a mapping.")
    generation = config["generation"]
    for name in ("batch_size", "cpu_workers_per_gpu", "max_episode_steps"):
        if int(generation.get(name, 0)) <= 0:
            raise ValueError(f"generation.{name} must be positive.")
    if float(generation.get("batch_wait_seconds", 0.02)) < 0:
        raise ValueError("generation.batch_wait_seconds cannot be negative.")
    if not str(config["metrics"].get("version", "")).strip():
        raise ValueError("metrics.version must be a non-empty string.")
    split = str(config["dataset"].get("split", "test"))
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split!r}.")
    checkpoint = Path(
        str(
            config["model"].get("checkpoint_path")
            or config.get("test", {}).get("checkpoint_path", "")
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")


def _task_rank(task_id: str, world_size: int) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % world_size


def _task_seed(base_seed: int, task_id: str) -> int:
    suffix = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12], 16)
    return (base_seed + suffix) % (2**32)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _atomic_write_yaml(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, sort_keys=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _read_result_shards(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for path in sorted(output_dir.glob("results-r*.jsonl")):
        file_size = path.stat().st_size
        valid_bytes = 0
        with path.open("rb") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line_end = file.tell()
                if not raw_line.strip():
                    valid_bytes = line_end
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if line_end != file_size:
                        raise ValueError(
                            f"Corrupt non-final result record in {path}:"
                            f"{line_number}."
                        )
                    logger.warning(
                        "Discarding incomplete final result record in {}:{}.",
                        path,
                        line_number,
                    )
                    break
                task_id = str(record["task_id"])
                result = record["result"]
                previous = results.get(task_id)
                if previous is not None and previous != result:
                    raise ValueError(
                        f"Conflicting results for {task_id} in {path}:"
                        f"{line_number}."
                    )
                results[task_id] = result
                valid_bytes = line_end
        if valid_bytes < file_size:
            with path.open("r+b") as file:
                file.truncate(valid_bytes)
    return results


def _append_result(path: Path, task_id: str, result: Dict[str, Any]) -> None:
    serialized = json.dumps(
        {"task_id": task_id, "result": result},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as file:
        file.write(serialized + "\n")
        file.flush()
        os.fsync(file.fileno())


def _error_result(
    task: EvaluationTask,
    termination_reason: str,
    error: str,
    elapsed_seconds: float = 0.0,
) -> Dict[str, Any]:
    return {
        "status": False,
        "error_message": error,
        "termination_reason": termination_reason,
        "num_generated_steps": 0,
        "chamfer distance": None,
        "f1": None,
        "is watertight": None,
        "metrics": {
            "termination_reason": termination_reason,
            "evaluation_error": error,
            "timings": {"evaluation_time_seconds": elapsed_seconds},
        },
        "task": {
            "chunk": task.chunk,
            "model_id": task.model_id,
            "prompt_variant": task.prompt_variant,
            "split": task.split,
        },
    }


def _export_models(
    task: EvaluationTask,
    environment: CADEnvironment,
    target_model,
    prediction_mesh: Optional[_LazyMesh],
    target_mesh: Optional[_LazyMesh],
    config: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str]]:
    outputs = config["outputs"]
    execution = config["execution"]
    errors: Dict[str, str] = {}
    timings = {"step_export_time_seconds": 0.0, "mesh_export_time_seconds": 0.0}
    paths: Dict[str, str] = {}
    safe_name = task.task_id.replace("/", "_")
    save_ground_truth = bool(outputs.get("save_ground_truth", False))

    if bool(outputs.get("save_step", False)) and environment.model.seq:
        step_dir = output_dir / "step"
        step_dir.mkdir(parents=True, exist_ok=True)
        for name, model in (
            ("prediction_step", environment.model),
            ("target_step", target_model if save_ground_truth else None),
        ):
            if model is None:
                continue
            suffix = "_gt" if name.startswith("target") else ""
            destination = step_dir / f"{safe_name}{suffix}.step"
            started_at = time.monotonic()
            try:
                with suppress_native_stdout(
                    bool(outputs.get("suppress_step_export_output", True))
                ):
                    exported = model.export_model(
                        str(destination),
                        timeout=float(execution.get("build_timeout_seconds", 300.0)),
                    )
                if exported is False or not destination.is_file():
                    raise RuntimeError("STEP export did not create a valid file.")
                paths[name] = destination.relative_to(output_dir).as_posix()
            except Exception as exc:
                errors[name] = exception_summary(exc)
            finally:
                timings["step_export_time_seconds"] += time.monotonic() - started_at

    if bool(outputs.get("save_stl", False)) and environment.model.seq:
        stl_dir = output_dir / "stl"
        stl_dir.mkdir(parents=True, exist_ok=True)
        mesh_items = [("prediction_stl", prediction_mesh)]
        if save_ground_truth:
            mesh_items.append(("target_stl", target_mesh))
        for name, lazy_mesh in mesh_items:
            if lazy_mesh is None:
                continue
            suffix = "_gt" if name.startswith("target") else ""
            destination = stl_dir / f"{safe_name}{suffix}.stl"
            started_at = time.monotonic()
            try:
                lazy_mesh.get().export(str(destination))
                if not destination.is_file():
                    raise RuntimeError("STL export did not create a file.")
                paths[name] = destination.relative_to(output_dir).as_posix()
            except Exception as exc:
                errors[name] = exception_summary(exc)
            finally:
                timings["mesh_export_time_seconds"] += time.monotonic() - started_at
    return errors, timings, paths


class _CPUWorkerEvaluation:
    def __init__(
        self,
        task: EvaluationTask,
        config: Mapping[str, Any],
        output_dir: Path,
    ):
        self.task = task
        self.config = config
        self.output_dir = output_dir
        self.started_at = time.monotonic()
        self.environment = CADEnvironment(
            strict=bool(config["execution"].get("strict", True)),
            surf_u_samples=int(config["execution"].get("surf_u_samples", 32)),
            surf_v_samples=int(config["execution"].get("surf_v_samples", 32)),
            curv_u_samples=int(config["execution"].get("curv_u_samples", 32)),
            build_timeout_seconds=float(
                config["execution"].get("build_timeout_seconds", 300.0)
            ),
        )
        self.step_index = 0
        self.termination_reason = "max_episode_steps"
        self.execution_error: Optional[str] = None
        self.plan_token_count = 0
        self.gpu_batch_sizes: List[int] = []
        self.timings: Dict[str, float] = {
            "target_load_time_seconds": 0.0,
            "state_graph_time_seconds": 0.0,
            "prompt_render_time_seconds": 0.0,
            "input_preparation_time_seconds": 0.0,
            "model_generation_time_seconds": 0.0,
            "generation_decode_time_seconds": 0.0,
            "execution_time_seconds": 0.0,
            "metrics_time_seconds": 0.0,
            "step_export_time_seconds": 0.0,
            "mesh_export_time_seconds": 0.0,
        }
        seed = _task_seed(
            int(config["generation"].get("base_seed", 0)), task.task_id
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.target_model = None
        self.target_model_error: Optional[str] = None
        started_at = time.monotonic()
        try:
            self.target_model = load_target_model(
                Path(str(config["dataset"]["dataset_dir"])), task.target_cad_path
            )
        except Exception as exc:
            self.target_model_error = exception_summary(exc)
        finally:
            self.timings["target_load_time_seconds"] += time.monotonic() - started_at

    def prepare_state(self) -> dgl.DGLGraph:
        started_at = time.monotonic()
        try:
            return self.environment.state_graph()
        finally:
            self.timings["state_graph_time_seconds"] += time.monotonic() - started_at

    def apply_action(
        self,
        action: GeneratedAction,
        action_timings: Mapping[str, float],
        gpu_batch_size: int,
    ) -> bool:
        for name, value in action_timings.items():
            self.timings[name] = self.timings.get(name, 0.0) + float(value)
        self.gpu_batch_sizes.append(int(gpu_batch_size))
        self.plan_token_count += action.plan_token_count
        started_at = time.monotonic()
        try:
            model_end, _ = self.environment.step(
                labels=action.labels,
                parameters=action.parameters,
                pointers=action.pointers,
                parameter_map=action.parameter_map,
            )
        except Exception as exc:
            self.termination_reason = "execution_error"
            self.execution_error = exception_summary(exc)
            return True
        finally:
            self.timings["execution_time_seconds"] += time.monotonic() - started_at

        self.step_index += 1
        if model_end:
            self.termination_reason = "model_end"
            return True
        if self.step_index >= int(self.config["generation"]["max_episode_steps"]):
            self.termination_reason = "max_episode_steps"
            self.execution_error = (
                f"Exceeded {self.config['generation']['max_episode_steps']} "
                "progressive CAD steps."
            )
            return True
        return False

    def fail(self, termination_reason: str, error: str) -> None:
        self.termination_reason = termination_reason
        self.execution_error = error

    def finalize(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "termination_reason": self.termination_reason,
            "metrics_version": str(self.config["metrics"]["version"]),
        }
        prediction_mesh: Optional[_LazyMesh] = None
        target_mesh: Optional[_LazyMesh] = None
        if self.environment.model.seq:
            prediction_mesh = _LazyMesh(lambda: create_mesh(self.environment.model))
            if self.target_model is not None:
                target_mesh = _LazyMesh(lambda: create_mesh(self.target_model))
                try:
                    evaluated = evaluate_models(
                        prediction=self.environment.model,
                        target=self.target_model,
                        enabled_metrics=self.config["metrics"].get("enabled", []),
                        chamfer_points=int(
                            self.config["metrics"].get("chamfer_points", 8192)
                        ),
                        prediction_mesh_factory=prediction_mesh.get,
                        target_mesh_factory=target_mesh.get,
                    )
                    metrics.update(evaluated)
                    self.timings["metrics_time_seconds"] += float(
                        evaluated.get("metrics_time_seconds", 0.0)
                    )
                except Exception as exc:
                    metrics["evaluation_error"] = exception_summary(exc)
        if self.target_model_error is not None:
            metrics["target_model_error"] = self.target_model_error

        output_errors, output_timings, output_paths = _export_models(
            task=self.task,
            environment=self.environment,
            target_model=self.target_model,
            prediction_mesh=prediction_mesh,
            target_mesh=target_mesh,
            config=self.config,
            output_dir=self.output_dir,
        )
        self.timings.update(output_timings)
        self.timings["prediction_mesh_build_time_seconds"] = (
            prediction_mesh.build_time_seconds if prediction_mesh else 0.0
        )
        self.timings["target_mesh_build_time_seconds"] = (
            target_mesh.build_time_seconds if target_mesh else 0.0
        )
        self.timings["evaluation_time_seconds"] = time.monotonic() - self.started_at
        metrics["timings"] = self.timings
        metrics["generation_counts"] = {
            "progressive_steps": self.step_index,
            "plan_tokens": self.plan_token_count,
            "gpu_batches": len(self.gpu_batch_sizes),
            "mean_gpu_batch_size": (
                sum(self.gpu_batch_sizes) / len(self.gpu_batch_sizes)
                if self.gpu_batch_sizes
                else 0.0
            ),
            "min_gpu_batch_size": min(self.gpu_batch_sizes, default=0),
            "max_gpu_batch_size": max(self.gpu_batch_sizes, default=0),
        }
        metrics["output_errors"] = output_errors
        metrics["output_paths"] = output_paths
        if self.execution_error:
            metrics["execution_error"] = self.execution_error

        chamfer = metrics.get("chamfer_distance")
        f1 = metrics.get("f1")
        valid = self.termination_reason == "model_end"
        result: Dict[str, Any] = {
            "status": valid,
            "termination_reason": self.termination_reason,
            "num_generated_steps": self.step_index,
            "chamfer distance": (
                float(chamfer) * 1000.0 if chamfer is not None else None
            ),
            "f1": (
                {str(name): float(value) * 100.0 for name, value in f1.items()}
                if isinstance(f1, dict)
                else None
            ),
            "is watertight": metrics.get("watertight"),
            "metrics": metrics,
            "task": {
                "chunk": self.task.chunk,
                "model_id": self.task.model_id,
                "prompt_variant": self.task.prompt_variant,
                "split": self.task.split,
            },
        }
        if not valid:
            result["error_message"] = self.execution_error or self.termination_reason
        return result


def _cpu_evaluation_worker(
    worker_id: int,
    input_queue,
    output_queue,
    config: Mapping[str, Any],
    output_dir: str,
) -> None:
    torch.set_num_threads(int(config["execution"].get("cpu_threads_per_worker", 1)))
    runner: Optional[_CPUWorkerEvaluation] = None
    active_task: Optional[EvaluationTask] = None
    while True:
        message = input_queue.get()
        kind = message[0]
        if kind == "stop":
            return
        try:
            if kind == "start":
                task = message[1]
                active_task = task
                runner = _CPUWorkerEvaluation(task, config, Path(output_dir))
                try:
                    graph = runner.prepare_state()
                except Exception as exc:
                    runner.fail("state_graph_error", exception_summary(exc))
                    result = runner.finalize()
                    output_queue.put(("done", worker_id, runner.task, result))
                    runner = None
                    active_task = None
                else:
                    output_queue.put(
                        ("ready", worker_id, task.task_id, _graph_to_payload(graph))
                    )
            elif kind == "action":
                if runner is None:
                    raise RuntimeError("CPU worker received action without a task.")
                _, action, timings, gpu_batch_size = message
                if runner.apply_action(action, timings, gpu_batch_size):
                    result = runner.finalize()
                    output_queue.put(("done", worker_id, runner.task, result))
                    runner = None
                    active_task = None
                else:
                    try:
                        graph = runner.prepare_state()
                    except Exception as exc:
                        runner.fail("state_graph_error", exception_summary(exc))
                        result = runner.finalize()
                        output_queue.put(("done", worker_id, runner.task, result))
                        runner = None
                        active_task = None
                    else:
                        output_queue.put(
                            (
                                "ready",
                                worker_id,
                                runner.task.task_id,
                                _graph_to_payload(graph),
                            )
                        )
            elif kind == "fail":
                if runner is None:
                    raise RuntimeError("CPU worker received failure without a task.")
                _, termination_reason, error = message
                runner.fail(termination_reason, error)
                result = runner.finalize()
                output_queue.put(("done", worker_id, runner.task, result))
                runner = None
                active_task = None
            else:
                raise ValueError(f"Unknown CPU worker message: {kind!r}")
        except Exception as exc:
            error = exception_summary(exc)
            if runner is not None:
                task = runner.task
                try:
                    runner.fail("evaluation_error", error)
                    result = runner.finalize()
                except Exception as finalization_exc:
                    result = _error_result(
                        task,
                        "evaluation_error",
                        exception_summary(finalization_exc),
                        time.monotonic() - runner.started_at,
                    )
                output_queue.put(("done", worker_id, task, result))
                runner = None
                active_task = None
            elif active_task is not None:
                output_queue.put(
                    (
                        "done",
                        worker_id,
                        active_task,
                        _error_result(active_task, "evaluation_error", error),
                    )
                )
                active_task = None
            else:
                output_queue.put(("worker_error", worker_id, error))


class EvaluationGenerator:
    def __init__(self, model, processor, device: torch.device, config):
        self.model = model
        self.processor = processor
        self.device = device
        self.config = config
        self._prompt_cache: OrderedDict[str, str] = OrderedDict()
        self._prompt_cache_size = max(
            64, int(config["generation"].get("cpu_workers_per_gpu", 1)) * 2
        )

    def _render_prompt(self, prompt: str) -> str:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached
        rendered = self.processor.apply_chat_template(
            prompt_message(prompt), tokenize=False, add_generation_prompt=True
        )
        if not isinstance(rendered, str):
            raise TypeError(
                "Expected one rendered prompt string, got "
                f"{type(rendered).__name__}."
            )
        self._prompt_cache[prompt] = rendered
        if len(self._prompt_cache) > self._prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        return rendered

    def generate(
        self, tasks: Sequence[EvaluationTask], graph_payloads: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Optional[GeneratedAction]], List[Optional[str]], Dict[str, float]]:
        if len(tasks) != len(graph_payloads) or not tasks:
            raise ValueError("Evaluation batch tasks and graphs must be aligned.")
        timings: Dict[str, float] = {}
        started_at = time.monotonic()
        rendered = [self._render_prompt(task.prompt) for task in tasks]
        timings["prompt_render_time_seconds"] = time.monotonic() - started_at

        started_at = time.monotonic()
        graphs = [_graph_from_payload(payload) for payload in graph_payloads]
        inputs = self.processor(
            text=rendered,
            breps=dgl.batch(graphs),
            max_length=int(self.config["generation"].get("max_input_length", 3072)),
        ).to(self.device)
        timings["input_preparation_time_seconds"] = time.monotonic() - started_at

        started_at = time.monotonic()
        outputs = self.model.predict(
            tokenizer=self.processor.tokenizer,
            max_steps=int(
                self.config["generation"].get("max_generation_steps", 1024)
            ),
            mode="argmax",
            **inputs,
        )
        timings["model_generation_time_seconds"] = time.monotonic() - started_at
        generated_ids, parameter_maps, labels, parameters, pointers = outputs

        started_at = time.monotonic()
        actions: List[Optional[GeneratedAction]] = []
        errors: List[Optional[str]] = []
        cad_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|cad_pad|>")
        for token_ids, parameter_map, label, parameter, pointer in zip(
            generated_ids, parameter_maps, labels, parameters, pointers
        ):
            try:
                label_values = [int(value) for value in label.detach().cpu().tolist()]
                parameter_values = [
                    int(value) for value in parameter.detach().cpu().tolist()
                ]
                pointer_values = [
                    int(value) for value in pointer.detach().cpu().tolist()
                ]
                if not label_values:
                    raise ValueError("Generation finished without a CAD action.")
                if not (
                    len(label_values)
                    == len(parameter_values)
                    == len(pointer_values)
                ):
                    raise ValueError(
                        "Generated structured CAD channels have different lengths."
                    )
                plan_token_count = sum(
                    int(value) != cad_pad_id
                    for value in token_ids.detach().cpu().tolist()
                )
                actions.append(
                    GeneratedAction(
                        parameter_map=parameter_map_to_lists(parameter_map),
                        labels=label_values,
                        parameters=parameter_values,
                        pointers=pointer_values,
                        plan_token_count=plan_token_count,
                    )
                )
                errors.append(None)
            except Exception as exc:
                actions.append(None)
                errors.append(exception_summary(exc))
        timings["generation_decode_time_seconds"] = time.monotonic() - started_at
        del inputs, outputs, graphs
        return actions, errors, timings


def _run_rank_evaluation(
    generator: EvaluationGenerator,
    tasks: Sequence[EvaluationTask],
    config: Mapping[str, Any],
    output_dir: Path,
    rank: int,
) -> None:
    if not tasks:
        logger.info("Rank {} has no unfinished evaluation tasks.", rank)
        return
    generation = config["generation"]
    batch_size = int(generation["batch_size"])
    effective_batch_size = batch_size
    oom_batch_limit = batch_size
    requested_workers = int(generation["cpu_workers_per_gpu"])
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    local_world_size = max(1, int(os.getenv("LOCAL_WORLD_SIZE", "1")))
    cpu_worker_count = min(
        requested_workers,
        max(1, available_cpus // local_world_size),
        len(tasks),
    )
    if cpu_worker_count < requested_workers:
        logger.warning(
            "Rank {} capped CPU workers from {} to {} (available CPUs={}, "
            "LOCAL_WORLD_SIZE={}).",
            rank,
            requested_workers,
            cpu_worker_count,
            available_cpus,
            local_world_size,
        )
    if cpu_worker_count < batch_size:
        logger.warning(
            "Rank {} CPU workers ({}) are below batch_size ({}); GPU batches "
            "cannot become full.",
            rank,
            cpu_worker_count,
            batch_size,
        )

    context = mp.get_context("spawn")
    output_queue = context.Queue(maxsize=max(4, cpu_worker_count * 2))
    input_queues = [context.Queue(maxsize=2) for _ in range(cpu_worker_count)]
    processes: List[mp.Process] = []
    worker_tasks: Dict[int, EvaluationTask] = {}
    ready: Deque[Tuple[int, str, Dict[str, Any]]] = deque()
    next_task_index = 0
    completed = 0
    result_shard = output_dir / f"results-r{rank:05d}.jsonl"
    progress = tqdm(
        total=len(tasks),
        desc=f"evaluation:rank{rank}",
        unit="model",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    termination_counts: Counter[str] = Counter()
    last_generation_error: Optional[str] = None
    consecutive_generation_errors = 0
    batch_wait_seconds = float(generation.get("batch_wait_seconds", 0.02))

    def start_process(worker_id: int) -> mp.Process:
        process = context.Process(
            target=_cpu_evaluation_worker,
            args=(
                worker_id,
                input_queues[worker_id],
                output_queue,
                config,
                str(output_dir),
            ),
            name=f"pointercad-eval-cpu-{rank}-{worker_id}",
        )
        process.start()
        return process

    def assign_task(worker_id: int) -> bool:
        nonlocal next_task_index
        if next_task_index >= len(tasks):
            worker_tasks.pop(worker_id, None)
            return False
        task = tasks[next_task_index]
        next_task_index += 1
        worker_tasks[worker_id] = task
        input_queues[worker_id].put(("start", task))
        return True

    def finish_result(task: EvaluationTask, result: Dict[str, Any]) -> None:
        nonlocal completed
        _append_result(result_shard, task.task_id, result)
        completed += 1
        termination = str(result.get("termination_reason", "unknown"))
        termination_counts[termination] += 1
        progress.update(1)
        progress.set_postfix(
            valid=termination_counts["model_end"],
            batch=effective_batch_size,
            terminations=dict(termination_counts),
            refresh=False,
        )

    def handle_message(message) -> None:
        kind = message[0]
        if kind == "ready":
            _, worker_id, task_id, graph_payload = message
            task = worker_tasks.get(worker_id)
            if task is None or task.task_id != task_id:
                raise RuntimeError("CPU worker returned an unexpected task state.")
            ready.append((worker_id, task_id, graph_payload))
            return
        if kind == "done":
            _, worker_id, task, result = message
            expected = worker_tasks.get(worker_id)
            if expected is None or expected.task_id != task.task_id:
                raise RuntimeError("CPU worker completed an unexpected task.")
            finish_result(task, result)
            assign_task(worker_id)
            return
        if kind == "worker_error":
            _, worker_id, error = message
            raise RuntimeError(f"CPU worker {worker_id} failed: {error}")
        raise ValueError(f"Unknown CPU worker output: {kind!r}")

    def recover_dead_workers() -> None:
        for worker_id, process in enumerate(processes):
            if process.is_alive() or process.exitcode is None:
                continue
            task = worker_tasks.pop(worker_id, None)
            retained = [item for item in ready if item[0] != worker_id]
            ready.clear()
            ready.extend(retained)
            if task is not None:
                error = f"CPU worker exited with code {process.exitcode}."
                finish_result(task, _error_result(task, "worker_error", error))
            if next_task_index < len(tasks):
                input_queues[worker_id].close()
                input_queues[worker_id] = context.Queue(maxsize=2)
                processes[worker_id] = start_process(worker_id)
                assign_task(worker_id)

    try:
        processes.extend(start_process(index) for index in range(cpu_worker_count))
        for worker_id in range(cpu_worker_count):
            assign_task(worker_id)

        with torch.inference_mode():
            while completed < len(tasks):
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
                batch_tasks = [worker_tasks[item[0]] for item in batch]
                try:
                    actions, errors, timings = generator.generate(
                        batch_tasks, [item[2] for item in batch]
                    )
                    last_generation_error = None
                    consecutive_generation_errors = 0
                    for index, (worker_id, _, _) in enumerate(batch):
                        if errors[index] is None:
                            input_queues[worker_id].put(
                                ("action", actions[index], timings, len(batch))
                            )
                        else:
                            input_queues[worker_id].put(
                                ("fail", "generation_error", errors[index])
                            )
                    effective_batch_size = min(
                        oom_batch_limit,
                        max(effective_batch_size, len(batch) * 2),
                    )
                except Exception as exc:
                    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                        generator.device.type == "cuda"
                        and "out of memory" in str(exc).lower()
                    )
                    if len(batch) > 1:
                        effective_batch_size = max(1, len(batch) // 2)
                        if is_oom:
                            oom_batch_limit = min(oom_batch_limit, effective_batch_size)
                            torch.cuda.empty_cache()
                        for item in reversed(batch):
                            ready.appendleft(item)
                        logger.warning(
                            "Rank {} batch {} failed; retrying with batch {}: {}",
                            rank,
                            len(batch),
                            effective_batch_size,
                            exception_summary(exc),
                        )
                        continue
                    error = exception_summary(exc)
                    if error == last_generation_error:
                        consecutive_generation_errors += 1
                    else:
                        last_generation_error = error
                        consecutive_generation_errors = 1
                    if consecutive_generation_errors >= SYSTEMIC_GENERATION_ERROR_LIMIT:
                        raise RuntimeError(
                            "Aborting evaluation after repeated identical generation "
                            f"errors: {error}"
                        ) from exc
                    input_queues[batch[0][0]].put(
                        ("fail", "generation_error", error)
                    )
                recover_dead_workers()
    finally:
        progress.close()
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

    logger.info(
        "Rank {} evaluation complete: tasks={} terminations={}",
        rank,
        len(tasks),
        dict(termination_counts),
    )


def _distributed_context() -> Tuple[int, int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        timeout_value = os.getenv(
            "POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS",
            str(DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS),
        )
        try:
            timeout_seconds = int(timeout_value)
        except ValueError as exc:
            raise ValueError(
                "POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS must be a "
                f"positive integer, got {timeout_value!r}."
            ) from exc
        if timeout_seconds <= 0:
            raise ValueError(
                "POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS must be a "
                f"positive integer, got {timeout_value!r}."
            )
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return rank, world_size, local_rank


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path).absolute()
    output_dir = Path(args.output_dir).absolute()
    config = load_config(config_path)
    validate_config(config)
    rank, world_size, local_rank = _distributed_context()

    if not torch.cuda.is_available():
        raise RuntimeError("The evaluation pipeline requires a CUDA device.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        output_dir / f"worker_{rank:05d}.log",
        enqueue=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    checkpoint_path = Path(
        str(
            config["model"].get("checkpoint_path")
            or config.get("test", {}).get("checkpoint_path")
        )
    ).absolute()
    checkpoint_hash = file_sha256(checkpoint_path) if rank == 0 else None
    if world_size > 1:
        values = [checkpoint_hash]
        dist.broadcast_object_list(values, src=0)
        checkpoint_hash = values[0]

    persisted_config = {
        "evaluator_version": EVALUATOR_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "config": config,
    }
    persisted_path = output_dir / "config.yaml"
    config_error = None
    if rank == 0:
        try:
            if persisted_path.is_file():
                with persisted_path.open("r", encoding="utf-8") as file:
                    previous = yaml.safe_load(file)
                if previous != persisted_config:
                    raise ValueError(
                        f"Evaluation output config mismatch: {persisted_path}. "
                        "Use a new output directory."
                    )
            else:
                _atomic_write_yaml(persisted_path, persisted_config)
        except Exception as exc:
            config_error = exception_summary(exc)
    if world_size > 1:
        config_errors = [config_error]
        dist.broadcast_object_list(config_errors, src=0)
        config_error = config_errors[0]
    if config_error is not None:
        raise RuntimeError(config_error)

    all_tasks = load_tasks(config)
    previous_results = _read_result_shards(output_dir)
    rank_tasks = [
        task
        for task in all_tasks
        if _task_rank(task.task_id, world_size) == rank
        and task.task_id not in previous_results
    ]
    logger.info(
        "Rank {}/{} on {}: total_tasks={} unfinished_rank_tasks={} resumed={}",
        rank,
        world_size,
        torch.cuda.get_device_properties(device).name,
        len(all_tasks),
        len(rank_tasks),
        len(previous_results),
    )

    if rank_tasks:
        dtype = torch_dtype(str(config["model"].get("dtype", "bfloat16")))
        model = PointerCAD(
            qwen_model=str(config["model"]["base_model"]), dtype=dtype
        )
        load_checkpoint(model, checkpoint_path)
        model.to(device)
        model.eval()
        processor = Text2CADProcessor.from_pretrained(
            pretrained_model_name_or_path=str(config["model"]["base_model"]),
            padding_side="left",
        )
        generator = EvaluationGenerator(model, processor, device, config)
        _run_rank_evaluation(generator, rank_tasks, config, output_dir, rank)
    _barrier(world_size)

    if rank == 0:
        merged = _read_result_shards(output_dir)
        expected = {task.task_id for task in all_tasks}
        missing = sorted(expected - set(merged))
        unexpected = sorted(set(merged) - expected)
        if missing or unexpected:
            raise RuntimeError(
                "Incomplete evaluation result set: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}."
            )
        ordered = {task.task_id: merged[task.task_id] for task in all_tasks}
        _atomic_write_json(output_dir / "results.json", ordered)
        logger.success(
            "Evaluation results merged: {} models -> {}",
            len(ordered),
            output_dir / "results.json",
        )
    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
