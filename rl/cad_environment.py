import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import dgl
import numpy as np
import torch
from dgl.data.utils import save_graphs
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from cadmodel.model import CADModel, convert_json_from_deepcad
from measurements.accuracy import accuracy
from measurements.chamfer_distance import chamfer_distance
from measurements.f1_score import f1_score
from measurements.intersection_over_union import intersection_over_union
from measurements.watertightness import is_watertight


def empty_brep_graph() -> dgl.DGLGraph:
    graph = dgl.graph(([], []))
    graph.ndata["x"] = torch.zeros((0,), dtype=torch.float32)
    graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)
    return graph


def build_brep_graph(
    model: Optional[CADModel],
    surf_u_samples: int,
    surf_v_samples: int,
    curv_u_samples: int,
    build_timeout_seconds: float,
) -> dgl.DGLGraph:
    if model is None or not model.seq:
        return empty_brep_graph()

    solid = model.build_model(timeout=build_timeout_seconds)
    if solid is None:
        raise RuntimeError("CAD model construction returned no solid.")
    adjacency = face_adjacency(solid).to_undirected(as_view=True)

    face_features = []
    for position, face_index in enumerate(adjacency.nodes):
        if position != face_index:
            raise ValueError(
                f"Face ID {face_index} is not continuous in the B-Rep graph."
            )
        face = adjacency.nodes[face_index]["face"]
        points = uvgrid(
            face, method="point", num_u=surf_u_samples, num_v=surf_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_u_samples, num_v=surf_v_samples
        )
        curvatures = uvgrid(
            face,
            method="gaussian_curvature",
            num_u=surf_u_samples,
            num_v=surf_v_samples,
        )
        visibility = uvgrid(
            face,
            method="visibility_status",
            num_u=surf_u_samples,
            num_v=surf_v_samples,
        )
        mask = np.logical_or(visibility == 0, visibility == 2)
        face_features.append(
            np.concatenate((points, normals, curvatures, mask), axis=-1)
        )

    edge_features = []
    for edge_index in adjacency.edges:
        edge = adjacency.edges[edge_index]["edge"]
        if not edge.has_curve():
            raise RuntimeError(
                f"Edge {edge_index} does not have a valid curve."
            )
        points = ugrid(edge, method="point", num_u=curv_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_u_samples)
        derivatives = ugrid(
            edge, method="first_derivative", num_u=curv_u_samples
        )
        edge_features.append(
            np.concatenate((points, tangents, -tangents, derivatives), axis=-1)
        )

    edges = list(adjacency.edges)
    source = [edge[0] for edge in edges]
    destination = [edge[1] for edge in edges]
    graph = dgl.graph(
        (source, destination), num_nodes=len(adjacency.nodes)
    )
    graph.ndata["x"] = torch.from_numpy(
        np.asarray(face_features)
    ).to(torch.float32)
    graph.edata["x"] = torch.from_numpy(
        np.asarray(edge_features)
    ).to(torch.float32)
    graph = dgl.add_reverse_edges(
        graph, copy_ndata=True, copy_edata=True
    )
    if graph.ndata["x"].ndim == 4:
        graph.ndata["x"][:, :, :, -2].clamp_(min=-10, max=10)
    if graph.edata["x"].ndim == 3:
        graph.edata["x"][:, :, -3:].clamp_(min=-100, max=100)
    for storage in (graph.ndata, graph.edata):
        if "_ID" in storage:
            del storage["_ID"]
    return graph


def save_brep_graph(graph: dgl.DGLGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_graphs(str(path), [graph.cpu()])


def load_target_model(source_root: Path, relative_path: str) -> CADModel:
    path = (Path(source_root).absolute() / relative_path).absolute()
    source_root = Path(source_root).absolute()
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"Target CAD path escapes source dataset: {relative_path!r}."
        ) from exc
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    data = copy.deepcopy(data)
    if "properties" in data:
        data = convert_json_from_deepcad(data)
    return CADModel.from_dict(data)


def vector_from_actions(
    labels: Iterable[int],
    parameters: Iterable[int],
    pointers: Iterable[int],
):
    labels = list(labels)
    parameters = list(parameters)
    pointers = list(pointers)
    if not (len(labels) == len(parameters) == len(pointers)):
        raise ValueError("Structured CAD action channels have different lengths.")
    return [
        [int(label), max(int(parameter), int(pointer))]
        for label, parameter, pointer in zip(labels, parameters, pointers)
    ]


class CADEnvironment:
    def __init__(
        self,
        strict: bool,
        surf_u_samples: int,
        surf_v_samples: int,
        curv_u_samples: int,
        build_timeout_seconds: float,
    ):
        self.model = CADModel()
        self.strict = strict
        self.surf_u_samples = surf_u_samples
        self.surf_v_samples = surf_v_samples
        self.curv_u_samples = curv_u_samples
        self.build_timeout_seconds = build_timeout_seconds

    def state_graph(self) -> dgl.DGLGraph:
        return build_brep_graph(
            self.model,
            surf_u_samples=self.surf_u_samples,
            surf_v_samples=self.surf_v_samples,
            curv_u_samples=self.curv_u_samples,
            build_timeout_seconds=self.build_timeout_seconds,
        )

    def step(
        self,
        labels,
        parameters,
        pointers,
        parameter_map: Dict[str, Any],
    ) -> Tuple[bool, float]:
        started_at = time.monotonic()
        vector = vector_from_actions(labels, parameters, pointers)
        if not vector:
            raise ValueError("Generated CAD action sequence is empty.")
        previous_length = len(self.model.seq)
        model_end = self.model.from_vector(
            vector,
            parameters=parameter_map,
            strict=self.strict,
        )
        if len(self.model.seq) != previous_length + 1:
            raise RuntimeError("CAD execution did not append exactly one operation.")
        if self.model.build_model(timeout=self.build_timeout_seconds) is None:
            raise RuntimeError("CAD operation produced no valid solid.")
        return model_end, time.monotonic() - started_at


def _numeric_metric(value):
    if isinstance(value, str):
        return None
    return float(value)


def topology_counts(model: CADModel) -> Dict[str, int]:
    solid = model.build_model()
    if solid is None:
        raise RuntimeError("Cannot count topology of an invalid CAD model.")
    return {
        "vertex_count": len(list(solid.vertices())),
        "edge_count": len(list(solid.edges())),
        "face_count": len(list(solid.faces())),
    }


def evaluate_models(
    prediction: CADModel,
    target: CADModel,
    enabled_metrics: Iterable[str],
    chamfer_points: int,
) -> Dict[str, Any]:
    enabled = set(enabled_metrics)
    supported = {"iou", "chamfer_distance", "accuracy", "f1", "watertight"}
    unknown = enabled - supported
    if unknown:
        raise ValueError(f"Unsupported rollout metrics: {sorted(unknown)}")

    metrics: Dict[str, Any] = {
        "operation_count": len(prediction.seq),
        "target_operation_count": len(target.seq),
        "operation_count_error": abs(len(prediction.seq) - len(target.seq)),
    }
    errors: Dict[str, str] = {}
    started_at = time.monotonic()

    try:
        prediction_counts = topology_counts(prediction)
        target_counts = topology_counts(target)
        for name, value in prediction_counts.items():
            metrics[name] = value
            metrics[f"target_{name}"] = target_counts[name]
            metrics[f"{name}_error"] = abs(value - target_counts[name])
    except Exception as exc:
        errors["topology_counts"] = f"{type(exc).__name__}: {exc}"

    if "iou" in enabled:
        try:
            metrics["iou"] = float(intersection_over_union(prediction, target))
        except Exception as exc:
            metrics["iou"] = None
            errors["iou"] = f"{type(exc).__name__}: {exc}"

    if "chamfer_distance" in enabled:
        try:
            metrics["chamfer_distance"] = float(
                chamfer_distance(
                    prediction, target, points=chamfer_points
                )
            )
        except Exception as exc:
            metrics["chamfer_distance"] = None
            errors["chamfer_distance"] = f"{type(exc).__name__}: {exc}"

    if "accuracy" in enabled:
        try:
            vertex, edge, face = accuracy(prediction, target)
            metrics["vertex_accuracy"] = _numeric_metric(vertex)
            metrics["edge_accuracy"] = _numeric_metric(edge)
            metrics["face_accuracy"] = _numeric_metric(face)
        except Exception as exc:
            metrics["vertex_accuracy"] = None
            metrics["edge_accuracy"] = None
            metrics["face_accuracy"] = None
            errors["accuracy"] = f"{type(exc).__name__}: {exc}"

    if "f1" in enabled:
        try:
            metrics["f1"] = {
                str(name): float(value)
                for name, value in f1_score(prediction, target).items()
            }
        except Exception as exc:
            metrics["f1"] = None
            errors["f1"] = f"{type(exc).__name__}: {exc}"

    if "watertight" in enabled:
        try:
            metrics["watertight"] = bool(is_watertight(prediction))
        except Exception as exc:
            metrics["watertight"] = None
            errors["watertight"] = f"{type(exc).__name__}: {exc}"

    metrics["metric_errors"] = errors
    metrics["metrics_time_seconds"] = time.monotonic() - started_at
    return metrics
