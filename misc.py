import os
import re
import torch
import trimesh
import tempfile
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from simpleeval import simple_eval
from torch.distributed import get_rank
from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.gp import gp_Pnt, gp_Trsf
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Edge, topods

from cadmodel.coordinatesys import CoordinateSystem



ROUND_JSON = 6

MAX_PART_LENGTH = 64
MAX_GENERATION_LENGTH = 1024

EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation", "IntersectFeatureOperation"]
EXTENT_TYPE = ["OneSideFeatureExtentType", "SymmetricFeatureExtentType", "TwoSidesFeatureExtentType"]

STANDARD_PLANES = {"Top": np.array([0, 0, 1]), "Right": np.array([1, 0, 0]), "Front": np.array([0, -1, 0])}

TOKEN = ["<|padding|>", "<|model_end|>", "<|part_end|>", "<|sketch_start|>", "<|extrude_start|>", "<|chamfer_start|>", "<|fillet_start|>", 
         "<|profile_start|>", "<|loop_start|>", "<|curve_start|>", "<|pointer_enable|>", "<|pointer_disable|>", "<|clockwise|>", 
         "<|counter_clockwise|>", "<|direction_x+|>", "<|direction_x-|>", "<|direction_y+|>", "<|direction_y-|>", "<|direction_z+|>", 
         "<|direction_z-|>",  "<|extrude_new|>", "<|extrude_join|>", "<|extrude_cut|>", "<|extrude_intersect|>", "<|length_value|>", "<|angle_value|>"]



def map_indices_to_parameter(indices, parameters=None, type="length"):
    """
    Map indices to a parameter. `type` must be either "length" or "angle".
    """
    allowed = ("length", "angle")
    type_str = str(type).lower()
    if type_str not in allowed:
        raise ValueError(f"'type' must be one of {allowed}, got {type!r}")

    if parameters is None:
        raise ValueError("Parameters dictionary is required for mapping indices to parameters.")

    p: list = parameters[type_str]
    def _mapper(x):
        idx = x - 1
        if idx < 0 or idx >= len(p):
            raise ValueError(f"Parameter index {x} not found in parameters list.")
        return p[idx]

    if isinstance(indices, (int, float)):
        return _mapper(indices)
    elif isinstance(indices, np.ndarray):
        return np.array([_mapper(x) for x in indices.flatten()]).reshape(indices.shape)
    elif isinstance(indices, list):
        return [_mapper(x) for x in indices]
    elif isinstance(indices, tuple):
        return tuple(_mapper(x) for x in indices)
    else:
        raise TypeError(f"Unsupported data type: {type_str(_mapper)}")


def map_patameter_to_indices(value, parameters=None, type="length"):
    """
    Map a parameter to indices. `type` must be either "length" or "angle".
    """
    allowed = ("length", "angle")
    type_str = str(type).lower()
    if type_str not in allowed:
        raise ValueError(f"'type' must be one of {allowed}, got {type!r}")

    if parameters is None:
        return value

    p = {k: v["value_m" if type_str == "length" else "value"] for k, v in parameters[type_str].items()}
    def _mapper(x):
        label, error = 1, abs(p[1] - x)
        for k, v in p.items():
            curr_error = abs(v - x)
            if curr_error < error:
                label, error = k, curr_error
        if (type_str == "length" and error >= 1e-5) or (type_str == "angle" and error >= 1e-3):
            raise ValueError(f"Parameter value {x} not found in parameters list.")
        return int(label)

    if isinstance(value, (int, float)):
        return _mapper(value)
    elif isinstance(value, np.ndarray):
        return np.array([_mapper(x) for x in value.flatten()]).reshape(value.shape)
    elif isinstance(value, list):
        return [_mapper(x) for x in value]
    elif isinstance(value, tuple):
        return tuple(_mapper(x) for x in value)
    else:
        raise TypeError(f"Unsupported data type: {type_str(_mapper)}")


#######################################################################################


def find_edge_by_points_and_length(shape: TopoDS_Compound, points: List[Tuple[float, float, float]], length: float, length_tolerance: float = 1e-2, point_tolerance: float = 1e-4) -> TopoDS_Edge:
    """
    查找与给定点序列最匹配且长度误差最小的边（TopoDS_Edge）。

    优先按长度误差最小匹配，再从中选点匹配误差最小者。

    参数:
        shape: TopoDS_Compound
        points: 点列表，每个为 (x, y, z)
        length: 期望边长度
        length_tolerance: 边长度允许的最大偏差
        point_tolerance: 点投影距离容差

    返回:
        最佳匹配的 TopoDS_Edge，若无匹配则返回 None
    """
    def points_on_edge(edge, pts, tol):
        sum_error = 0
        curve_handle, first, last = BRep_Tool.Curve(edge)
        if curve_handle is None:
            return -1
        for pt in pts:
            pnt = gp_Pnt(*pt)
            projector = GeomAPI_ProjectPointOnCurve(pnt, curve_handle)
            if projector.NbPoints() == 0:
                return -1
            dist = projector.LowerDistance()
            if dist > tol:
                return -1
            sum_error += dist
        return sum_error

    candidates = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        curve_handle, first, last = BRep_Tool.Curve(edge)
        if curve_handle is None:
            exp.Next()
            continue
        
        if length >= 0:
            edge_length = GCPnts_AbscissaPoint.Length(GeomAdaptor_Curve(curve_handle, first, last))
            length_err = abs(edge_length - length)
            if length_err > length_tolerance:
                exp.Next()
                continue
        else:
            length_err = 0

        point_error = points_on_edge(edge, points, point_tolerance)
        if point_error >= 0:
            candidates.append((edge, length_err, point_error))
        exp.Next()

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1], x[2]))
    best_edge = candidates[0][0]
    return best_edge


def project_point_to_edge(csys: CoordinateSystem, point: list, edge: Edge, tolerance: float = 0.2):
    def distance(p1, p2) -> float:
        return ((p1.X() - p2.X()) ** 2 +
                (p1.Y() - p2.Y()) ** 2 +
                (p1.Z() - p2.Z()) ** 2) ** 0.5
    point_coords = csys.relative2world(point)
    world_point = gp_Pnt(*point_coords)
    projector = GeomAPI_ProjectPointOnCurve(world_point, edge.curve())
    if projector.NbPoints() > 0:
        closest_point = projector.NearestPoint()
        if distance(world_point, closest_point) < tolerance:
            world_point = np.array([closest_point.X(), closest_point.Y(), closest_point.Z()])
            return csys.world2relative(world_point)[:2]
        else:
            return np.array(point)
    return None


def get_sketch_plane_normal(plane):
    if isinstance(plane, str):
        return STANDARD_PLANES[plane], np.array([0, 0, 0])
    elif isinstance(plane, Face):
        face_origin_uv = plane.uv_bounds().center()
        face_origin = plane.point(face_origin_uv)
        face_normal = plane.normal(face_origin_uv)
        face_normal = face_normal / np.linalg.norm(face_normal)
        return face_normal, face_origin
    else:
        raise ValueError("Unable to parse sketch plane normal vector.")


def match_sketch_plane_from_solid(
    solid: Solid,
    coordinate: CoordinateSystem,
    include_standard_planes: bool = False,
    normal_tol: float = 1e-4,
    point_tol: float = 1e-4,
):
    """
    匹配 sketch plane 到 solid 中的面。

    参数：
        solid: Solid 对象
        coordinate: 草图平面
        include_standard_planes: 是否将 XOY、YOZ、XOZ 平面也计入考虑范围
        normal_tol: 法向量容差
        point_tol: 点到面的距离容差

    返回：
        匹配的 Face 对象列表
    """
    matched_faces = []
    matched_planes = []
    
    sketch_origin = np.array(coordinate.origin)
    sketch_normal = np.array(coordinate.normal)
    sketch_normal = sketch_normal / np.linalg.norm(sketch_normal)  # 单位化

    if solid is not None:
        for face in solid.faces():
            if face.surface_type() != "plane":
                continue  # 跳过非平面面

            face_origin_uv = face.uv_bounds().center()
            face_origin = face.point(face_origin_uv)
            face_normal = face.normal(face_origin_uv)
            face_normal = face_normal / np.linalg.norm(face_normal)

            # 检查法向是否平行（允许方向相反）
            dot = np.dot(sketch_normal, face_normal)
            if abs(abs(dot) - 1.0) > normal_tol:
                continue

            # 检查面上任意一点到草图平面的距离
            vec = face_origin - sketch_origin
            dist = abs(np.dot(vec, sketch_normal))
            if dist > point_tol:
                continue

            matched_faces.append(face)

    # 额外考虑标准平面（仅在 solid 中未匹配成功的情况下有意义）
    if include_standard_planes:
        for name, std_n in STANDARD_PLANES.items():
            dot_std = np.dot(sketch_normal, std_n)
            if abs(abs(dot_std) - 1.0) > normal_tol:
                continue
                
            dist_std = abs(np.dot(std_n, sketch_origin))
            if dist_std > point_tol:
                continue
            matched_planes.append(name)

    return matched_faces, matched_planes


def match_curve_from_edges(
    edges: List[Edge], 
    csys: CoordinateSystem, 
    xy: Tuple[float, float], 
    tolerance: float = 1e-4
) -> List[Edge]:
    """
    判断给定坐标点与 Edge 是否重合（允许误差），返回所有匹配的 Edge 列表。

    参数:
    - edges: Edge 对象列表
    - csys: CoordinateSystem 对象，含 relative2world(xy) 方法
    - xy: 二维坐标 (x, y)
    - tolerance: 匹配误差范围

    返回:
    - 匹配的 Edge 列表（可能为空）
    """

    def distance(p1, p2) -> float:
        return ((p1.X() - p2.X()) ** 2 +
                (p1.Y() - p2.Y()) ** 2 +
                (p1.Z() - p2.Z()) ** 2) ** 0.5

    point_coords = csys.relative2world(xy)  # 返回 [x, y, z]
    world_point = gp_Pnt(*point_coords)
    matched_edges = []

    for edge in edges:
        curve = edge.curve()
        projector = GeomAPI_ProjectPointOnCurve(world_point, curve)
        if projector.NbPoints() > 0:
            closest_point = projector.NearestPoint()
            if distance(world_point, closest_point) <= tolerance:
                matched_edges.append(edge)

    return matched_edges


#######################################################################################


class DummyTQDM:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable if self.iterable else [])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


def get_progress_bar(iterable=None, **kwargs):
    if (get_rank() % torch.cuda.device_count()) == 0:
        return tqdm(iterable, **kwargs)
    else:
        return DummyTQDM(iterable)


#######################################################################################


def create_mesh(model, mode="ascii", linear_deflection=0.001, angular_deflection=0.5) -> trimesh.Trimesh:
    """
    Create a trimesh.Trimesh object from an occwl.Solid model.

    Parameters:
        model (Solid): The occwl Solid to be meshed.
        mode (str): 'ascii' or 'binary' STL export mode. Default is 'ascii'.
        linear_deflection (float): Controls mesh accuracy. Smaller = finer mesh.
        angular_deflection (float): Controls angular tolerance. Smaller = more accurate.

    Returns:
        trimesh.Trimesh: The mesh representation of the model.
    """
    a_shape = model.build_model().topods_shape()
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0, 0, 0), 1000)
    brep_trsf = BRepBuilderAPI_Transform(a_shape, trsf, True)
    a_shape_mm = brep_trsf.Shape()
    
    if a_shape_mm.IsNull():
        raise ValueError("Input model shape is null.")
    if mode not in ["ascii", "binary"]:
        raise ValueError("mode must be 'ascii' or 'binary'")

    # Perform OpenCascade mesh generation
    mesh = BRepMesh_IncrementalMesh(
        a_shape_mm, linear_deflection, False, angular_deflection, True
    )
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Mesh generation failed.")

    # Create a temporary STL file
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
        filename = temp_file.name

    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(mode == "ascii")
    stl_writer.Write(a_shape_mm, filename)

    if not os.path.exists(filename):
        raise IOError(f"Temporary STL file not created: {filename}")

    try:
        mesh = trimesh.load(filename, force='mesh')
    finally:
        os.remove(filename)

    return mesh


#######################################################################################


def extract_parameters_from_plan_by_tag(plan: str, tag: str) -> dict:
    tag_def_re = re.compile(rf"<{tag}(\d+)=([^>]+)>")
    unit_re = re.compile(r"\b(\d+(?:\.\d+)?)[ ]*(mm|cm|m|km|um|µm|nm|deg)\b")
    ref_re = re.compile(rf"\b{tag}(\d+)\b")
    values = {}

    for m in tag_def_re.finditer(plan):
        try:
            idx = int(m.group(1))
            rhs = m.group(2).strip()

            unit_m = unit_re.search(rhs)
            if unit_m:
                unit = unit_m.group(2)
                simple_rhs = rhs[:unit_m.start()] + rhs[unit_m.end():]
                simple_rhs = simple_rhs.strip()
            else:
                ref_tag_id = ref_re.search(rhs)
                if ref_tag_id is not None:
                    unit = values[f"{tag}{ref_tag_id.group(1)}"][1]
                else:
                    # 既没有单位也没有引用，无法确定单位，跳过该标签
                    continue

            simple_rhs = rhs.replace(unit, "").strip()
            value = simple_eval(simple_rhs, names={k: v[0] for k, v in values.items()})
            values[f"{tag}{idx}"] = (float(value), unit)
        except Exception:
            # 跳过解析失败的标签，但不要抛出（上层逻辑依赖容错）
            continue

    return values


def extract_parameters_from_plan(plan) -> dict:
    parameters = {"length": {}, "angle": {}}

    unit_to_m = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
    length = extract_parameters_from_plan_by_tag(plan, "L")
    for k, v in length.items():
        if v[1] not in unit_to_m:
            continue
        parameters["length"][int(k[1:])] = {"value": v[0], "unit": v[1], "value_m": v[0] * unit_to_m[v[1]]}
    
    angle = extract_parameters_from_plan_by_tag(plan, "A")
    for k, v in angle.items():
        if v[1] != "deg":
            continue
        parameters["angle"][int(k[1:])] = {"value": v[0], "unit": v[1]}

    return parameters
