import numpy as np

from scipy.spatial import cKDTree

from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.io import load_step
from occwl.compound import Compound

from OCC.Core import TopoDS
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid

from cadmodel.model import CADModel



class GeoCADModel:
    """封装 STEP 模型的读取与基础几何分析"""
    def __init__(self, cad_model: CADModel):
        self.shape = cad_model.build_model()
        if self.shape is None:
            raise ValueError("CADModel does not contain a valid OCC shape.")

        self._bbox = self._compute_bbox()

    def _iter_vertices(self):
        """遍历所有顶点对象"""
        if isinstance(self.shape, Solid):
            iterator = self.shape.vertices()
        else:
            comp = Compound(self.shape.Shape() if hasattr(self.shape, "Shape") else self.shape)
            iterator = comp.vertices()
        for v in iterator:
            yield v

    def vertices(self, dedup_tol=None):
        """返回所有顶点坐标 (N, 3)，可选容差去重"""
        verts = []
        circle_vertices = []
        for edge in self.shape.edges():
            if len(list(self.shape.faces_from_edge(edge))) > 0 and edge.curve_type() == "circle" and edge.closed_curve() and edge.closed_edge():
                circle_vertices.extend([v.point() for v in edge.vertices()])
                center = (edge.point(edge.u_bounds().a) + edge.point(edge.u_bounds().length() / 2 + edge.u_bounds().a)) / 2
                verts.append(center)

        vertices_iterator = self.shape.vertices()
        circle_vertices = np.array(circle_vertices, dtype=float)
        for v in vertices_iterator:
            p = v.point()
            if circle_vertices.size > 0:
                dists = np.linalg.norm(circle_vertices - p, axis=1)
                if np.any(dists < 1e-6):
                    continue
            verts.append(p)

        V = np.array(verts, dtype=float) if verts else np.zeros((0, 3), dtype=float)
        if dedup_tol is not None and len(V) > 0:
            Q = np.round(V / dedup_tol).astype(np.int64)
            _, idx = np.unique(Q, axis=0, return_index=True)
            V = V[np.sort(idx)]
        return V
    
    def edges(self):
        """返回所有边对象及其对应中间点"""
        for edge in self.shape.edges():
            if edge.curve_type() == "circle" and abs(edge.u_bounds().length() - 2 * np.pi) < 1e-6:
                center = (edge.point(edge.u_bounds().a) + edge.point(edge.u_bounds().length() / 2 + edge.u_bounds().a)) / 2
                yield edge, center
            else:
                if edge.curve_type() == "line" and len(list(self.shape.faces_from_edge(edge))) == 1 and len(list(edge.vertices())) == 2:
                    point1, point2 = list(edge.vertices())
                    faces1 = list(self.shape.faces_from_vertex(point1))
                    faces2 = list(self.shape.faces_from_vertex(point2))
                    if len(faces1) == 2 and len(faces2) == 2 and (faces1[0] in faces2 or faces1[1] in faces2):
                        side = faces1[0] if faces1[0] in faces2 else faces1[1]
                        if side.surface_type() == "cylinder" and side.closed_u() or side.closed_v():
                            continue
                    
                mid_u = (edge.u_bounds().a + edge.u_bounds().b) / 2
                mid_point = edge.point(mid_u)
                yield edge, mid_point

    def faces(self):
        """返回所有面对象"""
        for face in self.shape.faces():
            yield face

    def get_face_by_edge(self, edge):
        """通过边对象获取其所属面对象列表"""
        return list(self.shape.faces_from_edge(edge))
    
    def get_edge_by_face(self, face):
        """通过面对象获取其所属边对象列表"""
        return list(self.shape.edges_from_face(face))

    def _compute_bbox(self):
        """
        使用 OCCWL BoundingBoxMixin 计算模型整体包围盒。
        返回形状为 (2, 3) 的 ndarray：
            [[xmin, ymin, zmin],
            [xmax, ymax, zmax]]
        """
        mins, maxs = [], []

        if hasattr(self.shape, "exact_box"):
            try:
                box = self.shape.exact_box()
            except Exception:
                box = self.shape.box()
        elif hasattr(self.shape, "box"):
            box = self.shape.box()
        else:
            return np.zeros((2, 3), dtype=float)

        mins.append(box.min_point())
        maxs.append(box.max_point())

        if not mins or not maxs:
            return np.zeros((2, 3), dtype=float)

        mins = np.min(np.array(mins), axis=0)
        maxs = np.max(np.array(maxs), axis=0)
        return np.stack([mins, maxs], axis=0)

    @property
    def bbox(self):
        """模型的 bounding box (2, 3)：[[xmin, ymin, zmin], [xmax, ymax, zmax]]"""
        return self._bbox.copy()


class GeoMap:
    def __init__(self):
        self.curve_map = {}
        self.curve_to_face = {}
        self.face_to_curve = {}

    def add_curve_map(self, c1, c2):
        if c1 in self.curve_map or c2 in self.curve_map:
            raise ValueError("curve already mapped")
        
        self.curve_map[c1] = c2
        self.curve_map[c2] = c1

    def bind_curve_face(self, curve, faces):
        if curve not in self.curve_to_face:
            self.curve_to_face[curve] = set(faces)
        else:
            raise ValueError("curve already bound to a face")

        for face in faces:
            if face not in self.face_to_curve:
                self.face_to_curve[face] = set()
            self.face_to_curve[face].add(curve)

    def bind_all_curves_faces(self, model: GeoCADModel):
        for curve, _ in model.edges():
            if not curve in self.curve_map:
                continue

            faces = model.get_face_by_edge(curve)
            if len(faces) > 0:
                self.bind_curve_face(curve, faces)


def point_on_edge(edge: Edge, points, tol=1e-6):
    sum_error = 0
    curve_handle, first, last = BRep_Tool.Curve(TopoDS.topods_Edge(edge.topods_shape()))
    if curve_handle is None:
        return False
    for pt in points:
        pnt = gp_Pnt(*pt)
        projector = GeomAPI_ProjectPointOnCurve(pnt, curve_handle)
        if projector.NbPoints() == 0:
            return False
        dist = projector.LowerDistance()
        if dist > tol:
            return False
        sum_error += dist
    return sum_error < tol * len(points)

def vertex_match_ratio(gt_vertices, pred_vertices, tol=1e-4):
    """计算GT顶点中有多少比例在pred中出现"""
    matched = 0
    for gt_v in gt_vertices:
        # 计算gt_v与所有pred_v的欧氏距离
        dists = np.linalg.norm(pred_vertices - gt_v, axis=1)
        if np.any(dists < tol):  # 若存在一个预测顶点距离小于阈值
            matched += 1
    return matched / len(gt_vertices) if len(gt_vertices) > 0 else 0.0

def edge_match_ratio(gt_model: GeoCADModel, pred_edge_midpoints_tree: cKDTree, pred_edge_list: list[Edge], tol=1e-4):
    def edge_type(edge: Edge):
        etype = edge.curve_type()
        if etype == "line":
            return "line"
        elif etype == "circle":
            if edge.closed_curve() and edge.closed_edge():
                return "circle"
            else:
                return "arc"
        else:
            return etype

    total_edge = 0
    matched_edge = 0
    geo_map = GeoMap()
    for edge, midpoint in gt_model.edges():
        gt_type = edge_type(edge)
        total_edge += 1

        # 使用KD树查找最近的预测边中点
        dist, idx = pred_edge_midpoints_tree.query(midpoint)
        if dist < tol:
            pred_edge: Edge = pred_edge_list[idx]
            
            # check curve type
            pred_type = edge_type(pred_edge)
            if gt_type != pred_type:
                continue

            # check length
            gt_length = edge.length()
            pred_length = pred_edge.length()
            if abs(gt_length - pred_length) >= tol:
                continue

            # if is circle
            if gt_type == "circle":
                gt_u_bounds = edge.u_bounds()
                gt_point1 = edge.point(gt_u_bounds.a)
                gt_point2 = edge.point(gt_u_bounds.a + gt_u_bounds.length() / 4)
                # check point on pred
                if not point_on_edge(pred_edge, [gt_point1, gt_point2], tol=tol):
                    continue
            # if is line or arc
            elif gt_type == "line" or gt_type == "arc":
                gt_vertices = list(edge.vertices())
                pred_vertices = list(pred_edge.vertices())
                if len(gt_vertices) != 2 or len(pred_vertices) != 2:
                    continue
                gt_v1 = gt_vertices[0].point()
                gt_v2 = gt_vertices[1].point()
                pred_v1 = pred_vertices[0].point()
                pred_v2 = pred_vertices[1].point()
                if not ((np.linalg.norm(gt_v1 - pred_v1) < tol and np.linalg.norm(gt_v2 - pred_v2) < tol) or \
                   (np.linalg.norm(gt_v1 - pred_v2) < tol and np.linalg.norm(gt_v2 - pred_v1) < tol)):
                    continue
            else:
                raise NotImplementedError(f"Edge type {gt_type} not implemented for edge matching.")
            
            geo_map.add_curve_map(edge, pred_edge)
            matched_edge += 1

    return  matched_edge / total_edge if total_edge > 0 else "N/A", geo_map

def face_match_ratio(geo_map: GeoMap, gt_model: GeoCADModel):
    gt_faces = list(gt_model.faces())
    total_face = len(gt_faces)
    matched_face = 0

    for gt_edge, _ in gt_model.edges():
        if gt_edge in geo_map.curve_map:
            continue

        faces = gt_model.get_face_by_edge(gt_edge)
        for face in faces:
            if face in gt_faces: gt_faces.remove(face)
    
    for face in gt_faces:
        if face not in geo_map.face_to_curve:
            continue
            
        gt_curves = geo_map.face_to_curve[face]
        pred_curves = [geo_map.curve_map[c] for c in gt_curves]

        if len(pred_curves) == 0:
            continue

        pred_faces: set[Face] = set(geo_map.curve_to_face.get(pred_curves[0], []))
        for pred_curve in pred_curves[1:]:
            pred_faces = pred_faces.intersection(set(geo_map.curve_to_face.get(pred_curve, [])))
        
        if len(pred_faces) == 0:
            continue

        pred_faces = [pred_face for pred_face in pred_faces if pred_face.surface_type() == face.surface_type() and len(gt_curves) == len(geo_map.face_to_curve[pred_face])]

        if len(pred_faces) > 1:
            raise ValueError("Multiple predicted faces found for one ground-truth face.")
        elif len(pred_faces) == 0:
            continue
        
        matched_face += 1

    return  matched_face / total_face if total_face > 0 else "N/A"

def accuracy(pred: CADModel, gt: CADModel):
    gt_model = GeoCADModel(gt)
    pred_model = GeoCADModel(pred)

    abs_bbox: np.ndarray = np.abs(gt_model.bbox[1] - gt_model.bbox[0])
    tol = abs_bbox[abs_bbox > 0].min() * 1e-3

    vertex_acc = vertex_match_ratio(
        gt_model.vertices(),
        pred_model.vertices(),
        tol=tol,
    )

    pred_edge_midpoints = []
    pred_edge_list = []
    for edge, midpoint in pred_model.edges():
        pred_edge_list.append(edge)
        pred_edge_midpoints.append(midpoint)
    pred_edge_midpoints = np.array(pred_edge_midpoints, dtype=float)
    pred_edge_midpoints_tree = cKDTree(pred_edge_midpoints)
    edge_acc, geo_map = edge_match_ratio(
        gt_model,
        pred_edge_midpoints_tree,
        pred_edge_list,
        tol=tol,
    )

    geo_map.bind_all_curves_faces(gt_model)
    geo_map.bind_all_curves_faces(pred_model)
    face_acc = face_match_ratio(
        geo_map,
        gt_model,
    )

    return vertex_acc, edge_acc, face_acc



if __name__ == "__main__":
    import json
    from cadmodel.model import convert_json_from_deepcad

    with open("/mnt/afs_01e/mayi-folder/PointerCAD/dataset/pointercad/dataset/0000/00000007/json/00000007_00001.json", "r") as fp:
        data = json.load(fp)
        if 'properties' in data:
            data = convert_json_from_deepcad(data)
        model1 = CADModel.from_dict(data)
    gt = CADModel.from_dict(data)
    pred = CADModel.from_dict(data)

    v_acc, e_acc, f_acc = accuracy(pred, gt)
    print(f"Vertex Accuracy: {v_acc}")
    print(f"Edge Accuracy: {e_acc}")
    print(f"Face Accuracy: {f_acc}")