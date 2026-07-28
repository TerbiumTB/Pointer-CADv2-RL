import math
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse

from cadmodel.model import CADModel



def __compute_volume(solid):
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    return abs(props.Mass())


def __fit_shape_to_unit_cube(shape, use_triangulation=True, margin=0.0, tol=1e-12):
    """
    将 TopoDS_Shape 等比缩放并居中到以原点为中心的 [-1,1]^3 立方体内。
    - use_triangulation: 计算包围盒时是否使用三角化（更紧）。
    - margin: 可选的内缩边距（例如 0.02 表示四周留 2% 空隙）。
    - tol: 尺寸为“零”的判定阈值。
    返回：(scaled_shape, scale_factor, center_before)
    """
    # 1) 计算包围盒
    bbox = Bnd_Box()
    bbox.SetGap(0.0)
    brepbndlib.Add(shape, bbox, use_triangulation)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    # 基本健壮性检查
    if any(map(lambda v: math.isinf(v) or math.isnan(v), [xmin, ymin, zmin, xmax, ymax, zmax])):
        raise ValueError("Bounding box contains invalid numbers (inf/NaN).")

    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    cx, cy, cz = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5

    longest = max(dx, dy, dz)
    if longest < tol:
        # 形体几乎退化为一个点：只需把它平移到原点即可
        tr_translate = gp_Trsf()
        tr_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
        shape_centered = BRepBuilderAPI_Transform(shape, tr_translate, True).Shape()
        return shape_centered, 1.0, (cx, cy, cz)

    # 2) 先把形体移到原点（确保关于坐标轴对称）
    tr_translate = gp_Trsf()
    tr_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
    shape_centered = BRepBuilderAPI_Transform(shape, tr_translate, True).Shape()

    # 3) 按最大边做等比缩放，使其落入 [-1+margin, 1-margin]
    target_side = 2.0 * (1.0 - margin)
    s = target_side / longest

    tr_scale = gp_Trsf()
    tr_scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), s)  # 以原点为中心缩放
    shape_scaled = BRepBuilderAPI_Transform(shape_centered, tr_scale, True).Shape()

    return shape_scaled


def intersection_over_union(pred: CADModel, gt: CADModel):
    gt_normalized = __fit_shape_to_unit_cube(gt.build_model().topods_shape(), use_triangulation=True)
    pred_normalized = __fit_shape_to_unit_cube(pred.build_model().topods_shape(), use_triangulation=True)
    intersection = BRepAlgoAPI_Common(pred_normalized, gt_normalized).Shape()
    intersection_volume = __compute_volume(intersection)

    union = BRepAlgoAPI_Fuse(pred.build_model().topods_shape(), gt.build_model().topods_shape()).Shape()
    union_volume = __compute_volume(union)

    if union_volume == 0:
        return 0.0
    return max(0.0, min(1.0, intersection_volume / union_volume))



if __name__ == "__main__":
    from loguru import logger
    from cadmodel.model import CADModel

    @logger.catch()
    def test():
        import json
        from cadmodel.model import convert_json_from_deepcad

        with open("preprocessing/data.json", "r") as fp:
            data = json.load(fp)
            data = convert_json_from_deepcad(data)

        model = CADModel.from_dict(data)
        
        print(intersection_over_union(model, model))
    test()