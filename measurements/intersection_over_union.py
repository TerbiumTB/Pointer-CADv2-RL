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


def __normalization_transform(shape, use_triangulation=True, margin=0.0, tol=1e-12):
    bbox = Bnd_Box()
    bbox.SetGap(0.0)
    brepbndlib.Add(shape, bbox, use_triangulation)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    if any(map(lambda v: math.isinf(v) or math.isnan(v), [xmin, ymin, zmin, xmax, ymax, zmax])):
        raise ValueError("Bounding box contains invalid numbers (inf/NaN).")

    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    cx, cy, cz = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5
    longest = max(dx, dy, dz)
    if longest < tol:
        raise ValueError("Cannot normalize a degenerate shape.")
    return (cx, cy, cz), 2.0 * (1.0 - margin) / longest


def __apply_normalization(shape, center, scale):
    tr_translate = gp_Trsf()
    tr_translate.SetTranslation(gp_Vec(*(-value for value in center)))
    shape_centered = BRepBuilderAPI_Transform(shape, tr_translate, True).Shape()

    tr_scale = gp_Trsf()
    tr_scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale)
    shape_scaled = BRepBuilderAPI_Transform(shape_centered, tr_scale, True).Shape()
    return shape_scaled


def intersection_over_union(pred: CADModel, gt: CADModel):
    gt_shape = gt.build_model().topods_shape()
    pred_shape = pred.build_model().topods_shape()
    center, scale = __normalization_transform(gt_shape, use_triangulation=True)
    gt_normalized = __apply_normalization(gt_shape, center, scale)
    pred_normalized = __apply_normalization(pred_shape, center, scale)

    intersection = BRepAlgoAPI_Common(pred_normalized, gt_normalized).Shape()
    intersection_volume = __compute_volume(intersection)
    union = BRepAlgoAPI_Fuse(pred_normalized, gt_normalized).Shape()
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
