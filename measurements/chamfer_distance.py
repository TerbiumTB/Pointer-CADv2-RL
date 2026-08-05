import numpy as np

from scipy.spatial import cKDTree
from trimesh.sample import sample_surface, sample_surface_even

from misc import create_mesh
from cadmodel.model import CADModel



# def __normalize(points, centered=True):
#     if centered:
#         centroid = np.mean(points, axis=0)     # 计算点云的中心
#         points = points - centroid             # 平移到原点
#     scale = np.abs(np.max(points) - np.min(points))
#     points = points / scale
#     return points


def __normalize(points_gt, points_pred, centered=True):
    if centered:
        centroid = np.mean(points_gt, axis=0)
        points_gt = points_gt - centroid
        points_pred = points_pred - centroid

    scale = np.abs(np.max(points_gt) - np.min(points_gt))
    points_gt = points_gt / scale
    points_pred = points_pred / scale
    return points_gt, points_pred


def chamfer_distance_from_meshes(
    pred_mesh,
    gt_mesh,
    points=1024,
    type="uniform",
    normalize=True,
):
    pred_points, gt_points = None, None

    if type == "uniform":
        pred_points, _ = sample_surface(pred_mesh, points)
        gt_points, _ = sample_surface(gt_mesh, points)
    elif type == "even":
        pred_points, _ = sample_surface_even(pred_mesh, points)
        gt_points, _ = sample_surface_even(gt_mesh, points)
    else:
        raise AssertionError(f"Unknown sample type {type}")
    
    if normalize:
        # pred_points = __normalize(pred_points)
        # gt_points = __normalize(gt_points)
        gt_points, pred_points = __normalize(gt_points, pred_points)

    # one direction
    pred_points_kd_tree = cKDTree(pred_points)
    one_distances, one_vertex_ids = pred_points_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = cKDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_pred_chamfer + pred_to_gt_chamfer


def chamfer_distance(
    pred: CADModel,
    gt: CADModel,
    points=1024,
    type="uniform",
    normalize=True,
):
    return chamfer_distance_from_meshes(
        create_mesh(pred),
        create_mesh(gt),
        points=points,
        type=type,
        normalize=normalize,
    )

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
        print(chamfer_distance(model, model))
    test()
