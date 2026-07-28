import os
import sys
import time
import json
import argparse
sys.path.append("..")

from occwl.io import save_stl
from occwl.face import Face
from OCC.Core.gp import gp_Pnt
from OCC.Core.gp import gp_Vec
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepClass import BRepClass_FaceClassifier

import numpy as np
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Process
from cadmodel.extrude import Extrude
from misc import match_sketch_plane_from_solid
from cadmodel.model import CADModel, convert_json_from_deepcad


def face_to_thick_solid(face: Face, offset=0.1):
    """
    输入: occwl.face.Face 对象
    输出: TopoDS_Shape (Solid)
    """
    # 取法向
    normal_vec = face.normal([0.5, 0.5])
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    # face → TopoDS_Face
    face_shape = face.topods_shape()

    # 沿法向正负方向构建厚度体（对称，厚度=2*offset）
    vec = gp_Vec(*(normal_vec * (2 * offset)))
    solid_shape = BRepPrimAPI_MakePrism(face_shape, vec, True).Shape()

    return solid_shape


def fuse_solids(solids):
    """
    输入: List[TopoDS_Shape]
    输出: 合并后的 TopoDS_Shape
    """
    if not solids:
        return None
    result = solids[0]
    for s in solids[1:]:
        result = BRepAlgoAPI_Fuse(result, s).Shape()
    return result


def build_and_merge(matched_faces, offset=0.1, mesh_deflection=0.01):
    solids = []
    for f in matched_faces:
        s = face_to_thick_solid(f, offset)
        solids.append(s)

    merged_solid = fuse_solids(solids)

    # 生成 mesh
    mesh = BRepMesh_IncrementalMesh(merged_solid, mesh_deflection)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Mesh generation failed.")

    return merged_solid


def process_one_file(json_path, output_dir, mesh_deflection=0.01):
    file_name = os.path.splitext(os.path.basename(json_path))[0]
    with open(json_path, "r") as fp:
        data = json.load(fp)
        if 'properties' in data:
            data = convert_json_from_deepcad(data)

    model: CADModel = CADModel.from_dict(data)
    os.makedirs(output_dir, exist_ok=True)

    idx_range = range(len(model.seq))
    if "_" in file_name:
        file_name = file_name.split("_")[0]
        idx_range = [len(model.seq) - 1]

    for i in idx_range:
        model_sub = model.submodel(i)
        # model_sub.normalize(keep_sketch_plane=False, normalize_sketch=False)

        extrude = model_sub.seq[-1]
        if i == 0 or not isinstance(extrude, Extrude):
            continue

        base_model = model_sub.submodel(i - 1)
        solid = base_model.build_model()

        for j, sketch_coordinate in enumerate(iter(extrude.sketches.sketch_data.values())):
            face_path = os.path.join(output_dir, f"{file_name}_{model.seq[i].index:05d}_{j:03d}_face.stl")
            if os.path.exists(face_path):
                continue

            parallel_faces, _ = match_sketch_plane_from_solid(solid, sketch_coordinate, False)
            matched_faces = []

            for face in parallel_faces: 
                topo_face = face.topods_shape()
                pnt = gp_Pnt(*sketch_coordinate.origin)
                classifier = BRepClass_FaceClassifier()
                classifier.Perform(topo_face, pnt, 1e-4)
                state = classifier.State()

                if state in (TopAbs_IN, TopAbs_ON):
                    matched_faces.append(face)

            if len(matched_faces) == 0:
                continue

            shape = build_and_merge(matched_faces, offset=0.01, mesh_deflection=mesh_deflection)
            save_stl(shape, face_path)
        
        # 生成 base mesh
        base_path = os.path.join(output_dir, f"{file_name}_{model.seq[i].index:05d}_base.stl")
        if not os.path.exists(base_path):
            mesh = BRepMesh_IncrementalMesh(solid.topods_shape(), mesh_deflection)
            mesh.Perform()
            if not mesh.IsDone():
                raise RuntimeError("Mesh generation failed.")
            save_stl(solid.topods_shape(), base_path)


def multiprocessing(arguments):
    try:
        input_file, args = arguments
        process_one_file(input_file, os.path.join(args.output_dir, f"{args.chunk:04d}"))
    except Exception as e:
        tqdm.write(f"Error: {os.path.basename(input_file)} --- {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Process command line arguments")

    parser.add_argument('-c', '--chunk', type=int, default=0, help="Chunk value, default is 0")
    parser.add_argument('-i', '--input_dir', type=str, default="/public/home/qidacheng/workspace/cad/dataset/base/json", help="Input directory path")
    parser.add_argument('-o', '--output_dir', type=str, default="/public/home/qidacheng/workspace/cad/dataset/base/sketch_plane_stl", help="Output directory path")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes to use")
    parser.add_argument('--timeout', type=int, default=300, help="Maximum runtime for each task (seconds)")

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, f"{args.chunk:04d}")
    files_and_folders = os.listdir(input_dir)
    input_files = [os.path.join(input_dir, f) for f in files_and_folders if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".json")]

    max_workers = args.num_processes
    processes = []
    pending = list(zip(input_files, repeat(args)))

    try:
        with tqdm(total=len(pending)) as pbar:
            while pending or processes:
                while pending and len(processes) < max_workers:
                    task_args = pending.pop(0)
                    p = Process(target=multiprocessing, args=(task_args,))
                    p.start()
                    processes.append((p, time.time()))
                
                still_running = []
                for proc, start_time in processes:
                    proc.join(timeout=0)
                    if proc.is_alive():
                        if time.time() - start_time > args.timeout:
                            tqdm.write(f"Task timeout, terminating process PID {proc.pid}")
                            proc.terminate()
                            proc.join()
                            pbar.update(1)
                        else:
                            still_running.append((proc, start_time))
                    else:
                        pbar.update(1)
                processes = still_running
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Detected Ctrl+C, terminating all processes...")
        for proc, _ in processes:
            proc.terminate()
        for proc, _ in processes:
            proc.join()

if __name__ == "__main__":
    main()
    # process_one_file("/public/home/qidacheng/workspace/cad/dataset/addon/json/0000/00001241.json", "./")
