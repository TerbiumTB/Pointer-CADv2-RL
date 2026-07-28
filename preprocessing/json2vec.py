import os
import sys
import time
import json
import pickle
import argparse
sys.path.append("..")

from tqdm import tqdm
from itertools import repeat
from multiprocessing import Process
from cadmodel.model import CADModel, convert_json_from_deepcad
from dgl.data.utils import save_graphs


def process_one_file(json_path, part_id, ref_vec_path, parameter_path, output_dir, curv_u_samples, surf_u_samples, surf_v_samples):
    with open(json_path, "r") as fp:
        json_data = json.load(fp)
        if 'properties' in json_data:
            json_data = convert_json_from_deepcad(json_data)

    with open(parameter_path, "rb") as f:
        parameters = pickle.load(f)

    if part_id is None:
        model_id, part_id = os.path.splitext(os.path.basename(json_path))[0].split("_")
    else:
        model_id = os.path.splitext(os.path.basename(json_path))[0]

    json_data["sequence"] = [feature for feature in json_data["sequence"] if feature["index"] <= int(part_id)]

    model = CADModel.from_dict(json_data)
    graph, vec = model.to_vector(-1, parameters, curv_u_samples, surf_u_samples, surf_v_samples)

    with open(ref_vec_path, "rb") as f:
        ref_vec = pickle.load(f)
        vec[-1][0] = ref_vec["vec"][-1][0]

    output_path_vector = os.path.join(output_dir, model_id, "vector", f"{model_id}_{part_id}.pkl")
    output_path_graph = os.path.join(output_dir, model_id, "graph", f"{model_id}_{part_id}.bin")

    os.makedirs(os.path.dirname(output_path_vector), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_graph), exist_ok=True)

    with open(output_path_vector, "wb") as f:
        pickle.dump(vec, f)

    if graph is not None:
        save_graphs(output_path_graph, graph)


def multiprocessing(arguments):
    try:
        json_path, part_id, ref_vec_path, parameter_path, output_dir, curv_u_samples, surf_u_samples, surf_v_samples = arguments
        process_one_file(json_path, part_id, ref_vec_path, parameter_path, output_dir, curv_u_samples, surf_u_samples, surf_v_samples)
    except Exception as e:
        tqdm.write(f"Error: {os.path.basename(json_path)} --- {str(e)}")
        # raise e


def main():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="处理命令行传入的参数")

    # 定义命令行参数
    parser.add_argument('-c', '--chunk', type=int, default=0, help="chunk值，默认为0")
    parser.add_argument('--pointercad_dir', type=str, default="/mnt/afs_01e/mayi-folder/PointerCAD/dataset/pointercad-addon/dataset", help="Pointer CAD 数据集路径")
    parser.add_argument('--output_dir', type=str, default="/mnt/afs_01e/mayi-folder/ParamCAD/dataset/paramcad-addon/dataset-addon", help="输出路径")
    parser.add_argument('--curv_u_samples', type=int, default=32, help="curv_u_samples值")
    parser.add_argument('--surf_u_samples', type=int, default=32, help="surf_u_samples值")
    parser.add_argument('--surf_v_samples', type=int, default=32, help="surf_v_samples值")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes to use")
    parser.add_argument('--timeout', type=int, default=600, help="每个任务最大运行时间（秒）")

    # 解析命令行参数
    args = parser.parse_args()

    input_chunk_dir = os.path.join(args.pointercad_dir, f"{args.chunk:04d}")
    output_chunk_dir = os.path.join(args.output_dir, f"{args.chunk:04d}")

    if os.path.exists(output_chunk_dir):
        # 删除所有vector文件夹和graph文件夹（如果存在）
        print("正在删除旧的vector和graph文件夹...")
        for model_id in tqdm(list(os.listdir(output_chunk_dir)), desc="Deleting old data"):
            model_vector_dir = os.path.join(output_chunk_dir, model_id, "vector")
            model_graph_dir = os.path.join(output_chunk_dir, model_id, "graph")

            if os.path.exists(model_vector_dir):
                for f in os.listdir(model_vector_dir):
                    os.remove(os.path.join(model_vector_dir, f))
                os.rmdir(model_vector_dir)

            if os.path.exists(model_graph_dir):
                for f in os.listdir(model_graph_dir):
                    os.remove(os.path.join(model_graph_dir, f))
                os.rmdir(model_graph_dir)
        print("删除完成。")

    pending = []
    for model_id in tqdm(list(os.listdir(input_chunk_dir)), desc="Scanning models"):
        model_json_dir = os.path.join(input_chunk_dir, model_id, "json")
        model_vector_dir = os.path.join(input_chunk_dir, model_id, "vector")
        model_parameter_dir = os.path.join(output_chunk_dir, model_id, "parameter")

        if not os.path.exists(model_vector_dir) or not os.path.exists(model_parameter_dir):
            continue
        
        if os.path.exists(model_json_dir):
            json_set = set(os.path.splitext(f)[0] for f in os.listdir(model_json_dir) if f.endswith(".json"))
            vector_set = set(os.path.splitext(f)[0] for f in os.listdir(model_vector_dir) if f.endswith(".pkl"))
            parameter_set = set(os.path.splitext(f)[0] for f in os.listdir(model_parameter_dir) if f.endswith(".pkl"))

            valid_set = json_set & vector_set & parameter_set
            for base_name in valid_set:
                json_path = os.path.join(model_json_dir, f"{base_name}.json")
                ref_vec_path = os.path.join(model_vector_dir, f"{base_name}.pkl")
                parameter_path = os.path.join(model_parameter_dir, f"{base_name}.pkl")
                pending.append((json_path, None, ref_vec_path, parameter_path, output_chunk_dir, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples))
        else:
            json_path = os.path.join(input_chunk_dir, model_id, f"{model_id}.json")
            print(json_path)
            if not os.path.exists(json_path):
                continue

            vector_set = set(os.path.splitext(f)[0] for f in os.listdir(model_vector_dir) if f.endswith(".pkl"))
            parameter_set = set(os.path.splitext(f)[0] for f in os.listdir(model_parameter_dir) if f.endswith(".pkl"))

            print(vector_set, parameter_set)
            valid_set = vector_set & parameter_set
            for base_name in valid_set:
                ref_vec_path = os.path.join(model_vector_dir, f"{base_name}.pkl")
                parameter_path = os.path.join(model_parameter_dir, f"{base_name}.pkl")
                part_id = base_name.split("_")[-1]
                pending.append((json_path, part_id, ref_vec_path, parameter_path, output_chunk_dir, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples))

    max_workers = args.num_processes
    processes: list[tuple[Process, float]] = []

    try:
        with tqdm(total=len(pending)) as pbar:
            while pending or processes:
                # 启动新进程，最多同时 max_workers
                while pending and len(processes) < max_workers:
                    task_args = pending.pop(0)
                    p = Process(target=multiprocessing, args=(task_args,))
                    p.start()
                    processes.append((p, time.time()))
                
                # 检查进程状态
                still_running = []
                for proc, start_time in processes:
                    proc.join(timeout=0)
                    if proc.is_alive():
                        if time.time() - start_time > args.timeout:
                            tqdm.write(f"任务超时，正在终止进程PID {proc.pid}")
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
        print("检测到Ctrl+C，正在终止所有进程...")
        for proc, _ in processes:
            proc.terminate()
        for proc, _ in processes:
            proc.join()

if __name__ == "__main__":
    main()
    # process_one_file(
    #     "/mnt/afs_01e/mayi-folder/PointerCAD/dataset/pointercad-addon/dataset/0000/00000715/00000715.json", 2, 
    #     "/mnt/afs_01e/mayi-folder/PointerCAD/dataset/pointercad-addon/dataset/0000/00000715/vector/00000715_00002.pkl", 
    #     "/mnt/afs_01e/mayi-folder/ParamCAD/dataset/paramcad-addon/dataset-addon/0000/00000715/parameter/00000715_00002.pkl", "./", 10, 10, 10
    # )