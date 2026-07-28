import os
import gc
import dgl
import yaml
import torch
import requests
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from loguru import logger

from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from measurements.intersection_over_union import intersection_over_union
from measurements.chamfer_distance import chamfer_distance
from measurements.f1_score import f1_score
from measurements.watertightness import is_watertight
from misc import MAX_PART_LENGTH, create_mesh
from cadmodel.model import CADModel
from models.pointercad import PointerCAD
from dataset.dataset import get_dataloaders
from models.processor import Text2CADProcessor

# ---------------------------------------------------------------------------- #
#                              Text2CAD Test Code                              #
# ---------------------------------------------------------------------------- #


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def get_brep(model: CADModel, surf_u_samples=32, surf_v_samples=32, curv_u_samples=32):
    success = True

    if model is not None:
        try:
            adjacency = face_adjacency(model.build_model()).to_undirected(as_view=True)

            # Compute the UV-grids for faces
            graph_face_feat = []
            for idx, face_idx in enumerate(adjacency.nodes):
                if idx != face_idx:
                    raise ValueError(f"Face ID {face_idx} is not continuous. Unable to construct the graph.")
                # Get the B-rep face
                face = adjacency.nodes[face_idx]["face"]
                # Compute UV-grids
                points = uvgrid(face, method="point", num_u=surf_u_samples, num_v=surf_v_samples)
                normals = uvgrid(face, method="normal", num_u=surf_u_samples, num_v=surf_v_samples)
                curvatures = uvgrid(face, method="gaussian_curvature", num_u=surf_u_samples, num_v=surf_v_samples)
                visibility_status = uvgrid(face, method="visibility_status", num_u=surf_u_samples, num_v=surf_v_samples)
                mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
                # Concatenate channel-wise to form face feature tensor
                face_feat = np.concatenate((points, normals, curvatures, mask), axis=-1)
                graph_face_feat.append(face_feat)
            graph_face_feat = np.asarray(graph_face_feat)

            # Compute the U-grids for edges
            graph_edge_feat = []
            for edge_idx in adjacency.edges:
                # Get the B-rep edge
                edge = adjacency.edges[edge_idx]["edge"]
                # Ignore dgenerate edges, e.g. at apex of cone
                if not edge.has_curve():
                    raise RuntimeError(f"Unable to construct edge feature for edge ID {edge_idx}. The edge does not have a valid curve.")
                # Compute U-grids
                points = ugrid(edge, method="point", num_u=curv_u_samples)
                tangents = ugrid(edge, method="tangent", num_u=curv_u_samples)
                derivatives = ugrid(edge, method="first_derivative", num_u=curv_u_samples)
                # Concatenate channel-wise to form edge feature tensor
                edge_feat = np.concatenate((points, tangents, -tangents, derivatives), axis=-1)
                graph_edge_feat.append(edge_feat)
            graph_edge_feat = np.asarray(graph_edge_feat)

            # Convert face-adj graph to DGL format
            edges = list(adjacency.edges)
            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            graph = dgl.graph((src, dst), num_nodes=len(adjacency.nodes))
            graph.ndata["x"] = torch.from_numpy(graph_face_feat).to(torch.float32)
            graph.edata["x"] = torch.from_numpy(graph_edge_feat).to(torch.float32)
            graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
            if '_ID' in graph.ndata: del graph.ndata['_ID']
            if '_ID' in graph.edata: del graph.edata['_ID']
            return graph, success
        except Exception as e:
            logger.warning(f"An error occurred while constructing the B-rep graph: {e}")
            success = False

    graph = dgl.graph(([], []))
    graph.ndata['x'] = torch.zeros((0,), dtype=torch.float32)
    graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)
    return graph, success


@logger.catch()
def main():
    # Use add_help=False to free up '-h' for host; provide '--help' for help output
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config_path", type=str, default="./config/test.yaml")
    parser.add_argument("--help", action="help", help="Show this help message and exit.")
    args = parser.parse_args()

    config = parse_config_file(args.config_path)
    device = torch.device("cuda")
    logger.info(f"Current Device {torch.cuda.get_device_properties(device)}")

    # -------------------------------- Load Model -------------------------------- #
    text2cad = PointerCAD(qwen_model=config["model"]["base_model"]).to(device)

    text2cad.model.print_trainable_parameters()

    # -------------------------------- Load Processor -------------------------------- #
    processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(
        pretrained_model_name_or_path=config["model"]["base_model"],
        padding_side="left"
    )

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()

    logger.info(f"Current Date {date_str} Time {time_str}\n")

    # -------------------------------- Test Model ------------------------------- #
    test_model(
        model=text2cad,
        processor=processor,
        device=device,
        config=config,
    )


def test_model(
    model,
    processor,
    device,
    config,
):
    """
    Trains a deep learning model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        device (str): Device to train on ('cuda' for GPU, 'cpu' for CPU).
        config (dict): Additional configuration parameters.
        host (str): Server host name or IP (without port).
    """
    # Create the dataloader for test
    test_loader = get_dataloaders(
        dataset_dir=config["dataset"]["dataset_dir"],
        split_filepath=config["dataset"]["split_filepath"],
        subsets=["test"],
        batch_sizes=1,
        num_workers=config["test"]["num_workers"],
        pin_memory=False,
        shuffle=True,
        prefetch_factor=config["test"]["prefetch_factor"],
        prompt_choices=["exp"],
    )[0]

    checkpoint_file = config["test"]["checkpoint_path"]
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        logger.info(f"Using saved checkpoint at {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)

        if "epoch" in checkpoint:
            logger.info(f"Model was trained for epoch {checkpoint['epoch']}.")

        missing_keys_info = model.load_state_dict(checkpoint["model"], strict=False)
        if len(missing_keys_info.missing_keys) > 0:
            logger.warning(f"Missing keys in the checkpoint: {[key.split('.')[0] for key in missing_keys_info.missing_keys]}")
    else:
        logger.error("No checkpoint specified or checkpoint file does not exist. Unable to load pretrained model.")
        raise RuntimeError("Checkpoint not specified or not found. Cannot load pretrained model.")

    log_dir = os.path.join(config["test"]["log_dir"], f"{datetime.date.today()}/{datetime.datetime.now().strftime('%H:%M')}")
    logger.info(f"Logging directory: {log_dir}")

    # ---------------------------------- Test ---------------------------------- #
    batch_size = config["test"]["batch_size"]
    test_acc_uid = {}
    model.eval()
    topk = 1

    cd = None  # Chamfer Distance

    with torch.no_grad():
        with tqdm(total=len(test_loader), ascii=True, dynamic_ncols=True, desc=f"Test ✨") as pbar:
            outof_model = False
            test_iter = iter(test_loader)

            total_test = success_samples = 0
            batch_model_id = []
            gt_models: dict[str, CADModel] = {}
            pred_models: dict[str, CADModel] = {}
            messages: dict[str, dict] = {}
            results = {}
            while not outof_model or len(batch_model_id) > 0:
                try:
                    while len(batch_model_id) < batch_size:
                        iter_dict = next(test_iter)
                        model_id = iter_dict["model_id"][0]

                        # if model_id != "00783456":
                        #     pbar.update(1)
                        #     continue

                        gt_cadmodel = CADModel.from_dict(iter_dict["json"][0])
                        gt_models[model_id] = gt_cadmodel
                        messages[model_id] = [
                            {
                                "role": "system",
                                "content": [
                                    {"type": "text", "text": "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design."},
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "brep"},
                                    {"type": "text", "text": iter_dict["prompt"][0]},
                                ],
                            },
                        ]

                        batch_model_id.append(model_id)
                        total_test += 1
                except StopIteration:
                    outof_model = True

                if len(batch_model_id) > 0:
                    finished_model = {}
                    batch_breps = []
                    batch_messages = []
                    for key in batch_model_id:
                        brep, success = get_brep(pred_models[key] if key in pred_models else None)
                        if not success:
                            logger.warning(f"Failed to construct the B-rep graph for model_id {key}. Skipping this sample.")
                            finished_model[key] = "failed to construct B-rep"

                        batch_breps.append(brep)
                        batch_messages.append(messages[key])

                    text = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=text, breps=dgl.batch(batch_breps), max_length=3072).to(device)

                    generated_ids, generated_parameter_map, generated_label, generated_parameter, generated_pointer = model.predict(tokenizer=processor.tokenizer, **inputs)

                    gc.collect()
                    torch.cuda.empty_cache()

                    for model_id, parameter_map, pred_label, pred_parameter, pred_pointer in zip(batch_model_id, generated_parameter_map, generated_label, generated_parameter, generated_pointer):
                        if model_id not in pred_models:
                            pred_models[model_id] = CADModel()

                        assert pred_label.shape == pred_parameter.shape == pred_pointer.shape
                        pred_vector = [[pred_label[idx].item(), max(pred_parameter[idx].item(), pred_pointer[idx].item())] for idx in range(pred_label.shape[0])]

                        if len(pred_vector) == 0:
                            logger.warning(f"Predicted vector for model_id {model_id} is empty or exceeds the maximum allowed length. Skipping this sample.")
                            finished_model[model_id] = "out of length"
                            continue

                        try:
                            if pred_models[model_id].from_vector(pred_vector, {k: v.tolist() if torch.is_tensor(v) else v for k, v in parameter_map.items()}):
                                finished_model[model_id] = None
                            elif len(pred_models[model_id].seq) >= MAX_PART_LENGTH:
                                logger.warning(f"Predicted sequence for model_id {model_id} exceeds the maximum allowed part length ({MAX_PART_LENGTH}). Skipping this sample.")
                                finished_model[model_id] = "exceeded maximum part length"
                        except Exception as e:
                            logger.warning(f"Exception occurred while building CADModel from prediction for model_id {model_id}: {e}")
                            finished_model[model_id] = "failed while building CADModel"

                            print(pred_vector)

                    for model_id, error_reason in finished_model.items():
                        try:
                            if error_reason is not None:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": error_reason,
                                }
                                continue
                            
                            try:
                                gt_model = gt_models[model_id].build_model()
                            except Exception as e:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to build the CADModel (gt)",
                                }
                                continue
                            if gt_model is None:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to build the CADModel (gt)",
                                }
                                continue
                            
                            try:
                                pred_model = pred_models[model_id].build_model() if gt_model is not None else None
                            except Exception as e:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to build the CADModel (pred)",
                                }
                                continue
                            if pred_model is None:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to build the CADModel (pred)",
                                }
                                continue

                            try:
                                cd = chamfer_distance(pred_models[model_id], gt_models[model_id], 8192) * 1000
                            except Exception as e:
                                logger.warning(f"Error occurred while calculating Chamfer Distance for model_id {model_id}: {e}")
                                cd = None

                            try:
                                f1 = f1_score(pred_models[model_id], gt_models[model_id])
                                f1 = {k: v * 100 for k, v in f1.items()}
                            except Exception as e:
                                logger.warning(f"Error occurred while calculating F1 Score for model_id {model_id}: {e}")
                                f1 = None

                            try:
                                watertightness = is_watertight(pred_models[model_id])
                            except Exception as e:
                                logger.warning(f"Error occurred while checking watertightness for model_id {model_id}: {e}")
                                watertightness = None

                            if config["test"]["save_step"]:
                                save_dir = os.path.join(log_dir, "step")
                                os.makedirs(save_dir, exist_ok=True)

                                if not pred_models[model_id].export_model(os.path.join(save_dir, f"{model_id}.step")):
                                    logger.warning(f"Failed to export predicted model to STEP format for model_id {model_id}.")
                                if not gt_models[model_id].export_model(os.path.join(save_dir, f"{model_id}_gt.step")):
                                    logger.warning(f"Failed to export ground truth model to STEP format for model_id {model_id}.")

                            if config["test"]["save_stl"]:
                                save_dir = os.path.join(log_dir, "stl")
                                os.makedirs(save_dir, exist_ok=True)

                                try:
                                    mesh = create_mesh(pred_models[model_id])
                                    mesh.export(os.path.join(save_dir, f"{model_id}.stl"))
                                    mesh = create_mesh(gt_models[model_id])
                                    mesh.export(os.path.join(save_dir, f"{model_id}_gt.stl"))
                                except Exception as e:
                                    logger.warning(f"Error occurred while saving predicted STL for model_id {model_id}: {e}")

                            if cd is not None:
                                results[model_id] = {
                                    "status": True,
                                    "chamfer distance": cd,
                                    "f1": f1,
                                    "is watertight": watertightness,
                                }
                                success_samples += 1
                                tqdm.write(f"Model ID {model_id} - CD: {cd:.4f}, F1: {f1}, Watertight: {watertightness}")
                            else:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to measure result",
                                }
                        except Exception as e:
                            logger.warning(f"Error occurred while measuring result for model_id {model_id}: {e}")
                            results[model_id] = {
                                "status": False,
                                "error_message": "unknown error",
                            }

                    batch_model_id = [item for item in batch_model_id if item not in finished_model]

                    pbar.set_postfix({
                        "CD": f"{cd:.4f}" if cd is not None else "N/A",
                        "IR": f"{(total_test - success_samples - len(batch_model_id)) / total_test * 100:.2f}%" if total_test > 0 else "N/A",
                    })
                    pbar.update(len(finished_model))


if __name__ == "__main__":
    main()
