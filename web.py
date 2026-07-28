import os
import re
import dgl
import yaml
import json
import torch
import getpass
import tempfile
import datetime
import argparse
import numpy as np
import gradio as gr
from loguru import logger

from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from cadmodel.model import CADModel
from models.pointercad import PointerCAD
from models.processor import Text2CADProcessor
from misc import create_mesh, extract_parameters_from_plan

last_vector, last_plan = None, None



def load_model(config, device):
    # -------------------------------- Load Model -------------------------------- #
    logger.info("Loading model...", config["web"]["checkpoint_path"])
    pointercad = PointerCAD(qwen_model=config["model"]["base_model"]).to(device)
    checkpoint = torch.load(config["web"]["checkpoint_path"], map_location=device)
    missing_keys_info = pointercad.load_state_dict(checkpoint["model"], strict=False)
    if len(missing_keys_info.missing_keys) > 0:
        logger.warning(f"Missing keys in the checkpoint: {[key.split('.')[0] for key in missing_keys_info.missing_keys]}")

    return pointercad


def get_brep(model: CADModel, surf_u_samples=32, surf_v_samples=32, curv_u_samples=32):
    if model is not None and len(model.seq) > 0:
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
        return graph

    graph = dgl.graph(([], []))
    graph.ndata['x'] = torch.zeros((0,), dtype=torch.float32)
    graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)
    return graph


def format_prompt(user_prompt, system_message="You are a CAD designer"):
    """
    Formats the input prompt to match the model's expected structure.

    Args:
        user_prompt (str): The user's message.
        system_message (str, optional): The system message to guide the assistant.

    Returns:
        str: Formatted prompt.
    """
    message = [
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
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    return text


def replace_plan_parameters(plan: str, tag: str, value: str) -> str:
    pattern = fr"<{re.escape(tag)}(?:=[^>]*)>"
    return re.sub(pattern, value, plan)


def format_plan(plan):
    parameter_source = extract_parameters_from_plan(plan)
    parameter_map = {
        "length": {idx: f"{parameter['value']}{parameter['unit']}" for idx, parameter in parameter_source["length"].items()},
        "angle": {idx: f"{parameter['value']}{parameter['unit']}" for idx, parameter in parameter_source["angle"].items()},
    }

    for length_idx, length_value in parameter_map["length"].items():
        plan = replace_plan_parameters(plan, f"L{length_idx}", length_value)

    for angle_idx, angle_value in parameter_map["angle"].items():
        plan = replace_plan_parameters(plan, f"A{angle_idx}", angle_value)

    return plan


def test_model(model: PointerCAD, prompt, device, use_sampling, plan_temp=1.0, label_temp=1.0, parameter_temp=1.0, pointer_temp=1.0):
    mesh = None
    pred_model = CADModel()
    plans = []

    plan, pred_vector = None, None
    
    with torch.no_grad():
        logger.info("Starting generation task...")

        for i in range(10):
            brep = get_brep(pred_model)
            text = format_prompt(prompt, SYSTEM_MESSAGE)
            with open(os.path.join(OUTPUT_DIR, "input_prompt.txt"), "w") as f:
                f.write(text[0])
            inputs = processor(text=text, breps=dgl.batch([brep]), max_length=3072).to(device)

            if not use_sampling:
                generated_ids, generated_parameter_map, generated_label, generated_parameter, generated_pointer = model.predict(mode="argmax", tokenizer=processor.tokenizer, **inputs)
            else:
                generated_ids, generated_parameter_map, generated_label, generated_parameter, generated_pointer = model.predict(
                    mode="sample",
                    tokenizer=processor.tokenizer,
                    temperature_lm=plan_temp,
                    temperature_label=label_temp,
                    temperature_parameter=parameter_temp,
                    temperature_pointer=pointer_temp,
                    **inputs
                )
            pred_ids, parameter_map, pred_label, pred_parameter, pred_pointer = generated_ids[0], generated_parameter_map[0], generated_label[0], generated_parameter[0], generated_pointer[0]

            plan = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            # plans.append(format_plan(plan))
            plans.append(plan)

            pred_vector = list(zip(pred_label.tolist(), torch.max(pred_parameter, pred_pointer).tolist()))

            if pred_model.from_vector(pred_vector, {k: v.tolist() if torch.is_tensor(v) else v for k, v in parameter_map.items()}):
                logger.info(f"Generation task completed. The model contains {i+1} part(s) in total.")
                mesh = create_mesh(pred_model)
                break
            logger.info(f"Generation of part {i+1} completed, continuing to generate the next part")
        else:
            logger.warning("Exceeded the maximum allowed number of parts. All generation attempts failed.")

    return mesh, pred_model, plans, plan, pred_vector


def convert_negative_zero(obj):
    if isinstance(obj, dict):
        return {k: convert_negative_zero(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_negative_zero(item) for item in obj]
    elif isinstance(obj, float) and obj == 0.0:
        return 0.0  # 去除负号
    else:
        return obj


def remove_caption_and_visualization(obj):
    """
    Recursively remove any 'caption' or 'visualization' keys 
    from nested dicts/lists.
    """
    if isinstance(obj, dict):
        keys_to_remove = [k for k in obj if k in ("caption", "visualization")]
        for k in keys_to_remove:
            del obj[k]
        return {k: remove_caption_and_visualization(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [remove_caption_and_visualization(item) for item in obj]
    
    else:
        return obj


def compact_numeric_arrays(json_str: str) -> str:
    # 匹配一维数组，仅包含数字，允许多行和空白字符
    pattern = r'\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]'

    def compact_array(match):
        content = match.group()
        # 提取所有数字，保留顺序
        numbers = re.findall(r'-?\d+(?:\.\d+)?', content)
        return '[' + ', '.join(numbers) + ']'

    return re.sub(pattern, compact_array, json_str)


def normalize_to_unit_cube(mesh):
    """
    返回一个新的 mesh，将输入 mesh 归一化到 [-1, 1]^3 立方体中。
    原 mesh 不会被修改。
    """
    # 复制以避免修改原对象
    mesh_copy = mesh.copy()

    # 计算包围盒
    bounds = mesh_copy.bounds  # (2, 3)
    min_bound, max_bound = bounds
    center = (min_bound + max_bound) / 2.0
    size = (max_bound - min_bound).max()

    # 平移并缩放
    mesh_copy.apply_translation(-center)
    mesh_copy.apply_scale(2.0 / size)

    return mesh_copy


def generate_cad_model_from_text(text, use_sampling=False, plan_temp=1.0, label_temp=1.0, parameter_temp=1.0, pointer_temp=1.0):
    global last_vector, last_plan
    if text is None or text.strip() == "":
        raise ValueError("Text input cannot be empty. Please provide a valid text prompt.")

    mesh, cadmodel, plans, plan, vector = test_model(model=model, prompt=text, device=device, use_sampling=use_sampling, plan_temp=plan_temp, label_temp=label_temp, parameter_temp=parameter_temp, pointer_temp=pointer_temp)
    if mesh is not None:
        last_plan = plan
        last_vector = vector

        output_step_path = os.path.join(OUTPUT_DIR, "model.step")
        output_stl_path = os.path.join(OUTPUT_DIR, "model.stl")
        output_norm_path = os.path.join(OUTPUT_DIR, "model_norm.stl")

        norm_mesh = normalize_to_unit_cube(mesh)

        cadmodel.export_model(output_step_path)
        mesh.export(output_stl_path)
        norm_mesh.export(output_norm_path)

        json_dict = cadmodel._json()
        json_dict = convert_negative_zero(json_dict)
        json_dict = remove_caption_and_visualization(json_dict)
        json_str = json.dumps(json_dict, indent=2, ensure_ascii=False)
        json_str = compact_numeric_arrays(json_str)

        return gr.File(output_step_path), gr.File(output_stl_path), output_norm_path, json_str, "\n====================\n".join([f"Step {idx}:\n\n{t}" for idx, t in enumerate(plans)]),
    else:
        raise Exception("Error generating CAD model from text")


def modify_cad_model_from_text(text: str):
    text = text.replace(" ", "")
    tag = text.split('=')[0].strip()

    new_plan = re.sub(rf"<{tag}.*?>", f"<{text}>", last_plan)

    parameter_source = extract_parameters_from_plan(new_plan)
    parameter_tensor = {
        "length": [parameter_source["length"][idx]["value_m"] if idx in parameter_source["length"] else 0.0 for idx in range(1, max(parameter_source["length"].keys()) + 1)] if len(parameter_source["length"]) > 0 else [],
        "angle": [parameter_source["angle"][idx]["value"] if idx in parameter_source["angle"] else 0.0 for idx in range(1, max(parameter_source["angle"].keys()) + 1)] if len(parameter_source["angle"]) > 0 else []
    }

    print(last_vector, parameter_tensor)

    pred_model = CADModel()
    if pred_model.from_vector(last_vector, parameter_tensor):
        logger.info(f"Modification task completed")
        mesh = create_mesh(pred_model)
    else:
        mesh = None

    if mesh is not None:
        output_step_path = os.path.join(OUTPUT_DIR, "model.step")
        output_stl_path = os.path.join(OUTPUT_DIR, "model.stl")
        output_norm_path = os.path.join(OUTPUT_DIR, "model_norm.stl")

        norm_mesh = normalize_to_unit_cube(mesh)

        pred_model.export_model(output_step_path)
        mesh.export(output_stl_path)
        norm_mesh.export(output_norm_path)

        json_dict = pred_model._json()
        json_dict = convert_negative_zero(json_dict)
        json_dict = remove_caption_and_visualization(json_dict)
        json_str = json.dumps(json_dict, indent=2, ensure_ascii=False)
        json_str = compact_numeric_arrays(json_str)

        return gr.File(output_step_path), gr.File(output_stl_path), output_norm_path, json_str, new_plan
    else:
        raise Exception("Error modifying CAD model from text")

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data



if not torch.cuda.is_available():
    logger.error("CUDA is not available. Please check your PyTorch installation.")
    exit()

SYSTEM_MESSAGE = "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design."
config_path = "./config/web.yaml"
config = parse_config_file(config_path)
device = torch.device("cuda")
model = load_model(config, device)
processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(pretrained_model_name_or_path=config["model"]["base_model"], padding_side="left")
time_str = datetime.datetime.now().strftime("%H:%M")
date_str = datetime.date.today()
OUTPUT_DIR=os.path.join(config["web"]["log_dir"], f"{date_str}/{time_str}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

examples = [
["Open a new sketch on the XY-plane, draw a 1m × 1m square centered at the origin, and extrude it upward by 0.5m.", False],
["Start at (0m, 0m), draw to (8m, 0m), then to (6m, 3m), then to (2m, 3m), and back to (0m, 0m) counterclockwise. Extrude upward 2m.", False],
['''Starting at (–5m, –5m), draw to (5m, –5m), then to (5m, 5m), then to (–5m, 5m), and back to (–5m, –5m) counterclockwise to form a 10m square. Add a circle with a radius of 2m at the center (0m, 0m), then extrude the shape 2m upward.''', False],
['''Part 1: On the XY-plane, draw a circle with a radius of 10m and extrude it 10m along the positive normal direction.
Part 2: On the XY-plane, draw a circle with a radius of 5m and extrude it along the positive normal direction as a cut with a depth of 5m.''', False],
['''Part 1: On the XY-plane, draw a circle with a radius of 10m and extrude it 10m along the positive normal direction.
Part 2: On the XY-plane rotated 180° around the X-axis, draw a circle with a radius of 5m and extrude it 5m along its positive normal direction.''', False],
['''Part 1: On the XY-plane, draw a circle with a radius of 10m and extrude it 10m along the positive normal direction.
Part 2: On a sketch plane parallel to the XY-plane and translated 10m upward along the Z-axis, draw a circle with a radius of 5m and extrude it 5m.''', False],
['''Draw two concentric circles:
In your modeling workspace, create two circles sharing the same center point. Set the outer circle's radius to 10m and the inner circle's radius to half that of the outer circle. This will define the cross-sectional shape of the ring.

Select the area between the circles:
Highlight the ring-shaped region formed between the outer and inner circles. This will be the base profile for extrusion.

Extrude the profile:
Use the extrusion tool to extend the selected profile upward by 2m.

Complete the shape:
Confirm the operation to finalize the 3D ring with an outer radius of 10m, an inner radius of 5m, and a height of 2m.''', False],
]


title = "Pointer CAD v2"
description = """Plan-Then-Construct CAD Generation with Dimension-Aware Parametric Precision"""

base_tmp = tempfile.gettempdir()
cache_path = os.path.join(base_tmp, f"gradio_{getpass.getuser()}")
os.makedirs(cache_path, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = cache_path

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter a text prompt here"
            )
            use_sampling = gr.Checkbox(
                label="Use Sampling",
                value=False
            )

            # temperature sliders (hidden by default)
            plan_temp = gr.Number(
                label="Plan Temperature",
                value=1.0,
                visible=False
            )
            label_temp = gr.Number(
                label="Label Temperature",
                value=1.0,
                visible=False
            )
            parameter_temp = gr.Number(
                label="Parameter Temperature",
                value=1.0,
                visible=False
            )
            pointer_temp = gr.Number(
                label="Pointer Temperature",
                value=1.0,
                visible=False
            )
            # Generate button
            run_btn = gr.Button("Generate Model")

            modify_input = gr.Textbox(
                label="Modify parameter",
                placeholder="Enter the parameter you want to modify, e.g. L0=20mm or A0=90deg"
            )
            # Modify button
            modify_btn = gr.Button("Modify Model")

        # Right column
        with gr.Column(scale=2):
            output_3d = gr.Model3D(
                clear_color=[0.678, 0.847, 0.902, 1.0],
                label="3D CAD Model"
            )
            with gr.Row():
                download_step_btn = gr.DownloadButton(
                    label="Download Source Model (STEP)",
                    value=None
                )
                download_stl_btn = gr.DownloadButton(
                    label="Download Source Model (STL)",
                    value=None
                )
            with gr.Accordion("Model JSON Data", open=False):
                output_json = gr.Textbox(
                    label="JSON",
                    lines=20,
                    interactive=False,
                    show_copy_button=True
                )
            with gr.Accordion("Detailed Plans", open=False):
                output_plan = gr.Textbox(
                    label="Plans",
                    lines=20,
                    interactive=False,
                    show_copy_button=True
                )

    # Examples section (full width)
    gr.Examples(examples, [text_input, use_sampling])

    def toggle_sampling(show):
        return (
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
        )

    use_sampling.change(
        toggle_sampling,
        inputs=[use_sampling],
        outputs=[plan_temp, label_temp, parameter_temp, pointer_temp]
    )

    run_btn.click(
        fn=generate_cad_model_from_text,
        inputs=[text_input, use_sampling, plan_temp, label_temp, parameter_temp, pointer_temp],
        outputs=[download_step_btn, download_stl_btn, output_3d, output_json, output_plan]
    )

    modify_btn.click(
        fn=modify_cad_model_from_text,
        inputs=[modify_input],
        outputs=[download_step_btn, download_stl_btn, output_3d, output_json, output_plan]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=7860, help="Port to launch the Gradio app")
    args, unknown = parser.parse_known_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False, inbrowser=False)
