import dgl
import torch
from tqdm import tqdm
from loguru import logger
from dataset.dataset import Text2CAD_Dataset, get_dataloaders
from metrics.rewards import Text2CADVectorReward, Text2CADPlanReward
from models.pointercad import PointerCAD
from models.processor import Text2CADProcessor



def clear_vector(pred_label, pred_parameter, pred_pointer):
    pred_label_list, pred_parameter_list, pred_pointer_list = [], [], []
    for label, parameter, pointer in zip(pred_label, pred_parameter, pred_pointer):
        end_idx = (label == 0).nonzero(as_tuple=True)[0]
        if len(end_idx) > 0:
            end_idx = end_idx[0]
            pred_label_list.append(label[:end_idx])
            pred_parameter_list.append(parameter[:end_idx])
            pred_pointer_list.append(pointer[:end_idx])
        else:
            pred_label_list.append(label)
            pred_parameter_list.append(parameter)
            pred_pointer_list.append(pointer)
    return pred_label_list, pred_parameter_list, pred_pointer_list


@logger.catch
def test():
    rewards_evaluator = Text2CADVectorReward()
    dataloader = get_dataloaders(
        dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
        split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
        subsets=["train"],
        batch_sizes=1,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=16
    )[0]
    model = PointerCAD("Qwen/Qwen2.5-0.5B-Instruct").cuda()
    processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
    checkpoint = torch.load("/mnt/afs/wangchenyu/cad/CADv3/plan-param-tie/log/2025-10-23/04:08/model.pth", map_location="cuda")
    missing_keys_info  = model.load_state_dict(checkpoint["model"], strict=False)
    if len(missing_keys_info.missing_keys) > 0:
        print(f"Missing keys in the checkpoint: {missing_keys_info.missing_keys}")

    for item in tqdm(dataloader):
        messages, breps, parameter_maps = [], [], []
        for _ in range(8):
            messages.append([
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
                        {"type": "text", "text": item["prompt"][0]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": item["plan"][0]},
                        {"type": "cad"},
                    ],
                },
            ])
            breps.append(dgl.unbatch(item["graph"])[0].clone())
            parameter_maps.extend(item["parameter_map"])
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=text, breps=dgl.batch(breps), parameter_maps=parameter_maps, max_length=3072).to("cuda")
        pred_label, pred_parameter, pred_pointer = model.predict_vector(mode="sample", 
                                                                        temperature_label = 2,
                                                                        temperature_parameter = 3,
                                                                        temperature_pointer = 3, **inputs)

        pred_label, pred_parameter, pred_pointer = clear_vector(pred_label, pred_parameter, pred_pointer)

        vector_reward_dict = rewards_evaluator(
            item["parameter_map"][0],
            pred_label,
            pred_parameter,
            pred_pointer,
            item["label"][0].shape[0],
            item["json"][0]
        )
        print(vector_reward_dict)

test()