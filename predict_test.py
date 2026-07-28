import dgl
import torch
import time
from tqdm import tqdm
from loguru import logger
from torch import FloatTensor
from dgl.data.utils import load_graphs
from models.pointercad import PointerCAD as PointerCAD
from dataset.dataset import Text2CAD_Dataset, get_dataloaders
from models.processor import Text2CADProcessor
from misc import TOKEN


@logger.catch
def test():
    model = PointerCAD("Qwen/Qwen2.5-0.5B-Instruct").cuda()
    processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(padding_side="left")

    dataset = Text2CAD_Dataset(
        dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
        split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
        subset="test"
    )
    dataset = get_dataloaders(
        dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
        split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
        subsets=["train"],
        batch_sizes=[2],
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2,
        prompt_choices=["exp"]
    )[0]

    checkpoint = torch.load("/mnt/afs/wangchenyu/cad/CADv3/plan-param-tie/log/2025-10-23/04:08/model.pth", map_location="cuda")
    model.load_state_dict(checkpoint["model"], strict=False)
    
    for item in tqdm(dataset):
        messages = []
        for prompt in item["prompt"]:
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
                        {"type": "text", "text": prompt},
                    ],
                },
            ])
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, breps=item["graph"], max_length=3072)
        inputs = inputs.to("cuda")

        start_time = time.time()
        generated_ids, generated_parameter_map, generated_label, generated_parameter, generated_pointer = model.predict(mode="sample", tokenizer=processor.tokenizer, temperature_lm = 1.0, temperature_label = 0.1, temperature_parameter = 0.05, temperature_pointer = 0.05, **inputs)
        end_time = time.time()
        logger.info(f"Time taken for single generation : {end_time - start_time} seconds")

        for i in range(len(generated_ids)):
            label = generated_label[i].cpu().tolist()
            parameter = generated_parameter[i].cpu().tolist()
            parameter_map = generated_parameter_map[i]

            for l, p in zip(label, parameter):
                if l == TOKEN.index("<|length_value|>"):
                    assert 1 <= p <= parameter_map["length"].size(0)
                if l == TOKEN.index("<|angle_value|>"):
                    assert 1 <= p <= parameter_map["angle"].size(0)

        
if __name__ == "__main__":
    test()