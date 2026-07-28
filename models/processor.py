import dgl
import torch
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature



class Text2CADBatchFeature(BatchFeature):
    def to(self, *args, **kwargs) -> "Text2CADBatchFeature":
        for k, v in self.items():
            if isinstance(v, dgl.DGLGraph):
                self.data[k] = v.to(*args, **kwargs)
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                self.data[k] = [t.to(*args, **kwargs) for t in v]
            if isinstance(v, list) and isinstance(v[0], dict) and set(v[0].keys()) == {"length", "angle"}:
                self.data[k] = [{key: value.to(*args, **kwargs) for key, value in d.items()} for d in v]
        return super().to(*args, **kwargs)

class Text2CADTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, **kwargs):
        special_tokens_dict = {
            "additional_special_tokens": [
                "<|brep_edge_start|>",
                "<|brep_edge_end|>",
                "<|brep_edge_pad|>",
                "<|brep_face_start|>",
                "<|brep_face_end|>",
                "<|brep_face_pad|>",
                "<|cad_start|>",
                "<|cad_end|>",
                "<|cad_pad|>",
            ]
        }

        # 合并来自 kwargs 的 additional_special_tokens
        existing_specials = kwargs.pop("additional_special_tokens", [])
        special_tokens_dict["additional_special_tokens"].extend(
            t for t in existing_specials if t not in special_tokens_dict["additional_special_tokens"]
        )
        kwargs["additional_special_tokens"] = special_tokens_dict["additional_special_tokens"]

        super().__init__(*args, **kwargs)

        with open('./config/chat_template.jinja', 'r', encoding='utf-8') as f:
            self.chat_template = f.read()

        self.brep_face_token = "<|brep_face_pad|>"
        self.brep_edge_token = "<|brep_edge_pad|>"
        self.cad_token = "<|cad_pad|>"

        # 添加特殊 token
        self.add_special_tokens(special_tokens_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", **kwargs):
        # 调用父类的 from_pretrained 方法，然后转成子类实例
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 用 __class__ 重新绑定为子类
        tokenizer.__class__ = cls
        return tokenizer



class Text2CADProcessor(ProcessorMixin):

    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]

    tokenizer_class = ("Qwen2Tokenizer")

    def __init__(self, tokenizer:Text2CADTokenizer=None, **kwargs):
        super().__init__(tokenizer, chat_template=tokenizer.chat_template)
        self.brep_face_token = tokenizer.brep_face_token
        self.brep_edge_token = tokenizer.brep_edge_token
        self.cad_token = tokenizer.cad_token


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", **kwargs):
        processor = cls(Text2CADTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return processor


    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


    def __call__(self, text=None, breps=None, parameter_maps=None, labels=None, parameters=None, pointers=None, padding=True, max_length=None, return_tensors="pt"):
        extra_inputs = {}

        if not isinstance(text, list):
            text = [text]

        if breps is not None:
            extra_inputs["breps"] = breps
            sub_breps = dgl.unbatch(breps)
            assert len(text) == len(sub_breps)

            for i in range(len(text)):
                ct: str = text[i]
                cg: dgl.DGLGraph = sub_breps[i]
                if cg is not None:
                    ct = ct.replace(self.brep_edge_token, self.brep_edge_token * int(cg.num_edges() / 2), 1)
                    text[i] = ct.replace(self.brep_face_token, self.brep_face_token * cg.num_nodes(), 1)
        else:
            for i in range(len(text)):
                ct: str = text[i]
                ct = ct.replace(self.brep_edge_token, "", 1)
                text[i] = ct.replace(self.brep_face_token, "", 1)

        if parameter_maps is not None:
            extra_inputs["parameter_maps"] = parameter_maps

        if labels is not None and parameters is not None and pointers is not None:
            extra_inputs["labels"] = labels
            extra_inputs["parameters"] = parameters
            extra_inputs["pointers"] = pointers
            assert len(text) == len(labels) == len(parameters) == len(pointers)
            for i in range(len(text)):
                assert labels[i].shape == parameters[i].shape == pointers[i].shape 
                text[i] = text[i].replace(self.cad_token, self.cad_token * labels[i].shape[0], 1)
        else:
            for i in range(len(text)):
                text[i] = text[i].replace(self.cad_token, "", 1)


        text_inputs = self.tokenizer(text, padding=padding, return_tensors=return_tensors, max_length=max_length, truncation=True if max_length is not None else None)

        return Text2CADBatchFeature(data={**text_inputs, **extra_inputs})

if __name__ == "__main__":
    tokenizer = Text2CADTokenizer.from_pretrained()
    print("Length of tokenizer vocabulary:", len(tokenizer))
    for name, token in tokenizer.special_tokens_map.items():
        if isinstance(token, list):
            for t in token:
                print(f"{name:20s} {repr(t):15s} -> {tokenizer.convert_tokens_to_ids(t)}")
        else:
            print(f"{name:20s} {repr(token):15s} -> {tokenizer.convert_tokens_to_ids(token)}")


    # from dataset.dataset import get_dataloaders
    # dataloader = get_dataloaders(
    #     dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
    #     split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
    #     subsets=["train"],
    #     batch_sizes=1,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=4,
    #     prefetch_factor=16
    # )[0]

    # for batch in dataloader:
    #     message = [
    #         {
    #             "role": "system",
    #             "content": [
    #                 {"type": "text", "text": "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design."},
    #             ],
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "brep"},
    #                 {"type": "text", "text": batch["prompt"][0]},
    #             ],
    #         },
    #         {
    #             "role": "assistant",
    #             "content": [
    #                 {"type": "text", "text": batch["plan"][0]},
    #                 {"type": "cad"},
    #             ],
    #         },
    #     ]

    #     processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(padding_side="left")
    #     text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
    #     print(text)
    #     print("========================================")
    #     inputs = processor(text=text, breps=batch["graph"], labels=batch["label"], parameters=batch["parameter"], pointers=batch["pointer"])
    #     print(inputs)
    #     exit()