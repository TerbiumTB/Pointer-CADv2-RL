import torch
from collections import deque
import torch.nn.functional as F

from misc import TOKEN, STANDARD_PLANES



class SmoothedMetric:
    def __init__(self, window_size=20, ignore_invalid=True):
        self.window_size = window_size
        self.ignore_invalid = ignore_invalid
        self.values = deque(maxlen=window_size)

    def update(self, value):
        if value < 0:
            if self.ignore_invalid:
                return  # 忽略非法值
            value = 0.0  # 将非法值视为 0
        self.values.append(value)

    def average(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def max(self):
        if not self.values:
            return 0.0
        return max(self.values)

    def reset(self):
        self.values.clear()



class LabelAccuracyCalculator:
    def __init__(self, padding_token=TOKEN.index("<|padding|>")):
        self.padding_token = padding_token
    
    def calculateFromProbability2D(self, predProb, targetLabel):
        # Get the predicted classes
        return self.calculateFromLabel2D([pred.cpu().argmax(dim=-1) for pred in predProb], [f.cpu() for f in targetLabel])
    
    def calculateFromLabel2D(self, predLabel, targetLabel):
        """
        predLabel: list of tensor
        target: list of tensor
        """
        accuracy, num = 0, 0
        for pred, target in zip(predLabel, targetLabel):
            if len(target.shape) == 1:
                acc = self.calculateFromLabel(predLabel=pred, targetLabel=target)
            else:
                raise NotImplementedError("Multiple ground truth labels per prediction not supported yet.")
                acc_t, acc_k, acc_v = -1, -1, -1
                for i in range(target.shape[0]):
                    acc_t_, acc_k_, acc_v_ = self.calculateFromLabel(predLabel=pred, targetLabel=target[i])
                    acc_t = max(acc_t, acc_t_)
                    acc_k = max(acc_k, acc_k_)
                    acc_v = max(acc_v, acc_v_)

            if acc >= 0:
                accuracy += acc
                num += 1

        return accuracy / num if num > 0 else -1

    def calculateFromLabel(self, predLabel, targetLabel):
        """
        pred: tensor of shape (N)
        target: tensor of shape (N)
        """
        
        N_pred=predLabel.shape[0]
        N_gt=targetLabel.shape[0]

        if N_pred > N_gt:
            predLabel=predLabel[:N_gt]
        elif N_pred < N_gt:
            inf_tensor = torch.full((N_gt - N_pred,), float('inf'))
            predLabel = torch.cat([predLabel, inf_tensor])

        mask = (targetLabel != self.padding_token).to(targetLabel.device)
        correct = (predLabel == targetLabel) * 1 * mask

        return float(correct.sum() * 100 / mask.sum())



class ParameterAccuracyCalculator:
    def __init__(self):
        pass

    def calculateFromParameter2D(self, predParams, gtParams, gtLabels, param_embeds):
        # Get the predicted classes
        accuracy = 0
        num = 0
        for pred_param, gt_param, gt_label, param_embed in zip(predParams, gtParams, gtLabels, param_embeds):
            acc = self.calculateFromParameter(pred_param.cpu(), gt_param.cpu(), gt_label.cpu(), {k: v.cpu() for k, v in param_embed.items()})

            if acc >= 0:
                accuracy += acc
                num += 1

        return accuracy / num if num > 0 else -1
    

    def calculateFromParameter(self, predParams, gtParams, gtLabels, param_embeds):
        assert gtParams.shape[0] == gtLabels.shape[0]

        N_pred = predParams.shape[0]
        N_gt = gtParams.shape[0]
        N_min = min(N_pred, N_gt)

        pred_params = predParams[:N_min]
        gt_params = gtParams[:N_min]
        gt_labels = gtLabels[:N_min]

        match_num, total_num = 0, 0

        length_mask = gt_labels == TOKEN.index("<|length_value|>")
        if torch.any(length_mask):
            length_pred = pred_params[length_mask]
            length_candidate = param_embeds["length"].type_as(length_pred)

            length_pred_norm = F.normalize(length_pred, p=2, dim=1, eps=1e-6)
            length_candidate_norm = F.normalize(length_candidate, p=2, dim=1, eps=1e-6)
            sim_length = torch.matmul(length_pred_norm, length_candidate_norm.T)

            pred_idx = torch.argmax(sim_length, dim=1)
            gt_idx = gt_params[length_mask] - 1
            assert pred_idx.shape == gt_idx.shape

            match_num += torch.sum(pred_idx == gt_idx)
            total_num += gt_idx.shape[0]

        angle_mask = gt_labels == TOKEN.index("<|angle_value|>")
        if torch.any(angle_mask):
            angle_pred = pred_params[angle_mask]
            angle_candidate = param_embeds["angle"].type_as(angle_pred)

            angle_pred_norm = F.normalize(angle_pred, p=2, dim=1, eps=1e-6)
            angle_candidate_norm = F.normalize(angle_candidate, p=2, dim=1, eps=1e-6)
            sim_angle = torch.matmul(angle_pred_norm, angle_candidate_norm.T)

            pred_idx = torch.argmax(sim_angle, dim=1)
            gt_idx = gt_params[angle_mask] - 1
            assert pred_idx.shape == gt_idx.shape

            match_num += torch.sum(pred_idx == gt_idx)
            total_num += gt_idx.shape[0]

        return match_num.float() * 100 / total_num if total_num > 0 else -1



class PointerAccuracyCalculator:
    def __init__(self):
        pass
    
    def calculateFromLabel2D(self, pred_pointers, gt_pointers):
        # Get the predicted classes
        accuracy = 0
        num = 0
        for pred_pointer, gt_pointer in zip(pred_pointers, gt_pointers):
            acc = self.calculateFromLabel(pred_pointer.cpu(), gt_pointer)

            if acc >= 0:
                accuracy += acc
                num += 1

        return accuracy / num if num > 0 else -1
    
    def calculateFromPointer2D(self, pred_pointers, gt_pointers, gt_labels, pred_pointers_crv, pred_pointers_srf, standard_plane_pointers):
        # Get the predicted classes
        accuracy = 0
        num = 0
        for pred_pointer, gt_pointer, gt_label, pred_pointer_crv, pred_pointer_srf in zip(pred_pointers, gt_pointers, gt_labels, pred_pointers_crv, pred_pointers_srf):
            acc = self.calculateFromPointer(pred_pointer.cpu(), gt_pointer, gt_label.cpu(), pred_pointer_crv.cpu(), pred_pointer_srf.cpu(), standard_plane_pointers.cpu())

            if acc >= 0:
                accuracy += acc
                num += 1

        return accuracy / num if num > 0 else -1
    
    def calculateFromPointer(self, pred_pointers, gt_pointers, gt_labels, pred_pointers_crv, pred_pointers_srf, standard_plane_pointers):
        """
        predLabel: list of tensor
        target: list of tensor
        """
        pred_labels = []
        for i in range(pred_pointers.shape[0]):
            pointer_ids = gt_pointers[i]
            if isinstance(pointer_ids, list):
                if gt_labels[i - 1] == TOKEN.index("<|sketch_start|>"):
                    candidate_pointers = torch.vstack([standard_plane_pointers, pred_pointers_srf])
                    pred = pred_pointers[i].unsqueeze(0).expand(candidate_pointers.size(0), -1)
                    cos_sim = F.cosine_similarity(pred, candidate_pointers, dim=1)
                    index = torch.argmax(cos_sim)
                    pred_labels.append(index.item() - len(STANDARD_PLANES))
                else:
                    pred = pred_pointers[i].unsqueeze(0).expand(pred_pointers_crv.size(0), -1)
                    cos_sim = F.cosine_similarity(pred, pred_pointers_crv, dim=1)
                    index = torch.argmax(cos_sim)
                    pred_labels.append(index.item())

            else:
                pred_labels.append(pointer_ids)
        
        return self.calculateFromLabel(pred_labels, gt_pointers)

    def calculateFromLabel(self, predLabel, targetLabel):
        N_pred = len(predLabel)
        N_gt = len(targetLabel)
        N_min = min(N_pred, N_gt)
        predLabel = predLabel[:N_min]
        targetLabel = targetLabel[:N_min]

        correct, total = 0, 0.0
        for idx in range(N_min):
            pred, target = predLabel[idx], targetLabel[idx]
            if not (isinstance(target, list) or isinstance(target, set)):
                continue
            total += 1

            if len(target) == 0:
                target = [x for x in targetLabel[idx - 1] if x != predLabel[idx - 1]]
                targetLabel[idx] = target

            correct += 1 if pred in target else 0

        # Calculate the accuracy
        if total > 0: 
            return correct * 100 / total
        else:
            return -1



if __name__ == "__main__":
    from loguru import logger
    import dgl
    from torch import FloatTensor
    from dgl.data.utils import load_graphs
    from models.pointercad import PointerCAD
    from models.processor import Text2CADProcessor
    from dataset.dataset import get_dataloaders

    @logger.catch
    def test():

        model = PointerCAD("Qwen/Qwen2.5-0.5B-Instruct").cuda()
        processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(padding_side="left")
        dataloader = get_dataloaders(
            dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
            split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
            subsets=["train"],
            # batch_sizes=1,
            batch_sizes=6,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=16
        )[0]

        for iter_dict in dataloader:
            messages = []
            for prompt, plan in zip(iter_dict["prompt"], iter_dict["plan"]):
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
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": plan},
                            {"type": "cad"},
                        ],
                    },
                ]
                messages.append(message)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            breps = iter_dict["graph"]
            gt_parameter_maps = iter_dict["parameter_map"]
            gt_labels = iter_dict["label"]
            gt_parameters = iter_dict["parameter"]
            gt_pointers = iter_dict["pointer"]
            gt_pointer_sources = iter_dict["pointer_source"]
            inputs = processor(text=text, breps=breps, parameter_maps=gt_parameter_maps, labels=gt_labels, parameters=gt_parameters, pointers=gt_pointers, max_length=3072)
            inputs = inputs.to("cuda")
            gt_logits = inputs["input_ids"]
            gt_labels = inputs["labels"]
            gt_parameters = inputs["parameters"]
            gt_pointers = inputs["pointers"]

            pred_logits, pred_labels, pred_parameters, pred_pointers, param_embeds, param_tau, ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointer = model(**inputs)
            print(ParameterAccuracyCalculator().calculateFromParameter2D(pred_parameters, gt_parameters, gt_labels, param_embeds))

    with torch.no_grad():
        test()