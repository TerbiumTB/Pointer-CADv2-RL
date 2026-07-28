import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import TOKEN, STANDARD_PLANES



class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) Loss
    """

    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.mean(torch.abs((pred - target) / (target + 1e-8)))
        return loss

class Text2CADCriterion(nn.Module):
    """
    Cross Entropy Loss for Text2CAD
    """

    def __init__(self, weight_logits, weight_label, weight_parameter, weight_pointer):
        super(Text2CADCriterion, self).__init__()

        self.logits_ce = nn.CrossEntropyLoss(ignore_index=151643)
        self.label_ce = nn.CrossEntropyLoss()
        self.parameter_ce = nn.CrossEntropyLoss()
        self.pointer_bce = nn.BCEWithLogitsLoss()
        self.scale_mape = MAPELoss()

        self.weight_logits = weight_logits
        self.weight_label = weight_label
        self.weight_parameter = weight_parameter
        self.weight_pointer = weight_pointer

    def forward(self, pred_logits, gt_logits, pred_labels, gt_labels, pred_parameters, gt_parameters, param_embeds, param_tau, pred_pointers, gt_pointers, gt_pointer_sources, pred_pointer_crv, pred_pointer_srf, pointer_tau, standard_plane_pointers):
        ##################### Logits Loss #####################
        loss_logits = 0
        logits_mask = (gt_logits != 151667) & (gt_logits != 151670) & (gt_logits != 151673)
        for pred_l, gt_l, mask in zip(pred_logits, gt_logits, logits_mask):
            gt_masked = gt_l[mask][1:pred_l.shape[0]+1]
            loss_logits = loss_logits + self.logits_ce(pred_l, gt_masked)
        loss_logits = loss_logits / len(gt_logits)

        ##################### Label Loss #####################
        loss_label = 0
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            ce_loss = self.label_ce(pred_label, gt_label[:pred_label.shape[0]])
            if torch.isnan(ce_loss): continue
            loss_label = loss_label + ce_loss
        loss_label = loss_label / len(gt_labels)

        ##################### Parameter Loss #####################
        loss_parameter = 0
        for pred_param, gt_param, gt_label, embed in zip(pred_parameters, gt_parameters, gt_labels, param_embeds):
            if pred_param.shape[0] == 0:
                continue
            
            gt_param = gt_param[:pred_param.shape[0]]
            gt_label = gt_label[:pred_param.shape[0]]

            loss_length = 0
            length_mask = gt_label == TOKEN.index("<|length_value|>")
            if torch.any(length_mask):
                length_pred = pred_param[length_mask]
                length_candidate = embed["length"].type_as(length_pred)

                length_pred_norm = F.normalize(length_pred, p=2, dim=1, eps=1e-6)
                length_candidate_norm = F.normalize(length_candidate, p=2, dim=1, eps=1e-6)

                sim_length = torch.matmul(length_pred_norm, length_candidate_norm.T) * param_tau
                gt_length = gt_param[length_mask] - 1
                loss_length = self.parameter_ce(sim_length, gt_length)

            loss_angle = 0
            angle_mask = gt_label == TOKEN.index("<|angle_value|>")
            if torch.any(angle_mask):
                angle_pred = pred_param[angle_mask]
                angle_candidate = embed["angle"].type_as(angle_pred)

                angle_pred_norm = F.normalize(angle_pred, p=2, dim=1, eps=1e-6)
                angle_candidate_norm = F.normalize(angle_candidate, p=2, dim=1, eps=1e-6)

                sim_angle = torch.matmul(angle_pred_norm, angle_candidate_norm.T) * param_tau
                gt_angle = gt_param[angle_mask] - 1
                loss_angle = self.parameter_ce(sim_angle, gt_angle)

            if length_mask.sum() + angle_mask.sum() > 0:
                loss_parameter = loss_parameter + (loss_length * length_mask.sum() + loss_angle * angle_mask.sum()) / (length_mask.sum() + angle_mask.sum())
        loss_parameter = loss_parameter / len(gt_parameters)

        ##################### Pointer Loss #####################
        loss_pointer = 0
        gt_pointer_sources = gt_pointer_sources.copy()
        for pred_pointer, gt_pointer, gt_pointer_source, gt_label, crv_pointer, srf_pointer in zip(pred_pointers, gt_pointers, gt_pointer_sources, gt_labels, pred_pointer_crv, pred_pointer_srf):
            if pred_pointer.shape[0] == 0:
                continue

            pointer_mask = gt_label[:pred_pointer.shape[0]] == TOKEN.index("<|pointer_enable|>")
            face_mask = torch.zeros_like(pointer_mask, dtype=torch.bool)
            face_mask[1:] = gt_label[:pred_pointer.shape[0] - 1] == TOKEN.index("<|sketch_start|>")
            face_mask &= pointer_mask
            curve_mask = (~face_mask) & pointer_mask

            face_idx = torch.nonzero(face_mask, as_tuple=False).squeeze(1)
            curve_idx = torch.nonzero(curve_mask, as_tuple=False).squeeze(1)

            face_num, curve_num = face_idx.numel(), curve_idx.numel()

            loss_pointer_srf = 0
            loss_pointer_crv = 0

            if face_num > 0:
                face_pred = pred_pointer[face_idx]
                face_target = torch.vstack([standard_plane_pointers, srf_pointer]).type_as(face_pred)

                face_pred_norm = F.normalize(face_pred, p=2, dim=1, eps=1e-6)
                face_target_norm = F.normalize(face_target, p=2, dim=1, eps=1e-6)

                sim_face = torch.matmul(face_pred_norm, face_target_norm.T) * pointer_tau
                gt_face = torch.zeros_like(sim_face, dtype=sim_face.dtype)

                for i, idx in enumerate(face_idx):
                    # surface index offset is len(STANDARD_PLANES)
                    gt_face[i, [x + len(STANDARD_PLANES) for x in gt_pointer_source[idx]]] = 1

                loss_pointer_srf = self.pointer_bce(sim_face, gt_face)
            
            if curve_num > 0:
                curve_pred = pred_pointer[curve_idx]
                curve_target = crv_pointer.type_as(curve_pred)

                curve_pred_norm = F.normalize(curve_pred, p=2, dim=1, eps=1e-6)
                curve_target_norm = F.normalize(curve_target, p=2, dim=1, eps=1e-6)

                sim_curve = torch.matmul(curve_pred_norm, curve_target_norm.T) * pointer_tau
                gt_curve = torch.zeros_like(sim_curve, dtype=sim_curve.dtype)

                for i, idx in enumerate(curve_idx):
                    if len(gt_pointer_source[idx]) == 0:
                        gt_pointer_source[idx] = [x for x in gt_pointer_source[curve_idx[i - 1]] if x != gt_pointer[curve_idx[i - 1]]]
                    gt_curve[i, gt_pointer_source[idx]] = 1  # no offset for crv_pointer

                loss_pointer_crv = self.pointer_bce(sim_curve, gt_curve)

            if face_num + curve_num > 0:
                loss_pointer = loss_pointer + (loss_pointer_srf * face_num + loss_pointer_crv * curve_num) / (face_num + curve_num)
        loss_pointer = loss_pointer / len(gt_pointer_sources)

        loss = loss_logits * self.weight_logits + loss_label * self.weight_label + loss_parameter * self.weight_parameter + loss_pointer * self.weight_pointer
        if not torch.is_tensor(loss):
            loss = pointer_tau.sum() * 0.0

        return loss, {
            "logit": loss_logits.item() if isinstance(loss_logits, torch.Tensor) else loss_logits,
            "label": loss_label.item() if isinstance(loss_label, torch.Tensor) else loss_label, 
            "parameter": loss_parameter.item() if isinstance(loss_parameter, torch.Tensor) else loss_parameter,
            "pointer": loss_pointer.item() if isinstance(loss_pointer, torch.Tensor) else loss_pointer
        }



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

        checkpoint = torch.load("/mnt/afs/wangchenyu/cad/CADv3/plan-param/log/2025-10-21/08:04/model.pth", map_location="cuda")
        missing_keys_info  = model.load_state_dict(checkpoint["model"], strict=False)
        if len(missing_keys_info.missing_keys) > 0:
            print(f"Missing keys in the checkpoint: {missing_keys_info.missing_keys}")

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
            loss, loss_dict = Text2CADCriterion(1, 1, 1, 1)(pred_logits, gt_logits, pred_labels, gt_labels, pred_parameters, gt_parameters, param_embeds, param_tau, pred_pointers, gt_pointers, gt_pointer_sources, ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointer)
            print(loss, loss_dict)

    with torch.no_grad():
        test()