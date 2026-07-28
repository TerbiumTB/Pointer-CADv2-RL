import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from misc import TOKEN, STANDARD_PLANES, MAX_GENERATION_LENGTH_PLAN, MAX_GENERATION_LENGTH_VECTOR, extract_parameters_from_plan
from .brep_embed import UVNetEmbedder
from .parameter_encoder import ParameterEncoder



class PointerCAD(nn.Module):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}

    def __init__(self, qwen_model="Qwen/Qwen2.5-7B-Instruct", crv_channels=12, surf_channels=8, pointer_size=128, max_parameter_tau=100.0, max_pointer_tau=100.0, dtype=torch.bfloat16):
        super().__init__()

        self.pointer_size = pointer_size
        self.dtype = dtype

        self.model: Qwen2Model = Qwen2Model.from_pretrained(qwen_model, torch_dtype=dtype, attn_implementation="flash_attention_2")
        self.brep = UVNetEmbedder(crv_channels, surf_channels, pointer_size, pointer_size, self.model.config.hidden_size)
        self.parameter = ParameterEncoder(pointer_size)
        self.label_embedding = nn.Embedding(len(TOKEN), self.model.config.hidden_size, TOKEN.index("<|padding|>"))
        self.pointer_projection = nn.Linear(self.pointer_size, self.model.config.hidden_size, bias=False)
        self.parameter_projection = nn.Linear(self.pointer_size, self.model.config.hidden_size, bias=False)
        self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False, dtype=dtype)
        self.label_head = nn.Linear(self.model.config.hidden_size, len(TOKEN), bias=False, dtype=dtype)
        self.pointer_head = nn.Linear(self.model.config.hidden_size, self.pointer_size, bias=False, dtype=dtype)
        self.parameter_head = nn.Linear(self.model.config.hidden_size, self.pointer_size, bias=False, dtype=dtype)
        self.cad_position_embedding = nn.Embedding(2, self.model.config.hidden_size, dtype=dtype)

        self.parameter_tau = nn.Parameter(torch.log(torch.tensor(1 / 0.07, dtype=torch.float32)))
        self.parameter_tau_max = math.log(max_parameter_tau)

        self.pointer_tau = nn.Parameter(torch.log(torch.tensor(1 / 0.07, dtype=torch.float32)))
        self.pointer_tau_max = math.log(max_pointer_tau)
        self.standard_plane_pointer = nn.Parameter(torch.randn(len(STANDARD_PLANES), pointer_size))

        with torch.no_grad():
            self.cad_position_embedding.weight.zero_()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, lora_config)

        self.model.get_input_embeddings().weight.requires_grad = True
        self.lm_head.weight = self.model.get_input_embeddings().weight

    @torch.no_grad()
    def clamp_parameters(self):
        with torch.no_grad():
            self.parameter_tau.clamp_(max=self.parameter_tau_max)
            self.pointer_tau.clamp_(max=self.pointer_tau_max)

    def forward(self, input_ids, attention_mask, breps, parameter_maps, labels=None, parameters=None, pointers=None, logits_to_keep=0, **kwargs):
        ##################  Build Brep Embeded  ##################
        text_embeds: torch.Tensor = self.model.get_input_embeddings()(input_ids)

        crv_mask: torch.Tensor = input_ids == 151667  # TODO: 使用Config指定
        srf_mask: torch.Tensor = input_ids == 151670  # TODO: 使用Config指定
        cad_mask: torch.Tensor = input_ids == 151673  # TODO: 使用Config指定
        lm_mask = ~crv_mask & ~srf_mask & ~cad_mask

        if breps.num_edges() > 0:
            pointer_crv, pointer_srf, feat_crv, feat_srf = self.brep(breps)

            feat_crv_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_crv, crv_mask)
            ]
            feat_srf_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_srf, srf_mask)
            ]

            crv_mask_expanded = crv_mask.unsqueeze(-1).expand_as(text_embeds)
            srf_mask_expanded = srf_mask.unsqueeze(-1).expand_as(text_embeds)

            feat_crv = torch.vstack(feat_crv_clipped).type_as(text_embeds)
            feat_srf = torch.vstack(feat_srf_clipped).type_as(text_embeds)

            assert crv_mask.sum() == feat_crv.shape[0]
            assert srf_mask.sum() == feat_srf.shape[0]

            text_embeds = text_embeds.masked_scatter(crv_mask_expanded, feat_crv)
            text_embeds = text_embeds.masked_scatter(srf_mask_expanded, feat_srf)
        else:
            pointer_crv = [torch.zeros((0, self.pointer_size), dtype=torch.float, device=text_embeds.device) for _ in range(input_ids.shape[0])]
            pointer_srf = pointer_crv

        ##################  Build Parameter Embeded  ##################
        if parameter_maps is not None:
            param_embeds = self.parameter(parameter_maps)
        else:
            param_embeds = None

        ##################  Build CAD Embeded  ##################
        if labels is not None and len(labels) > 0:
            assert len(labels) == len(parameters) == len(pointers)  # same batch size 
            cad_feature = []
            for idx, (batch_pointer_crv, batch_pointer_srf, batch_param_embeds, batch_label, batch_parameter, batch_pointer, batch_cad_mask) in enumerate(zip(pointer_crv, pointer_srf, param_embeds, labels, parameters, pointers, cad_mask)):
                total_cad_num = batch_cad_mask.sum()
                if total_cad_num == 0:
                    continue

                ##################  Build Label Embeded  ##################
                label_embeds: torch.Tensor = self.label_embedding(batch_label[:total_cad_num])

                ##################  Build Parameter Embeded  ##################
                length_mask = batch_label[:total_cad_num] == TOKEN.index("<|length_value|>")
                if torch.any(length_mask):
                    length_idx: torch.Tensor = batch_parameter[:total_cad_num][length_mask] - 1
                    assert length_idx.min() >= 0
                    assert length_idx.max() < batch_param_embeds["length"].shape[0]
                    length_embeds: torch.Tensor = self.parameter_projection(batch_param_embeds["length"][length_idx])
                    assert length_embeds.shape[0] == length_mask.sum()
                    label_embeds[length_mask] += length_embeds
                
                angle_mask = batch_label[:total_cad_num] == TOKEN.index("<|angle_value|>")
                if torch.any(angle_mask):
                    angle_idx: torch.Tensor = batch_parameter[:total_cad_num][angle_mask] - 1
                    assert angle_idx.min() >= 0
                    assert angle_idx.max() < batch_param_embeds["angle"].shape[0]
                    angle_embeds: torch.Tensor = self.parameter_projection(batch_param_embeds["angle"][angle_idx])
                    assert angle_embeds.shape[0] == angle_mask.sum()
                    label_embeds[angle_mask] += angle_embeds

                ##################  Build Pointer Embeded  ##################
                valid_srf_crv_mask = batch_pointer[:total_cad_num] >= 0
                srf_mask = torch.zeros((total_cad_num), dtype=torch.bool, device=valid_srf_crv_mask.device)
                srf_mask[1:] = batch_label[:total_cad_num - 1] == TOKEN.index("<|sketch_start|>")
                srf_mask &= valid_srf_crv_mask
                crv_mask: torch.Tensor = ~srf_mask & valid_srf_crv_mask
                plane_mask: torch.Tensor = (batch_pointer[:total_cad_num] >= -len(STANDARD_PLANES)) & ~valid_srf_crv_mask
                valid_mask: torch.Tensor = plane_mask | valid_srf_crv_mask

                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                srf_idx = torch.nonzero(srf_mask, as_tuple=False).squeeze(1)    
                crv_idx = torch.nonzero(crv_mask, as_tuple=False).squeeze(1)
                plane_idx = torch.nonzero(plane_mask, as_tuple=False).squeeze(1)

                pointer_valid = torch.empty((valid_idx.size(0), self.pointer_size), device=batch_pointer.device)

                if srf_idx.numel() > 0:
                    valid_srf_idx = torch.nonzero(srf_mask[valid_idx], as_tuple=False).squeeze(1)
                    pointer_valid[valid_srf_idx] = batch_pointer_srf[batch_pointer[srf_idx]]

                if crv_idx.numel() > 0:
                    valid_crv_idx = torch.nonzero(crv_mask[valid_idx], as_tuple=False).squeeze(1)
                    pointer_valid[valid_crv_idx] = batch_pointer_crv[batch_pointer[crv_idx]]

                if plane_idx.numel() > 0:
                    valid_plane_idx = torch.nonzero(plane_mask[valid_idx], as_tuple=False).squeeze(1)
                    pointer_valid[valid_plane_idx] = self.standard_plane_pointer[batch_pointer[plane_idx] + len(STANDARD_PLANES)]

                if valid_idx.numel():
                    pointer_embeds = self.pointer_projection(pointer_valid)
                    label_embeds_idx = torch.nonzero(valid_mask).squeeze(1)

                    assert label_embeds_idx.shape[0] == pointer_embeds.shape[0]

                    label_embeds = label_embeds.index_add(0, label_embeds_idx, pointer_embeds)

                cad_feature.append(label_embeds)

            if cad_feature:
                cad_mask_expanded = cad_mask.unsqueeze(-1).expand_as(text_embeds)
                cad_feature = torch.vstack(cad_feature).type_as(text_embeds)
                assert cad_mask.sum() == cad_feature.shape[0]
                text_embeds = text_embeds.masked_scatter(cad_mask_expanded, cad_feature)
            else:
                assert not torch.any(cad_mask)

        ##################  Absolute Position Embed  ##################
        last_feature_class = cad_mask[:, -1].clone()
        for i in range(last_feature_class.shape[0]):
            if last_feature_class[i]:
                last_label = labels[i][cad_mask[i].sum() - 1]
                if last_label == TOKEN.index("<|model_end|>") or last_label == TOKEN.index("<|part_end|>"):
                    last_feature_class[i] = False
        last_feature_class |= input_ids[:, -1] == 151671
        shifted_cad_mask = torch.cat([cad_mask[:, 1:], last_feature_class.unsqueeze(1)], dim=1)
        position_embeds = self.cad_position_embedding(shifted_cad_mask.type_as(input_ids))

        ##################  Forward  ##################
        outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=text_embeds + position_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        shifted_lm_mask = torch.cat([lm_mask[:, 1:], ~(lm_mask[:, :1] == lm_mask[:, :1])], dim=1)[:, slice_indices]
        lm_states = hidden_states[shifted_lm_mask]
        num_lm_per_batch = shifted_lm_mask.sum(dim=1).tolist()
        pred_logits = self.lm_head(lm_states)

        shifted_cad_mask = shifted_cad_mask[:, slice_indices]
        cad_states = hidden_states[shifted_cad_mask]
        num_cad_per_batch = shifted_cad_mask.sum(dim=1).tolist()

        pred_labels = self.label_head(cad_states)
        pred_parameters = self.parameter_head(cad_states)
        pred_pointers = self.pointer_head(cad_states)

        return torch.split(pred_logits, num_lm_per_batch), torch.split(pred_labels, num_cad_per_batch), torch.split(pred_parameters, num_cad_per_batch), torch.split(pred_pointers, num_cad_per_batch), param_embeds, self.parameter_tau.exp().clone(), pointer_crv, pointer_srf, self.pointer_tau.exp().clone(), self.standard_plane_pointer.clone()

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        breps: dgl.DGLGraph,
        tokenizer=None,
        temperature_lm=1.0,
        temperature_label=1.0,
        temperature_parameter=1.0,
        temperature_pointer=1.0,
        max_steps: int = MAX_GENERATION_LENGTH_PLAN + MAX_GENERATION_LENGTH_VECTOR,
        mode="argmax", # can be "argmax" or "sample"
        **kwargs
    ):
        assert mode in ["argmax", "sample"], f"Invalid mode: {mode}. Must be 'argmax' or 'sample'."

        self.eval()

        batch_size, seq_length = input_ids.shape
        finish_mark = [False] * batch_size
        generated_ids = input_ids.clone()
        generated_mask = attention_mask.clone()
        generated_label = [torch.zeros((0), dtype=input_ids.dtype, device=input_ids.device) for _ in range(batch_size)]
        generated_parameter = [torch.zeros((0), dtype=input_ids.dtype, device=input_ids.device) for _ in range(batch_size)]
        generated_pointer = [torch.zeros((0), dtype=input_ids.dtype, device=input_ids.device) for _ in range(batch_size)]
        past_key_values = None

        text_embeds: torch.Tensor = self.model.get_input_embeddings()(generated_ids)

        crv_mask = generated_ids == 151667  # TODO: 使用Config指定
        srf_mask = generated_ids == 151670  # TODO: 使用Config指定

        if breps.num_edges() > 0:
            pointer_crv, pointer_srf, feat_crv, feat_srf = self.brep(breps)

            feat_crv_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_crv, crv_mask)
            ]
            feat_srf_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_srf, srf_mask)
            ]

            crv_mask_expanded = crv_mask.unsqueeze(-1).expand_as(text_embeds)
            srf_mask_expanded = srf_mask.unsqueeze(-1).expand_as(text_embeds)

            feat_crv = torch.vstack(feat_crv_clipped).type_as(text_embeds)
            feat_srf = torch.vstack(feat_srf_clipped).type_as(text_embeds)

            assert crv_mask.sum() == feat_crv.shape[0]
            assert srf_mask.sum() == feat_srf.shape[0]

            text_embeds = text_embeds.masked_scatter(crv_mask_expanded, feat_crv)
            text_embeds = text_embeds.masked_scatter(srf_mask_expanded, feat_srf)
        else:
            pointer_crv = [torch.zeros((0, self.pointer_size), dtype=torch.float, device=text_embeds.device) for _ in range(batch_size)]
            pointer_srf = pointer_crv

        pointer_srf = [torch.vstack([self.standard_plane_pointer, srf_pointer]) for srf_pointer in pointer_srf]
        parameter_maps = [{} for _ in range(batch_size)]  # extract_parameters_from_plan will provide parameters during generation
        param_embeds = [None] * batch_size  # extract_parameters_from_plan will provide parameters during generation

        inputs_embeds = text_embeds + self.cad_position_embedding((input_ids == 151671).type_as(input_ids))

        for step in range(max_steps):
            if past_key_values is None:
                outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)
            else:
                outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=inputs_embeds[:, -1:, :], attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state[:, -1, :]  # shape: batch * hidden_size
            
            generated_ids_list = [151643] * batch_size
            for idx in range(batch_size):
                if finish_mark[idx]:
                    continue

                if generated_ids[idx][-1] == 151671 or ((generated_ids[idx][-1] == 151673) and (generated_label[idx][-1] != TOKEN.index("<|model_end|>")) and (generated_label[idx][-1] != TOKEN.index("<|part_end|>"))):
                    # is CAD generation step
                    if mode == "sample":
                        logits = self.label_head(hidden_states[idx]) / temperature_label
                        logits = logits - torch.max(logits)
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                        pred_label = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                    else:
                        pred_label = torch.argmax(self.label_head(hidden_states[idx]), dim=0)

                    if pred_label in [TOKEN.index("<|length_value|>"), TOKEN.index("<|angle_value|>")]:
                        pred_parameter: torch.Tensor = self.parameter_head(hidden_states[idx])

                        if pred_label == TOKEN.index("<|length_value|>"):
                            pred_length = F.normalize(pred_parameter, p=2, dim=0, eps=1e-6)
                            cand_length = F.normalize(param_embeds[idx]["length"], p=2, dim=1, eps=1e-6).type_as(pred_length)

                            if cand_length.numel() == 0:
                                pred_parameter = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_parameter[idx].dtype, device=generated_parameter[idx].device)
                                pred_label = torch.tensor(TOKEN.index("<|padding|>"), dtype=pred_label.dtype, device=pred_label.device)
                            else:
                                sim_length = torch.matmul(pred_length, cand_length.T) * self.parameter_tau.exp()
                                if mode == "argmax":
                                    pred_parameter = torch.argmax(sim_length) + 1  # +1 for 1-based index
                                else:
                                    logits = sim_length / temperature_parameter
                                    logits = logits - torch.max(logits)
                                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                    pred_parameter = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0) + 1  # +1 for 1-based index
                        else:
                            pred_angle = F.normalize(pred_parameter, p=2, dim=0, eps=1e-6)
                            cand_angle = F.normalize(param_embeds[idx]["angle"], p=2, dim=1, eps=1e-6).type_as(pred_angle)

                            if cand_angle.numel() == 0:
                                pred_parameter = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_parameter[idx].dtype, device=generated_parameter[idx].device)
                                pred_label = torch.tensor(TOKEN.index("<|padding|>"), dtype=pred_label.dtype, device=pred_label.device)
                            else:
                                sim_angle = torch.matmul(pred_angle, cand_angle.T) * self.parameter_tau.exp()
                                if mode == "argmax":
                                    pred_parameter = torch.argmax(sim_angle) + 1  # +1 for 1-based index
                                else:
                                    logits = sim_angle / temperature_parameter
                                    logits = logits - torch.max(logits)
                                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                    pred_parameter = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0) + 1  # +1 for 1-based index
                    else:
                        pred_parameter = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_parameter[idx].dtype, device=generated_parameter[idx].device)


                    if step > 0 and pred_label == TOKEN.index("<|pointer_enable|>"):
                        pred_pointer: torch.Tensor = self.pointer_head(hidden_states[idx])
                        if generated_label[idx][-1] == TOKEN.index("<|sketch_start|>"):
                            pred = pred_pointer.unsqueeze(0).expand(pointer_srf[idx].size(0), -1)
                            cos_sim = F.cosine_similarity(pred, pointer_srf[idx], dim=1) * self.pointer_tau.exp()
                            if mode == "argmax":
                                index = torch.argmax(cos_sim)
                            else:
                                logits = cos_sim / temperature_pointer
                                logits = logits - torch.max(logits)
                                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                index = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                            pred_pointer = index - len(STANDARD_PLANES)
                        else:
                            if pointer_crv[idx].shape[0] > 0:
                                pred = pred_pointer.unsqueeze(0).expand(pointer_crv[idx].size(0), -1)
                                cos_sim = F.cosine_similarity(pred, pointer_crv[idx], dim=1) * self.pointer_tau.exp()
                                if mode == "argmax":
                                    pred_pointer = torch.argmax(cos_sim)
                                else:
                                    logits = cos_sim / temperature_pointer
                                    logits = logits - torch.max(logits)
                                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                    pred_pointer = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                            else:
                                pred_pointer = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_pointer[idx].dtype, device=generated_pointer[idx].device)
                                pred_label = torch.tensor(TOKEN.index("<|pointer_disable|>"), dtype=pred_label.dtype, device=pred_label.device)
                    else:
                        pred_pointer = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_pointer[idx].dtype, device=generated_pointer[idx].device)

                    generated_label[idx] = torch.cat([generated_label[idx], pred_label.unsqueeze(0)])
                    generated_parameter[idx] = torch.cat([generated_parameter[idx], pred_parameter.unsqueeze(0)])
                    generated_pointer[idx] = torch.cat([generated_pointer[idx], pred_pointer.unsqueeze(0)])
                    generated_ids_list[idx] = 151673
                else:
                    # is LM generation step
                    pred_logit = self.lm_head(hidden_states[idx])
                    if mode == "argmax":
                        pred_id = torch.argmax(pred_logit, dim=0)
                    else:
                        logits = pred_logit / temperature_lm
                        logits = logits - torch.max(logits)
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                        pred_id = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                    pred_id[pred_id >= len(tokenizer)] = 151643  # unknown token handling
                    generated_ids_list[idx] = pred_id.item()
                    if pred_id in [151643, 151645]:
                        finish_mark[idx] = True

                    if pred_id == 151671:
                        # finish plan, start CAD generation
                        plan_ids = generated_ids[idx][seq_length:]
                        plan_txt = tokenizer.decode(plan_ids)
                        parameter_source = extract_parameters_from_plan(plan_txt)
                        parameter_tensor = {
                            "length": torch.tensor([parameter_source["length"][idx]["value_m"] if idx in parameter_source["length"] else 0.0 for idx in range(1, max(parameter_source["length"].keys()) + 1)] if len(parameter_source["length"]) > 0 else [], dtype=torch.float32, device=input_ids.device),
                            "angle": torch.tensor([parameter_source["angle"][idx]["value"] if idx in parameter_source["angle"] else 0.0 for idx in range(1, max(parameter_source["angle"].keys()) + 1)] if len(parameter_source["angle"]) > 0 else [], dtype=torch.float32, device=input_ids.device)
                        }

                        parameter_maps[idx] = parameter_tensor
                        param_embeds[idx] = self.parameter(parameter_tensor)



            generated_ids = torch.cat([generated_ids, torch.tensor(generated_ids_list, dtype=generated_ids.dtype, device=generated_ids.device).unsqueeze(1)], dim=1)

            if all(finish_mark):
                break

            inputs_embeds_next: torch.Tensor = self.model.get_input_embeddings()(generated_ids[:, -1])  # shape: batch_size * hidden_size
            for idx in range(batch_size):
                if generated_ids_list[idx] == 151673:
                    value_embeds = self.label_embedding(generated_label[idx][-1])  # shape: hiddensize
                    generated_parameter_idx = generated_parameter[idx][-1] - 1
                    generated_pointer_idx = generated_pointer[idx][-1]
                    
                    if generated_parameter_idx >= 0:
                        if generated_label[idx][-1] == TOKEN.index("<|length_value|>"):
                            parameter_embeds = self.parameter_projection(param_embeds[idx]["length"][generated_parameter_idx])
                        else:
                            parameter_embeds = self.parameter_projection(param_embeds[idx]["angle"][generated_parameter_idx])
                        value_embeds = value_embeds + parameter_embeds
                    
                    if generated_pointer_idx >= -len(STANDARD_PLANES):
                        pointer_embeds = pointer_srf[idx][generated_pointer_idx + len(STANDARD_PLANES)] if generated_label[idx][-2] == TOKEN.index("<|sketch_start|>") else pointer_crv[idx][generated_pointer_idx]
                        pointer_embeds = self.pointer_projection(pointer_embeds)
                        value_embeds = value_embeds + pointer_embeds
                    
                    inputs_embeds_next[idx] = value_embeds
            next_token_class = generated_ids[:, -1] == 151671
            for idx in range(batch_size):
                if (generated_ids[idx, -1] == 151673) and (generated_label[idx][-1] != TOKEN.index("<|model_end|>")) and (generated_label[idx][-1] != TOKEN.index("<|part_end|>")):
                    next_token_class[idx] = True
            inputs_embeds_next = inputs_embeds_next + self.cad_position_embedding(next_token_class.type_as(input_ids))
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_next.unsqueeze(1)], dim=1)

            generated_mask_pad = torch.ones((batch_size, generated_ids.shape[1] - generated_mask.shape[1]), device=generated_mask.device, dtype=generated_mask.dtype)
            generated_mask = torch.cat([generated_mask, generated_mask_pad], dim=-1)

        return generated_ids[:, seq_length:], parameter_maps, generated_label, generated_parameter, generated_pointer

    @torch.no_grad()
    def predict_plan(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        breps: dgl.DGLGraph,
        tokenizer=None,
        temperature=1.0,
        max_steps: int = MAX_GENERATION_LENGTH_PLAN,
        mode="argmax", # can be "argmax" or "sample"
        **kwargs
    ):
        assert mode in ["argmax", "sample"], f"Invalid mode: {mode}. Must be 'argmax' or 'sample'."

        self.eval()

        batch_size, seq_length = input_ids.shape
        finish_mark = torch.zeros((batch_size,), dtype=torch.bool, device=input_ids.device)
        generated_ids = torch.full((batch_size, max_steps), 151643, dtype=input_ids.dtype, device=input_ids.device)
        generated_mask = attention_mask.clone()
        past_key_values = None

        text_embeds: torch.Tensor = self.model.get_input_embeddings()(input_ids)

        crv_mask = input_ids == 151667  # TODO: 使用Config指定
        srf_mask = input_ids == 151670  # TODO: 使用Config指定

        if breps.num_edges() > 0:
            _, _, feat_crv, feat_srf = self.brep(breps)

            feat_crv_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_crv, crv_mask)
            ]
            feat_srf_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_srf, srf_mask)
            ]

            crv_mask_expanded = crv_mask.unsqueeze(-1).expand_as(text_embeds)
            srf_mask_expanded = srf_mask.unsqueeze(-1).expand_as(text_embeds)

            feat_crv = torch.vstack(feat_crv_clipped).type_as(text_embeds)
            feat_srf = torch.vstack(feat_srf_clipped).type_as(text_embeds)

            assert crv_mask.sum() == feat_crv.shape[0]
            assert srf_mask.sum() == feat_srf.shape[0]

            text_embeds = text_embeds.masked_scatter(crv_mask_expanded, feat_crv)
            text_embeds = text_embeds.masked_scatter(srf_mask_expanded, feat_srf)

        inputs_embeds = text_embeds + self.cad_position_embedding(torch.zeros_like(input_ids, dtype=input_ids.dtype, device=input_ids.device)).type_as(text_embeds)

        for step in range(max_steps):
            if past_key_values is None:
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=generated_mask,
                    past_key_values=None,
                    use_cache=True
                )
            else:
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=generated_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state[:, -1, :]  # [B, H]

            # 并行预测 logits
            pred_logits = self.lm_head(hidden_states)  # [B, V]
            
            if mode == "argmax":
                pred_ids = torch.argmax(pred_logits, dim=-1)  # [B]
            else:
                logits = pred_logits / temperature
                logits = logits - logits.max(dim=-1, keepdim=True).values
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                probs = F.softmax(logits, dim=-1)
                pred_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

            # 防止越界
            pred_ids = torch.where(pred_ids < len(tokenizer), pred_ids, torch.tensor(151643, device=pred_ids.device))

            # 对已完成的样本不再更新
            pred_ids = torch.where(finish_mark, torch.tensor(151643, device=pred_ids.device), pred_ids)

            # 写入到序列缓存
            generated_ids[:, step] = pred_ids

            finish_mark = finish_mark | (pred_ids == 151671)
            if finish_mark.all():
                break

            # 增量嵌入
            next_embeds = self.model.get_input_embeddings()(pred_ids)
            next_embeds = next_embeds + self.cad_position_embedding(
                torch.zeros((batch_size), dtype=input_ids.dtype, device=input_ids.device)
            ).type_as(next_embeds)
            inputs_embeds = next_embeds.unsqueeze(1)
            generated_mask = torch.cat(
                [generated_mask, torch.ones((batch_size, 1), device=generated_mask.device, dtype=generated_mask.dtype)],
                dim=-1
            )

        parameter_maps = [{} for _ in range(batch_size)]
        plan_txt = tokenizer.batch_decode(generated_ids[:, :step], skip_special_tokens=True)
        for idx in range(batch_size):
            parameter_source = extract_parameters_from_plan(plan_txt[idx])
            parameter_tensor = {
                "length": torch.tensor(
                    [parameter_source["length"][i]["value_m"] if i in parameter_source["length"] else 0.0
                    for i in range(1, max(parameter_source["length"].keys()) + 1)] if len(parameter_source["length"]) > 0 else [],
                    dtype=torch.float32,
                    device=input_ids.device
                ),
                "angle": torch.tensor(
                    [parameter_source["angle"][i]["value"] if i in parameter_source["angle"] else 0.0
                    for i in range(1, max(parameter_source["angle"].keys()) + 1)] if len(parameter_source["angle"]) > 0 else [],
                    dtype=torch.float32,
                    device=input_ids.device
                ),
            }
            parameter_maps[idx] = parameter_tensor

        return generated_ids[:, :step], parameter_maps, plan_txt

    @torch.no_grad()
    def predict_vector(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        breps: dgl.DGLGraph,
        parameter_maps,
        temperature_label=1.0,
        temperature_parameter=1.0,
        temperature_pointer=1.0,
        max_steps: int = MAX_GENERATION_LENGTH_VECTOR,
        mode="argmax", # can be "argmax" or "sample"
        **kwargs
    ):
        assert mode in ["argmax", "sample"], f"Invalid mode: {mode}. Must be 'argmax' or 'sample'."

        self.eval()
        
        print(input_ids[:, -1])
        if not torch.all(input_ids[:, -1] == 151671):
            input_ids = input_ids.clone()
            input_ids[:, -1] = 151671  # ensure CAD generation start token

        batch_size, seq_length = input_ids.shape
        finish_mark = [False] * batch_size
        generated_mask = attention_mask.clone()
        generated_label = torch.zeros((batch_size, max_steps), dtype=input_ids.dtype, device=input_ids.device)
        generated_parameter = torch.zeros((batch_size, max_steps), dtype=input_ids.dtype, device=input_ids.device)
        generated_pointer = torch.zeros((batch_size, max_steps), dtype=input_ids.dtype, device=input_ids.device)
        past_key_values = None

        text_embeds: torch.Tensor = self.model.get_input_embeddings()(input_ids)

        crv_mask = input_ids == 151667  # TODO: 使用Config指定
        srf_mask = input_ids == 151670  # TODO: 使用Config指定

        if breps.num_edges() > 0:
            pointer_crv, pointer_srf, feat_crv, feat_srf = self.brep(breps)

            feat_crv_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_crv, crv_mask)
            ]
            feat_srf_clipped = [
                feats[:mask.sum().item()] if feats.shape[0] > mask.sum().item() else feats
                for feats, mask in zip(feat_srf, srf_mask)
            ]

            crv_mask_expanded = crv_mask.unsqueeze(-1).expand_as(text_embeds)
            srf_mask_expanded = srf_mask.unsqueeze(-1).expand_as(text_embeds)

            feat_crv = torch.vstack(feat_crv_clipped).type_as(text_embeds)
            feat_srf = torch.vstack(feat_srf_clipped).type_as(text_embeds)

            assert crv_mask.sum() == feat_crv.shape[0]
            assert srf_mask.sum() == feat_srf.shape[0]

            text_embeds = text_embeds.masked_scatter(crv_mask_expanded, feat_crv)
            text_embeds = text_embeds.masked_scatter(srf_mask_expanded, feat_srf)
        else:
            pointer_crv = [torch.zeros((0, self.pointer_size), dtype=torch.float, device=text_embeds.device) for _ in range(batch_size)]
            pointer_srf = pointer_crv

        pointer_srf = [torch.vstack([self.standard_plane_pointer, srf_pointer]) for srf_pointer in pointer_srf]

        if parameter_maps is not None:
            param_embeds = self.parameter(parameter_maps)
        else:
            param_embeds = None

        inputs_embeds = text_embeds + self.cad_position_embedding((input_ids == 151671).type_as(input_ids))

        for step in range(max_steps):
            if past_key_values is None:
                outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)
            else:
                outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state[:, -1, :]  # shape: batch * hidden_size

            generated_label_last = torch.zeros((batch_size,), dtype=generated_label.dtype, device=generated_label.device)
            generated_parameter_last = torch.ones((batch_size,), dtype=generated_parameter.dtype, device=generated_parameter.device) * (-len(STANDARD_PLANES) - 1)
            generated_pointer_last = torch.ones((batch_size,), dtype=generated_pointer.dtype, device=generated_pointer.device) * (-len(STANDARD_PLANES) - 1)
            for idx in range(batch_size):
                if finish_mark[idx]:
                    continue

                # is CAD generation step
                if mode == "sample":
                    logits = self.label_head(hidden_states[idx]) / temperature_label
                    logits = logits - torch.max(logits)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                    pred_label = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                else:
                    pred_label = torch.argmax(self.label_head(hidden_states[idx]), dim=0)

                if pred_label in [TOKEN.index("<|length_value|>"), TOKEN.index("<|angle_value|>")]:
                    pred_parameter: torch.Tensor = self.parameter_head(hidden_states[idx])

                    if pred_label == TOKEN.index("<|length_value|>"):
                        pred_length = F.normalize(pred_parameter, p=2, dim=0, eps=1e-6)
                        cand_length = F.normalize(param_embeds[idx]["length"], p=2, dim=1, eps=1e-6).type_as(pred_length)

                        if cand_length.numel() == 0:
                            pred_parameter = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_parameter.dtype, device=generated_parameter.device)
                            pred_label = torch.tensor(TOKEN.index("<|padding|>"), dtype=pred_label.dtype, device=pred_label.device)
                        else:
                            sim_length = torch.matmul(pred_length, cand_length.T) * self.parameter_tau.exp()
                            if mode == "argmax":
                                pred_parameter = torch.argmax(sim_length) + 1  # +1 for 1-based index
                            else:
                                logits = sim_length / temperature_parameter
                                logits = logits - torch.max(logits)
                                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                pred_parameter = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0) + 1  # +1 for 1-based index
                    else:
                        pred_angle = F.normalize(pred_parameter, p=2, dim=0, eps=1e-6)
                        cand_angle = F.normalize(param_embeds[idx]["angle"], p=2, dim=1, eps=1e-6).type_as(pred_angle)

                        if cand_angle.numel() == 0:
                            pred_parameter = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_parameter.dtype, device=generated_parameter.device)
                            pred_label = torch.tensor(TOKEN.index("<|padding|>"), dtype=pred_label.dtype, device=pred_label.device)
                        else:
                            sim_angle = torch.matmul(pred_angle, cand_angle.T) * self.parameter_tau.exp()
                            if mode == "argmax":
                                pred_parameter = torch.argmax(sim_angle) + 1  # +1 for 1-based index
                            else:
                                logits = sim_angle / temperature_parameter
                                logits = logits - torch.max(logits)
                                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                pred_parameter = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0) + 1  # +1 for 1-based index
                else:
                    pred_parameter = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_parameter.dtype, device=generated_parameter.device)


                if step > 0 and pred_label == TOKEN.index("<|pointer_enable|>"):
                    pred_pointer: torch.Tensor = self.pointer_head(hidden_states[idx])
                    if generated_label[idx][step - 1] == TOKEN.index("<|sketch_start|>"):
                        pred = pred_pointer.unsqueeze(0).expand(pointer_srf[idx].size(0), -1)
                        cos_sim = F.cosine_similarity(pred, pointer_srf[idx], dim=1) * self.pointer_tau.exp()
                        if mode == "argmax":
                            index = torch.argmax(cos_sim)
                        else:
                            logits = cos_sim / temperature_pointer
                            logits = logits - torch.max(logits)
                            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                            index = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                        pred_pointer = index - len(STANDARD_PLANES)
                    else:
                        if pointer_crv[idx].shape[0] > 0:
                            pred = pred_pointer.unsqueeze(0).expand(pointer_crv[idx].size(0), -1)
                            cos_sim = F.cosine_similarity(pred, pointer_crv[idx], dim=1) * self.pointer_tau.exp()
                            if mode == "argmax":
                                pred_pointer = torch.argmax(cos_sim)
                            else:
                                logits = cos_sim / temperature_pointer
                                logits = logits - torch.max(logits)
                                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                pred_pointer = torch.multinomial(F.softmax(logits, dim=0), num_samples=1).squeeze(0)
                        else:
                            pred_pointer = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_pointer.dtype, device=generated_pointer.device)
                            pred_label = torch.tensor(TOKEN.index("<|pointer_disable|>"), dtype=pred_label.dtype, device=pred_label.device)
                else:
                    pred_pointer = torch.tensor(-len(STANDARD_PLANES) - 1, dtype=generated_pointer.dtype, device=generated_pointer.device)

                generated_label_last[idx] = pred_label
                generated_parameter_last[idx] = pred_parameter
                generated_pointer_last[idx] = pred_pointer
                finish_mark[idx] = pred_label == TOKEN.index("<|model_end|>") or pred_label == TOKEN.index("<|part_end|>")

            generated_label[:, step] = generated_label_last
            generated_parameter[:, step] = generated_parameter_last
            generated_pointer[:, step] = generated_pointer_last

            if all(finish_mark):
                break

            inputs_embeds_next = []
            value_embeds = self.label_embedding(generated_label_last)  # shape: batch_size * hiddensize
            for idx in range(batch_size):
                if finish_mark[idx]:
                    inputs_embeds_next.append(value_embeds[idx, :].unsqueeze(0))
                    continue

                value_embeds_idx = value_embeds[idx]
                generated_parameter_idx = generated_parameter_last[idx] - 1
                generated_pointer_idx = generated_pointer_last[idx]
                
                if generated_parameter_idx >= 0:
                    if generated_label_last[idx] == TOKEN.index("<|length_value|>"):
                        parameter_embeds = self.parameter_projection(param_embeds[idx]["length"][generated_parameter_idx])
                    else:
                        parameter_embeds = self.parameter_projection(param_embeds[idx]["angle"][generated_parameter_idx])
                    value_embeds_idx = value_embeds_idx + parameter_embeds
                
                if generated_pointer_idx >= -len(STANDARD_PLANES):
                    pointer_embeds = pointer_srf[idx][generated_pointer_idx + len(STANDARD_PLANES)] if generated_label[idx][step - 1] == TOKEN.index("<|sketch_start|>") else pointer_crv[idx][generated_pointer_idx]
                    pointer_embeds = self.pointer_projection(pointer_embeds)
                    value_embeds_idx = value_embeds_idx + pointer_embeds
                
                inputs_embeds_next.append(value_embeds_idx.unsqueeze(0))
            inputs_embeds_next = torch.stack(inputs_embeds_next, dim=0)
            inputs_embeds_next = inputs_embeds_next + self.cad_position_embedding(torch.ones((batch_size,), dtype=input_ids.dtype, device=input_ids.device)).type_as(inputs_embeds_next)
            inputs_embeds = inputs_embeds_next.to(inputs_embeds.dtype)

            generated_mask = torch.cat([generated_mask, torch.ones((batch_size, 1), device=generated_mask.device, dtype=generated_mask.dtype)], dim=-1)

        return generated_label[:, :step], generated_parameter[:, :step], generated_pointer[:, :step]

    def get_param_groups(self, base_lr: float, tau_lr: float, weight_decay: float):
        other_params, tau_params = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("pointer_tau") or name.endswith("parameter_tau"):
                tau_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
        if tau_params:
            groups.append({"params": tau_params, "lr": tau_lr, "weight_decay": 0.0})
        return groups



if __name__ == "__main__":
    from loguru import logger

    @logger.catch
    def test():
        import dgl
        from torch import FloatTensor
        from dgl.data.utils import load_graphs
        from .processor import Text2CADProcessor
        from dataset.dataset import get_dataloaders

        model = PointerCAD("Qwen/Qwen2.5-0.5B-Instruct").cuda()
        processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(padding_side="left")
        dataloader = get_dataloaders(
            dataset_dir="/mnt/afs/wangchenyu/dataset/cadv3/Base",
            split_filepath="/mnt/afs/wangchenyu/dataset/cadv3/train_val_test.json",
            subsets=["validation"],
            batch_sizes=2,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=16
        )[0]

        checkpoint = torch.load("/mnt/afs/wangchenyu/cad/CADv3/plan-param-tie/log/2025-10-21/13:57/model.pth", map_location="cuda")
        missing_keys_info  = model.load_state_dict(checkpoint["model"], strict=False)
        if len(missing_keys_info.missing_keys) > 0:
            print(f"Missing keys in the checkpoint: {missing_keys_info.missing_keys}")

        for iter_dict in dataloader:
            # ###################### train ######################
            # messages = []
            # for prompt, plan in zip(iter_dict["prompt"], iter_dict["plan"]):
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
            #                 {"type": "text", "text": prompt},
            #             ],
            #         },
            #         {
            #             "role": "assistant",
            #             "content": [
            #                 {"type": "text", "text": plan},
            #                 {"type": "cad"},
            #             ],
            #         },
            #     ]
            #     messages.append(message)
            # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # breps = iter_dict["graph"]
            # gt_parameter_maps = iter_dict["parameter_map"]
            # gt_labels = iter_dict["label"]
            # gt_parameters = iter_dict["parameter"]
            # gt_pointers = iter_dict["pointer"]
            # inputs = processor(text=text, breps=breps, parameter_maps=gt_parameter_maps, labels=gt_labels, parameters=gt_parameters, pointers=gt_pointers, max_length=3072)
            # inputs = inputs.to("cuda")

            # with torch.no_grad():
            #     model(**inputs)


            ###################### validation ######################
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
                        ],
                    },
                ]
                messages.append(message)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            breps = iter_dict["graph"]
            gt_plans = iter_dict["plan"]
            gt_label_sources = iter_dict["label_source"]
            gt_pointer_sources = iter_dict["pointer_source"]
            inputs = processor(text=text, breps=breps, parameter_maps=iter_dict["parameter_map"], max_length=3072)
            inputs = inputs.to("cuda")
            print("=========================================")
            print(gt_plans[0])
            print("====================")
            for i in range(5):
                generated_label, generated_parameter, generated_pointer = model.predict_vector(mode="sample", **inputs)
                print(list(zip(generated_label[0].tolist(), generated_parameter[0].tolist(), generated_pointer[0].tolist())))
                print("====================")

            # generated_value, generated_pointer = model.predict(**inputs)
            # for idx in range(len(generated_value)):
            #     print(generated_value[idx].shape)
            #     print(generated_value[idx])
            #     print(gt_label_sources[idx])
            #     print(generated_pointer[idx])
            #     print(gt_pointer_sources[idx])
            #     print("=" * 30)
            #     print()

    test()