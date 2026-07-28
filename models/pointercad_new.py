import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from misc import TOKEN, STANDARD_PLANES, MAX_GENERATION_LENGTH_PLAN, extract_parameters_from_plan
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

    def forward(self, input_ids, attention_mask, breps, parameter_maps, labels, parameters, pointers, logits_to_keep=0, **kwargs):
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
            pointer_crv = [torch.zeros((0, self.pointer_size), dtype=torch.float, device=text_embeds.device) for _ in range(len(pointers))]
            pointer_srf = pointer_crv

        ##################  Build Parameter Embeded  ##################
        if parameter_maps is not None:
            param_embeds = self.parameter(parameter_maps)
        else:
            param_embeds = None

        ##################  Build CAD Embeded  ##################
        assert len(labels) == len(pointers)  # same batch size 
        if len(labels) > 0:
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
                    length_embeds: torch.Tensor = self.parameter_projection(batch_param_embeds["length"][length_idx])
                    label_embeds[length_mask] += length_embeds
                
                angle_mask = batch_label[:total_cad_num] == TOKEN.index("<|angle_value|>")
                if torch.any(angle_mask):
                    angle_idx: torch.Tensor = batch_parameter[:total_cad_num][angle_mask] - 1
                    assert angle_idx.min() >= 0
                    angle_embeds: torch.Tensor = self.parameter_projection(batch_param_embeds["angle"][angle_idx])
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
        max_steps: int = MAX_GENERATION_LENGTH_PLAN,
        mode="argmax", # can be "argmax" or "sample"
        **kwargs
    ):
        assert mode in ["argmax", "sample"], f"Invalid mode: {mode}. Must be 'argmax' or 'sample'."

        self.eval()

        batch_size, seq_length = input_ids.shape
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

        def first_occurrence_indices(t, n, default=0):
            idx = torch.full((n,), fill_value=default, dtype=t.dtype, device=t.device)
            seen = set()
            for i in range(t.numel()):
                val = int(t[i])
                if 0 <= val < n and val not in seen:
                    idx[val] = i
                    seen.add(val)
            return idx

        pointer_crv_all = torch.vstack(pointer_crv)
        pointer_crv_id = [torch.full((t.shape[0],), i, dtype=input_ids.dtype, device=input_ids.device) for i, t in enumerate(pointer_crv)]
        pointer_crv_id = torch.cat(pointer_crv_id, dim=0)
        pointer_proj_crv = self.pointer_projection(pointer_crv_all)
        pointer_proj_crv_id = first_occurrence_indices(pointer_crv_id, batch_size, default=-1)
        pointer_crv_all = F.normalize(pointer_crv_all, p=2, dim=1, eps=1e-6).T.to(self.dtype)

        pointer_srf_all = torch.vstack(pointer_srf)
        pointer_srf_id = [torch.full((t.shape[0],), i, dtype=input_ids.dtype, device=input_ids.device) for i, t in enumerate(pointer_srf)]
        pointer_srf_id = torch.cat(pointer_srf_id, dim=0)
        pointer_srf_all = torch.vstack([self.standard_plane_pointer, pointer_srf_all])
        pointer_srf_id = torch.cat([torch.full((self.standard_plane_pointer.size(0),), -1, dtype=input_ids.dtype, device=input_ids.device), pointer_srf_id], dim=0)
        pointer_proj_srf = self.pointer_projection(pointer_srf_all)
        pointer_proj_srf_id = first_occurrence_indices(pointer_srf_id, batch_size, default=-1)
        pointer_srf_all = F.normalize(pointer_srf_all, p=2, dim=1, eps=1e-6).T.to(self.dtype)

        inputs_embeds = text_embeds + self.cad_position_embedding((input_ids == 151671).type_as(input_ids))  # todo: not only start token

        parameter_maps = [{} for _ in range(batch_size)]  # extract_parameters_from_plan will provide parameters during generation
        param_proj_length = torch.zeros((0, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        param_proj_length_id = torch.full((batch_size,), fill_value=-1, dtype=input_ids.dtype, device=input_ids.device)
        param_proj_angle = torch.zeros((0, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        param_proj_angle_id = torch.full((batch_size,), fill_value=-1, dtype=input_ids.dtype, device=input_ids.device)
        param_embeds_length_norm_T = torch.zeros((self.pointer_size, 0), dtype=self.dtype, device=input_ids.device)
        param_embeds_length_id = torch.full((0,), fill_value=-1, dtype=input_ids.dtype, device=input_ids.device)
        param_embeds_angle_norm_T = torch.zeros((self.pointer_size, 0), dtype=self.dtype, device=input_ids.device)
        param_embeds_angle_id = torch.full((0,), fill_value=-1, dtype=input_ids.dtype, device=input_ids.device)
        generated_ids_last = torch.full((batch_size,), fill_value=151643, dtype=generated_ids.dtype, device=generated_ids.device)
        generated_label_last = torch.full((batch_size,), fill_value=TOKEN.index("<|padding|>"), dtype=input_ids.dtype, device=input_ids.device)
        generated_parameter_last = torch.full((batch_size,), fill_value=-len(STANDARD_PLANES) - 1, dtype=input_ids.dtype, device=input_ids.device)
        generated_pointer_last = torch.full((batch_size,), fill_value=-len(STANDARD_PLANES) - 1, dtype=input_ids.dtype, device=input_ids.device)
        finish_mark = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        for step in range(max_steps):
            if past_key_values is None:
                outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)
            else:
                outputs: BaseModelOutputWithPast = self.model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state[:, -1, :]  # shape: batch * hidden_size

            is_cad = (generated_ids_last == 151671) | ((generated_ids_last == 151673) & (generated_label_last != TOKEN.index("<|model_end|>")) & (generated_label_last != TOKEN.index("<|part_end|>")))  # [B]
            is_lm  = ~is_cad  # [B]
            is_cad = is_cad & ~finish_mark
            is_lm  = is_lm & ~finish_mark
            is_sketch_start = generated_label_last == TOKEN.index("<|sketch_start|>")

            generated_ids_last.fill_(151643)
            if torch.any(is_lm):
                lm_hidden_states = hidden_states[is_lm]
                pred_logit = self.lm_head(lm_hidden_states)
                if mode == "sample":
                    pred_id = torch.multinomial(F.softmax(pred_logit / temperature_lm, dim=-1), num_samples=1).squeeze(1)
                else:
                    pred_id = torch.argmax(pred_logit, dim=-1)

                pred_id[pred_id >= len(tokenizer)] = 151643  # unknown token handling
                generated_ids_last[is_lm] = pred_id

            generated_label_last.fill_(TOKEN.index("<|padding|>"))
            generated_parameter_last.fill_(-len(STANDARD_PLANES) - 1)
            generated_pointer_last.fill_(-len(STANDARD_PLANES) - 1)
            if torch.any(is_cad):
                cad_hidden_states = hidden_states[is_cad]
                pred_label_logit = self.label_head(cad_hidden_states)
                if mode == "sample":
                    pred_label = torch.multinomial(F.softmax(pred_label_logit / temperature_label, dim=-1), num_samples=1).squeeze(1)
                else:
                    pred_label = torch.argmax(pred_label_logit, dim=-1)

                generated_label_last[is_cad] = pred_label
                generated_ids_last[is_cad] = 151673

                is_length = generated_label_last == TOKEN.index("<|length_value|>")
                is_angle = generated_label_last == TOKEN.index("<|angle_value|>")

                if torch.any(is_length):
                    if param_embeds_length_norm_T.numel() > 0:
                        length_hidden_states = hidden_states[is_length]
                        pred_parameter_length = self.parameter_head(length_hidden_states)

                        length_vec = F.normalize(pred_parameter_length, p=2, dim=1, eps=1e-6)
                        sim_length = torch.matmul(length_vec, param_embeds_length_norm_T) * self.parameter_tau.exp()
                        length_mask = param_embeds_length_id.unsqueeze(0) == torch.arange(batch_size, device=param_embeds_length_id.device)[is_length].unsqueeze(1)
                        sim_length_masked = sim_length.masked_fill(~length_mask, float('-inf'))
                        valid_length_mask = torch.any(length_mask, dim=1)
                        if not torch.all(valid_length_mask):
                            # Some length parameters have no candidates
                            invalid_length_mask = is_length.masked_scatter(is_length, ~valid_length_mask)
                            generated_label_last[invalid_length_mask] = TOKEN.index("<|padding|>")
                            sim_length_masked = sim_length_masked[valid_length_mask]
                            length_mask = length_mask[valid_length_mask]
                            is_length &= ~invalid_length_mask
                        if mode == "argmax":
                            length_idx = torch.argmax(sim_length_masked, dim=1)
                        else:
                            probs = F.softmax(sim_length_masked / temperature_parameter, dim=1)
                            length_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                        length_mask_cum = length_mask.cumsum(dim=1)
                        generated_parameter_last[is_length] = length_mask_cum[torch.arange(length_mask_cum.size(0)), length_idx]  # +1 for 1-based index
                    else:
                        generated_label_last[is_length] = TOKEN.index("<|padding|>")
                        is_length.fill_(False)

                if torch.any(is_angle):
                    if param_embeds_angle_norm_T.numel() > 0:
                        angle_hidden_states = hidden_states[is_angle]
                        pred_parameter_angle = self.parameter_head(angle_hidden_states)

                        angle_vec = F.normalize(pred_parameter_angle, p=2, dim=1, eps=1e-6)
                        sim_angle = torch.matmul(angle_vec, param_embeds_angle_norm_T) * self.parameter_tau.exp()
                        angle_mask = param_embeds_angle_id.unsqueeze(0) == torch.arange(batch_size, device=param_embeds_angle_id.device)[is_angle].unsqueeze(1)
                        sim_angle_masked = sim_angle.masked_fill(~angle_mask, float('-inf'))
                        valid_angle_mask = torch.any(angle_mask, dim=1)
                        if not torch.all(valid_angle_mask):
                            # Some angle parameters have no candidates
                            invalid_angle_mask = is_angle.masked_scatter(is_angle, ~valid_angle_mask)
                            generated_label_last[invalid_angle_mask] = TOKEN.index("<|padding|>")
                            sim_angle_masked = sim_angle_masked[valid_angle_mask]
                            angle_mask = angle_mask[valid_angle_mask]
                            is_angle &= ~invalid_angle_mask
                        if mode == "argmax":
                            angle_idx = torch.argmax(sim_angle_masked, dim=1)
                        else:
                            probs = F.softmax(sim_angle_masked / temperature_parameter, dim=1)
                            angle_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                        angle_mask_cum = angle_mask.cumsum(dim=1)
                        generated_parameter_last[is_angle] = angle_mask_cum[torch.arange(angle_mask_cum.size(0)), angle_idx]  # +1 for 1-based index
                    else:
                        generated_label_last[is_angle] = TOKEN.index("<|padding|>")
                        is_angle.fill_(False)

                is_pointer = generated_label_last == TOKEN.index("<|pointer_enable|>")
                is_pointer_srf = is_pointer & is_sketch_start
                is_pointer_crv = is_pointer & ~is_sketch_start

                if torch.any(is_pointer_srf):
                    if pointer_srf_all.numel() > 0:
                        pointer_hidden_states = hidden_states[is_pointer_srf]
                        pred_pointer_srf = self.pointer_head(pointer_hidden_states)

                        pred_pointer_srf_vec = F.normalize(pred_pointer_srf, p=2, dim=1, eps=1e-6)
                        sim_pointer_srf = torch.matmul(pred_pointer_srf_vec, pointer_srf_all) * self.pointer_tau.exp()
                        pointer_srf_mask = (pointer_srf_id.unsqueeze(0) == torch.arange(batch_size, device=pointer_srf_id.device)[is_pointer_srf].unsqueeze(1)) | (pointer_srf_id.unsqueeze(0) == -1)
                        sim_pointer_srf_masked = sim_pointer_srf.masked_fill(~pointer_srf_mask, float('-inf'))
                        valid_pointer_srf_mask = torch.any(pointer_srf_mask, dim=1)
                        if not torch.all(valid_pointer_srf_mask):
                            # Some pointer_srf parameters have no candidates
                            invalid_pointer_srf_mask = is_pointer_srf.masked_scatter(is_pointer_srf, ~valid_pointer_srf_mask)
                            generated_label_last[invalid_pointer_srf_mask] = TOKEN.index("<|pointer_disable|>")
                            sim_pointer_srf_masked = sim_pointer_srf_masked[valid_pointer_srf_mask]
                            pointer_srf_mask = pointer_srf_mask[valid_pointer_srf_mask]
                            is_pointer_srf &= ~invalid_pointer_srf_mask
                            is_pointer &= ~invalid_pointer_srf_mask
                        if mode == "argmax":
                            pointer_idx = torch.argmax(sim_pointer_srf_masked, dim=1)
                        else:
                            probs = F.softmax(sim_pointer_srf_masked / temperature_pointer, dim=1)
                            pointer_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                        pointer_srf_mask_cum = pointer_srf_mask.cumsum(dim=1)
                        generated_pointer_last[is_pointer_srf] = pointer_srf_mask_cum[torch.arange(pointer_srf_mask_cum.size(0)), pointer_idx] - 1 - len(STANDARD_PLANES)  # adjust for standard planes
                    else:
                        generated_label_last[is_pointer_srf] = TOKEN.index("<|pointer_disable|>")
                        is_pointer[is_pointer_srf] = False
                        is_pointer_srf.fill_(False)
                
                if torch.any(is_pointer_crv):
                    if pointer_crv_all.numel() > 0:
                        pointer_hidden_states = hidden_states[is_pointer_crv]
                        pred_pointer_crv = self.pointer_head(pointer_hidden_states)

                        pred_pointer_crv_vec = F.normalize(pred_pointer_crv, p=2, dim=1, eps=1e-6)
                        sim_pointer_crv = torch.matmul(pred_pointer_crv_vec, pointer_crv_all) * self.pointer_tau.exp()
                        pointer_crv_mask = pointer_crv_id.unsqueeze(0) == torch.arange(batch_size, device=pointer_crv_id.device)[is_pointer_crv].unsqueeze(1)
                        sim_pointer_crv_masked = sim_pointer_crv.masked_fill(~pointer_crv_mask, float('-inf'))
                        valid_pointer_crv_mask = torch.any(pointer_crv_mask, dim=1)
                        if not torch.all(valid_pointer_crv_mask):
                            # Some pointer_crv parameters have no candidates
                            invalid_pointer_crv_mask = is_pointer_crv.masked_scatter(is_pointer_crv, ~valid_pointer_crv_mask)
                            generated_label_last[invalid_pointer_crv_mask] = TOKEN.index("<|pointer_disable|>")
                            sim_pointer_crv_masked = sim_pointer_crv_masked[valid_pointer_crv_mask]
                            pointer_crv_mask = pointer_crv_mask[valid_pointer_crv_mask]
                            is_pointer_crv &= ~invalid_pointer_crv_mask
                            is_pointer &= ~invalid_pointer_crv_mask
                        if mode == "argmax":
                            pointer_idx = torch.argmax(sim_pointer_crv_masked, dim=1)
                        else:
                            probs = F.softmax(sim_pointer_crv_masked / temperature_pointer, dim=1)
                            pointer_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                        pointer_crv_mask_cum = pointer_crv_mask.cumsum(dim=1)
                        generated_pointer_last[is_pointer_crv] = pointer_crv_mask_cum[torch.arange(pointer_crv_mask_cum.size(0)), pointer_idx] - 1  # adjust for 0-based index
                    else:
                        generated_label_last[is_pointer_crv] = TOKEN.index("<|pointer_disable|>")
                        is_pointer[is_pointer_crv] = False
                        is_pointer_crv.fill_(False)
                
                for idx in torch.where(is_cad)[0]:
                    generated_label[idx] = torch.cat([generated_label[idx], generated_label_last[idx].unsqueeze(0)], dim=0)
                    generated_parameter[idx] = torch.cat([generated_parameter[idx], generated_parameter_last[idx].unsqueeze(0)], dim=0)
                    generated_pointer[idx] = torch.cat([generated_pointer[idx], generated_pointer_last[idx].unsqueeze(0)], dim=0)

            generated_ids = torch.cat([generated_ids, generated_ids_last.unsqueeze(1)], dim=1)

            finish_mark |= (generated_ids_last == 151643) | (generated_ids_last == 151645)

            for idx in torch.where(generated_ids_last == 151671)[0]:
                # finish plan, start CAD generation
                plan_ids = generated_ids[idx][seq_length:-1]
                plan_txt = tokenizer.decode(plan_ids)
                parameter_source = extract_parameters_from_plan(plan_txt)
                parameter_tensor = {
                    "length": torch.tensor([parameter_source["length"][idx]["value_m"] if idx in parameter_source["length"] else 0.0 for idx in range(1, max(parameter_source["length"].keys()) + 1)] if len(parameter_source["length"]) > 0 else [], dtype=torch.float32, device=input_ids.device),
                    "angle": torch.tensor([parameter_source["angle"][idx]["value"] if idx in parameter_source["angle"] else 0.0 for idx in range(1, max(parameter_source["angle"].keys()) + 1)] if len(parameter_source["angle"]) > 0 else [], dtype=torch.float32, device=input_ids.device)
                }

                parameter_maps[idx] = parameter_tensor
                param_embeds_source = self.parameter(parameter_tensor)
                if param_embeds_source["length"].numel() > 0:
                    param_embeds_length_norm_T = torch.cat([param_embeds_length_norm_T, F.normalize(param_embeds_source["length"], p=2, dim=1, eps=1e-6).T], dim=1).to(self.dtype)
                    param_embeds_length_id = torch.cat([param_embeds_length_id, torch.full((param_embeds_source["length"].size(0),), fill_value=idx, dtype=input_ids.dtype, device=input_ids.device)], dim=0)
                    param_proj_length_id[idx] = param_proj_length.size(0)
                    param_proj_length = torch.cat([param_proj_length, self.parameter_projection(param_embeds_source["length"])], dim=0)
                if param_embeds_source["angle"].numel() > 0:
                    param_embeds_angle_norm_T = torch.cat([param_embeds_angle_norm_T, F.normalize(param_embeds_source["angle"], p=2, dim=1, eps=1e-6).T], dim=1).to(self.dtype)
                    param_embeds_angle_id = torch.cat([param_embeds_angle_id, torch.full((param_embeds_source["angle"].size(0),), fill_value=idx, dtype=input_ids.dtype, device=input_ids.device)], dim=0)
                    param_proj_angle_id[idx] = param_proj_angle.size(0)
                    param_proj_angle = torch.cat([param_proj_angle, self.parameter_projection(param_embeds_source["angle"])], dim=0)

            if torch.all(finish_mark):
                break

            inputs_embeds: torch.Tensor = self.model.get_input_embeddings()(generated_ids_last)  # shape: batch_size * hidden_size
            if torch.any(is_cad):
                inputs_embeds[is_cad] = self.label_embedding(generated_label_last[is_cad]).type_as(inputs_embeds)  # shape: num_cad * hidden_size
                
                if torch.any(is_length):
                    length_embeds_idx = (param_proj_length_id + generated_parameter_last - 1)[is_length]
                    length_projection = param_proj_length[length_embeds_idx]
                    inputs_embeds[is_length] += length_projection.type_as(inputs_embeds)
                
                if torch.any(is_angle):
                    angle_embeds_idx = (param_proj_angle_id + generated_parameter_last - 1)[is_angle]
                    angle_projection = param_proj_angle[angle_embeds_idx]
                    inputs_embeds[is_angle] += angle_projection.type_as(inputs_embeds)

                if torch.any(is_pointer_crv):
                    pointer_crv_idx = (generated_pointer_last + pointer_proj_crv_id)[is_pointer_crv]
                    pointer_crv_projection = pointer_proj_crv[pointer_crv_idx]
                    inputs_embeds[is_pointer_crv] += pointer_crv_projection.type_as(inputs_embeds)

                if torch.any(is_pointer_srf):
                    pointer_srf_idx = (generated_pointer_last + pointer_proj_srf_id)
                    pointer_srf_idx[generated_pointer_last < 0] = generated_pointer_last[generated_pointer_last < 0] + len(STANDARD_PLANES)
                    pointer_srf_idx = pointer_srf_idx[is_pointer_srf]
                    pointer_srf_projection = pointer_proj_srf[pointer_srf_idx]
                    inputs_embeds[is_pointer_srf] += pointer_srf_projection.type_as(inputs_embeds)

            inputs_embeds = (inputs_embeds + self.cad_position_embedding(((generated_ids_last == 151671) | (is_cad & (generated_label_last != TOKEN.index("<|model_end|>")) & (generated_label_last != TOKEN.index("<|part_end|>")))).type_as(input_ids))).unsqueeze(1)
            generated_mask = torch.cat([generated_mask, torch.ones((batch_size, 1), dtype=generated_mask.dtype, device=generated_mask.device)], dim=-1)

        return generated_ids[:, seq_length:], parameter_maps, generated_label, generated_parameter, generated_pointer

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
            for prompt in iter_dict["prompt"]:
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
                ]
                messages.append(message)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            breps = iter_dict["graph"]
            gt_plans = iter_dict["plan"]
            inputs = processor(text=text, breps=breps, max_length=3072)
            inputs = inputs.to("cuda")
            pred_plans = []
            for _ in range(5):
                generated_ids, generated_parameter_map, generated_value, generated_parameter, generated_pointer = model.predict(mode="argmax", tokenizer=processor.tokenizer, **inputs)
                pred_plans.append(processor.batch_decode(generated_ids))
            
            for i in range(len(gt_plans)):
                print("=========================================")
                print(gt_plans[i])
                print("====================")
                for j in range(5):
                    print(pred_plans[j][i])
                    print("====================")
                print("\n\n")

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