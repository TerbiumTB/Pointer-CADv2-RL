from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import dgl
import torch
import torch.nn.functional as F
from dgl.data.utils import load_graphs
from torch.utils.checkpoint import checkpoint

from misc import STANDARD_PLANES, TOKEN
from rl.dpo_data import DPOCompletion
from rl.prompts import prompt_message
POINTER_ENABLE_ID = TOKEN.index("<|pointer_enable|>")
SKETCH_START_ID = TOKEN.index("<|sketch_start|>")
LENGTH_VALUE_ID = TOKEN.index("<|length_value|>")
ANGLE_VALUE_ID = TOKEN.index("<|angle_value|>")
MODEL_END_ID = TOKEN.index("<|model_end|>")
PART_END_ID = TOKEN.index("<|part_end|>")
LM_LOG_PROB_CHUNK_SIZE = 8


@dataclass(frozen=True)
class ScoringTemperatures:
    plan: float = 1.0
    label: float = 1.0
    parameter: float = 1.0
    pointer: float = 1.0

    def validate(self) -> None:
        for name, value in (
            ("plan", self.plan),
            ("label", self.label),
            ("parameter", self.parameter),
            ("pointer", self.pointer),
        ):
            if value <= 0:
                raise ValueError(f"{name} scoring temperature must be positive.")


@dataclass
class TrajectoryLogProbs:
    plan: torch.Tensor
    label: torch.Tensor
    parameter: torch.Tensor
    pointer: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.plan + self.label + self.parameter + self.pointer


def _load_state_graph(rollout_root: Path, relative_path: str) -> dgl.DGLGraph:
    root = rollout_root.absolute()
    path = (root / relative_path).absolute()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"State graph path escapes rollout root: {relative_path!r}."
        ) from exc
    graph = load_graphs(str(path))[0][0]
    graph.ndata["x"] = graph.ndata["x"].float()
    graph.edata["x"] = graph.edata["x"].float()
    if graph.ndata["x"].ndim == 4:
        graph.ndata["x"][:, :, :, -2].clamp_(min=-10, max=10)
    if graph.edata["x"].ndim == 3:
        graph.edata["x"][:, :, -3:].clamp_(min=-100, max=100)
    for storage in (graph.ndata, graph.edata):
        if "_ID" in storage:
            del storage["_ID"]
    return graph


def _prompt_message(prompt: str):
    return prompt_message(prompt)


def _replay_completion_token_ids(
    stored_lm_token_ids: Sequence[int],
    num_cad_actions: int,
    cad_start_id: int,
    cad_pad_id: int,
) -> List[int]:
    """Restore the exact generated sequence, including structured CAD slots."""
    token_ids = [int(token_id) for token_id in stored_lm_token_ids]
    if num_cad_actions <= 0:
        raise ValueError("A replay step must contain at least one CAD action.")
    if token_ids.count(cad_start_id) != 1:
        raise ValueError(
            "Stored LM token IDs must contain exactly one <|cad_start|> token."
        )
    if cad_pad_id in token_ids:
        raise ValueError(
            "Stored LM token IDs unexpectedly contain <|cad_pad|>; rollout "
            "records must store that structured channel separately."
        )
    cad_start_position = token_ids.index(cad_start_id)
    return (
        token_ids[: cad_start_position + 1]
        + [cad_pad_id] * num_cad_actions
        + token_ids[cad_start_position + 1 :]
    )


def _build_replay_token_batch(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    stored_lm_token_ids: Sequence[Sequence[int]],
    cad_action_counts: Sequence[int],
    cad_start_id: int,
    cad_pad_id: int,
    pad_token_id: int,
    padding_side: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append exact rollout tokens to tokenized prompts and pad the batch."""
    batch_size = prompt_input_ids.shape[0]
    if not (
        prompt_attention_mask.shape == prompt_input_ids.shape
        and len(stored_lm_token_ids) == batch_size
        and len(cad_action_counts) == batch_size
    ):
        raise ValueError("Replay prompts and stored trajectory steps are misaligned.")
    if padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported tokenizer padding side {padding_side!r}.")

    rows: List[torch.Tensor] = []
    for index in range(batch_size):
        prompt_ids = prompt_input_ids[index][
            prompt_attention_mask[index].bool()
        ]
        completion_ids = _replay_completion_token_ids(
            stored_lm_token_ids[index],
            cad_action_counts[index],
            cad_start_id,
            cad_pad_id,
        )
        completion = torch.tensor(
            completion_ids,
            dtype=prompt_input_ids.dtype,
            device=prompt_input_ids.device,
        )
        row = torch.cat([prompt_ids, completion], dim=0)
        if row.numel() > max_length:
            raise ValueError(
                "Exact trajectory replay requires "
                f"{row.numel()} tokens, exceeding "
                f"model.max_replay_length={max_length}. Increase "
                "model.max_replay_length; truncating a stored trajectory would "
                "change the DPO likelihood."
            )
        rows.append(row)

    replay_length = max(row.numel() for row in rows)
    input_ids = torch.full(
        (batch_size, replay_length),
        int(pad_token_id),
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    attention_mask = torch.zeros(
        (batch_size, replay_length),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    for index, row in enumerate(rows):
        if padding_side == "left":
            start = replay_length - row.numel()
            input_ids[index, start:] = row
            attention_mask[index, start:] = 1
        else:
            input_ids[index, : row.numel()] = row
            attention_mask[index, : row.numel()] = 1
    return input_ids, attention_mask


def _parameter_map_to_tensors(parameter_map, device=None):
    result = {}
    for name in ("length", "angle"):
        values = parameter_map.get(name, [])
        if not isinstance(values, list):
            raise ValueError(
                f"Stored parameter map field {name!r} must be a JSON list."
            )
        result[name] = torch.tensor(values, dtype=torch.float32, device=device)
    return result


def _special_token_ids(processor):
    tokenizer = processor.tokenizer
    return {
        tokenizer.convert_tokens_to_ids(processor.brep_edge_token),
        tokenizer.convert_tokens_to_ids(processor.brep_face_token),
        tokenizer.convert_tokens_to_ids(processor.cad_token),
    }


def _valid_lm_token_count(input_ids, attention_mask, special_ids) -> int:
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special_mask |= input_ids == token_id
    return int((~special_mask & attention_mask.bool()).sum().item())


def _completion_lm_logit_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lm_token_counts: Sequence[int],
    special_ids,
) -> torch.Tensor:
    """Select hidden positions whose next LM token belongs to a completion."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have matching shapes.")
    if input_ids.ndim != 2:
        raise ValueError("Completion LM masking expects a rank-2 token batch.")
    if len(prompt_lm_token_counts) != input_ids.shape[0]:
        raise ValueError("Expected one prompt LM token count per batch item.")

    lm_mask = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        lm_mask &= input_ids != token_id
    attended_lm_mask = lm_mask & attention_mask.bool()
    valid_lm_rank = torch.cumsum(attended_lm_mask.long(), dim=1) - 1
    prompt_counts = torch.tensor(
        prompt_lm_token_counts,
        dtype=valid_lm_rank.dtype,
        device=valid_lm_rank.device,
    ).unsqueeze(1)
    completion_target_mask = attended_lm_mask & (valid_lm_rank >= prompt_counts)

    # PointerCAD hidden state at position t predicts the LM token at t + 1.
    result = torch.zeros_like(completion_target_mask)
    result[:, :-1] = completion_target_mask[:, 1:]
    return result


def _selected_completion_lm_log_prob(
    logits: torch.Tensor,
    expected_token_ids: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    """Score logits already projected only at completion target positions."""
    expected = [int(token_id) for token_id in expected_token_ids]
    if logits.shape[0] != len(expected):
        raise ValueError(
            "Completion-only LM logits do not align with stored token IDs: "
            f"model produced {logits.shape[0]}, rollout stores {len(expected)}."
        )
    if not expected:
        raise ValueError("A replay step must contain at least one completion LM token.")
    targets = torch.tensor(expected, dtype=torch.long, device=logits.device)
    total_logp = torch.zeros((), dtype=torch.float32, device=logits.device)

    def chunk_log_prob(chunk_logits, chunk_targets):
        chunk_logits = chunk_logits.float()
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        return -F.cross_entropy(
            chunk_logits,
            chunk_targets,
            reduction="sum",
        )

    for start in range(0, targets.shape[0], LM_LOG_PROB_CHUNK_SIZE):
        end = start + LM_LOG_PROB_CHUNK_SIZE
        chunk_logits = logits[start:end]
        chunk_targets = targets[start:end]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            # Cross entropy saves a float32 [tokens, vocabulary] intermediate.
            # Non-reentrant checkpointing discards it after forward and
            # recomputes it during backward, while the small chunk bounds the
            # temporary allocation in both passes.
            chunk_value = checkpoint(
                chunk_log_prob,
                chunk_logits,
                chunk_targets,
                use_reentrant=False,
            )
        else:
            chunk_value = chunk_log_prob(chunk_logits, chunk_targets)
        total_logp = total_logp + chunk_value
    return total_logp


def _completion_lm_log_prob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lm_token_count: int,
    special_ids,
    temperature: float,
    expected_token_ids,
) -> torch.Tensor:
    lm_mask = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        lm_mask &= input_ids != token_id

    lm_positions = torch.nonzero(lm_mask, as_tuple=False).squeeze(1)
    if logits.shape[0] + 1 > lm_positions.shape[0]:
        raise ValueError("PointerCAD LM logits cannot be aligned with input IDs.")
    target_positions = lm_positions[1 : logits.shape[0] + 1]
    targets = input_ids[target_positions]

    valid_lm_rank = torch.cumsum(
        (lm_mask & attention_mask.bool()).long(), dim=0
    ) - 1
    completion_mask = attention_mask[target_positions].bool()
    completion_mask &= valid_lm_rank[target_positions] >= prompt_lm_token_count
    if not torch.any(completion_mask):
        raise ValueError(
            "No completion LM tokens remain after prompt masking; the stored "
            "trajectory may be truncated."
        )

    completion_targets = targets[completion_mask]
    if completion_targets.detach().cpu().tolist() != list(expected_token_ids):
        raise ValueError(
            "Stored LM token IDs do not match the tokenized trajectory replay."
        )
    # Projecting PointerCAD hidden states already materializes a large
    # [tokens, vocabulary] tensor. A full float32 log_softmax over prompt and
    # completion tokens can add several GiB even though DPO only needs the
    # sampled completion targets. Select those rows first and evaluate their
    # NLL in bounded chunks. Cross entropy is exactly
    # -log_softmax(logits)[target] without retaining the full FP32 matrix.
    return _selected_completion_lm_log_prob(
        logits=logits[completion_mask],
        expected_token_ids=completion_targets.detach().cpu().tolist(),
        temperature=temperature,
    )


def _candidate_log_prob(
    prediction: torch.Tensor,
    candidates: torch.Tensor,
    target_index: int,
    scale: torch.Tensor,
    temperature: float,
    candidate_kind: str,
) -> torch.Tensor:
    if candidates.shape[0] == 0:
        raise ValueError(f"Empty {candidate_kind} candidate set.")
    if target_index < 0 or target_index >= candidates.shape[0]:
        raise ValueError(
            f"{candidate_kind} target {target_index} is outside "
            f"{candidates.shape[0]} candidates."
        )
    prediction = F.normalize(prediction.float(), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    logits = candidates @ prediction
    logits = logits * scale.float() / temperature
    return F.log_softmax(logits, dim=0)[target_index]


def _structured_step_log_probs(
    pred_labels,
    pred_parameters,
    pred_pointers,
    labels,
    parameters,
    pointers,
    parameter_candidates,
    parameter_tau,
    curve_candidates,
    surface_candidates,
    pointer_tau,
    standard_plane_candidates,
    temperatures: ScoringTemperatures,
    allow_trailing_prediction: bool = False,
    step_description: str = "trajectory step",
):
    prediction_counts = {
        pred_labels.shape[0],
        pred_parameters.shape[0],
        pred_pointers.shape[0],
    }
    stored_counts = {
        labels.shape[0],
        parameters.shape[0],
        pointers.shape[0],
    }
    if len(prediction_counts) != 1 or len(stored_counts) != 1:
        raise ValueError(
            f"Structured channels have inconsistent lengths for {step_description}: "
            f"model={sorted(prediction_counts)}, stored={sorted(stored_counts)}."
        )
    prediction_count = next(iter(prediction_counts))
    stored_count = next(iter(stored_counts))
    if prediction_count == stored_count + 1 and allow_trailing_prediction:
        # Generation can hit max_generation_steps immediately after a
        # non-terminal CAD action. PointerCAD then exposes the distribution of
        # the next action as its final state, but that action was never sampled
        # and is not part of the stored trajectory likelihood.
        pred_labels = pred_labels[:stored_count]
        pred_parameters = pred_parameters[:stored_count]
        pred_pointers = pred_pointers[:stored_count]
        prediction_count = stored_count
    if prediction_count != stored_count:
        raise ValueError(
            f"Stored structured actions do not align with model outputs for "
            f"{step_description}: model produced {prediction_count}, rollout "
            f"stores {stored_count}."
        )

    label_logps = F.log_softmax(
        pred_labels.float() / temperatures.label, dim=-1
    )
    label_total = label_logps.gather(
        -1, labels.long().unsqueeze(-1)
    ).squeeze(-1).sum()
    parameter_total = pred_parameters.sum() * 0.0
    for candidates in parameter_candidates.values():
        parameter_total = parameter_total + candidates.sum() * 0.0
    parameter_total = parameter_total + parameter_tau.sum() * 0.0
    pointer_total = pred_pointers.sum() * 0.0
    pointer_total = pointer_total + curve_candidates.sum() * 0.0
    pointer_total = pointer_total + surface_candidates.sum() * 0.0
    pointer_total = pointer_total + standard_plane_candidates.sum() * 0.0
    pointer_total = pointer_total + pointer_tau.sum() * 0.0
    plane_count = len(STANDARD_PLANES)

    for action_index, label_tensor in enumerate(labels):
        label = int(label_tensor.item())
        if label == LENGTH_VALUE_ID:
            parameter_total = parameter_total + _candidate_log_prob(
                pred_parameters[action_index],
                parameter_candidates["length"],
                int(parameters[action_index].item()) - 1,
                parameter_tau,
                temperatures.parameter,
                "length parameter",
            )
        elif label == ANGLE_VALUE_ID:
            parameter_total = parameter_total + _candidate_log_prob(
                pred_parameters[action_index],
                parameter_candidates["angle"],
                int(parameters[action_index].item()) - 1,
                parameter_tau,
                temperatures.parameter,
                "angle parameter",
            )

        if label != POINTER_ENABLE_ID:
            continue
        if action_index == 0:
            raise ValueError("Pointer cannot be enabled for the first CAD action.")
        pointer_target = int(pointers[action_index].item())
        if int(labels[action_index - 1].item()) == SKETCH_START_ID:
            candidates = torch.cat(
                [standard_plane_candidates, surface_candidates], dim=0
            )
            target_index = pointer_target + plane_count
            candidate_kind = "surface pointer"
        else:
            candidates = curve_candidates
            target_index = pointer_target
            candidate_kind = "curve pointer"
        pointer_total = pointer_total + _candidate_log_prob(
            pred_pointers[action_index],
            candidates,
            target_index,
            pointer_tau,
            temperatures.pointer,
            candidate_kind,
        )
    return label_total, parameter_total, pointer_total


def score_trajectories(
    model,
    processor,
    completions: Sequence[DPOCompletion],
    rollout_root: str,
    device,
    max_replay_length: int,
    temperatures: ScoringTemperatures,
    max_input_length: int = 3072,
) -> TrajectoryLogProbs:
    """Score variable-length episodes in a single batched model forward."""
    temperatures.validate()
    flat_steps = []
    trajectory_ranges = []
    for completion in completions:
        start = len(flat_steps)
        flat_steps.extend((completion, step) for step in completion.steps)
        trajectory_ranges.append((start, len(flat_steps)))
    if not flat_steps:
        raise ValueError("Cannot score an empty trajectory batch.")

    graphs = [
        _load_state_graph(
            Path(rollout_root), step.state_before_graph_path
        )
        for _, step in flat_steps
    ]
    graph_batch = dgl.batch(graphs)
    prompt_messages = [
        _prompt_message(completion.prompt) for completion, _ in flat_steps
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    prompt_inputs = processor(
        text=prompt_text,
        breps=graph_batch,
        # Match rollout generation exactly. The generated completion is
        # appended only after this input-side truncation.
        max_length=max_input_length,
    )
    special_ids = _special_token_ids(processor)
    prompt_lm_counts = [
        _valid_lm_token_count(ids, mask, special_ids)
        for ids, mask in zip(
            prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        )
    ]

    labels = [
        torch.tensor(step.labels, dtype=torch.long) for _, step in flat_steps
    ]
    parameters = [
        torch.tensor(step.parameters, dtype=torch.long) for _, step in flat_steps
    ]
    pointers = [
        torch.tensor(step.pointers, dtype=torch.long) for _, step in flat_steps
    ]
    parameter_maps = [
        _parameter_map_to_tensors(step.parameter_map)
        for _, step in flat_steps
    ]
    tokenizer = processor.tokenizer
    cad_start_id = tokenizer.convert_tokens_to_ids("<|cad_start|>")
    cad_pad_id = tokenizer.convert_tokens_to_ids(processor.cad_token)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Exact trajectory replay requires a tokenizer pad token.")
    replay_input_ids, replay_attention_mask = _build_replay_token_batch(
        prompt_input_ids=prompt_inputs["input_ids"],
        prompt_attention_mask=prompt_inputs["attention_mask"],
        stored_lm_token_ids=[
            step.plan_token_ids for _, step in flat_steps
        ],
        cad_action_counts=[len(step.labels) for _, step in flat_steps],
        cad_start_id=cad_start_id,
        cad_pad_id=cad_pad_id,
        pad_token_id=pad_token_id,
        padding_side=tokenizer.padding_side,
        max_length=max_replay_length,
    )
    model_inputs = prompt_inputs
    model_inputs["input_ids"] = replay_input_ids
    model_inputs["attention_mask"] = replay_attention_mask
    model_inputs["parameter_maps"] = parameter_maps
    model_inputs["labels"] = labels
    model_inputs["parameters"] = parameters
    model_inputs["pointers"] = pointers
    model_inputs["lm_logits_mask"] = _completion_lm_logit_mask(
        replay_input_ids,
        replay_attention_mask,
        prompt_lm_counts,
        special_ids,
    )
    # DPO disables DDP's find_unused_parameters traversal to avoid _DDPSink
    # cloning large model outputs. Conditional modules that a particular replay
    # batch skips are instead attached to the loss with exact zero gradients.
    model_inputs["ensure_conditional_parameter_usage"] = True
    model_inputs = model_inputs.to(device)

    outputs = model(**model_inputs)
    (
        pred_logits,
        pred_labels,
        pred_parameters,
        pred_pointers,
        parameter_candidates,
        parameter_tau,
        curve_candidates,
        surface_candidates,
        pointer_tau,
        standard_plane_candidates,
    ) = outputs

    step_plan = []
    step_label = []
    step_parameter = []
    step_pointer = []
    for index in range(len(flat_steps)):
        completion, step = flat_steps[index]
        step_plan.append(
            _selected_completion_lm_log_prob(
                logits=pred_logits[index],
                expected_token_ids=step.plan_token_ids,
                temperature=temperatures.plan,
            )
        )
        truncated_inside_cad = (
            bool(step.plan_token_ids)
            and step.plan_token_ids[-1] == cad_start_id
            and bool(step.labels)
            and step.labels[-1] not in {MODEL_END_ID, PART_END_ID}
        )
        label_logp, parameter_logp, pointer_logp = (
            _structured_step_log_probs(
                pred_labels[index],
                pred_parameters[index],
                pred_pointers[index],
                model_inputs["labels"][index],
                model_inputs["parameters"][index],
                model_inputs["pointers"][index],
                parameter_candidates[index],
                parameter_tau,
                curve_candidates[index],
                surface_candidates[index],
                pointer_tau,
                standard_plane_candidates,
                temperatures,
                allow_trailing_prediction=truncated_inside_cad,
                step_description=(
                    f"trajectory {completion.trajectory_id!r}, "
                    f"step {step.step_index}"
                ),
            )
        )
        step_label.append(label_logp)
        step_parameter.append(parameter_logp)
        step_pointer.append(pointer_logp)

    def reduce_steps(values: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(values)
        return torch.stack(
            [stacked[start:end].sum() for start, end in trajectory_ranges]
        )

    return TrajectoryLogProbs(
        plan=reduce_steps(step_plan),
        label=reduce_steps(step_label),
        parameter=reduce_steps(step_parameter),
        pointer=reduce_steps(step_pointer),
    )
