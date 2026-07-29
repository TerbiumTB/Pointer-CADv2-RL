from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import dgl
import torch
import torch.nn.functional as F
from dgl.data.utils import load_graphs

from misc import STANDARD_PLANES, TOKEN
from rl.dpo_data import DPOCompletion
from rl.prompts import prompt_message
POINTER_ENABLE_ID = TOKEN.index("<|pointer_enable|>")
SKETCH_START_ID = TOKEN.index("<|sketch_start|>")
LENGTH_VALUE_ID = TOKEN.index("<|length_value|>")
ANGLE_VALUE_ID = TOKEN.index("<|angle_value|>")


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


def _full_message(prompt: str, plan: str):
    return _prompt_message(prompt) + [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": plan},
                {"type": "cad"},
            ],
        }
    ]


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
    token_logps = F.log_softmax(logits.float() / temperature, dim=-1)
    selected = token_logps.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    return selected[completion_mask].sum()


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
):
    if not (
        pred_labels.shape[0]
        == pred_parameters.shape[0]
        == pred_pointers.shape[0]
        == labels.shape[0]
        == parameters.shape[0]
        == pointers.shape[0]
    ):
        raise ValueError(
            "Stored structured actions do not align with model outputs. "
            "The sequence may have been truncated."
        )

    label_logps = F.log_softmax(
        pred_labels.float() / temperatures.label, dim=-1
    )
    label_total = label_logps.gather(
        -1, labels.long().unsqueeze(-1)
    ).squeeze(-1).sum()
    parameter_total = pred_parameters.sum() * 0.0
    pointer_total = pred_pointers.sum() * 0.0
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
    max_length: int,
    temperatures: ScoringTemperatures,
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
    full_messages = [
        _full_message(completion.prompt, step.plan_text)
        for completion, step in flat_steps
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_inputs = processor(
        text=prompt_text,
        breps=graph_batch,
        max_length=max_length,
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
    model_inputs = processor(
        text=full_text,
        breps=graph_batch,
        parameter_maps=parameter_maps,
        labels=labels,
        parameters=parameters,
        pointers=pointers,
        max_length=max_length,
    ).to(device)

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
        step_plan.append(
            _completion_lm_log_prob(
                pred_logits[index],
                model_inputs["input_ids"][index],
                model_inputs["attention_mask"][index],
                prompt_lm_counts[index],
                special_ids,
                temperatures.plan,
                flat_steps[index][1].plan_token_ids,
            )
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
