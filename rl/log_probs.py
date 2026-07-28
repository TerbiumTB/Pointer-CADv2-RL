from typing import Sequence

import torch
import torch.nn.functional as F

from misc import STANDARD_PLANES, TOKEN


POINTER_ENABLE_ID = TOKEN.index("<|pointer_enable|>")
SKETCH_START_ID = TOKEN.index("<|sketch_start|>")


def _candidate_log_prob(
    prediction: torch.Tensor,
    candidates: torch.Tensor,
    target_index: int,
    pointer_tau: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if candidates.shape[0] == 0:
        raise ValueError("A preference sequence points to an empty B-Rep candidate set.")
    if not 0 <= target_index < candidates.shape[0]:
        raise ValueError(
            f"Pointer index {target_index} is outside a candidate set of "
            f"size {candidates.shape[0]}."
        )

    prediction = F.normalize(prediction.float(), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    logits = candidates @ prediction
    logits = logits * pointer_tau.float() / temperature
    return F.log_softmax(logits, dim=0)[target_index]


def pointercad_action_log_probs(
    outputs: tuple,
    values: Sequence[torch.Tensor],
    pointers: Sequence[torch.Tensor],
    value_temperature: float = 1.0,
    pointer_temperature: float = 1.0,
) -> list[torch.Tensor]:
    """Return log p(value_t, pointer_t | history) for every CAD action.

    Pointer probability is included only for ``<|pointer_enable|>`` actions.
    Standard planes are prepended to the surface candidate set, matching
    ``PointerCAD.predict``.
    """
    if value_temperature <= 0 or pointer_temperature <= 0:
        raise ValueError("Sampling temperatures must be positive.")

    (
        pred_values,
        pred_pointers,
        _,
        pointer_crv,
        pointer_srf,
        pointer_tau,
        standard_plane_pointer,
    ) = outputs
    plane_count = len(STANDARD_PLANES)
    batch_log_probs = []

    for sample_index, (
        sample_value_logits,
        sample_pointer_predictions,
        sample_values,
        sample_pointers,
        sample_crv,
        sample_srf,
    ) in enumerate(
        zip(
            pred_values,
            pred_pointers,
            values,
            pointers,
            pointer_crv,
            pointer_srf,
        )
    ):
        if sample_values.shape != sample_pointers.shape:
            raise ValueError(
                f"Values and pointers differ in sample {sample_index}: "
                f"{sample_values.shape} != {sample_pointers.shape}."
            )
        if sample_value_logits.shape[0] != sample_values.shape[0]:
            raise ValueError(
                "The tokenizer truncated CAD placeholders. Increase max_seq_len or "
                f"shorten sample {sample_index}."
            )

        value_log_probs = F.log_softmax(
            sample_value_logits.float() / value_temperature, dim=-1
        )
        action_log_probs = value_log_probs.gather(
            dim=-1, index=sample_values.long().unsqueeze(-1)
        ).squeeze(-1)

        pointer_terms = torch.zeros_like(action_log_probs)
        for step in range(sample_values.shape[0]):
            if sample_values[step].item() != POINTER_ENABLE_ID:
                continue
            if step == 0:
                raise ValueError("A sequence cannot enable a pointer at its first action.")

            target_pointer = int(sample_pointers[step].item())
            if sample_values[step - 1].item() == SKETCH_START_ID:
                candidates = torch.cat(
                    [standard_plane_pointer, sample_srf], dim=0
                )
                target_index = target_pointer + plane_count
            else:
                candidates = sample_crv
                target_index = target_pointer

            pointer_terms[step] = _candidate_log_prob(
                sample_pointer_predictions[step],
                candidates,
                target_index,
                pointer_tau,
                pointer_temperature,
            )

        batch_log_probs.append(action_log_probs + pointer_terms)

    return batch_log_probs


def sequence_log_probs(
    action_log_probs: Sequence[torch.Tensor], average: bool = False
) -> torch.Tensor:
    """Reduce action log probabilities to one scalar per sequence."""
    reduced = []
    for log_probs in action_log_probs:
        if log_probs.numel() == 0:
            raise ValueError("DPO preference completions cannot be empty.")
        reduced.append(log_probs.mean() if average else log_probs.sum())
    return torch.stack(reduced)


def score_pointercad_sequences(
    model,
    model_inputs,
    average: bool = False,
    value_temperature: float = 1.0,
    pointer_temperature: float = 1.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    outputs = model(**model_inputs)
    action_log_probs = pointercad_action_log_probs(
        outputs,
        model_inputs["values"],
        model_inputs["pointers"],
        value_temperature=value_temperature,
        pointer_temperature=pointer_temperature,
    )
    return sequence_log_probs(action_log_probs, average=average), action_log_probs

