from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from rl.dpo_data import DPOCompletion, DPOPair
from rl.likelihood import (
    ScoringTemperatures,
    TrajectoryLogProbs,
    score_trajectories,
)


@dataclass
class DPOLossOutput:
    loss: torch.Tensor
    per_example_loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def dpo_loss(
    policy_preferred_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_preferred_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")
    if label_smoothing < 0 or label_smoothing >= 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5).")

    policy_margin = policy_preferred_logps - policy_rejected_logps
    reference_margin = reference_preferred_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(logits)
        + label_smoothing * F.logsigmoid(-logits)
    )
    preferred_rewards = beta * (
        policy_preferred_logps - reference_preferred_logps
    ).detach()
    rejected_rewards = beta * (
        policy_rejected_logps - reference_rejected_logps
    ).detach()
    metrics = {
        "loss": losses.detach().mean(),
        "reward_accuracy": (
            preferred_rewards > rejected_rewards
        ).float().mean(),
        "reward_margin": (preferred_rewards - rejected_rewards).mean(),
        "preferred_reward": preferred_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "policy_preferred_logp": policy_preferred_logps.detach().mean(),
        "policy_rejected_logp": policy_rejected_logps.detach().mean(),
        "reference_preferred_logp": reference_preferred_logps.detach().mean(),
        "reference_rejected_logp": reference_rejected_logps.detach().mean(),
    }
    return DPOLossOutput(
        loss=losses.mean(),
        per_example_loss=losses,
        metrics=metrics,
    )


class PointerCADDPO:
    def __init__(
        self,
        processor,
        rollout_root: str,
        max_replay_length: int,
        max_input_length: int,
        beta: float,
        label_smoothing: float,
        temperatures: ScoringTemperatures,
        reference_model=None,
    ):
        self.processor = processor
        self.rollout_root = rollout_root
        self.max_replay_length = max_replay_length
        self.max_input_length = max_input_length
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.temperatures = temperatures
        self.reference_model = reference_model

    @staticmethod
    def _ordered_completions(
        pairs: Sequence[DPOPair],
    ) -> List[DPOCompletion]:
        return [pair.preferred for pair in pairs] + [
            pair.rejected for pair in pairs
        ]

    @staticmethod
    def _cached_reference_logps(
        pairs: Sequence[DPOPair], device
    ) -> torch.Tensor:
        completions = PointerCADDPO._ordered_completions(pairs)
        values = []
        for completion in completions:
            if completion.reference_logp is None:
                raise ValueError(
                    "Cached-reference mode requires a score for every trajectory."
                )
            values.append(completion.reference_logp)
        return torch.tensor(values, dtype=torch.float32, device=device)

    def _score(
        self, model, completions: Sequence[DPOCompletion], device
    ) -> TrajectoryLogProbs:
        return score_trajectories(
            model=model,
            processor=self.processor,
            completions=completions,
            rollout_root=self.rollout_root,
            device=device,
            max_replay_length=self.max_replay_length,
            max_input_length=self.max_input_length,
            temperatures=self.temperatures,
        )

    def compute(self, policy_model, pairs: Sequence[DPOPair], device):
        if not pairs:
            raise ValueError("DPO batch cannot be empty.")
        completions = self._ordered_completions(pairs)
        pair_count = len(pairs)
        policy = self._score(policy_model, completions, device)

        if self.reference_model is None:
            reference_total = self._cached_reference_logps(pairs, device)
        else:
            with torch.no_grad():
                reference_total = self._score(
                    self.reference_model, completions, device
                ).total

        output = dpo_loss(
            policy_preferred_logps=policy.total[:pair_count],
            policy_rejected_logps=policy.total[pair_count:],
            reference_preferred_logps=reference_total[:pair_count],
            reference_rejected_logps=reference_total[pair_count:],
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
        output.metrics.update(
            {
                "plan_logp": policy.plan.detach().mean(),
                "label_logp": policy.label.detach().mean(),
                "parameter_logp": policy.parameter.detach().mean(),
                "pointer_logp": policy.pointer.detach().mean(),
            }
        )
        return output
