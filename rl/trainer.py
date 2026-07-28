from typing import Any

import dgl
import torch
from trl import DPOTrainer

from rl.log_probs import score_pointercad_sequences


SYSTEM_PROMPT = (
    "You are an expert mechanical engineer. Based on the user's text "
    "requirements, generate the corresponding CAD model design."
)


class PointerCADDPOTrainer(DPOTrainer):
    """TRL DPOTrainer with PointerCAD's structured action likelihood."""

    def __init__(
        self,
        *args,
        pointercad_processor,
        max_seq_len: int,
        value_temperature: float = 1.0,
        pointer_temperature: float = 1.0,
        **kwargs,
    ):
        self.pointercad_processor = pointercad_processor
        self.pointercad_max_seq_len = max_seq_len
        self.value_temperature = value_temperature
        self.pointer_temperature = pointer_temperature
        super().__init__(*args, processing_class=pointercad_processor, **kwargs)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        # Values/pointers are already tokenized structured actions. Let datasets
        # handle storage/shuffling and skip TRL's text-completion tokenization.
        return dataset

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "graph_path",
                "chosen_values",
                "chosen_pointers",
                "rejected_values",
                "rejected_pointers",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        messages = []
        for prompt in prompts:
            messages.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "brep"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {"role": "assistant"},
                ]
            )
        return self.pointercad_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def concatenated_forward(
        self, model, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        batch_size = len(batch["prompt"])
        graphs = dgl.unbatch(batch["graph"])
        paired_graph = dgl.batch(graphs + graphs)
        values = batch["chosen_values"] + batch["rejected_values"]
        pointers = batch["chosen_pointers"] + batch["rejected_pointers"]
        text = self._format_prompts(batch["prompt"] + batch["prompt"])

        model_inputs = self.pointercad_processor(
            text=text,
            breps=paired_graph,
            values=values,
            pointers=pointers,
            max_length=self.pointercad_max_seq_len,
        ).to(self.accelerator.device)

        sequence_logps, action_logps = score_pointercad_sequences(
            model,
            model_inputs,
            average=self.loss_type == "ipo",
            value_temperature=self.value_temperature,
            pointer_temperature=self.pointer_temperature,
        )
        mean_action_logps = torch.stack([logps.mean() for logps in action_logps])
        return {
            "chosen_logps": sequence_logps[:batch_size],
            "rejected_logps": sequence_logps[batch_size:],
            # TRL logs these under logits/*. For PointerCAD the closest useful
            # analogue is the mean joint action log probability.
            "mean_chosen_logits": mean_action_logps[:batch_size].mean(),
            "mean_rejected_logits": mean_action_logps[batch_size:].mean(),
        }
