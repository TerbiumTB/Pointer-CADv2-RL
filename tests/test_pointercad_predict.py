import unittest

import dgl
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from misc import STANDARD_PLANES, TOKEN
from models.pointercad import PointerCAD
from rl.cad_environment import empty_brep_graph


VOCAB_SIZE = 151674
HIDDEN_SIZE = 4
POINTER_SIZE = 2


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.gradient_checkpointing_kwargs = None

    def get_input_embeddings(self):
        return self.embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
    ):
        return BaseModelOutputWithPast(
            last_hidden_state=inputs_embeds,
            past_key_values=((torch.zeros(1), torch.zeros(1)),),
        )


class CountingHead(nn.Module):
    def __init__(self, output_size, selected_indices=()):
        super().__init__()
        self.output_size = output_size
        self.selected_indices = tuple(selected_indices)
        self.calls = 0
        self.batch_sizes = []

    def forward(self, hidden_states):
        self.calls += 1
        self.batch_sizes.append(hidden_states.shape[0])
        result = torch.full(
            (hidden_states.shape[0], self.output_size),
            -100.0,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for index in self.selected_indices:
            result[:, index] = 0.0
        return result


class FakeTokenizer:
    def __len__(self):
        return VOCAB_SIZE

    def decode(self, token_ids):
        return "plan"


def empty_brep_batch(batch_size):
    return dgl.batch([empty_brep_graph() for _ in range(batch_size)])


def lightweight_pointercad(lm_indices=(151643,), label_indices=(0,)):
    model = PointerCAD.__new__(PointerCAD)
    nn.Module.__init__(model)
    model.pointer_size = POINTER_SIZE
    model.dtype = torch.float32
    model.model = FakeBackbone()
    model.lm_head = CountingHead(VOCAB_SIZE, lm_indices)
    model.label_head = CountingHead(len(TOKEN), label_indices)
    model.parameter_head = CountingHead(POINTER_SIZE)
    model.pointer_head = CountingHead(POINTER_SIZE)
    model.label_embedding = nn.Embedding(len(TOKEN), HIDDEN_SIZE)
    model.parameter_projection = nn.Linear(
        POINTER_SIZE, HIDDEN_SIZE, bias=False
    )
    model.pointer_projection = nn.Linear(
        POINTER_SIZE, HIDDEN_SIZE, bias=False
    )
    model.cad_position_embedding = nn.Embedding(2, HIDDEN_SIZE)
    model.standard_plane_pointer = nn.Parameter(
        torch.zeros(len(STANDARD_PLANES), POINTER_SIZE)
    )
    model.parameter_tau = nn.Parameter(torch.tensor(0.0))
    model.pointer_tau = nn.Parameter(torch.tensor(0.0))
    return model


class BatchedPredictTest(unittest.TestCase):
    def test_gradient_checkpointing_uses_non_reentrant_mode(self):
        model = lightweight_pointercad()

        model.enable_gradient_checkpointing()

        self.assertEqual(
            model.model.gradient_checkpointing_kwargs,
            {"use_reentrant": False},
        )

    def test_forward_projects_only_requested_lm_positions(self):
        input_ids = torch.tensor(
            [[42, 43, 44, 45], [46, 47, 48, 49]]
        )
        attention_mask = torch.ones_like(input_ids)

        default_model = lightweight_pointercad()
        default_logits = default_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            breps=empty_brep_batch(2),
            parameter_maps=None,
        )[0]
        self.assertEqual(default_model.lm_head.batch_sizes, [6])
        self.assertEqual([value.shape[0] for value in default_logits], [3, 3])

        masked_model = lightweight_pointercad()
        lm_logits_mask = torch.tensor(
            [[False, True, False, False], [True, False, True, False]]
        )
        masked_logits = masked_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            breps=empty_brep_batch(2),
            parameter_maps=None,
            lm_logits_mask=lm_logits_mask,
        )[0]
        self.assertEqual(masked_model.lm_head.batch_sizes, [3])
        self.assertEqual([value.shape[0] for value in masked_logits], [1, 2])

    def test_forward_anchor_connects_skipped_conditional_modules(self):
        model = lightweight_pointercad()
        model.brep = nn.Linear(2, 2)
        model.parameter = nn.Linear(2, 2)
        input_ids = torch.tensor([[42, 43]])
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            breps=empty_brep_batch(1),
            parameter_maps=None,
            ensure_conditional_parameter_usage=True,
        )

        (outputs[5] * 0.0).backward()

        conditional_modules = (
            model.brep,
            model.parameter,
            model.parameter_projection,
            model.pointer_projection,
        )
        for module in conditional_modules:
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_mixed_lm_and_structured_rows_preserve_order(self):
        model_end = TOKEN.index("<|model_end|>")
        model = lightweight_pointercad(label_indices=(model_end,))
        input_ids = torch.tensor([[42], [151671]])

        outputs = model.predict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            breps=empty_brep_batch(2),
            tokenizer=FakeTokenizer(),
            max_steps=1,
            mode="argmax",
            return_log_probs=True,
        )

        generated_ids, _, labels, _, _, behavior = outputs
        self.assertEqual(model.lm_head.batch_sizes, [1])
        self.assertEqual(model.label_head.batch_sizes, [1])
        self.assertEqual(generated_ids.tolist(), [[151643], [151673]])
        self.assertEqual(labels[0].tolist(), [])
        self.assertEqual(labels[1].tolist(), [model_end])
        self.assertEqual(len(behavior[0]["plan"]), 1)
        self.assertEqual(len(behavior[1]["label"]), 1)

    def test_lm_head_runs_once_for_the_active_batch(self):
        model = lightweight_pointercad()
        input_ids = torch.tensor([[42], [43]])

        outputs = model.predict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            breps=empty_brep_batch(2),
            tokenizer=FakeTokenizer(),
            max_steps=1,
            mode="argmax",
            return_log_probs=True,
        )

        generated_ids, _, labels, _, _, behavior = outputs
        self.assertEqual(model.lm_head.calls, 1)
        self.assertEqual(model.lm_head.batch_sizes, [2])
        self.assertEqual(generated_ids.tolist(), [[151643], [151643]])
        self.assertEqual([value.numel() for value in labels], [0, 0])
        self.assertEqual(
            [len(value["plan"]) for value in behavior], [1, 1]
        )
        self.assertIsInstance(behavior[0]["plan"][0], float)
        self.assertAlmostEqual(behavior[0]["plan"][0], 0.0, places=5)

    def test_structured_heads_run_once_for_the_active_batch(self):
        model_end = TOKEN.index("<|model_end|>")
        model = lightweight_pointercad(label_indices=(model_end,))
        input_ids = torch.tensor([[151671], [151671]])

        outputs = model.predict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            breps=empty_brep_batch(2),
            tokenizer=FakeTokenizer(),
            max_steps=1,
            mode="argmax",
            return_log_probs=True,
        )

        _, _, labels, _, _, behavior = outputs
        self.assertEqual(model.label_head.batch_sizes, [2])
        self.assertEqual(model.parameter_head.batch_sizes, [2])
        self.assertEqual(model.pointer_head.batch_sizes, [2])
        self.assertEqual([value.tolist() for value in labels], [[model_end]] * 2)
        self.assertEqual(
            [len(value["label"]) for value in behavior], [1, 1]
        )

    def test_sampling_generators_are_schedule_independent(self):
        batched_model = lightweight_pointercad(lm_indices=(10, 11))
        batch_generators = [torch.Generator(), torch.Generator()]
        batch_generators[0].manual_seed(7)
        batch_generators[1].manual_seed(19)
        input_ids = torch.tensor([[42], [43]])
        batched_ids = batched_model.predict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            breps=empty_brep_batch(2),
            tokenizer=FakeTokenizer(),
            max_steps=1,
            mode="sample",
            sampling_generators=batch_generators,
        )[0]

        independent_ids = []
        for token_id, seed in ((42, 7), (43, 19)):
            single_model = lightweight_pointercad(lm_indices=(10, 11))
            generator = torch.Generator().manual_seed(seed)
            generated = single_model.predict(
                input_ids=torch.tensor([[token_id]]),
                attention_mask=torch.ones((1, 1), dtype=torch.long),
                breps=empty_brep_batch(1),
                tokenizer=FakeTokenizer(),
                max_steps=1,
                mode="sample",
                sampling_generators=[generator],
            )[0]
            independent_ids.append(generated[0, 0].item())

        self.assertEqual(
            batched_ids[:, 0].tolist(), independent_ids
        )


if __name__ == "__main__":
    unittest.main()
