import torch

from rl.likelihood import (
    ScoringTemperatures,
    _build_replay_token_batch,
    _completion_lm_logit_mask,
    _completion_lm_log_prob,
    _replay_completion_token_ids,
    _selected_completion_lm_log_prob,
    _structured_step_log_probs,
)


def test_replay_completion_inserts_structured_actions_after_cad_start():
    stored_ids = [20, 21, 30, 40, 41]

    replay_ids = _replay_completion_token_ids(
        stored_lm_token_ids=stored_ids,
        num_cad_actions=3,
        cad_start_id=30,
        cad_pad_id=31,
    )

    assert replay_ids == [20, 21, 30, 31, 31, 31, 40, 41]


def test_replay_token_batch_uses_exact_ids_without_retokenizing_plan_text():
    prompt_input_ids = torch.tensor(
        [
            [0, 0, 10, 11],
            [12, 13, 14, 15],
        ]
    )
    prompt_attention_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ]
    )

    input_ids, attention_mask = _build_replay_token_batch(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        stored_lm_token_ids=[
            [20, 30, 40, 41],
            [21, 30, 42],
        ],
        cad_action_counts=[2, 1],
        cad_start_id=30,
        cad_pad_id=31,
        pad_token_id=0,
        padding_side="left",
        max_length=16,
    )

    assert input_ids.tolist() == [
        [10, 11, 20, 30, 31, 31, 40, 41],
        [12, 13, 14, 15, 21, 30, 31, 42],
    ]
    assert attention_mask.tolist() == [[1] * 8, [1] * 8]


def test_completion_log_prob_scores_stored_lm_ids_and_skips_cad_slots():
    # Prompt: [10, 11]. Completion: [20, cad_start, cad_end, im_end], with
    # two structured CAD slots between cad_start and cad_end.
    input_ids = torch.tensor([10, 11, 20, 30, 31, 31, 32, 33])
    attention_mask = torch.ones_like(input_ids)
    special_ids = {31}
    lm_token_count = input_ids.numel() - 2
    logits = torch.zeros((lm_token_count - 1, 64), dtype=torch.float32)

    result = _completion_lm_log_prob(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lm_token_count=2,
        special_ids=special_ids,
        temperature=1.0,
        expected_token_ids=[20, 30, 32, 33],
    )

    assert torch.allclose(result, -4.0 * torch.log(torch.tensor(64.0)))


def test_completion_log_prob_matches_full_log_softmax_value_and_gradient():
    torch.manual_seed(7)
    input_ids = torch.tensor([2, 3, 5, 7, 11, 13, 17, 19])
    attention_mask = torch.ones_like(input_ids)
    special_ids = {11}
    logits = torch.randn((6, 32), dtype=torch.float32, requires_grad=True)
    expected_token_ids = [5, 7, 13, 17, 19]

    actual = _completion_lm_log_prob(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lm_token_count=2,
        special_ids=special_ids,
        temperature=0.7,
        expected_token_ids=expected_token_ids,
    )
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]

    targets = input_ids[torch.tensor([1, 2, 3, 5, 6, 7])]
    expected = torch.log_softmax(logits / 0.7, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)[1:].sum()
    expected_gradient = torch.autograd.grad(expected, logits)[0]

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6
    )


def test_completion_only_projection_matches_full_replay_value_and_gradient():
    # Left padding and CAD slots are deliberately included because neither is
    # an LM target, while cad_start/cad_end remain regular LM tokens.
    input_ids = torch.tensor([[0, 0, 10, 11, 20, 30, 31, 31, 32, 33]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    special_ids = {31}
    expected_token_ids = [20, 30, 32, 33]
    full_lm_mask = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        full_lm_mask &= input_ids != token_id
    shifted_lm_mask = torch.cat(
        [full_lm_mask[:, 1:], torch.zeros_like(full_lm_mask[:, :1])], dim=1
    )

    torch.manual_seed(11)
    full_logits = torch.randn(
        (int(shifted_lm_mask.sum().item()), 64), requires_grad=True
    )
    full_value = _completion_lm_log_prob(
        logits=full_logits,
        input_ids=input_ids[0],
        attention_mask=attention_mask[0],
        prompt_lm_token_count=2,
        special_ids=special_ids,
        temperature=0.8,
        expected_token_ids=expected_token_ids,
    )
    full_gradient = torch.autograd.grad(
        full_value, full_logits, retain_graph=True
    )[0]

    completion_mask = _completion_lm_logit_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lm_token_counts=[2],
        special_ids=special_ids,
    )
    selection_in_full_logits = completion_mask[shifted_lm_mask]
    selected_logits = full_logits[selection_in_full_logits]
    selected_value = _selected_completion_lm_log_prob(
        logits=selected_logits,
        expected_token_ids=expected_token_ids,
        temperature=0.8,
    )
    selected_gradient = torch.autograd.grad(selected_value, full_logits)[0]

    assert completion_mask.nonzero().squeeze(1).tolist() == [[0, 3], [0, 4], [0, 7], [0, 8]]
    assert torch.allclose(selected_value, full_value, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        selected_gradient, full_gradient, atol=1e-6, rtol=1e-6
    )


def test_structured_log_probs_ignore_unsampled_trailing_prediction():
    stored_count = 2
    prediction_count = stored_count + 1
    embedding_size = 4

    label_logp, parameter_logp, pointer_logp = _structured_step_log_probs(
        pred_labels=torch.zeros((prediction_count, 16)),
        pred_parameters=torch.zeros((prediction_count, embedding_size)),
        pred_pointers=torch.zeros((prediction_count, embedding_size)),
        labels=torch.zeros(stored_count, dtype=torch.long),
        parameters=torch.zeros(stored_count, dtype=torch.long),
        pointers=torch.full((stored_count,), -100, dtype=torch.long),
        parameter_candidates={
            "length": torch.zeros((0, embedding_size)),
            "angle": torch.zeros((0, embedding_size)),
        },
        parameter_tau=torch.tensor(1.0),
        curve_candidates=torch.zeros((0, embedding_size)),
        surface_candidates=torch.zeros((0, embedding_size)),
        pointer_tau=torch.tensor(1.0),
        standard_plane_candidates=torch.zeros((3, embedding_size)),
        temperatures=ScoringTemperatures(),
        allow_trailing_prediction=True,
    )

    assert torch.allclose(label_logp, -2.0 * torch.log(torch.tensor(16.0)))
    assert parameter_logp.item() == 0.0
    assert pointer_logp.item() == 0.0


def test_unused_structured_candidates_remain_connected_to_zero_gradient():
    embedding_size = 4
    pred_labels = torch.zeros((1, 16), requires_grad=True)
    pred_parameters = torch.zeros((1, embedding_size), requires_grad=True)
    pred_pointers = torch.zeros((1, embedding_size), requires_grad=True)
    length_candidates = torch.randn((2, embedding_size), requires_grad=True)
    angle_candidates = torch.randn((1, embedding_size), requires_grad=True)
    curve_candidates = torch.randn((2, embedding_size), requires_grad=True)
    surface_candidates = torch.randn((1, embedding_size), requires_grad=True)
    standard_planes = torch.randn((3, embedding_size), requires_grad=True)
    parameter_tau = torch.tensor(1.0, requires_grad=True)
    pointer_tau = torch.tensor(1.0, requires_grad=True)

    totals = _structured_step_log_probs(
        pred_labels=pred_labels,
        pred_parameters=pred_parameters,
        pred_pointers=pred_pointers,
        labels=torch.zeros(1, dtype=torch.long),
        parameters=torch.zeros(1, dtype=torch.long),
        pointers=torch.full((1,), -100, dtype=torch.long),
        parameter_candidates={
            "length": length_candidates,
            "angle": angle_candidates,
        },
        parameter_tau=parameter_tau,
        curve_candidates=curve_candidates,
        surface_candidates=surface_candidates,
        pointer_tau=pointer_tau,
        standard_plane_candidates=standard_planes,
        temperatures=ScoringTemperatures(),
    )
    sum(totals).backward()

    conditionally_used = (
        pred_parameters,
        pred_pointers,
        length_candidates,
        angle_candidates,
        curve_candidates,
        surface_candidates,
        standard_planes,
        parameter_tau,
        pointer_tau,
    )
    for value in conditionally_used:
        assert value.grad is not None
        assert torch.count_nonzero(value.grad).item() == 0
