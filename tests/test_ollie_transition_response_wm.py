"""True intermediate inputs and strictly H1 response are distinct contracts."""

import copy

import numpy as np
import pytest
import torch

from scripts.local.fit_ollie_transition_response_wm import (
    ARMS,
    h1_response_loss,
    input_coverage,
    make_values,
    paired_commit,
    transition_loss,
)


class DummySpec:
    @staticmethod
    def error_vector(predicted, truth):
        return predicted - truth


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.3))
        self.spec = DummySpec()
        self.inputs = []

    def step(self, state, action):
        self.inputs.append(state.detach().clone())
        next_state = state.clone()
        next_state[..., 84] = state[..., 84] + self.gain * action[..., 0]
        return next_state


def test_tf1_receives_actual_intermediate_states_not_only_anchor():
    states = np.zeros((2, 9, 92))
    states[:, :, 0] = np.arange(9)[None] + np.arange(2)[:, None] * 10
    data = {"state": states, "action": np.zeros((2, 8, 29))}
    v = make_values(data, np.array([[0, 1]]), np.array([0]), np.zeros((1, 2)), np.ones(6), "cpu")
    rows = torch.tensor([1, 7, 8, 15])
    model = CaptureModel()
    loss = transition_loss(
        model, v["current"][rows], v["transition_actions"][rows], v["next"][rows]
    )
    assert torch.equal(model.inputs[0][:, 0], torch.tensor([1.0, 7.0, 10.0, 17.0]))
    assert len(model.inputs) == 1 and torch.isfinite(loss)


def test_transition_alignment_is_real_k_to_real_kplus1():
    states = np.arange(2 * 9 * 92).reshape(2, 9, 92).astype(float)
    actions = np.arange(2 * 8 * 29).reshape(2, 8, 29).astype(float)
    v = make_values(
        {"state": states, "action": actions},
        np.array([[0, 1]]),
        np.array([0]),
        np.zeros((1, 2)),
        np.ones(6),
        "cpu",
    )
    for branch in range(2):
        for k in range(8):
            row = branch * 8 + k
            assert np.array_equal(v["current"][row].numpy(), states[branch, k])
            assert np.array_equal(v["next"][row].numpy(), states[branch, k + 1])
            assert np.array_equal(v["transition_actions"][row].numpy(), actions[branch, k])


def test_h1_response_has_exactly_two_one_step_model_calls_and_finite_gain_gradient():
    model = CaptureModel()
    s = torch.zeros(6, 92)
    plus = torch.zeros(6, 29)
    minus = plus.clone()
    plus[:, 0] = 0.03
    minus[:, 0] = -0.03
    loss = h1_response_loss(
        model, s, plus, minus, torch.full((6,), 0.06), torch.arange(6), torch.full((6,), 0.02)
    )
    assert len(model.inputs) == 2 and torch.equal(model.inputs[0], model.inputs[1])
    assert torch.allclose(loss, torch.tensor(((0.018 - 0.06) / 0.02) ** 2))
    loss.backward()
    assert torch.isfinite(model.gain.grad) and model.gain.grad < 0


def test_h1_phase_balance_does_not_weight_one_role_by_its_sample_count():
    model = CaptureModel()
    s = torch.zeros(6, 92)
    plus = torch.zeros(6, 29)
    minus = plus.clone()
    plus[:, 0] = 0.03
    minus[:, 0] = -0.03
    target = torch.ones(6) * 0.06
    target[0] = 0
    role = torch.arange(6)
    scale = torch.ones(6) * 0.02
    original = h1_response_loss(model, s, plus, minus, target, role, scale)
    rows = torch.tensor([0] * 20 + [1, 2, 3, 4, 5])
    repeated = h1_response_loss(
        model, s[rows], plus[rows], minus[rows], target[rows], role[rows], scale
    )
    assert torch.allclose(original, repeated)


def test_input_coverage_counts_intermediate_real_states_without_heldout():
    states = np.zeros((3, 9, 92))
    states[:, :, 0] = np.arange(9) + np.arange(3)[:, None] * 10
    data = {"state": states, "train_mask": np.array([True, False, True])}
    rows = np.r_[np.arange(8), np.arange(16, 24)]
    result = input_coverage(data, rows)
    assert result["unique_real_transition_rows"] == 16
    assert result["unique_real_state92_inputs"] == 16
    assert result["unique_row_counts_by_step"] == [2] * 8
    assert result["all_sampled_rows_are_training"]


def test_renamed_arms_keep_transactional_rollback(monkeypatch):
    models = {arm: torch.nn.Linear(2, 1) for arm in ARMS}
    optimizers = {arm: torch.optim.Adam(models[arm].parameters(), lr=0.01) for arm in ARMS}
    for arm in ARMS:
        models[arm](torch.ones(1, 2)).sum().backward()
    before = {arm: copy.deepcopy(models[arm].state_dict()) for arm in ARMS}
    original = optimizers[ARMS[1]].step

    def fail_after_mutation():
        original()
        raise RuntimeError("synthetic96 second arm failure")

    monkeypatch.setattr(optimizers[ARMS[1]], "step", fail_after_mutation)
    with pytest.raises(RuntimeError, match="synthetic96"):
        paired_commit(models, optimizers)
    for arm in ARMS:
        assert all(
            torch.equal(value, before[arm][key]) for key, value in models[arm].state_dict().items()
        )
        assert not optimizers[arm].state_dict()["state"]
