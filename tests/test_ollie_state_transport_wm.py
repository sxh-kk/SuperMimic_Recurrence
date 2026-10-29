"""Finite different-state transport is not a same-state action intervention."""

import numpy as np
import pytest
import torch

from scripts.local.fit_ollie_state_transport_wm import (
    losses,
    make_values,
    previous,
    transport_loss,
    transport_metrics,
    transport_scales,
    transport_targets,
)


def test_transport_scale_does_not_read_heldout():
    roles = np.repeat(np.arange(6), 4)
    train = np.tile([True, True, False, False], 6)
    target = np.arange(24 * 7).reshape(24, 7) * 0.001
    frozen = transport_scales(target, roles, train)
    target[~train] = 1e10
    assert np.array_equal(frozen, transport_scales(target, roles, train))
    assert np.array_equal(
        transport_scales(np.zeros((6, 7)), np.arange(6), np.ones(6, dtype=bool)),
        np.full((6, 7), 0.001),
    )


def test_targets_are_actual_next_state_differences_and_require_same_later_actions():
    states = np.zeros((2, 9, 92))
    states[0, :, 84] = np.arange(9)
    states[1, :, 84] = -np.arange(9)
    actions = np.zeros((2, 8, 29))
    actions[:, 0, 0] = [0.03, -0.03]
    data = {"state": states, "action": actions}
    pairs = np.array([[0, 1]])
    assert np.array_equal(transport_targets(data, pairs), np.arange(2, 9)[None] * 2)
    actions[1, 3, 0] = 0.01
    with pytest.raises(ValueError, match="identical"):
        transport_targets(data, pairs)


class FakeSpec:
    @staticmethod
    def error_vector(predicted, truth):
        return predicted - truth


class StateResponse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.3))
        self.spec = FakeSpec()
        self.calls = []

    def step(self, state, action):
        self.calls.append((state.detach().clone(), action.detach().clone()))
        predicted = state.clone()
        predicted[:, 84] = state[:, 84] + self.gain * (state[:, 0] + action[:, 0])
        return predicted


def transport_fixture():
    positive = torch.zeros(6, 7, 92)
    negative = positive.clone()
    positive[:, :, 0] = 1
    negative[:, :, 0] = -1
    return (
        StateResponse(),
        positive,
        negative,
        torch.zeros(6, 7, 29),
        torch.ones(6, 7) * 2,
        torch.arange(6),
        torch.ones(6, 7) * 0.5,
    )


def test_transport_uses_different_true_states_equal_actions_without_recursive_rollout():
    model, positive, negative, actions, target, roles, scale = transport_fixture()
    loss = transport_loss(model, positive, negative, actions, target, roles, scale)
    assert len(model.calls) == 2
    assert not torch.equal(model.calls[0][0], model.calls[1][0])
    assert torch.equal(model.calls[0][1], model.calls[1][1])
    assert model.calls[0][0].shape == (42, 92)
    assert torch.allclose(loss, torch.tensor(((0.6 - 2) / 0.5) ** 2))
    loss.backward()
    assert torch.isfinite(model.gain.grad) and model.gain.grad < 0


def test_transport_gives_each_role_and_k_equal_weight():
    model, positive, negative, actions, target, roles, scale = transport_fixture()
    target[0] = torch.arange(7) * 0.1
    original = transport_loss(model, positive, negative, actions, target, roles, scale)
    expected = (((torch.ones(6, 7) * 0.6 - target) / scale) ** 2).mean()
    assert torch.allclose(original, expected)
    ids = torch.tensor([0] * 12 + [1, 2, 3, 4, 5])
    assert torch.allclose(
        original,
        transport_loss(
            model, positive[ids], negative[ids], actions[ids], target[ids], roles[ids], scale
        ),
    )


def test_arm_A_matches_the_frozen96_loss_without_transport_model_calls():
    data = {"state": np.zeros((12, 9, 92)), "action": np.zeros((12, 8, 29))}
    data["action"][::2, 0, 0] = 0.03
    data["action"][1::2, 0, 0] = -0.03
    pairs = np.arange(12).reshape(6, 2)
    roles = np.arange(6)
    response = np.ones((6, 2)) * 0.06
    values = make_values(
        data, pairs, roles, response, np.ones(6) * 0.02, np.zeros((6, 7)), np.ones((6, 7)), "cpu"
    )
    model = StateResponse()
    rows = torch.arange(8)
    pair_rows = torch.arange(6)
    a, b, c = losses(model, values, rows, pair_rows, False)
    assert len(model.calls) == 3 and c == 0
    old_a, old_b = previous.losses(model, values, rows, pair_rows, True)
    assert torch.equal(a, old_a) and torch.equal(b, old_b)
    assert torch.equal(a + 0.25 * b, old_a + 0.25 * old_b)


def test_native_transport_diagnostics_align_h2_through_h8():
    pairs = np.arange(24).reshape(12, 2)
    roles = np.tile(np.arange(6), 2)
    states = np.zeros((24, 9, 92))
    predicted = np.zeros((24, 8, 92))
    for i, (p, n) in enumerate(pairs):
        states[p, :, 84] = (np.arange(9) + 1) * (0.1 + i * 0.01)
        predicted[p, :, 84] = states[p, 1:, 84]
    data = {"state": states, "train_mask": np.arange(24) < 12, "heldout_mask": np.arange(24) >= 12}
    result = transport_metrics(predicted, data, pairs, roles)
    for h in range(2, 9):
        row = result["heldout"]["all"][f"H{h}"]
        assert row["pairs"] == 6 and row["rmse_m_s"] == 0
        assert row["gain_projection"] == 1 and row["sign_agreement"] == 1
    predicted[pairs[6:, 0], 7, 84] *= -1
    result = transport_metrics(predicted, data, pairs, roles)
    assert result["heldout"]["all"]["H8"]["gain_projection"] == -1
