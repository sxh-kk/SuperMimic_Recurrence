"""The94 loss must isolate a true action response without heldout leakage."""

import copy

import numpy as np
import pytest
import torch

from scripts.local.fit_ollie_phase_response_wm import (
    ARMS,
    paired_commit,
    response_loss,
    train_scales,
)


def test_train_scale_cannot_read_heldout():
    roles = np.repeat(np.arange(6), 4)
    response = np.ones((24, 2)) * 0.01
    train = np.tile([True, True, False, False], 6)
    original = train_scales(response, roles, train)
    response[~train] = 1e20
    assert np.array_equal(original, train_scales(response, roles, train))


def test_scale_floors_keep_zero_response_finite():
    roles = np.arange(6)
    assert np.array_equal(
        train_scales(np.zeros((6, 2)), roles, np.ones(6, dtype=bool)),
        np.tile([0.001, 0.005], (6, 1)),
    )


class SimpleResponse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.3))

    def step(self, state, action):
        updated = state.clone()
        updated[..., 84] = state[..., 84] + self.gain * action[..., 0]
        return updated


def response_fixture():
    model = SimpleResponse()
    state = torch.zeros(6, 92)
    plus, minus = torch.zeros(6, 8, 29), torch.zeros(6, 8, 29)
    plus[:, 0, 0] = 0.03
    minus[:, 0, 0] = -0.03
    role = torch.arange(6)
    target = torch.ones(6, 2) * 0.06
    scale = torch.ones(6, 2) * 0.02
    return model, state, plus, minus, target, role, scale


def test_loss_measures_positive_minus_negative_and_reaches_action_gain():
    model, state, plus, minus, target, role, scale = response_fixture()
    loss = response_loss(model, state, plus, minus, target, role, scale)
    assert torch.allclose(loss, torch.tensor(((0.018 - 0.06) / 0.02) ** 2))
    loss.backward()
    assert torch.isfinite(model.gain.grad) and model.gain.grad < 0


def test_each_phase_has_equal_weight_even_when_repeated():
    model, state, plus, minus, target, role, scale = response_fixture()
    target[0] = 0
    expected = response_loss(model, state, plus, minus, target, role, scale)
    ids = torch.tensor([0] * 30 + [1, 2, 3, 4, 5])
    repeated = response_loss(
        model, state[ids], plus[ids], minus[ids], target[ids], role[ids], scale
    )
    assert torch.allclose(expected, repeated)


def test_equal_actions_cannot_create_effect_even_with_wrong_absolute_velocity():
    model, state, plus, _, _, role, scale = response_fixture()
    state[:, 84] = 20
    loss = response_loss(model, state, plus, plus, torch.zeros(6, 2), role, scale)
    assert loss == 0


def tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tensor_tree_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            tensor_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def test_failed_second_optimizer_commit_restores_both_weights_and_adam_state(monkeypatch):
    models = {name: torch.nn.Linear(2, 1) for name in ARMS}
    optimizers = {name: torch.optim.Adam(models[name].parameters(), lr=0.001) for name in ARMS}
    for name in ARMS:
        models[name](torch.ones(1, 2)).sum().backward()
    paired_commit(models, optimizers)
    for name in ARMS:
        optimizers[name].zero_grad()
        models[name](torch.ones(1, 2)).square().sum().backward()
    before = {
        name: (
            copy.deepcopy(models[name].state_dict()),
            copy.deepcopy(optimizers[name].state_dict()),
        )
        for name in ARMS
    }
    original_step = optimizers[ARMS[1]].step

    def fail_after_mutating():
        original_step()
        raise RuntimeError("synthetic failure after second arm mutation")

    monkeypatch.setattr(optimizers[ARMS[1]], "step", fail_after_mutating)
    with pytest.raises(RuntimeError, match="synthetic"):
        paired_commit(models, optimizers)
    for name in ARMS:
        assert tensor_tree_equal(models[name].state_dict(), before[name][0])
        assert tensor_tree_equal(optimizers[name].state_dict(), before[name][1])
