"""Supervised contact auxiliaries for joint dynamics; no policy or force injection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from supermimic.kinematics.rotations import quat_apply, quat_mul, quat_normalize, rotvec_from_quat
from supermimic.ollie.learning import JointLearningConfig, JointWorldModel
from supermimic.ollie.state import JointStateLayout
from supermimic.world_model.state_spec import quat_apply_inverse_quat

SCHEMA = "supermimic.ollie.contact_aux_wm.v1"
RESPONSE_NAMES = tuple(
    f"{entity}_{kind}_{axis}"
    for entity in ("robot", "board")
    for kind in ("position_m", "rotation_rad", "velocity_m_s", "angular_world_rad_s")
    for axis in "xyz"
)
FORCE_NAMES = (
    "left_foot_deck_upward_support_n",
    "right_foot_deck_upward_support_n",
    "front_nose_board_fz_n",
    "tail_ground_normal_force_n",
)
GEOMETRY_NAMES = (
    "board_center_clearance_m",
    "board_tail_clearance_m",
    "board_nose_clearance_m",
    "board_front_axle_clearance_m",
    "board_rear_axle_clearance_m",
    "wheel_front_left_clearance_m",
    "wheel_front_right_clearance_m",
    "wheel_rear_left_clearance_m",
    "wheel_rear_right_clearance_m",
)
FLAG_NAMES = (
    "tail_ground",
    "nose_ground",
    "left_foot_board",
    "right_foot_board",
    "left_foot_ground",
    "right_foot_ground",
    "any_robot_ground",
    "any_board_ground",
)
AUX_NAMES = FORCE_NAMES + GEOMETRY_NAMES + ("system_com_vz_m_s",) + FLAG_NAMES


@dataclass(frozen=True)
class ContactFitConfig:
    seed: int = 73
    updates: int = 500
    batch_size: int = 64
    learning_rate: float = 2e-4
    wheel_weight: float = 0.05
    paired_weight: float = 1.0
    auxiliary_weight: float = 0.1
    wall_seconds: float = 240.0
    heldout_anchors: tuple[int, ...] = (146, 160)
    horizons: tuple[int, ...] = (1, 8)
    hidden: tuple[int, ...] = (512, 512, 512)


class ContactWorldModel(JointWorldModel):
    """The unchanged DynamicsMLP state head and one shared-trunk auxiliary head.

    Aux outputs describe all eight post-forward samples of the next transition.
    They never enter the 92D state, integration, simulator, or action.
    """

    def __init__(self, layout: JointStateLayout, config=None, device="cpu"):
        super().__init__(layout, config or JointLearningConfig(), device)
        self.contact_head = nn.Linear(self.cfg.hidden[-1], 8 * len(AUX_NAMES)).to(device)

    def step_with_aux(self, state, action):
        latent = torch.cat((self.spec.canonical(state), action), dim=-1)
        for layer in self.dynamics.mlp.net[:-1]:
            latent = layer(latent)
        delta = self.dynamics.mlp.net[-1](latent)
        return self.spec.integrate(state, delta), self.contact_head(latent).reshape(
            *state.shape[:-1], 8, len(AUX_NAMES)
        )

    def step(self, state, joint_target):
        # Preserve the original state-only forward contract exactly.
        return super().step(state, joint_target)


def auxiliary_targets(metrics, metric_names):
    """Fixed SI transforms; no fitting scales to train or heldout values."""
    indices = [tuple(metric_names).index(name) for name in AUX_NAMES]
    target = metrics[..., indices].clone()
    target[..., :4] = torch.asinh(target[..., :4] / 100.0)
    target[..., 4:13] /= 0.05
    # system vz / 1 m/s, flags raw 0/1. MSE is used for all fixed outputs.
    return target


def root_response(positive, negative):
    """Same 24 SI world-root response features used by experiment 42."""
    result = []
    for pos, quat, vel, ang in (
        (slice(0, 3), slice(3, 7), slice(36, 39), slice(39, 42)),
        (slice(71, 74), slice(74, 78), slice(82, 85), slice(85, 88)),
    ):
        qp, qn = quat_normalize(positive[..., quat]), quat_normalize(negative[..., quat])
        result.extend(
            (
                positive[..., pos] - negative[..., pos],
                rotvec_from_quat(quat_mul(qp, quat_apply_inverse_quat(qn))),
                positive[..., vel] - negative[..., vel],
                quat_apply(qp, positive[..., ang]) - quat_apply(qn, negative[..., ang]),
            )
        )
    return torch.cat(result, -1)


def response_scales(device="cpu"):
    value = torch.tensor(
        [0.001] * 3
        + [0.005] * 3
        + [0.05] * 3
        + [0.1] * 3
        + [0.001] * 3
        + [0.005] * 3
        + [0.01] * 3
        + [0.05] * 3,
        device=device,
    ).repeat(2, 1)
    value[0, 20], value[1, 20] = 1e-4, 0.01
    return value


def rollout(model, states0, actions, *, auxiliary=False):
    state, predictions, auxiliaries = states0, [], []
    for action in actions.unbind(1):
        if auxiliary:
            state, aux = model.step_with_aux(state, action)
            auxiliaries.append(aux)
        else:
            state = model.step(state, action)
        predictions.append(state)
    return torch.stack(predictions, 1), torch.stack(auxiliaries, 1) if auxiliary else None


def training_losses(model, states, actions, aux_target, pair_actions, pair_target, config):
    predicted, auxiliary = rollout(model, states[:, 0], actions, auxiliary=True)
    error = model.spec.error_vector(predicted, states[:, 1:])
    weights = error.new_ones(error.shape[-1])
    weights[-8:] = config.wheel_weight
    state_loss = (error.square() * weights).sum(-1).mean() / weights.sum()
    plus, _ = rollout(model, pair_actions[0], pair_actions[1])
    minus, _ = rollout(model, pair_actions[0], pair_actions[2])
    response = root_response(plus[:, [0, 7]], minus[:, [0, 7]])
    pair_loss = ((response - pair_target) / response_scales(states.device)).square().mean()
    aux_loss = (auxiliary - aux_target).square().mean()
    return state_loss, pair_loss, aux_loss
