"""Joint-state model and residual goal policy for the Ollie research extension.

These networks are independent of the flat-ground checkpoint's incompatible
input/output contract. Fitting this model alone does not train an Ollie policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from supermimic.ollie.state import JointStateLayout, JointStateSpec
from supermimic.world_model.dynamics import MLP, DynamicsMLP


@dataclass
class JointLearningConfig:
    hidden: tuple[int, ...] = (512, 512, 512)
    policy_hidden: tuple[int, ...] = (512, 256)
    model_lr: float = 2e-4
    policy_lr: float = 1e-4
    residual_limit_rad: float = 0.25
    residual_limits_rad: tuple[float, ...] | None = None
    reference_offsets: tuple[int, ...] = (0, 5, 10)


class JointWorldModel(nn.Module):
    def __init__(self, layout: JointStateLayout, config: JointLearningConfig | None = None, device="cpu"):
        super().__init__()
        self.cfg = config or JointLearningConfig()
        self.spec = JointStateSpec(layout, device=device)
        self.layout = layout
        # Fixed scene for the first pilot; no domain-randomization xi is assumed.
        self.dynamics = DynamicsMLP(
            feat_dim=layout.feature_dim, act_dim=29, xi_dim=0,
            hidden=self.cfg.hidden, out_dim=layout.delta_dim,
        )
        self.to(device)

    def step(self, state: torch.Tensor, joint_target: torch.Tensor) -> torch.Tensor:
        features = self.spec.canonical(state)
        context = state.new_empty(*state.shape[:-1], 0)
        delta = self.dynamics(features, joint_target, context)
        return self.spec.integrate(state, delta)

    def window_loss(self, states: torch.Tensor, actions: torch.Tensor, wheel_weight: float = 1.0):
        if len(states) != len(actions) + 1:
            raise ValueError("window needs k+1 states and k actions")
        if not 0 <= wheel_weight <= 1:
            raise ValueError("wheel loss weight must be within [0,1]")
        predicted = states[0]
        errors, raw_errors, board_errors, board_root_errors, robot_root_errors = [], [], [], [], []
        for step, action in enumerate(actions):
            predicted = self.step(predicted, action)
            vector = self.spec.error_vector(predicted, states[step + 1])
            weights = vector.new_ones(vector.shape[-1])
            if self.layout.board_joints:
                weights[-2 * self.layout.board_joints:] = wheel_weight
            errors.append((vector.square() * weights).sum(-1).mean() / weights.sum())
            raw_errors.append(vector.square().mean())
            board_errors.append(vector[..., 70:].square().mean())
            board_root_errors.append(vector[..., 70:82].square().mean())
            robot_root_errors.append(vector[..., :6].square().mean())
        values = torch.stack(errors)
        return values.mean(), {
            "nse_1step": raw_errors[0].detach(), "nse_kstep": raw_errors[-1].detach(),
            "board_nse_kstep": board_errors[-1].detach(),
            "board_root_nse_kstep": board_root_errors[-1].detach(),
            "robot_root_pose_nse_kstep": robot_root_errors[-1].detach(),
        }


class JointResidualPolicy(nn.Module):
    """A goal encoder plus state decoder, with bounded robot-only PD residuals.

Full simulator state is available in this prototype. This is not the paper's
930/680-dimensional sensing setup. Reference wheel motion is unobserved and is
masked from the tracking goal; actual wheel motion remains a dynamics input.
"""

    def __init__(self, layout: JointStateLayout, config: JointLearningConfig | None = None, device="cpu"):
        super().__init__()
        self.cfg = config or JointLearningConfig()
        self.layout = layout
        self.spec = JointStateSpec(layout, device=device)
        limits = ((self.cfg.residual_limit_rad,) * 29 if self.cfg.residual_limits_rad is None
                  else self.cfg.residual_limits_rad)
        if len(limits) != 29 or any(not 0 < value < float("inf") for value in limits):
            raise ValueError("policy requires 29 finite positive residual limits")
        self.register_buffer("residual_limits", torch.tensor(limits, dtype=torch.float32, device=device),
                             persistent=False)
        self.encoder = MLP(layout.delta_dim * len(self.cfg.reference_offsets), (256, 128), 32)
        self.decoder = MLP(layout.feature_dim + 32, self.cfg.policy_hidden, 29, zero_last=True)
        self.to(device)

    def forward(self, state: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        if references.shape[-2:] != (len(self.cfg.reference_offsets), self.layout.state_dim):
            raise ValueError("future references must match the policy goal contract")
        # Quaternion primitives require identical batch shapes, not implicit broadcast.
        errors = self.spec.error_vector(references, state.unsqueeze(-2).expand_as(references))
        if self.layout.board_joints:
            errors = torch.cat([
                errors[..., :-2 * self.layout.board_joints],
                torch.zeros_like(errors[..., -2 * self.layout.board_joints:]),
            ], dim=-1)
        latent = torch.tanh(self.encoder(errors.clamp(-20, 20).flatten(-2)))
        residual = torch.tanh(self.decoder(torch.cat([self.spec.canonical(state), latent], -1)))
        return references[..., 0, 7:36] + self.residual_limits * residual


def tracking_loss(
    spec: JointStateSpec, state: torch.Tensor, target: torch.Tensor,
    board_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable engineering tracking objective; no fictitious contact force.

Reference wheel angles/speeds are never treated as observations. Real simulation
is still required for contact timing, success, penetration, and energy checks.
"""
    error = spec.error_vector(state, target)
    robot = error[..., :70]
    board_root = error[..., 70:82]
    confidence = 1.0 if board_confidence is None else board_confidence.clamp(0.1, 1)
    return 0.5 * robot.square().mean(-1) + confidence * board_root.square().mean(-1)
