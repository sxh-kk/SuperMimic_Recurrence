"""Versioned robot + passive-board state primitives, separate from the 71-d baseline.

Entity layout: [position_w(3), quaternion_wxyz(4), joint_position(K),
                linear_velocity_w(3), angular_velocity_body(3), joint_velocity(K)].
The robot uses K=29 (71 values), followed by the board entity with arbitrary K.
Both positions are relative to the same environment origin. Actions remain robot-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg, quat_apply_inverse_quat

STATE_SCHEMA = "supermimic.ollie.entity_state.v1"
ROBOT_STATE_DIM = 71
ROBOT_DELTA_DIM = 70


@dataclass(frozen=True)
class JointStateLayout:
    board_joint_names: tuple[str, ...] = ()

    def __post_init__(self):
        if len(set(self.board_joint_names)) != len(self.board_joint_names):
            raise ValueError("board joint names must be unique")
        if any(not isinstance(name, str) or not name for name in self.board_joint_names):
            raise ValueError("board joint names must be nonempty strings")

    @property
    def board_joints(self) -> int:
        return len(self.board_joint_names)

    @property
    def state_dim(self) -> int:
        return 84 + 2 * self.board_joints

    @property
    def delta_dim(self) -> int:
        return 82 + 2 * self.board_joints

    @property
    def feature_dim(self) -> int:
        return 86 + 2 * self.board_joints

    def split(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state dimension {self.state_dim}, got {state.shape[-1]}")
        return state[..., :ROBOT_STATE_DIM], state[..., ROBOT_STATE_DIM:]

    def split_board(self, board: torch.Tensor) -> tuple[torch.Tensor, ...]:
        k = self.board_joints
        if board.shape[-1] != 13 + 2 * k:
            raise ValueError("board state dimension does not match the layout")
        return (
            board[..., :3], board[..., 3:7], board[..., 7:7 + k],
            board[..., 7 + k:10 + k], board[..., 10 + k:13 + k], board[..., 13 + k:],
        )


@dataclass(frozen=True)
class JointStateIndices:
    """Map by MuJoCo joint addresses, independent of entity ordering in the scene."""

    layout: JointStateLayout
    robot_qpos: tuple[int, ...]
    robot_qvel: tuple[int, ...]
    board_qpos: tuple[int, ...]
    board_qvel: tuple[int, ...]
    nq: int
    nv: int

    @classmethod
    def from_model(
        cls, model, *, robot_free_joint: str = "floating_base_joint",
        board_free_joint: str = "board_free",
        robot_joint_names: Sequence[str] = G1_JOINT_NAMES,
        board_joint_names: Sequence[str] = (),
    ) -> JointStateIndices:
        import mujoco

        if len(robot_joint_names) != 29 or len(set(robot_joint_names)) != 29:
            raise ValueError("exactly 29 unique robot joint names are required")
        layout = JointStateLayout(tuple(board_joint_names))

        def addresses(name: str, free: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index < 0:
                raise ValueError(f"joint is absent from the model: {name}")
            kind = int(model.jnt_type[index])
            if free and kind != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"expected a free joint: {name}")
            if not free and kind not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"expected a scalar hinge/slide joint: {name}")
            q, v = int(model.jnt_qposadr[index]), int(model.jnt_dofadr[index])
            return tuple(range(q, q + (7 if free else 1))), tuple(range(v, v + (6 if free else 1)))

        def entity(free: str, names: Sequence[str]):
            q, v = addresses(free, True)
            for name in names:
                qi, vi = addresses(name, False)
                q, v = q + qi, v + vi
            return q, v

        rq, rv = entity(robot_free_joint, robot_joint_names)
        bq, bv = entity(board_free_joint, board_joint_names)
        if len(set(rq + bq)) != len(rq + bq) or len(set(rv + bv)) != len(rv + bv):
            raise ValueError("robot and board joint addresses overlap")
        return cls(layout, rq, rv, bq, bv, model.nq, model.nv)

    def pack(self, qpos, qvel, origins=None) -> torch.Tensor:
        qpos = torch.as_tensor(qpos)
        if not qpos.dtype.is_floating_point:
            raise TypeError("qpos must use a floating-point dtype")
        qvel = torch.as_tensor(qvel, dtype=qpos.dtype, device=qpos.device)
        if qpos.shape[-1] != self.nq or qvel.shape[-1] != self.nv:
            raise ValueError("qpos/qvel dimensions do not match the indexed model")
        origin = 0 if origins is None else torch.as_tensor(origins, dtype=qpos.dtype, device=qpos.device)
        pieces = []
        for qi, vi in ((self.robot_qpos, self.robot_qvel), (self.board_qpos, self.board_qvel)):
            q = qpos[..., list(qi)]
            pieces.extend((q[..., :3] - origin, q[..., 3:], qvel[..., list(vi)]))
        return torch.cat(pieces, dim=-1)

    def unpack(self, state: torch.Tensor, origins=None, *, qpos_template=None, qvel_template=None):
        robot, board = self.layout.split(state)
        qpos = state.new_zeros(*state.shape[:-1], self.nq) if qpos_template is None else torch.as_tensor(
            qpos_template, dtype=state.dtype, device=state.device
        ).clone()
        qvel = state.new_zeros(*state.shape[:-1], self.nv) if qvel_template is None else torch.as_tensor(
            qvel_template, dtype=state.dtype, device=state.device
        ).clone()
        if qpos.shape != (*state.shape[:-1], self.nq) or qvel.shape != (*state.shape[:-1], self.nv):
            raise ValueError("template shape differs from the state batch/model dimensions")
        origin = 0 if origins is None else torch.as_tensor(origins, dtype=state.dtype, device=state.device)
        for entity, qi, vi in (
            (robot, self.robot_qpos, self.robot_qvel), (board, self.board_qpos, self.board_qvel),
        ):
            nq = len(qi)
            qpos[..., list(qi)] = torch.cat([entity[..., :3] + origin, entity[..., 3:nq]], dim=-1)
            qvel[..., list(vi)] = entity[..., nq:]
        return qpos, qvel


class JointStateSpec:
    """Differentiable joint-state integration/errors; no contact forces or trained WM.

    Robot behavior/scales reuse StateSpec. Board scales are engineering defaults,
    not paper hyperparameters. Equality coupling must be enforced by a later adapter.
    """

    def __init__(self, layout: JointStateLayout, cfg: StateSpecCfg | None = None, device="cpu", dtype=torch.float32):
        self.layout = layout
        self.robot = StateSpec(cfg or StateSpecCfg(), device, dtype)
        self.dt = self.robot.dt
        k = layout.board_joints
        self.board_out_scale = torch.tensor(
            [0.3] * 3 + [0.5] * 3 + [3.0] * k + [0.01] * 3 + [0.02] * 3 + [0.02] * k,
            device=device, dtype=dtype,
        )
        self.board_error_scale = torch.tensor(
            [0.05] * 3 + [0.1] * 3 + [1.0] * 3 + [2.0] * 3 + [0.1] * k + [5.0] * k,
            device=device, dtype=dtype,
        )
        self.board_feature_scale = torch.tensor(
            [1.0] * 12 + [2.0] * 3 + [1.0] * k + [5.0] * k, device=device, dtype=dtype,
        )

    def canonical(self, state: torch.Tensor) -> torch.Tensor:
        robot, board = self.layout.split(state)
        p, q, joints, velocity, angular, joint_velocity = self.layout.split_board(board)
        heading = heading_quat(quat_normalize(robot[..., 3:7]))
        relative_q = quat_canonical(quat_mul(quat_apply_inverse_quat(heading), quat_normalize(q)))
        features = torch.cat([
            quat_apply_inverse(heading, p - robot[..., :3]), rot6d_from_quat(relative_q),
            quat_apply_inverse(heading, velocity), angular, joints, joint_velocity,
        ], dim=-1)
        return torch.cat([self.robot.canonical(robot), features / self.board_feature_scale], dim=-1)

    def integrate(self, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        robot, board = self.layout.split(state)
        if delta.shape[-1] != self.layout.delta_dim:
            raise ValueError(f"expected delta dimension {self.layout.delta_dim}")
        p, q, joints, velocity, angular, joint_velocity = self.layout.split_board(board)
        k = self.layout.board_joints
        d = delta[..., ROBOT_DELTA_DIM:] * self.board_out_scale
        dv, dw, dqd = d[..., :3], d[..., 3:6], d[..., 6:6 + k]
        dp, dtheta, dq = d[..., 6 + k:9 + k], d[..., 9 + k:12 + k], d[..., 12 + k:]
        q = quat_normalize(q)
        heading = heading_quat(q)
        new_velocity = velocity + quat_apply(heading, dv)
        new_angular = angular + dw
        new_joint_velocity = joint_velocity + dqd
        dt = self.dt if self.robot.cfg.explicit_integration else 0.0
        new_board = torch.cat([
            p + dt * new_velocity + quat_apply(heading, dp),
            quat_normalize(quat_mul(q, quat_from_rotvec(dt * new_angular + dtheta))),
            joints + dt * new_joint_velocity + dq, new_velocity, new_angular, new_joint_velocity,
        ], dim=-1)
        return torch.cat([self.robot.integrate(robot, delta[..., :ROBOT_DELTA_DIM]), new_board], dim=-1)

    def error_vector(self, predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        robot_p, board_p = self.layout.split(predicted)
        robot_t, board_t = self.layout.split(truth)
        p, q, j, v, w, jd = self.layout.split_board(board_p)
        pt, qt, jt, vt, wt, jdt = self.layout.split_board(board_t)
        rotation = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(quat_normalize(qt)), quat_normalize(q)))
        board_error = torch.cat([p - pt, rotation, v - vt, w - wt, j - jt, jd - jdt], dim=-1)
        return torch.cat([self.robot.error_vector(robot_p, robot_t), board_error / self.board_error_scale], dim=-1)

    def normalized_sq_error(self, predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.error_vector(predicted, truth).square().mean(-1)
