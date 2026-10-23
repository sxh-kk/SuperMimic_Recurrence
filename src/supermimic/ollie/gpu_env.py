"""Small independent MuJoCo-Warp vector scene for joint-model data collection.

No state is overwritten during step, no automatic reset is performed, and only
the robot's 29 position actuators receive control. This is separate from the
flat-ground ManagerBasedRlEnv and does not load its incompatible world model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from supermimic.ollie.physics import BOARD_JOINT_NAMES, WHEEL_RADIUS, compile_scene
from supermimic.ollie.state import JointStateIndices


@dataclass
class JointEnvConfig:
    num_envs: int = 64
    physics_dt: float = 0.0025
    control_dt: float = 0.02
    max_contacts: int = 128
    max_constraints: int = 512
    cuda_graph: bool = True


class JointPhysicsEnv:
    def __init__(self, config: JointEnvConfig | None = None, device: str = "cuda:0"):
        import mujoco_warp as mjw
        import warp as wp

        self.cfg = config or JointEnvConfig()
        if self.cfg.num_envs < 1 or self.cfg.physics_dt <= 0 or self.cfg.control_dt <= 0:
            raise ValueError("environment count and time steps must be positive")
        self.decimation = round(self.cfg.control_dt / self.cfg.physics_dt)
        if not np.isclose(self.decimation * self.cfg.physics_dt, self.cfg.control_dt):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.device = torch.device(device)
        self.num_envs = self.cfg.num_envs
        self.native_model, native_data, self.index = compile_scene(timestep=self.cfg.physics_dt)
        self.indices = JointStateIndices.from_model(
            self.native_model, board_joint_names=BOARD_JOINT_NAMES
        )
        self.layout = self.indices.layout
        wp.init()
        self.wp, self.mjw = wp, mjw
        # Warp 1.18 cannot capture the externally wrapped null Torch stream.
        # Use its own capturable stream with explicit boundary synchronization.
        self.stream = wp.Stream(str(self.device))
        with wp.ScopedDevice(str(self.device)), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(self.native_model)
            self.data = mjw.put_data(
                self.native_model, native_data, nworld=self.num_envs,
                nconmax=self.cfg.max_contacts, njmax=self.cfg.max_constraints,
            )
        wp.synchronize_stream(self.stream)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.initial_qpos = torch.as_tensor(native_data.qpos.copy(), device=self.device, dtype=self.qpos.dtype)
        self.initial_qvel = torch.as_tensor(native_data.qvel.copy(), device=self.device, dtype=self.qvel.dtype)
        self.actuator_ids = torch.tensor(self.index.robot_actuators, device=self.device)
        self.joint_ids = torch.tensor(self.index.robot_joint_qpos, device=self.device)
        self.wheel_body_ids = torch.tensor(
            [self.native_model.body(name.removesuffix("_joint")).id for name in BOARD_JOINT_NAMES],
            device=self.device,
        )
        joint_ids = [self.native_model.actuator_trnid[a, 0] for a in self.index.robot_actuators]
        self.joint_limits = torch.as_tensor(
            self.native_model.jnt_range[joint_ids].copy(), device=self.device, dtype=self.qpos.dtype
        )
        self.graph = None
        self.reset()
        if self.cfg.cuda_graph:
            # Compile kernels outside capture and restore the actual initial state afterwards.
            self._advance()
            wp.synchronize_stream(self.stream)
            self.reset()
            with wp.ScopedDevice(str(self.device)), wp.ScopedStream(self.stream):
                with wp.ScopedCapture(stream=self.stream) as capture:
                    self._advance()
                self.graph = capture.graph
            wp.synchronize_stream(self.stream)
            self.reset()

    @torch.no_grad()
    def reset(self, ids: torch.Tensor | None = None, state: torch.Tensor | None = None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids.to(self.device)
        if state is None:
            self.qpos[ids] = self.initial_qpos
            self.qvel[ids] = self.initial_qvel
        else:
            qpos, qvel = self.indices.unpack(state.to(self.device))
            self.qpos[ids], self.qvel[ids] = qpos, qvel
        self.ctrl[ids] = 0
        self.ctrl[ids[:, None], self.actuator_ids] = self.qpos[ids[:, None], self.joint_ids]
        self.wp.to_torch(self.data.qacc_warmstart)[ids] = 0
        self.wp.to_torch(self.data.time)[ids] = 0
        torch.cuda.synchronize(self.device)
        with self.wp.ScopedDevice(str(self.device)), self.wp.ScopedStream(self.stream):
            self.mjw.forward(self.model, self.data)
        self.wp.synchronize_stream(self.stream)
        return self.state()

    def _advance(self):
        with self.wp.ScopedDevice(str(self.device)), self.wp.ScopedStream(self.stream):
            for _ in range(self.decimation):
                self.mjw.step(self.model, self.data)
            self.mjw.forward(self.model, self.data)

    @torch.no_grad()
    def step(self, joint_targets: torch.Tensor):
        if joint_targets.shape != (self.num_envs, 29):
            raise ValueError("joint_targets must be [num_envs,29] in G1_JOINT_NAMES order")
        targets = joint_targets.to(self.device).clamp(
            self.joint_limits[:, 0] + 0.005, self.joint_limits[:, 1] - 0.005
        )
        self.ctrl[:, self.actuator_ids] = targets
        torch.cuda.synchronize(self.device)
        if self.graph is None:
            self._advance()
        else:
            with self.wp.ScopedDevice(str(self.device)), self.wp.ScopedStream(self.stream):
                self.wp.capture_launch(self.graph, stream=self.stream)
        self.wp.synchronize_stream(self.stream)
        return self.state()

    def state(self) -> torch.Tensor:
        return self.indices.pack(self.qpos, self.qvel)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        state = self.state()
        robot, board = self.layout.split(state)
        wheels = self.wp.to_torch(self.data.xpos)[:, self.wheel_body_ids, 2] - WHEEL_RADIUS
        upright = 1 - 2 * (robot[:, 4].square() + robot[:, 5].square())
        return {
            "board_height": board[:, 2], "minimum_wheel_clearance": wheels.min(-1).values,
            "root_height": robot[:, 2], "root_upright": upright,
            "robot_board_xy_distance": (robot[:, :2] - board[:, :2]).norm(dim=-1),
            "finite": torch.isfinite(state).all(-1),
        }
