"""Joint robot/board reference NPZ: SI units, wxyz, two explicit time axes.

Schema validation is structural/numerical; it does not certify contact forces,
dynamic feasibility, video reconstruction accuracy, or controller performance.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.state import STATE_SCHEMA, JointStateLayout

REFERENCE_SCHEMA = "supermimic.ollie.joint_reference.v1"
PHASE_NAMES = ("prepare", "crouch", "pop", "flight", "landing", "recovery")
CONTACT_NAMES = (
    "left_foot_board", "right_foot_board", "left_foot_ground", "right_foot_ground",
    "tail_ground", "front_left_wheel_ground", "front_right_wheel_ground",
    "rear_left_wheel_ground", "rear_right_wheel_ground",
)


@dataclass
class JointReference:
    """Arrays share one frame axis. Contacts use -1 unknown, 0 off, 1 on.

    metadata requires time_scale, video_time_origin_s, physics_fps, and provenance.
    t_video = video_time_origin_s + time_scale * t_physics. The first physics
    time is zero; the last time denotes a sampled pose, not container end/hold.
    Robot joints use G1_JOINT_NAMES. Board joint order is explicit in layout.
    """

    layout: JointStateLayout
    time_video: np.ndarray
    time_physics: np.ndarray
    robot_state: np.ndarray
    board_state: np.ndarray
    phase: np.ndarray
    contacts: np.ndarray
    confidence: np.ndarray
    metadata: dict
    phase_names: tuple[str, ...] = PHASE_NAMES
    contact_names: tuple[str, ...] = CONTACT_NAMES

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.robot_state, self.board_state], axis=-1)

    def validate(self) -> None:
        frames = len(self.time_physics)
        if frames < 2:
            raise ValueError("a joint reference requires at least two sampled poses")
        shapes = {
            "time_video": (frames,), "time_physics": (frames,), "robot_state": (frames, 71),
            "board_state": (frames, 13 + 2 * self.layout.board_joints), "phase": (frames,),
            "contacts": (frames, len(self.contact_names)), "confidence": (frames,),
        }
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape or not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{name} must be a numeric array with shape {shape}")
            if np.iscomplexobj(array) or not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite or complex values")
        for name in ("phase_names", "contact_names"):
            labels = getattr(self, name)
            if not labels or len(set(labels)) != len(labels) or any(
                not isinstance(label, str) or not label for label in labels
            ):
                raise ValueError(f"{name} must contain unique nonempty labels")
        if not np.isin(self.phase, np.arange(-1, len(self.phase_names))).all():
            raise ValueError("phase must be -1 (unknown) or a valid phase index")
        if not np.isin(self.contacts, [-1, 0, 1]).all():
            raise ValueError("contacts must use -1 unknown, 0 off, or 1 on")
        if (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("confidence must lie in [0, 1]")
        for state in (self.robot_state, self.board_state):
            if np.max(np.abs(np.linalg.norm(state[:, 3:7], axis=-1) - 1)) > 1e-4:
                raise ValueError("root quaternions must be normalized wxyz")
        scale = self.metadata.get("time_scale")
        fps = self.metadata.get("physics_fps")
        origin = self.metadata.get("video_time_origin_s")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (scale, fps, origin)
        ):
            raise ValueError("metadata needs finite time_scale, physics_fps, video_time_origin_s")
        if scale <= 0 or fps <= 0 or origin < 0:
            raise ValueError("time scale/fps must be positive; source video time must be nonnegative")
        if not isinstance(self.metadata.get("provenance"), dict) or not self.metadata["provenance"]:
            raise ValueError("explicit nonempty provenance is required")
        if abs(float(self.time_physics[0])) > 1e-8:
            raise ValueError("the first physical sample must be at t=0")
        delta = np.diff(self.time_physics)
        if (delta <= 0).any() or not np.allclose(delta, 1 / fps, rtol=1e-5, atol=1e-8):
            raise ValueError("physical sample times must be strictly increasing and match physics_fps")
        if (np.diff(self.time_video) <= 0).any() or not np.allclose(
            self.time_video, origin + scale * self.time_physics, rtol=0, atol=1e-7
        ):
            raise ValueError("video and physics time axes do not match the declared scale/origin")
        # Reject non-JSON metadata, including NaN, before writing an otherwise valid archive.
        json.dumps(self.metadata, allow_nan=False)

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("joint references use .npz files")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream, schema=np.array(REFERENCE_SCHEMA), state_schema=np.array(STATE_SCHEMA),
                    quaternion_order=np.array("wxyz"), units=np.array("SI"),
                    robot_joint_names=np.asarray(G1_JOINT_NAMES),
                    board_joint_names=np.asarray(self.layout.board_joint_names, dtype=str),
                    phase_names=np.asarray(self.phase_names), contact_names=np.asarray(self.contact_names),
                    metadata_json=np.array(json.dumps(self.metadata, allow_nan=False)),
                    **{key: getattr(self, key) for key in (
                        "time_video", "time_physics", "robot_state", "board_state", "phase",
                        "contacts", "confidence",
                    )},
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> JointReference:
        with np.load(path, allow_pickle=False) as arrays:
            for key, value in (
                ("schema", REFERENCE_SCHEMA), ("state_schema", STATE_SCHEMA),
                ("quaternion_order", "wxyz"), ("units", "SI"),
            ):
                if key not in arrays or str(arrays[key].item()) != value:
                    raise ValueError(f"unsupported or missing joint-reference {key}")
            if tuple(arrays["robot_joint_names"].tolist()) != G1_JOINT_NAMES:
                raise ValueError("robot joint names/order differ from the 29 DoF G1 contract")
            result = cls(
                layout=JointStateLayout(tuple(arrays["board_joint_names"].tolist())),
                phase_names=tuple(arrays["phase_names"].tolist()),
                contact_names=tuple(arrays["contact_names"].tolist()),
                metadata=json.loads(str(arrays["metadata_json"].item())),
                **{key: arrays[key].copy() for key in (
                    "time_video", "time_physics", "robot_state", "board_state", "phase",
                    "contacts", "confidence",
                )},
            )
        result.validate()
        return result


def _resample_entity(state: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Interpolate poses and recompute derivatives in physical time (not video time)."""
    joints = (state.shape[1] - 13) // 2
    nq = 7 + joints
    positions = np.stack([np.interp(target_time, source_time, state[:, i]) for i in range(3)], -1)
    source_rotations = Rotation.from_quat(state[:, [4, 5, 6, 3]])
    rotations = Slerp(source_time, source_rotations)(target_time)
    quaternions = rotations.as_quat()[:, [3, 0, 1, 2]]
    joint_position = np.stack(
        [np.interp(target_time, source_time, state[:, i]) for i in range(7, nq)], -1
    ) if joints else np.empty((len(target_time), 0))
    dt = float(target_time[1] - target_time[0])
    linear = np.gradient(positions, dt, axis=0)
    joint_velocity = np.gradient(joint_position, dt, axis=0)
    if len(target_time) > 2:
        world_angular = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * dt)
        world_angular = np.concatenate([world_angular[:1], world_angular, world_angular[-1:]])
    else:
        velocity = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
        world_angular = np.repeat(velocity[None], 2, axis=0)
    body_angular = rotations.inv().apply(world_angular)
    return np.concatenate([positions, quaternions, joint_position, linear, body_angular, joint_velocity], -1)


def resample_reference(reference: JointReference, fps: float = 50.0) -> JointReference:
    """Uniform pose resampling, velocity reconstruction, and held categorical labels.

    Does not synthesize a container-end pose. Wheel angles must be continuous
    (unwrapped) before calling; no generic joint unwrap/physical repair is applied.
    """
    reference.validate()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("target fps must be finite and positive")
    count = math.floor(float(reference.time_physics[-1]) * fps + 1e-8) + 1
    if count < 2:
        raise ValueError("reference duration is shorter than one target sample interval")
    times = np.arange(count, dtype=np.float64) / fps
    # Floating point roundoff at an exact endpoint must not put Slerp outside its support.
    times = np.minimum(times, reference.time_physics[-1])
    indices = np.searchsorted(reference.time_physics, times, side="right") - 1
    metadata = dict(reference.metadata)
    metadata.update(physics_fps=fps, resampling={
        "source_fps": reference.metadata["physics_fps"], "target_fps": fps,
        "velocities": "recomputed from pose in physical time", "categorical_labels": "left hold",
    })
    result = JointReference(
        layout=reference.layout, time_physics=times,
        time_video=metadata["video_time_origin_s"] + metadata["time_scale"] * times,
        robot_state=_resample_entity(reference.robot_state, reference.time_physics, times),
        board_state=_resample_entity(reference.board_state, reference.time_physics, times),
        phase=reference.phase[indices].copy(), contacts=reference.contacts[indices].copy(),
        confidence=np.interp(times, reference.time_physics, reference.confidence),
        metadata=metadata, phase_names=reference.phase_names, contact_names=reference.contact_names,
    )
    result.validate()
    return result
