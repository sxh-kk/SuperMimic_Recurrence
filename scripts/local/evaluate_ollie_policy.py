"""Run one uninterrupted native-MuJoCo Ollie episode and render actual states.

Only 29 robot position targets are controlled. The passive board is initialized
once and subsequently evolves through contact, with no external forces or resets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.learning import JointLearningConfig, JointResidualPolicy
from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import STATE_SCHEMA, JointStateIndices

CONTROL_DT = 0.02
PHYSICS_DT = 0.0025
ACTOR_SCHEMA = "supermimic.ollie.actor.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path: Path, layout, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != ACTOR_SCHEMA:
        raise ValueError(f"Expected actor schema {ACTOR_SCHEMA}")
    if tuple(checkpoint["board_joint_names"]) != BOARD_JOINT_NAMES:
        raise ValueError("Checkpoint board joint names/order differ from the scene")
    values = dict(checkpoint["learning_config"])
    allowed = {field.name for field in fields(JointLearningConfig)}
    if set(values) - allowed:
        raise ValueError(f"Unknown actor learning settings: {set(values) - allowed}")
    for name in ("hidden", "policy_hidden", "reference_offsets"):
        if name in values:
            values[name] = tuple(values[name])
    config = JointLearningConfig(**values)
    if (
        not config.reference_offsets
        or config.reference_offsets[0] != 0
        or any(not isinstance(offset, int) or offset < 0 for offset in config.reference_offsets)
    ):
        raise ValueError("Actor offsets must start at zero and be nonnegative integers")
    policy = JointResidualPolicy(layout, config, device=device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    return policy, checkpoint


class ContactObserver:
    """Physical contacts use normal force, with wheels counted by unique geom ID."""

    def __init__(self, model, index, force_threshold: float):
        self.model = model
        self.index = index
        self.threshold = force_threshold
        self.names = [model.geom(i).name or "" for i in range(model.ngeom)]
        self.ground = model.geom("ground").id
        self.board = {
            i for i, name in enumerate(self.names) if name.startswith(("board_", "wheel_"))
        }
        self.deck = {model.geom(name).id for name in BOARD_GEOM_NAMES}
        self.wheels = [
            model.geom(name.removesuffix("_joint") + "_collision").id for name in BOARD_JOINT_NAMES
        ]
        self.feet = {
            side: {
                i
                for i, name in enumerate(self.names)
                if name.startswith(f"{side}_foot") and name.endswith("_collision")
            }
            for side in ("left", "right")
        }
        self.robot_body = int(model.jnt_bodyid[model.joint("floating_base_joint").id])

    def observe(self, data) -> dict:
        wheel_contact = set()
        foot_board = {"left": False, "right": False}
        foot_ground = {"left": False, "right": False}
        board_ground, robot_ground, tail_ground, nose_ground = set(), set(), False, False
        active_pairs = set()
        maximum_penetration, maximum_normal_force = 0.0, 0.0
        force = np.empty(6)
        for contact_index, contact in enumerate(data.contact[: data.ncon]):
            geoms = {int(contact.geom[0]), int(contact.geom[1])}
            maximum_penetration = max(maximum_penetration, -float(contact.dist))
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            normal_force = float(force[0])
            maximum_normal_force = max(maximum_normal_force, normal_force)
            if normal_force < self.threshold:
                continue
            active_pairs.add(tuple(sorted(geoms)))
            if self.ground in geoms:
                other = next(iter(geoms - {self.ground}))
                if other in self.board:
                    board_ground.add(other)
                    if other in self.wheels:
                        wheel_contact.add(other)
                    tail_ground |= self.names[other] == "board_tail"
                    nose_ground |= self.names[other] == "board_nose"
                else:
                    robot_ground.add(other)
                for side, feet in self.feet.items():
                    foot_ground[side] |= other in feet
            for side, feet in self.feet.items():
                foot_board[side] |= bool(geoms & feet) and bool(geoms & self.deck)
        clearance = []
        for geom in self.wheels:
            # MuJoCo cylinder axis is geom-local Z, including its authored quat.
            axis_z = data.geom_xmat[geom].reshape(3, 3)[2, 2]
            radius, half_width = self.model.geom_size[geom, :2]
            extent = radius * np.sqrt(max(0, 1 - axis_z**2)) + half_width * abs(axis_z)
            clearance.append(float(data.geom_xpos[geom, 2] - extent))
        board_clearance = list(clearance)
        for geom in self.deck:
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Board support observer expects deck/axle collision boxes")
            extent = np.dot(
                np.abs(data.geom_xmat[geom].reshape(3, 3)[2]), self.model.geom_size[geom]
            )
            board_clearance.append(float(data.geom_xpos[geom, 2] - extent))
        record = {
            "time_physics": float(data.time),
            "unique_wheels_ground": len(wheel_contact),
            "unique_board_geoms_ground": len(board_ground),
            "unique_robot_geoms_ground": len(robot_ground),
            "tail_ground": bool(tail_ground),
            "nose_ground": bool(nose_ground),
            "left_foot_board": bool(foot_board["left"]),
            "right_foot_board": bool(foot_board["right"]),
            "left_foot_ground": bool(foot_ground["left"]),
            "right_foot_ground": bool(foot_ground["right"]),
            "active_contact_pairs": len(active_pairs),
            "maximum_penetration_m": maximum_penetration,
            "maximum_normal_force_n": maximum_normal_force,
            "minimum_wheel_clearance_m": min(clearance),
            "minimum_board_clearance_m": min(board_clearance),
            "board_root_height_m": float(data.qpos[self.index.board_qpos + 2]),
            "robot_root_height_m": float(data.qpos[self.index.robot_qpos + 2]),
            "board_upright_cosine": float(data.xmat[self.index.board_body, 8]),
            "robot_upright_cosine": float(data.xmat[self.robot_body, 8]),
            "board_linear_speed_m_s": float(
                np.linalg.norm(data.qvel[self.index.board_qvel : self.index.board_qvel + 3])
            ),
            "board_angular_speed_rad_s": float(
                np.linalg.norm(data.qvel[self.index.board_qvel + 3 : self.index.board_qvel + 6])
            ),
            "robot_linear_speed_m_s": float(
                np.linalg.norm(data.qvel[self.index.robot_qvel : self.index.robot_qvel + 3])
            ),
            "robot_angular_speed_rad_s": float(
                np.linalg.norm(data.qvel[self.index.robot_qvel + 3 : self.index.robot_qvel + 6])
            ),
            "maximum_abs_actuator_force_nm": float(np.abs(data.actuator_force).max()),
            "external_generalized_force_norm": float(np.linalg.norm(data.qfrc_applied)),
            "external_body_force_norm": float(np.linalg.norm(data.xfrc_applied)),
        }
        for slot, geom in enumerate(self.wheels):
            name = BOARD_JOINT_NAMES[slot].removesuffix("_joint")
            record[f"{name}_ground"] = geom in wheel_contact
            record[f"{name}_clearance_m"] = clearance[slot]
        return record


def continuous_windows(mask: np.ndarray, times: np.ndarray) -> list[dict]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [
        {
            "start_s": float(times[start]),
            "end_s": float(times[end - 1]),
            "duration_s": float(times[end - 1] - times[start] + PHYSICS_DT),
        }
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]


def outcome(records: list[dict], *, full_reference: bool, numerical_failure: str | None) -> dict:
    times = np.array([row["time_physics"] for row in records])
    flight = np.array(
        [
            row["unique_board_geoms_ground"] == 0 and row["minimum_board_clearance_m"] > 0.005
            for row in records
        ]
    )
    flights = continuous_windows(flight, times)
    qualifying_flights = [window for window in flights if window["duration_s"] >= 0.04 - 1e-8]
    tails = np.array([row["tail_ground"] for row in records])
    tail_windows = continuous_windows(tails, times)
    tail_before_air = any(
        np.any(tails & (times < window["start_s"])) for window in qualifying_flights
    )
    final = [row for row in records if row["time_physics"] >= times[-1] - 0.3 + 1e-8]
    fractions = {
        name: float(np.mean([row[name] for row in final]))
        for name in ("left_foot_board", "right_foot_board", "left_foot_ground", "right_foot_ground")
    }
    fractions["all_four_wheels_ground"] = float(
        np.mean([row["unique_wheels_ground"] == 4 for row in final])
    )
    stable = (
        len(final) >= 100
        and fractions["all_four_wheels_ground"] >= 0.8
        and fractions["left_foot_board"] >= 0.8
        and fractions["right_foot_board"] >= 0.8
        and min(row["robot_upright_cosine"] for row in final) >= 0.8
        and min(row["board_upright_cosine"] for row in final) >= 0.95
        and max(row["robot_linear_speed_m_s"] for row in final) <= 0.3
        and max(row["robot_angular_speed_rad_s"] for row in final) <= 0.8
        and max(row["board_linear_speed_m_s"] for row in final) <= 0.35
        and max(row["board_angular_speed_rad_s"] for row in final) <= 0.8
        and min(row["robot_root_height_m"] - row["board_root_height_m"] for row in final) >= 0.55
    )
    maximum_penetration = max(row["maximum_penetration_m"] for row in records)
    robot_ground = any(row["unique_robot_geoms_ground"] for row in records)
    external_forces = any(
        row["external_generalized_force_norm"] != 0 or row["external_body_force_norm"] != 0
        for row in records
    )
    rules = {
        "full_reference_episode": full_reference,
        "numerically_continuous": numerical_failure is None,
        "initial_penetration_below_2mm": records[0]["maximum_penetration_m"] <= 0.002,
        "tail_contact_precedes_40ms_board_flight": bool(tail_before_air),
        "board_flight_clearance_at_least_15mm": max(
            row["minimum_wheel_clearance_m"] for row in records
        )
        >= 0.015,
        "no_robot_geom_touches_ground": not robot_ground,
        "penetration_at_most_10mm": maximum_penetration <= 0.01,
        "final_300ms_standing_stable_on_board": bool(stable),
        "no_external_forces": not external_forces,
    }
    return {
        "success": all(rules.values()),
        "outcome": "SUCCESS" if all(rules.values()) else "FAILED",
        "criteria": rules,
        "failure_reasons": [name for name, passed in rules.items() if not passed],
        "numerical_failure": numerical_failure,
        "flight_windows": flights,
        "tail_contact_windows": tail_windows,
        "final_contact_fractions": fractions,
        "maximum_penetration_m": maximum_penetration,
        "maximum_all_wheel_clearance_m": max(row["minimum_wheel_clearance_m"] for row in records),
        "maximum_whole_board_clearance_m": max(row["minimum_board_clearance_m"] for row in records),
        "initial_penetration_m": records[0]["maximum_penetration_m"],
        "robot_ground_first_time_s": next(
            (row["time_physics"] for row in records if row["unique_robot_geoms_ground"]), None
        ),
        "final_window_seconds": 0.3,
        "final_window_contact_samples": len(final),
        "tail_rule": "Tail contact occurs before a continuous >=40ms fully airborne board window; tail duration is reported separately.",
    }


def render_actual(model, recorded_qpos: np.ndarray, times: np.ndarray, summary: dict, args):
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 2.8, 130, -18
    label = "Reference PD baseline" if args.reference_pd else "Learned full-state prototype"
    text_filter = (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        f"text='{label}':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7,"
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        f"text='Outcome - {summary['outcome']} / actual uninterrupted physics':"
        "x=12:y=h-38:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7"
    )
    destination = args.out_dir / "rollout.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "800x600",
            "-r",
            str(args.video_fps),
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            text_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    render_data = mujoco.MjData(model)
    robot = model.jnt_qposadr[model.joint("floating_base_joint").id]
    board = model.jnt_qposadr[model.joint("board_free").id]
    frame_times = np.arange(int(np.floor(times[-1] * args.video_fps)) + 1) / args.video_fps
    frame_ids = np.abs(times[:, None] - frame_times[None]).argmin(axis=0)
    try:
        with mujoco.Renderer(model, width=800, height=600) as renderer:
            for frame in frame_ids:
                # This separate rendering data never feeds back into the episode.
                render_data.qpos[:] = recorded_qpos[frame]
                mujoco.mj_forward(model, render_data)
                camera.lookat[:] = 0.5 * (
                    render_data.qpos[robot : robot + 3] + render_data.qpos[board : board + 3]
                )
                camera.lookat[2] = max(0.35, camera.lookat[2])
                renderer.update_scene(render_data, camera=camera)
                encoder.stdin.write(renderer.render().tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        _, errors = encoder.communicate(timeout=45)
        if encoder.returncode:
            raise RuntimeError(errors.decode(errors="replace"))
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.communicate()
    return {
        "path": str(destination),
        "fps": args.video_fps,
        "frames": len(frame_ids),
        "size": [800, 600],
        "codec": "H264",
        "source": "recorded actual simulation qpos only",
        "camera": "Rendering camera follows midpoint of actual robot and board; no reference ghost",
    }


def _evaluate(args) -> dict:
    reference = JointReference.load(args.reference)
    if reference.layout.board_joint_names != BOARD_JOINT_NAMES:
        raise ValueError("Reference board joints differ from this scene")
    if not np.isclose(reference.metadata["physics_fps"], 1 / CONTROL_DT):
        raise ValueError("Evaluation requires a 50Hz reference")
    model, data, index = compile_scene(include_robot=True, timestep=PHYSICS_DT)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    if model.nu != 29 or model.neq != 0:
        raise ValueError("Only 29 robot actuators and no equality constraints are supported")
    initial = torch.tensor(reference.state[0], dtype=torch.float64)
    qpos, qvel = indices.unpack(initial)
    data.qpos[:] = qpos.numpy()
    data.qvel[:] = qvel.numpy() if args.initial_velocity == "reference" else 0
    data.ctrl[index.robot_actuators] = reference.robot_state[0, 7:36]
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    data.time = 0
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, index, args.contact_force_threshold)
    records = [observer.observe(data)]
    qposes, qvels = [data.qpos.copy()], [data.qvel.copy()]
    actual_times = [float(data.time)]
    states = [indices.pack(data.qpos.copy(), data.qvel.copy()).numpy()]
    actions, raw_actions, goal_indices, control_times = [], [], [], []
    policy, checkpoint = (
        (None, None) if args.reference_pd else load_policy(args.ckpt, reference.layout, args.device)
    )
    refs = torch.tensor(reference.state, dtype=torch.float32, device=args.device)
    seconds = reference.time_physics[-1] if args.seconds is None else args.seconds
    steps = round(seconds / CONTROL_DT)
    if seconds <= 0 or not np.isclose(seconds, steps * CONTROL_DT):
        raise ValueError("seconds must be a positive integer multiple of 0.02")
    low = model.jnt_range[model.dof_jntid[index.robot_joint_qvel], 0] + 0.005
    high = model.jnt_range[model.dof_jntid[index.robot_joint_qvel], 1] - 0.005
    previous_callback = mujoco.get_mjcb_control()
    numerical_failure, absolute_work = None, 0.0
    mujoco.set_mjcb_control(None)
    try:
        with torch.inference_mode():
            for step in range(steps):
                offsets = (0,) if policy is None else policy.cfg.reference_offsets
                goals = np.minimum(step + 1 + np.array(offsets), len(reference.state) - 1)
                if policy is None:
                    raw_action = reference.robot_state[goals[0], 7:36].copy()
                else:
                    state = torch.tensor(states[-1], dtype=torch.float32, device=args.device)[None]
                    raw_action = policy(state, refs[goals][None])[0].cpu().numpy().astype(float)
                if not np.isfinite(raw_action).all():
                    numerical_failure = "nonfinite policy action"
                    break
                target = np.clip(raw_action, low, high) if args.clip_joint_targets else raw_action
                data.ctrl[index.robot_actuators] = target
                actions.append(target.copy())
                raw_actions.append(raw_action.copy())
                goal_indices.append(goals.copy())
                control_times.append(float(data.time))
                for _ in range(8):
                    before_time = float(data.time)
                    mujoco.mj_step(model, data)
                    if not np.isclose(data.time, before_time + PHYSICS_DT, rtol=0, atol=1e-10):
                        numerical_failure = (
                            "MuJoCo time discontinuity or automatic instability reset detected"
                        )
                        break
                    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                        numerical_failure = "nonfinite native simulation state"
                        break
                    mujoco.mj_forward(model, data)
                    record = observer.observe(data)
                    record["time_video"] = reference.metadata["time_scale"] * float(data.time)
                    records.append(record)
                    absolute_work += (
                        float(
                            np.sum(
                                np.abs(
                                    data.actuator_force[index.robot_actuators]
                                    * data.qvel[index.robot_joint_qvel]
                                )
                            )
                        )
                        * PHYSICS_DT
                    )
                qposes.append(data.qpos.copy())
                qvels.append(data.qvel.copy())
                actual_times.append(float(data.time))
                states.append(indices.pack(data.qpos.copy(), data.qvel.copy()).numpy())
                if numerical_failure:
                    break
    finally:
        mujoco.set_mjcb_control(previous_callback)
    records[0]["time_video"] = 0.0
    actual_times = np.asarray(actual_times)
    complete = bool(numerical_failure is None and data.time >= reference.time_physics[-1] - 1e-8)
    summary = outcome(records, full_reference=complete, numerical_failure=numerical_failure)
    summary.update(
        {
            "schema": "supermimic.ollie.evaluation.v1",
            "reference": str(args.reference),
            "reference_sha256": sha256(args.reference),
            "checkpoint": str(args.ckpt) if args.ckpt else None,
            "checkpoint_sha256": sha256(args.ckpt) if args.ckpt else None,
            "checkpoint_iteration": checkpoint.get("iteration") if checkpoint else None,
            "mode": "Reference PD baseline"
            if args.reference_pd
            else "Learned full-state prototype",
            "physics": "native CPU MuJoCo",
            "mujoco_version": mujoco.__version__,
            "policy_device": None if policy is None else args.device,
            "initial_velocity": args.initial_velocity,
            "initializations": 1,
            "explicit_resets_during_episode": 0,
            "board_pose_writes_after_initialization": 0,
            "robot_root_pose_writes_after_initialization": 0,
            "board_actuators": 0,
            "external_force_injection": False,
            "timestep_s": PHYSICS_DT,
            "control_dt_s": CONTROL_DT,
            "substeps": 8,
            "episode_seconds": float(data.time),
            "control_steps": len(actions),
            "contact_force_threshold_n": args.contact_force_threshold,
            "clip_joint_targets": args.clip_joint_targets,
            "clipped_action_values": int(
                np.count_nonzero(np.array(actions) != np.array(raw_actions))
            ),
            "absolute_actuator_work_j": absolute_work,
            "reference_goal_offsets": [0] if policy is None else list(policy.cfg.reference_offsets),
            "goal_index_rule": "min(current_step + 1 + offset, last_reference_frame)",
            "time_scale_unverified": reference.metadata["time_scale"],
            "scope": "Full-state research prototype / reference-PD evaluation; not a complete SuperMimic paper implementation.",
        }
    )
    for warning_index, warning in enumerate(data.warning):
        if warning.number:
            name = mujoco.mjtWarning(warning_index).name
            summary.setdefault("mujoco_warnings", {})[name] = int(warning.number)
    actual_state = np.array(states)
    reference_index = np.minimum(
        np.rint(actual_times / CONTROL_DT).astype(int), len(reference.state) - 1
    )
    comparison = reference.state[reference_index]
    robot_error = actual_state[:, :3] - comparison[:, :3]
    board_error = actual_state[:, 71:74] - comparison[:, 71:74]
    joint_error = actual_state[:, 7:36] - comparison[:, 7:36]
    summary["tracking"] = {
        "robot_position_rmse_m": float(np.sqrt(np.mean(robot_error**2))),
        "board_position_rmse_m": float(np.sqrt(np.mean(board_error**2))),
        "robot_joint_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
        "board_position_final_error_m": float(np.linalg.norm(board_error[-1])),
        "robot_position_final_error_m": float(np.linalg.norm(robot_error[-1])),
    }
    ground_fraction = float(np.mean([row["unique_robot_geoms_ground"] > 0 for row in records]))
    summary["ranking_cost"] = (
        summary["tracking"]["robot_position_rmse_m"]
        + summary["tracking"]["board_position_rmse_m"]
        + 0.1 * summary["tracking"]["robot_joint_rmse_rad"]
        + 5 * ground_fraction
        + 10 * max(0, summary["maximum_penetration_m"] - 0.01)
        + (0 if complete else 10)
    )
    summary["ranking_cost_meaning"] = (
        "Engineering CPU screening cost; lower is better, never a replacement for success criteria."
    )
    np.savez_compressed(
        args.out_dir / "rollout.npz",
        state=np.array(states),
        qpos=np.array(qposes),
        qvel=np.array(qvels),
        control=np.array(actions),
        raw_control=np.array(raw_actions),
        time_physics=actual_times,
        time_video=actual_times * reference.metadata["time_scale"],
        control_time_physics=np.array(control_times),
        reference_goal_indices=np.array(goal_indices),
        board_joint_names=np.array(BOARD_JOINT_NAMES),
        robot_joint_names=np.array(G1_JOINT_NAMES),
        state_schema=np.array(STATE_SCHEMA),
        quaternion_order=np.array("wxyz"),
        units=np.array("SI"),
        metadata_json=np.array(json.dumps(summary)),
    )
    with (args.out_dir / "contacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if not args.no_render:
        summary["video"] = render_actual(model, np.array(qposes), actual_times, summary, args)
    return summary


def evaluate(
    ckpt: str | Path | None = None,
    reference: str | Path = "data/ollie/joint_reference_50hz.npz",
    output: str | Path = "logs/ollie_eval/evaluation",
    render: bool = False,
    *,
    device: str = "cpu",
    initial_velocity: str = "zero",
    seconds: float | None = None,
    contact_force_threshold: float = 0.1,
    clip_joint_targets: bool = True,
    video_fps: int = 50,
) -> dict:
    """Importable native CPU screening API; ckpt=None selects reference PD.

    Each call creates one independent continuous episode and a new output folder.
    It does not alter Torch thread counts or the caller's checkpoint/model.
    """
    args = argparse.Namespace(
        ckpt=Path(ckpt).resolve() if ckpt else None,
        reference_pd=ckpt is None,
        reference=Path(reference).resolve(),
        out_dir=Path(output).resolve(),
        no_render=not render,
        device=device,
        initial_velocity=initial_velocity,
        seconds=seconds,
        contact_force_threshold=contact_force_threshold,
        clip_joint_targets=clip_joint_targets,
        video_fps=video_fps,
    )
    if contact_force_threshold <= 0 or initial_velocity not in ("zero", "reference"):
        raise ValueError(
            "Positive contact threshold and a supported initial velocity mode are required"
        )
    if video_fps not in (25, 50):
        raise ValueError("Video FPS must be 25 or 50")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "result.json").exists() or (args.out_dir / "rollout.npz").exists():
        raise ValueError("Choose a new output directory to preserve the previous evaluation")
    started = time.monotonic()
    # A caller may use a global MuJoCo control callback elsewhere. It must never
    # add hidden controls/forces to initialization, evaluation, or rendering.
    callback = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(None)
    try:
        result = _evaluate(args)
    finally:
        mujoco.set_mjcb_control(callback)
    result["wall_seconds"] = time.monotonic() - started
    (args.out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("data/ollie/joint_reference_50hz.npz")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference-pd", action="store_true")
    mode.add_argument("--ckpt", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial-velocity", choices=("zero", "reference"), default="zero")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--contact-force-threshold", type=float, default=0.1)
    parser.add_argument("--clip-joint-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, choices=(25, 50), default=50)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(
        ckpt=args.ckpt,
        reference=args.reference,
        output=args.out_dir,
        render=not args.no_render,
        device=args.device,
        initial_velocity=args.initial_velocity,
        seconds=args.seconds,
        contact_force_threshold=args.contact_force_threshold,
        clip_joint_targets=args.clip_joint_targets,
        video_fps=args.video_fps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
