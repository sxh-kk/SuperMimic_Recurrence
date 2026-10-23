"""Training-only native-MuJoCo freefall/collision augmentation.

Initial conditions are synthetic, not video observations or policy evaluations.
Only 29 bounded robot targets are applied after the single episode reset. The
board stays passive, gravity stays normal, and no external support is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie import physics
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import STATE_SCHEMA, JointStateIndices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/ollie/self_collision_corrected/joint_reference_50hz.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--max-attempt-factor", type=int, default=16)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 40, 80, 100, 110, 128, 138])
    parser.add_argument("--height", type=float, nargs=2, default=[0.4, 0.8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-penetration-m", type=float, default=0.03)
    parser.add_argument(
        "--max-angle-overshoot-rad",
        type=float,
        default=0.05,
        help="Dataset quality filter for actual soft-limit overshoot, not a change to the physical model",
    )
    args = parser.parse_args()
    if (
        args.out_dir.exists()
        or args.episodes < 2
        or args.max_attempt_factor < 1
        or not 0 < args.height[0] <= args.height[1]
        or not np.isfinite(args.max_penetration_m)
        or args.max_penetration_m < 0.002
        or not np.isfinite(args.max_angle_overshoot_rad)
        or args.max_angle_overshoot_rad < 0
    ):
        parser.error(
            "Need a fresh directory, >=2 episodes and positive ordered heights/quality threshold"
        )
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    reference = JointReference.load(args.reference)
    if any(frame < 0 or frame >= len(reference.state) for frame in args.frames):
        parser.error("Invalid source-pose frame")
    model, data, index = compile_scene(timestep=0.0025)
    if model.nu != 29 or model.neq != 0 or mujoco.get_mjcb_control() is not None:
        raise ValueError("Requires only robot actuators, no equality constraints or hidden callback")
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    poses, _ = indices.unpack(torch.as_tensor(reference.state))
    poses = poses.numpy()
    limits = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]]
    low, high = limits[:, 0] + 0.005, limits[:, 1] - 0.005
    observer = ContactObserver(model, index, 0.1)
    args.out_dir.mkdir(parents=True)
    accepted, rejected = [], []
    states, actions = [], []
    for attempt in range(args.episodes * args.max_attempt_factor):
        if len(accepted) == args.episodes:
            break
        frame = int(rng.choice(args.frames))
        mujoco.mj_resetData(model, data)
        data.qpos[:] = poses[frame]
        center = data.qpos[index.board_qpos : index.board_qpos + 3].copy()
        rotation = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
        shift = np.r_[rng.uniform(-0.25, 0.25, 2), rng.uniform(*args.height)]
        for address in (index.robot_qpos, index.board_qpos):
            data.qpos[address : address + 3] = (
                center + shift + rotation.apply(data.qpos[address : address + 3] - center)
            )
            original = Rotation.from_quat(data.qpos[address + np.array([4, 5, 6, 3])])
            data.qpos[address + 3 : address + 7] = (rotation * original).as_quat()[[3, 0, 1, 2]]
        raw_initial = data.qpos[index.robot_joint_qpos] + rng.normal(0, 0.01, 29)
        data.qpos[index.robot_joint_qpos] = np.clip(raw_initial, low, high)
        data.qpos[index.wheel_qpos] = rng.uniform(-np.pi, np.pi, 4)
        mujoco.mj_forward(model, data)
        shared_velocity = np.r_[rng.uniform(-0.4, 0.4, 2), rng.uniform(-1.0, 0.6)]
        shared_angular = rng.uniform(-1.0, 1.0, 3)
        for qaddress, vaddress, body in (
            (index.robot_qpos, index.robot_qvel, observer.robot_body),
            (index.board_qpos, index.board_qvel, index.board_body),
        ):
            data.qvel[vaddress : vaddress + 3] = shared_velocity + rng.normal(0, 0.12, 3)
            world_angular = shared_angular + rng.normal(0, 0.25, 3)
            data.qvel[vaddress + 3 : vaddress + 6] = data.xmat[body].reshape(3, 3).T @ world_angular
        data.qvel[index.robot_joint_qvel] = rng.uniform(-0.6, 0.6, 29)
        data.qvel[index.wheel_qvel] = rng.uniform(-8, 8, 4)
        nominal = data.qpos[index.robot_joint_qpos].copy()
        data.ctrl[index.robot_actuators] = nominal
        mujoco.mj_forward(model, data)
        initial = observer.observe(data)
        if (
            initial["maximum_penetration_m"] > 0.002
            or initial["unique_robot_geoms_ground"]
            or initial["unique_board_geoms_ground"]
            or initial["minimum_board_clearance_m"] <= 0.1
        ):
            rejected.append(
                {
                    "attempt": attempt,
                    "source_frame": frame,
                    "reason": "initial_geometry_gate",
                    "initial": initial,
                }
            )
            continue
        episode_states = [indices.pack(data.qpos, data.qvel).numpy()]
        episode_actions, records = [], [initial]
        initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
        amplitude, frequency, phase = (
            rng.uniform(0, 0.25, 29),
            rng.uniform(0.5, 2.5, 29),
            rng.uniform(0, 2 * np.pi, 29),
        )
        numerical_failure, clipped_targets, angle_violation = None, 0, 0.0
        for step in range(25):
            raw = (
                nominal
                + amplitude * np.sin(2 * np.pi * frequency * step * 0.02 + phase)
                + rng.normal(0, 0.015, 29)
            )
            target = np.clip(raw, low, high)
            clipped_targets += int(np.count_nonzero(target != raw))
            data.ctrl[index.robot_actuators] = target
            episode_actions.append(target.copy())
            for _ in range(8):
                before = data.time
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if (
                    not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                    or not np.isclose(data.time, before + 0.0025, rtol=0, atol=1e-10)
                ):
                    numerical_failure = "nonfinite_or_discontinuous_physics"
                    break
                records.append(observer.observe(data))
                angles = data.qpos[index.robot_joint_qpos]
                angle_violation = max(
                    angle_violation,
                    float(np.maximum(limits[:, 0] - angles, 0).max()),
                    float(np.maximum(angles - limits[:, 1], 0).max()),
                )
            if numerical_failure:
                break
            episode_states.append(indices.pack(data.qpos, data.qvel).numpy())
        peak_penetration = max(row["maximum_penetration_m"] for row in records)
        external = any(
            row["external_generalized_force_norm"] or row["external_body_force_norm"]
            for row in records
        )
        reason = numerical_failure or (
            "unexpected_external_force"
            if external
            else "penetration_quality_filter"
            if peak_penetration > args.max_penetration_m
            else "joint_angle_quality_filter"
            if angle_violation > args.max_angle_overshoot_rad
            else None
        )
        if reason:
            rejected.append(
                {
                    "attempt": attempt,
                    "source_frame": frame,
                    "reason": reason,
                    "maximum_penetration_m": peak_penetration,
                    "actual_joint_angle_soft_limit_violation_max_rad": angle_violation,
                }
            )
            continue
        states.append(np.asarray(episode_states))
        actions.append(np.asarray(episode_actions))
        accepted.append(
            {
                "episode_id": len(accepted),
                "attempt": attempt,
                "source_frame": frame,
                "source_video_time_s": float(reference.time_video[frame]),
                "initial_translation_m": shift.tolist(),
                "initial_qpos": initial_qpos.tolist(),
                "initial_qvel": initial_qvel.tolist(),
                "initial_penetration_m": initial["maximum_penetration_m"],
                "initial_minimum_board_clearance_m": initial["minimum_board_clearance_m"],
                "initial_joint_target_clips": int(np.count_nonzero(raw_initial != nominal)),
                "executed_joint_target_clips": clipped_targets,
                "actual_joint_angle_soft_limit_violation_max_rad": angle_violation,
                "maximum_penetration_m": peak_penetration,
                "board_ground_first_time_s": next(
                    (row["time_physics"] for row in records if row["unique_board_geoms_ground"]),
                    None,
                ),
                "robot_ground_first_time_s": next(
                    (row["time_physics"] for row in records if row["unique_robot_geoms_ground"]),
                    None,
                ),
                "whole_board_free_samples": sum(
                    row["minimum_board_clearance_m"] > 0.005
                    and not row["unique_board_geoms_ground"]
                    for row in records
                ),
                "physics_samples": len(records),
                "no_external_forces": not external,
                "final_robot_speed_m_s": records[-1]["robot_linear_speed_m_s"],
            }
        )
        if len(accepted) % 16 == 0:
            print(
                json.dumps(
                    {"accepted": len(accepted), "attempts": attempt + 1, "rejected": len(rejected)}
                ),
                flush=True,
            )
    if len(accepted) < args.episodes:
        (args.out_dir / "incomplete_attempts.json").write_text(
            json.dumps(
                {
                    "training_only": True,
                    "requested_episodes": args.episodes,
                    "accepted": accepted,
                    "rejected": rejected,
                    "dataset_exported": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise RuntimeError(
            f"Only {len(accepted)}/{args.episodes} quality-controlled episodes; no dataset exported"
        )
    states, actions = np.stack(states), np.stack(actions)
    before, after = states[:, :-1].reshape(-1, 92), states[:, 1:].reshape(-1, 92)
    actual = actions.reshape(-1, 29)
    assert np.all(actual >= low - 1e-12) and np.all(actual <= high + 1e-12)
    manifest = {
        "schema": "supermimic.ollie.training_air_augmentation.v1",
        "training_only": True,
        "video_observed_trajectory": False,
        "verified_teacher": False,
        "ollie_success_certified": False,
        "reference_pose_source": str(args.reference.resolve()),
        "reference_sha256": sha(args.reference),
        "physics_module_sha256": sha(physics.__file__),
        "board_xml_sha256": sha(physics.BOARD_XML),
        "mujoco_version": mujoco.__version__,
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "max_attempt_factor": args.max_attempt_factor,
        "episodes": len(accepted),
        "transitions": len(actual),
        "physics_dt": 0.0025,
        "control_dt": 0.02,
        "episode_seconds": 0.5,
        "source_frames": args.frames,
        "height_range_m": args.height,
        "accepted": accepted,
        "rejected": rejected,
        "initial_geometry_limit_m": 0.002,
        "dynamic_penetration_quality_limit_m": args.max_penetration_m,
        "actual_joint_angle_soft_limit_quality_limit_rad": args.max_angle_overshoot_rad,
        "robot_joint_target_limits_rad": np.stack([low, high], axis=1).tolist(),
        "original_robot_joint_limits_rad": limits.tolist(),
        "original_actuator_force_limits_nm": model.actuator_forcerange[
            index.robot_actuators
        ].tolist(),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "board_actuators": 0,
        "initializations_per_episode": 1,
        "resets_within_episode": 0,
        "root_or_board_pose_writes_after_initialization": 0,
        "external_forces_applied": False,
        "initialization": "Synthetic shared-origin/yaw translation, positive air height, sampled true velocities; geometry gated",
        "controls": "Actual held robot position targets; original actuators/force limits; no expert labels or policy",
    }
    np.savez_compressed(
        args.out_dir / "transitions.npz",
        state=before.astype(np.float32),
        next_state=after.astype(np.float32),
        action=actual.astype(np.float32),
        ctrl=actual.astype(np.float32),
        state_float64=before,
        next_state_float64=after,
        ctrl_float64=actual,
        episode_id=np.repeat(np.arange(len(accepted), dtype=np.int32), 25),
        time_s=np.tile(np.arange(25) * 0.02, len(accepted)),
        control_dt=np.array(0.02),
        schema=np.array(STATE_SCHEMA),
        robot_joint_names=np.asarray(G1_JOINT_NAMES),
        board_joint_names=np.asarray(BOARD_JOINT_NAMES),
        verified_teacher=np.array(False),
        source=np.array(
            "Training-only native MuJoCo freefall/collision augmentation; synthetic initial conditions, not video observations or demonstrations"
        ),
        metadata_json=np.asarray(json.dumps(manifest)),
    )
    manifest["dataset_sha256"] = sha(args.out_dir / "transitions.npz")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "episodes": len(accepted),
        "transitions": len(actual),
        "rejections": len(rejected),
        "physics_numerical_failures": sum(
            row["reason"] == "nonfinite_or_discontinuous_physics" for row in rejected
        ),
        "episodes_with_board_ground_impact": sum(
            row["board_ground_first_time_s"] is not None for row in accepted
        ),
        "episodes_with_robot_ground_impact": sum(
            row["robot_ground_first_time_s"] is not None for row in accepted
        ),
        "maximum_penetration_m": max(row["maximum_penetration_m"] for row in accepted),
        "maximum_actual_joint_angle_soft_limit_violation_rad": max(
            row["actual_joint_angle_soft_limit_violation_max_rad"] for row in accepted
        ),
        "training_only": True,
        "not_a_video_or_ollie_success_result": True,
        "dataset_sha256": manifest["dataset_sha256"],
    }
    (args.out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
