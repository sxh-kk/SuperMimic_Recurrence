"""Physical reference-PD/bias/ankle-feedback probe, independent of learned actor.

All candidates evolve the same passive board freely. Only 29 bounded robot PD
targets are written. This diagnostic is not a SuperMimic policy result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from evaluate_ollie_policy import ContactObserver, outcome, render_actual
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


def run(reference, bias, sign, scale, velocity_gain, reference_hash, *, keep=False):
    model, data, index = compile_scene(timestep=0.0025)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    qpos, _ = indices.unpack(torch.tensor(reference.state[0], dtype=torch.float64))
    data.qpos[:], data.qvel[:], data.qfrc_applied[:], data.xfrc_applied[:] = qpos, 0, 0, 0
    mujoco.mj_forward(model, data)
    limits = model.jnt_range[model.actuator_trnid[index.robot_actuators, 0]]
    data.ctrl[index.robot_actuators] = np.clip(
        reference.robot_state[0, 7:36] + bias, limits[:, 0] + .005, limits[:, 1] - .005)
    mujoco.mj_forward(model, data)
    ankle_offsets = np.array([4, 5, 10, 11])
    ankle_joints = model.actuator_trnid[index.robot_actuators[ankle_offsets], 0]
    root_body = model.body("pelvis").id
    feet = [model.site(f"{side}_foot").id for side in ("left", "right")]
    initial_rotation = data.xmat[root_body].reshape(3, 3)
    yaw = np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0])
    heading = Rotation.from_euler("z", yaw).as_matrix()
    forward, lateral = heading[:, 0], heading[:, 1]
    observer = ContactObserver(model, index, 0.1)
    records = [observer.observe(data)]
    poses, states, actions, times = [data.qpos.copy()], [indices.pack(data.qpos, data.qvel).numpy()], [], [0.0]
    for step in range(1, len(reference.state)):
        target = reference.robot_state[step, 7:36].copy() + bias
        if sign and (records[-1]["left_foot_board"] or records[-1]["right_foot_board"]):
            root_rotation = data.xmat[root_body].reshape(3, 3)
            roll, pitch, _ = Rotation.from_matrix(heading.T @ root_rotation).as_euler("xyz")
            omega_world = root_rotation @ data.qvel[index.robot_qvel + 3:index.robot_qvel + 6]
            velocity = np.r_[data.qvel[index.robot_qvel:index.robot_qvel + 2], 0.0]
            com_error = np.r_[data.subtree_com[root_body, :2] - data.site_xpos[feet, :2].mean(0), 0.0]
            pitch_error = scale * (1.4 * pitch + .14 * (omega_world @ lateral)
                                   + 1.2 * (com_error @ forward) + velocity_gain * (velocity @ forward))
            roll_error = scale * (.8 * roll + .10 * (omega_world @ forward)
                                  - .7 * (com_error @ lateral) - velocity_gain * (velocity @ lateral))
            tilt_world = forward * np.clip(roll_error, -.15, .15) + lateral * np.clip(pitch_error, -.35, .35)
            correction = np.clip(data.xaxis[ankle_joints] @ tilt_world,
                                 [-.35, -.15, -.35, -.15], [.35, .15, .35, .15])
            target[ankle_offsets] += sign * correction
        target = np.clip(target, limits[:, 0] + .005, limits[:, 1] - .005)
        actions.append(target.copy())
        data.ctrl[index.robot_actuators] = target
        for _ in range(8):
            before = data.time
            mujoco.mj_step(model, data)
            if not np.isclose(data.time, before + .0025, atol=1e-10, rtol=0):
                raise RuntimeError("physical time discontinuity")
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite physical probe")
            records.append(observer.observe(data))
        poses.append(data.qpos.copy())
        states.append(indices.pack(data.qpos.copy(), data.qvel.copy()).numpy())
        times.append(data.time)
    result = outcome(records, full_reference=True, numerical_failure=None)
    prepare = [row for row in records if row["time_physics"] <= 2.76 + 1e-8]
    result.update(sign=sign, scale=scale, velocity_gain=velocity_gain,
                  prepare_min_upright=min(row["robot_upright_cosine"] for row in prepare),
                  prepare_robot_ground_samples=sum(row["unique_robot_geoms_ground"] > 0 for row in prepare),
                  prepare_final_root_speed_m_s=prepare[-1]["robot_linear_speed_m_s"],
                  reference_sha256=reference_hash,
                  mode="PD bias and ankle feedback probe", full_episode_seconds=float(data.time),
                  initializations=1, intermediate_resets=0, board_actuators=0,
                  no_external_forces=True, untrained_controller=True)
    # Rank physical survival before tracking. Full contact success remains separate.
    first_ground = result["robot_ground_first_time_s"] or float(data.time)
    result["selection_score"] = first_ground + 2 * result["prepare_min_upright"]
    if keep:
        return result, model, records, np.array(poses), np.array(states), np.array(actions), np.array(times)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[.25, .5, 1.0])
    parser.add_argument("--velocity-gains", type=float, nargs="+", default=[.16, .5, 1.0])
    parser.add_argument("--signs", type=int, choices=[-1, 1], nargs="+", default=[-1, 1])
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a new output directory")
    args.out_dir.mkdir(parents=True)
    reference = JointReference.load(args.reference)
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    bias_document = json.loads(args.bias.read_text())
    if tuple(bias_document["joint_names"]) != G1_JOINT_NAMES:
        raise ValueError("bias joint order differs")
    bias = np.asarray(bias_document["bias_rad"])
    if bias.shape != (29,) or not np.isfinite(bias).all():
        raise ValueError("bias must be 29 finite values")
    results = [run(reference, bias, 0, 0, 0, reference_hash)]
    for sign in args.signs:
        for scale in args.scales:
            for velocity_gain in args.velocity_gains:
                result = run(reference, bias, sign, scale, velocity_gain, reference_hash)
                results.append(result)
                print(json.dumps({key: result[key] for key in (
                    "sign", "scale", "velocity_gain", "robot_ground_first_time_s",
                    "prepare_min_upright", "success")}), flush=True)
    best = max(results, key=lambda row: row["selection_score"])
    best, model, records, poses, states, actions, times = run(
        reference, bias, best["sign"], best["scale"], best["velocity_gain"], reference_hash, keep=True)
    best["video"] = render_actual(model, poses, times, best,
                                  SimpleNamespace(out_dir=args.out_dir.resolve(), video_fps=50))
    np.savez_compressed(args.out_dir / "actual_rollout.npz", state=states, qpos=poses,
                        action=actions, time_physics=times,
                        robot_joint_names=G1_JOINT_NAMES, board_joint_names=BOARD_JOINT_NAMES)
    (args.out_dir / "result.json").write_text(json.dumps(best, indent=2) + "\n")
    (args.out_dir / "all_candidates.json").write_text(json.dumps(results, indent=2) + "\n")
    with (args.out_dir / "contacts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
