"""Four fixed native branches of one rear-ankle target after a warm prefix.

This is a causal diagnostic, not an Ollie controller, teacher, or optimization.
Only one robot joint target changes for H8; all later controls are the recorded
baseline. Complete mjData copies preserve contact solver history.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("requires no global control callback")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.native_dir / "result.json").read_text())
    reference_path = Path(source["reference"])
    if file_hash(reference_path) != source["reference_sha256"]:
        raise ValueError("native reference changed")
    reference = JointReference.load(reference_path)
    archive_path = args.native_dir / "rollout.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        qp, qv = archive["qpos"].copy(), archive["qvel"].copy()
        actions, goals = archive["control"].copy(), archive["reference_goal_indices"].copy()
    start, end = round(args.time / 0.02), round(args.end_time / 0.02)
    if (not np.isclose(start * 0.02, args.time) or not np.isclose(end * 0.02, args.end_time)
            or start < 0 or end > len(actions) or end < start + 8):
        raise ValueError("times must be valid 20ms boundaries with at least H8 remaining")
    model, data, index = compile_scene(timestep=0.0025)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    observer = ContactObserver(model, index, 0.1)
    pelvis = model.body("pelvis").id
    rear_site = model.site("right_foot").id
    robot_mass = float(model.body_subtreemass[pelvis])
    joint = G1_JOINT_NAMES.index("right_ankle_roll_joint")
    jid = model.dof_jntid[index.robot_joint_qvel[joint]]
    lower, upper = model.jnt_range[jid] + np.array([0.005, -0.005])

    def record(branch):
        mujoco.mj_subtreeVel(model, branch)
        board_rotation = branch.xmat[index.board_body].reshape(3, 3)
        rear_rotation = branch.site_xmat[rear_site].reshape(3, 3)
        return {**observer.observe(branch),
                "robot_com_z_m": float(branch.subtree_com[pelvis, 2]),
                "robot_com_vz_m_s": float(branch.subtree_linvel[pelvis, 2]),
                "robot_com_xy_speed_m_s": float(np.linalg.norm(branch.subtree_linvel[pelvis, :2])),
                "robot_root_vz_m_s": float(branch.qvel[index.robot_qvel + 2]),
                "board_root_vz_m_s": float(branch.qvel[index.board_qvel + 2]),
                "rear_foot_toe_angle_deg": float(np.degrees(np.arcsin(np.clip(
                    (board_rotation.T @ rear_rotation)[2, 0], -1, 1)))),
                "rear_roll_q_rad": float(branch.qpos[index.robot_joint_qpos[joint]]),
                "rear_roll_ctrl_rad": float(branch.ctrl[index.robot_actuators[joint]])}

    data.qpos[:], data.qvel[:] = qp[0], qv[0]
    data.ctrl[index.robot_actuators] = reference.robot_state[0, 7:36]
    mujoco.mj_forward(model, data)
    prefix_error = 0.0
    for step in range(start):
        data.ctrl[index.robot_actuators] = actions[step]
        for _ in range(8):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        prefix_error = max(prefix_error, float(np.max(abs(data.qpos - qp[step + 1]))),
                           float(np.max(abs(data.qvel - qv[step + 1]))))
    if prefix_error > 1e-10:
        raise RuntimeError(f"prefix replay mismatch {prefix_error}")
    history = {field: np.asarray(getattr(data, field)).copy() for field in
               ("qpos", "qvel", "ctrl", "qacc_warmstart", "efc_force", "qfrc_applied", "xfrc_applied")}
    np.savez_compressed(args.out_dir / "warm_snapshot.npz", **history)
    summaries = []
    for mode in ("baseline", "reference", "zero", "fade_reference"):
        branch = copy.copy(data)
        if np.shares_memory(branch.qpos, data.qpos):
            raise RuntimeError("copy aliases warm source")
        if any(not np.array_equal(getattr(branch, field), value) for field, value in history.items()):
            raise RuntimeError("copy lost warm solver history")
        records = [record(branch)]
        states, qposes, qvels = [indices.pack(branch.qpos, branch.qvel).numpy()], [branch.qpos.copy()], [branch.qvel.copy()]
        executed = []
        baseline_error = 0.0
        for step in range(start, end):
            action = actions[step].copy()
            if step < start + 8:
                target = float(reference.robot_state[goals[step, 0], 7 + joint])
                if mode == "zero":
                    action[joint] = 0
                elif mode == "reference":
                    action[joint] = target
                elif mode == "fade_reference":
                    blend = (step - start + 1) / 8
                    action[joint] += blend * (target - action[joint])
                action[joint] = np.clip(action[joint], lower, upper)
            unchanged = np.arange(29) != joint
            if not np.array_equal(action[unchanged], actions[step, unchanged]):
                raise RuntimeError("another joint changed")
            branch.ctrl[index.robot_actuators] = action
            executed.append(action)
            for _ in range(8):
                mujoco.mj_step(model, branch)
                mujoco.mj_forward(model, branch)
                row = record(branch)
                if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                    raise RuntimeError("unexpected external force")
                records.append(row)
            states.append(indices.pack(branch.qpos, branch.qvel).numpy())
            qposes.append(branch.qpos.copy())
            qvels.append(branch.qvel.copy())
            if mode == "baseline":
                baseline_error = max(baseline_error, float(np.max(abs(branch.qpos - qp[step + 1]))),
                                     float(np.max(abs(branch.qvel - qv[step + 1]))))
        if not np.isfinite(states).all():
            raise RuntimeError("nonfinite branch")
        states = np.asarray(states)
        windows = {}
        for label, subset in (("h8", records[:65]), ("continuation", records)):
            dynamic = subset[1:]
            windows[label] = {
                "end_time_s": subset[-1]["time_physics"],
                "end_robot_com_vz_m_s": subset[-1]["robot_com_vz_m_s"],
                "peak_robot_com_vz_m_s": max(row["robot_com_vz_m_s"] for row in subset),
                "end_robot_com_z_m": subset[-1]["robot_com_z_m"],
                "end_robot_com_xy_speed_m_s": subset[-1]["robot_com_xy_speed_m_s"],
                "end_robot_root_vz_m_s": subset[-1]["robot_root_vz_m_s"],
                "end_board_root_vz_m_s": subset[-1]["board_root_vz_m_s"],
                "peak_whole_board_clearance_m": max(row["minimum_board_clearance_m"] for row in subset),
                "right_foot_support_fraction": sum(row["right_foot_deck_upward_support_n"] > 1 for row in dynamic) / len(dynamic),
                "left_foot_support_fraction": sum(row["left_foot_deck_upward_support_n"] > 1 for row in dynamic) / len(dynamic),
                "end_right_foot_support_n": subset[-1]["right_foot_deck_upward_support_n"],
                "end_rear_foot_toe_angle_deg": subset[-1]["rear_foot_toe_angle_deg"],
                "maximum_penetration_m": max(row["maximum_penetration_m"] for row in subset),
                "maximum_actual_joint_limit_violation_rad": max(row["maximum_actual_joint_limit_violation_rad"] for row in subset),
                "robot_ground_first_s": next((row["time_physics"] for row in subset if row["unique_robot_geoms_ground"]), None),
                "tail_contact_samples": sum(row["tail_ground"] for row in dynamic),
            }
            value = windows[label]
            value["quality_valid"] = bool(value["maximum_penetration_m"] <= 0.01
                                           and value["maximum_actual_joint_limit_violation_rad"] <= 0.05
                                           and value["robot_ground_first_s"] is None)
        np.savez_compressed(args.out_dir / f"{mode}.npz", state=states, action=executed,
                            qpos=qposes, qvel=qvels, time_physics=np.arange(start, end + 1) * 0.02,
                            control_dt=np.array(0.02), joint_names=np.asarray(G1_JOINT_NAMES))
        (args.out_dir / f"{mode}_contacts.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        summaries.append({"mode": mode, "baseline_replay_error": baseline_error if mode == "baseline" else None,
                          "windows": windows})
        print(json.dumps(summaries[-1], allow_nan=False), flush=True)
    summary = {"schema": "supermimic.ollie.rear_roll_probe.v1", "native_source": str(args.native_dir.resolve()),
               "archive_sha256": file_hash(archive_path), "reference_sha256": source["reference_sha256"],
               "source_checkpoint_sha256": source["checkpoint_sha256"], "robot_mass_kg": robot_mass,
               "mujoco_version": mujoco.__version__, "physical_dt_s": 0.0025, "control_dt_s": 0.02,
               "time_s": args.time, "pulse_seconds": 0.16, "prefix_replay_error": prefix_error,
               "snapshot_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in history.items()},
               "changed_joint": G1_JOINT_NAMES[joint], "branches": summaries,
               "scope": "four fixed conditional branches; later baseline controls are open-loop continuation, not a recovery controller or verified teacher",
               "ollie_success_certified": False}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time", type=float, default=2.76)
    parser.add_argument("--end-time", type=float, default=3.30)
    args = parser.parse_args()
    if not np.isfinite(args.time) or not np.isfinite(args.end_time):
        parser.error("times must be finite")
    run(args)


if __name__ == "__main__":
    main()
