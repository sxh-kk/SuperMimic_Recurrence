"""Verify and export one hash-bound native-MPC preparation prefix only.

No training is run. The original MPC files remain unverified and unchanged;
only a new export may be certified after complete native replay and all gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices

REFERENCE_SHA = "38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7"
ROLLOUT_SHA = "559fdfb376120a1529bbe5e3de8f0537103c7f4001a4ed3934e5be7152b9b96e"
FEEDFORWARD_SHA = "08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(reference_path, source_dir):
    source_path = source_dir / "actual_rollout.npz"
    require(digest(reference_path) == REFERENCE_SHA, "reference must be the explicitly bound v4 file")
    require(digest(source_path) == ROLLOUT_SHA, "actual trajectory SHA mismatch")
    source_result = json.loads((source_dir / "result.json").read_text())
    source_contacts = json.loads((source_dir / "contacts.json").read_text())
    require(source_result["reference_sha256"] == REFERENCE_SHA, "source reference metadata mismatch")
    require(source_result["feedforward_sha256"] == FEEDFORWARD_SHA, "source FF metadata mismatch")
    ff_path = Path(source_result["feedforward"])
    require(digest(ff_path) == FEEDFORWARD_SHA, "source static FF SHA mismatch")
    require(source_result["verified_teacher"] is False, "do not rewrite the source pilot certification")
    require(source_result["ollie_success_certified"] is False, "source scope must remain preparation only")
    require(source_result["initializations"] == 1 and source_result["root_board_writes_after_initialization"] == 0,
            "source contains extra resets or root/board writes")
    require(source_result["board_actuators"] == 0 and source_result["contact_forces_applied"] is False,
            "source contains board actuation or applied forces")
    require(source_result["mujoco_version"] == mujoco.__version__, "native MuJoCo version differs")
    require(source_result["physics_dt_s"] == .0025 and source_result["control_dt_s"] == .02,
            "source physical/control dt differs")
    with np.load(source_path, allow_pickle=False) as arrays:
        source = {key: arrays[key].copy() for key in arrays.files}
    for key, shape in (("state", (139, 92)), ("action", (138, 29)), ("qpos", (139, 47)),
                       ("qvel", (139, 45)), ("time_physics", (139,))):
        require(source[key].shape == shape and np.isfinite(source[key]).all(), f"invalid {key} shape/values")
    require(tuple(source["robot_joint_names"].tolist()) == G1_JOINT_NAMES, "robot joint order mismatch")
    require(tuple(source["board_joint_names"].tolist()) == BOARD_JOINT_NAMES, "board joint order mismatch")
    require(source["control_dt"].item() == .02 and not source["verified_teacher"].item(), "source flags/dt differ")
    require(np.array_equal(source["time_physics"], np.arange(139) * .02), "source time grid differs")
    require(len(source_contacts) == 1105, "source must contain initial + 1104 physical-step observations")
    require(mujoco.get_mjcb_control() is None, "no global control callback allowed")
    reference = JointReference.load(reference_path)
    require(reference.state.shape == (258, 92), "bound v4 reference shape differs")
    model, data, index = compile_scene(timestep=.0025)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    qpos_ref, qvel_ref = [v.numpy() for v in indices.unpack(torch.tensor(reference.state))]
    require(np.array_equal(source["qpos"][0], qpos_ref[0]), "source initial pose differs from bound reference")
    require(np.count_nonzero(source["qvel"][0]) == 0, "source initial velocity is not zero")
    require(model.nu == 29 and model.na == 0, "unexpected robot actuators/activation")
    limits = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]].copy()
    target_limits = limits + np.array([.005, -.005])
    target_violation = float(np.maximum(target_limits[:, 0] - source["action"],
                                        source["action"] - target_limits[:, 1]).max())
    require(target_violation <= 1e-12, "actual target exceeds original joint range with source margin")
    feet = [model.site(side + "_foot").id for side in ("left", "right")]
    pelvis = model.body("pelvis").id
    ref_feet = []
    scratch = mujoco.MjData(model)
    for q, v in zip(qpos_ref[:139], qvel_ref[:139], strict=True):
        scratch.qpos[:], scratch.qvel[:] = q, v
        mujoco.mj_forward(model, scratch)
        rotation = scratch.xmat[index.board_body].reshape(3, 3)
        ref_feet.append((scratch.site_xpos[feet] - scratch.xpos[index.board_body]) @ rotation)
    ref_feet = np.asarray(ref_feet)
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = source["qpos"][0], source["qvel"][0]
    data.ctrl[index.robot_actuators] = reference.robot_state[0, 7:36]
    mujoco.mj_forward(model, data)
    require(np.array_equal(indices.pack(data.qpos, data.qvel).numpy(), source["state"][0]), "initial state92 mismatch")
    observer = ContactObserver(model, index, .1)
    records, torques = [], []
    exact_observer = True
    external_forces_zero = True

    def observe():
        nonlocal exact_observer, external_forces_zero
        row = observer.observe(data)
        previous = source_contacts[len(records)]
        exact_observer &= all(previous.get(key) == value for key, value in row.items())
        external_forces_zero &= not (np.any(data.qfrc_applied) or np.any(data.xfrc_applied))
        coordinate = np.clip(data.time / .02, 0, 138)
        lo, hi = int(np.floor(coordinate)), min(int(np.floor(coordinate)) + 1, 138)
        weight = coordinate - lo
        desired = (1 - weight) * ref_feet[lo] + weight * ref_feet[hi]
        rotation = data.xmat[index.board_body].reshape(3, 3)
        error = (data.site_xpos[feet] - data.xpos[index.board_body]) @ rotation - desired
        robot_ref = reference.robot_state[lo] * (1 - weight) + reference.robot_state[hi] * weight
        row.update(foot_gap_m=float(abs(error[:, 2]).max()),
                   foot_planar_slip_m=float(np.linalg.norm(error[:, :2], axis=1).max()),
                   root_height_error_m=float(data.qpos[index.robot_qpos + 2] - robot_ref[2]),
                   root_board_xy_m=float(np.linalg.norm(data.qpos[index.robot_qpos:index.robot_qpos + 2]
                                                       - data.qpos[index.board_qpos:index.board_qpos + 2])),
                   joint_position_error_rad=(data.qpos[index.robot_joint_qpos] - robot_ref[7:36]).tolist(),
                   joint_velocity_error_rad_s=(data.qvel[index.robot_joint_qvel] - robot_ref[42:71]).tolist(),
                   board_xy_displacement_m=float(np.linalg.norm(data.xpos[index.board_body, :2]
                                                               - source["qpos"][0, index.board_qpos:index.board_qpos + 2])))
        records.append(row)
        torques.append(data.actuator_force[index.robot_actuators].copy())

    observe()
    q_error = v_error = state_error = 0.
    for step, action in enumerate(source["action"]):
        data.ctrl[index.robot_actuators] = action
        for _ in range(8):
            previous_time = float(data.time)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            require(np.isclose(data.time, previous_time + .0025, rtol=0, atol=1e-12), "physical time discontinuity")
            require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "native nonfinite state")
            observe()
        q_error = max(q_error, float(abs(data.qpos - source["qpos"][step + 1]).max()))
        v_error = max(v_error, float(abs(data.qvel - source["qvel"][step + 1]).max()))
        state_error = max(state_error, float(abs(indices.pack(data.qpos, data.qvel).numpy()
                                                  - source["state"][step + 1]).max()))
    final = [r for r in records if r["time_physics"] >= data.time - .3]
    torques = np.asarray(torques)
    force_limits = model.actuator_forcerange[index.robot_actuators]
    tau_violation = float(np.maximum(force_limits[:, 0] - torques, torques - force_limits[:, 1]).max())
    metrics = {
        "minimum_upright": min(r["robot_upright_cosine"] for r in records),
        "robot_ground": any(r["unique_robot_geoms_ground"] for r in records),
        "maximum_penetration_m": max(r["maximum_penetration_m"] for r in records),
        "maximum_actual_joint_limit_violation_rad": max(r["maximum_actual_joint_limit_violation_rad"] for r in records),
        "initial_penetration_m": records[0]["maximum_penetration_m"],
        "foot_gap_max_m": max(r["foot_gap_m"] for r in records),
        "foot_planar_slip_max_m": max(r["foot_planar_slip_m"] for r in records),
        "root_board_xy_max_m": max(r["root_board_xy_m"] for r in records),
        "root_height_rmse_m": float(np.mean([r["root_height_error_m"] ** 2 for r in records]) ** .5),
        "root_height_max_error_m": max(abs(r["root_height_error_m"]) for r in records),
        "foot_deck_upward_support_above_1n_fraction": {side: float(np.mean([r[f"{side}_foot_deck_upward_support_n"] > 1 for r in records])) for side in ("left", "right")},
        "final_300ms_foot_deck_upward_support_above_1n_fraction": {side: float(np.mean([r[f"{side}_foot_deck_upward_support_n"] > 1 for r in final])) for side in ("left", "right")},
        "four_wheels_ground_fraction": float(np.mean([r["unique_wheels_ground"] == 4 for r in records])),
        "joint_position_rmse_rad": float(np.mean(np.square([r["joint_position_error_rad"] for r in records])) ** .5),
        "joint_velocity_rmse_rad_s": float(np.mean(np.square([r["joint_velocity_error_rad_s"] for r in records])) ** .5),
        "maximum_board_xy_displacement_m": max(r["board_xy_displacement_m"] for r in records),
        "qpos_replay_max_error": q_error, "qvel_replay_max_error": v_error, "state92_replay_max_error": state_error,
        "all_source_contact_observer_fields_exact": bool(exact_observer),
        "maximum_target_limit_violation_rad": max(0., target_violation),
        "maximum_actual_torque_limit_violation_nm": max(0., tau_violation),
        "maximum_actual_torque_cap_fraction": float((abs(torques) / np.max(abs(force_limits), axis=1)).max()),
    }
    s = metrics
    gates = {
        "full_138_control_1104_physics_steps": len(records) == 1105 and abs(data.time - 2.76) < 1e-9,
        "native_exact_replay": q_error == v_error == state_error == 0,
        "all_physical_source_observations_exact": bool(exact_observer),
        "no_external_forces": bool(external_forces_zero),
        "targets_inside_original_limits": target_violation <= 1e-12,
        "actual_torque_inside_original_caps": tau_violation <= 1e-12,
        "upright": s["minimum_upright"] >= .85, "no_robot_ground": not s["robot_ground"],
        "maximum_penetration": s["maximum_penetration_m"] <= .01,
        "maximum_actual_joint_violation": s["maximum_actual_joint_limit_violation_rad"] <= .05,
        "initial_penetration": s["initial_penetration_m"] <= .002,
        "foot_gap": s["foot_gap_max_m"] <= .025, "foot_planar_error": s["foot_planar_slip_max_m"] <= .04,
        "root_board_xy": s["root_board_xy_max_m"] <= .2,
        "root_height_rmse": s["root_height_rmse_m"] <= .08,
        "root_height_maximum": s["root_height_max_error_m"] <= .16,
        "both_foot_upward_support": min(s["foot_deck_upward_support_above_1n_fraction"].values()) >= .8,
        "final_both_foot_upward_support": min(s["final_300ms_foot_deck_upward_support_above_1n_fraction"].values()) >= .8,
        "four_unique_wheels_ground": s["four_wheels_ground_fraction"] >= .8,
    }
    mujoco.mj_subtreeVel(model, data)
    root_rotation = data.xmat[pelvis].reshape(3, 3)
    terminal = {
        "time_physics_s": float(data.time), "time_video_s": float(reference.time_video[138]),
        "robot_root_linear_world_m_s": data.qvel[index.robot_qvel:index.robot_qvel + 3].tolist(),
        "robot_root_angular_world_rad_s": (root_rotation @ data.qvel[index.robot_qvel + 3:index.robot_qvel + 6]).tolist(),
        "robot_com_velocity_world_m_s": data.subtree_linvel[pelvis].tolist(),
        "board_linear_world_m_s": data.qvel[index.board_qvel:index.board_qvel + 3].tolist(),
        "pelvis_upright": float(data.xmat[pelvis, 8]),
        "foot_upward_support_n": [records[-1][side + "_foot_deck_upward_support_n"] for side in ("left", "right")],
        "unique_wheels_ground": records[-1]["unique_wheels_ground"],
    }
    integration = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    return source, reference, {**metrics, "gates": {key: bool(value) for key, value in gates.items()},
                               "terminal": terminal}, integration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"))
    parser.add_argument("--source-dir", type=Path, default=Path("logs/ollie_sampling_mpc_prep_01"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out_dir.exists(), "output directory must be fresh")
    source_files = [args.source_dir / name for name in ("actual_rollout.npz", "result.json", "contacts.json")]
    before = {str(path): digest(path) for path in source_files}
    source, reference, metrics, integration = verify(args.reference, args.source_dir)
    metadata = {
        "schema": "supermimic.ollie.preparation_teacher.v1", "teacher_kind": "native_sampling_mpc",
        "scope": "single_initial_state_preparation_prefix_only", "steady_state": False,
        "robustness_verified": False, "ollie_success_certified": False, "valid_physics_interval_s": [0., 2.76],
        "reference_sha256": REFERENCE_SHA, "source_trajectory_sha256": ROLLOUT_SHA,
        "feedforward_sha256": FEEDFORWARD_SHA, "source_files_sha256": before,
        "initializations": 1, "intermediate_resets": 0, "board_actuators": 0, "forces_applied": False,
        "physical_dt": .0025, "control_dt": .02, "mujoco_version": mujoco.__version__,
        "future_reference_outside_preparation_verified": False,
        "terminal_continuation_requires_warm_state_or_complete_prefix_replay": True,
        "source_pilot_verified_teacher_remains_false": True,
        **metrics,
    }
    args.out_dir.mkdir(parents=True)
    passed = bool(all(metrics["gates"].values()))
    metadata["verified_teacher"] = passed
    if not passed:
        (args.out_dir / "rejected.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        raise RuntimeError("native preparation gates failed; no teacher exported")
    count = len(source["action"])
    offsets = np.array([0, 5, 10])
    goals = np.minimum(np.arange(1, count + 1)[:, None] + offsets, 257)
    target = args.out_dir / "teacher.npz"
    pending = args.out_dir / ".teacher.pending.npz"
    np.savez_compressed(pending, state=source["state"][:-1], next_state=source["state"][1:],
                        all_boundary_states=source["state"], action=source["action"],
                        qpos=source["qpos"], qvel=source["qvel"],
                        episode_id=np.zeros(count, dtype=np.int32), time_s=np.arange(count) * .02,
                        time_video_s=reference.time_video[:count], control_dt=np.array(.02),
                        reference_sha256=np.array(REFERENCE_SHA), reference_offsets=offsets,
                        reference_goal_indices=goals, future_goals=reference.state[goals].astype(np.float32),
                        joint_names=np.asarray(G1_JOINT_NAMES), board_joint_names=np.asarray(BOARD_JOINT_NAMES),
                        terminal_state=source["state"][-1], terminal_integration_state=integration,
                        terminal_integration_state_spec=np.array(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
                        verified_teacher=np.array(True), metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
    if not all(digest(path) == before[str(path)] for path in source_files):
        pending.unlink()
        raise RuntimeError("source pilot files changed during export; no teacher published")
    pending.replace(target)
    report = {**metadata, "teacher_sha256": digest(target), "teacher": str(target.resolve()),
              "source_files_unchanged": True, "counts": {"state": count, "next_state": count, "actions": count,
                                                         "boundaries": count + 1, "physical_substeps": 1104}}
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
