"""One fixed mixed native-MPC/BC preparation episode and supervised repair.

Actual mixed actions and counterfactual MPC labels are separate. This is not
the legacy static-controller DAgger path, WM/BPTT, or Ollie certification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from scripts.local.evaluate_ollie_policy import evaluate, load_policy
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC, quality_summary
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES
from supermimic.ollie.reference import JointReference

REFERENCE = Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz")
FEEDFORWARD = Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz")
CHECKPOINT = Path("logs/ollie_bc_mpc_prep_v4_01/bc_03000.pt")
TEACHER = Path("logs/ollie_mpc_teacher_v4_02/teacher.npz")
HASHES = {REFERENCE: "38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7",
          FEEDFORWARD: "08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115",
          CHECKPOINT: "734c4230e69a7feaa9b0192104709ea855ae2688259dc0cd82497260e7b2fcb1",
          TEACHER: "274963f820973826cb993b9f239f979769e1d06d5a2e8e9b4ff6f3ae15d645e5"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_valid():
    for path, expected in HASHES.items():
        if digest(path) != expected:
            raise ValueError(f"bound input hash mismatch: {path}")


def gates(summary, controls, target_violation, torque_violation):
    s = summary
    return {"complete_continuous_preparation": controls == 138 and abs(s["seconds"] - 2.76) < 1e-9 and s["failure"] is None,
            "upright": s["minimum_upright"] >= .85, "no_robot_ground": s["robot_ground_first_s"] is None,
            "penetration": s["maximum_penetration_m"] <= .01,
            "actual_joint_soft_limits": s["maximum_actual_joint_limit_violation_rad"] <= .05,
            "initial_penetration": s["initial_penetration_m"] <= .002,
            "foot_height": s["maximum_foot_gap_m"] <= .025,
            "foot_planar": s["maximum_foot_planar_slip_m"] <= .04,
            "root_xy": s["maximum_root_board_xy_m"] <= .2,
            "root_height_rmse": s["root_height_rmse_m"] <= .08,
            "root_height_max": s["root_height_max_error_m"] <= .16,
            "both_foot_upward_support": min(s["foot_support_fraction"].values()) >= .8,
            "final_both_foot_upward_support": min(s["final_foot_support_fraction"].values()) >= .8,
            "four_unique_wheels": s["four_wheels_ground_fraction"] >= .8,
            "targets_inside_original_limits": target_violation <= 1e-12,
            "actual_tau_inside_original_caps": torque_violation <= 1e-12}


def replay(qpos, qvel, states, actions):
    """Fresh original physics, actual controls only, independent 400Hz QA."""
    c = NativeSamplingMPC(REFERENCE, FEEDFORWARD, seed=58)
    c.pool.close()
    m, d, x = c.model, c.data, c.index
    if not np.array_equal(d.qpos, qpos[0]) or not np.array_equal(d.qvel, qvel[0]):
        raise ValueError("initial state differs from the unique bound start")
    records = [c.observe()]
    error_q = error_v = error_s = 0.
    forces_zero, time_valid = True, True
    tau = []
    for step, action in enumerate(actions):
        d.ctrl[x.robot_actuators] = action
        for _ in range(8):
            before = float(d.time)
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            time_valid &= bool(np.isclose(d.time, before + .0025, rtol=0, atol=1e-12))
            forces_zero &= not (np.any(d.qfrc_applied) or np.any(d.xfrc_applied))
            records.append(c.observe())
            tau.append(d.actuator_force[x.robot_actuators].copy())
        error_q = max(error_q, float(abs(d.qpos - qpos[step + 1]).max()))
        error_v = max(error_v, float(abs(d.qvel - qvel[step + 1]).max()))
        error_s = max(error_s, float(abs(c.indices.pack(d.qpos, d.qvel).numpy() - states[step + 1]).max()))
    summary = quality_summary(records, float(d.time), None if time_valid else "time_discontinuity")
    limits, fl = c.target_limits, m.actuator_forcerange[x.robot_actuators]
    tv = max(0., float(np.maximum(limits[:, 0] - actions, actions - limits[:, 1]).max())) if len(actions) else 0.
    fv = max(0., float(np.maximum(fl[:, 0] - tau, np.asarray(tau) - fl[:, 1]).max())) if len(tau) else 0.
    passed = gates(summary, len(actions), tv, fv)
    passed.update(exact_native_replay=error_q == error_v == error_s == 0, no_external_forces=bool(forces_zero),
                  finite=bool(all(np.isfinite(a).all() for a in (qpos, qvel, states, actions))))
    return {**summary, "gates": {key: bool(value) for key, value in passed.items()},
            "preparation_quality_passed": bool(all(passed.values())),
            "qpos_replay_max_error": error_q, "qvel_replay_max_error": error_v, "state92_replay_max_error": error_s,
            "maximum_target_limit_violation_rad": tv, "maximum_actual_torque_limit_violation_nm": fv,
            "verified_teacher": False, "ollie_success_certified": False}


def collect(output):
    if output.exists():
        raise ValueError("fresh collection output required")
    inputs_valid()
    output.mkdir(parents=True)
    c = NativeSamplingMPC(REFERENCE, FEEDFORWARD, seed=58)
    proof = c.proof()
    policy, source = load_policy(CHECKPOINT, c.reference.layout, "cpu")
    if source["reference_sha256"] != HASHES[REFERENCE] or policy.cfg.reference_offsets != (0, 5, 10):
        raise ValueError("reference/head contract mismatch")
    if policy.cfg.frozen_prefix_frame is not None:
        raise ValueError("unexpected frozen policy head")
    refs = torch.tensor(c.reference.state, dtype=torch.float32)
    residual_limits = policy.residual_limits.cpu().numpy().reshape(29)
    noise_rng = np.random.default_rng(58)
    m, d, x = c.model, c.data, c.index
    states, poses, velocities = [c.indices.pack(d.qpos, d.qvel).numpy()], [d.qpos.copy()], [d.qvel.copy()]
    actual, labels, bc_raw, bc_target, noises, goals, counterfactual, choices, records = [], [], [], [], [], [], [], [], [c.observe()]
    started, failure = time.monotonic(), None
    max_label_violation = maximum_expert_error = 0.
    for step in range(138):
        if time.monotonic() - started >= 280:
            failure = "collector_wall_budget"
            break
        expert, report, _, predicted = c.choose(step)
        goal_indices = np.minimum(step + 1 + np.array([0, 5, 10]), 257)
        # Independent native counterfactual verifies the expert action. This
        # next-state is not the next-state of the mixed main trajectory.
        shadow = copy.copy(d)
        shadow.ctrl[x.robot_actuators] = expert
        for _ in range(8):
            mujoco.mj_step(m, shadow)
            mujoco.mj_forward(m, shadow)
        expert_error = float(abs(c.state(shadow) - predicted[7]).max())
        maximum_expert_error = max(maximum_expert_error, expert_error)
        if expert_error > 1e-10:
            failure = "expert_counterfactual_branch_mismatch"
            break
        violation = float((abs(expert - c.reference.robot_state[goal_indices[0], 7:36]) - residual_limits).max())
        max_label_violation = max(max_label_violation, violation)
        if violation > 1e-7:
            failure = "expert_label_outside_existing_residual_limits"
            choices.append({**report, "label_limit_violation_rad": violation})
            break
        with torch.inference_mode():
            raw = policy(torch.tensor(states[-1], dtype=torch.float32)[None], refs[goal_indices][None],
                         frame=step + 1)[0].numpy().astype(float)
        clipped = np.clip(raw, c.target_limits[:, 0], c.target_limits[:, 1])
        noise = noise_rng.normal(0, .002, 29)
        executed = np.clip(.75 * expert + .25 * clipped + noise, c.target_limits[:, 0], c.target_limits[:, 1])
        d.ctrl[x.robot_actuators] = executed
        actual.append(executed.copy())
        labels.append(expert.copy())
        bc_raw.append(raw.copy())
        bc_target.append(clipped.copy())
        noises.append(noise.copy())
        goals.append(goal_indices)
        counterfactual.append(c.indices.pack(shadow.qpos, shadow.qvel).numpy())
        choices.append({**report, "expert_first_control_prediction_max_error": expert_error,
                        "label_limit_violation_rad": max(0., violation)})
        for _ in range(8):
            before = float(d.time)
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            row = c.observe()
            records.append(row)
            if (not np.isclose(d.time, before + .0025, rtol=0, atol=1e-12)
                    or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                    or row["external_generalized_force_norm"] or row["external_body_force_norm"]):
                raise RuntimeError("unexpected numerical reset, nonfinite state, or external force")
        states.append(c.indices.pack(d.qpos, d.qvel).numpy())
        poses.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        if step % 10 == 0:
            print(json.dumps({"stage": "collect", "step": step, "time_s": d.time,
                              "upright": records[-1]["robot_upright_cosine"], "wall_seconds": time.monotonic() - started}), flush=True)
        if any(r["robot_upright_cosine"] < .85 or r["maximum_penetration_m"] > .01
               or r["maximum_actual_joint_limit_violation_rad"] > .05 or r["unique_robot_geoms_ground"]
               or r["foot_gap_m"] > .025 or r["foot_planar_slip_m"] > .04
               or r["root_board_xy_m"] > .2 or abs(r["root_height_error_m"]) > .16 for r in records[-8:]):
            failure = "irreversible_preparation_gate_violation"
            break
    collection_wall = time.monotonic() - started
    c.pool.close()
    states, poses, velocities = map(np.asarray, (states, poses, velocities))
    actual = np.asarray(actual).reshape(-1, 29)
    labels = np.asarray(labels).reshape(-1, 29)
    goal_array = np.asarray(goals, dtype=np.int64).reshape(-1, 3)
    audit = replay(poses, velocities, states, actual)
    if failure:
        audit["gates"]["collector_completed_without_failure"] = False
    audit["gates"]["expert_labels_inside_existing_residual_limits"] = max_label_violation <= 1e-7
    audit["gates"]["expert_counterfactual_exact"] = maximum_expert_error <= 1e-10
    passed = bool(all(audit["gates"].values()))
    metadata = {"schema": "supermimic.ollie.mpc_dagger_collection.v1", "scope": "single_initial_state_preparation_only",
                "seed": 58, "expert_seed": 58, "noise_seed": 58, "beta": .75, "noise_std_rad": .002,
                "initializations": 1, "intermediate_resets": 0, "root_board_writes_after_initialization": 0,
                "forces_applied": False, "board_actuators": 0, "input_sha256": {str(p): h for p, h in HASHES.items()},
                "reference_sha256": HASHES[REFERENCE], "collection_wall_seconds": collection_wall,
                "failure": failure, "maximum_expert_prediction_error": maximum_expert_error,
                "maximum_expert_residual_limit_violation_rad": max(0., max_label_violation),
                "mixed_preparation_quality_passed": passed, "verified_teacher": False,
                "label_kind": "counterfactual_native_MPC_action_at_actual_mixed_state",
                "actual_next_state_kind": "actual_mixed_control_transition", "ollie_success_certified": False,
                "audit": audit, "branch_proof": proof}
    np.savez_compressed(output / "mixed_transitions.npz", state=states[:-1], next_state=states[1:],
                        all_boundary_states=states, actual_action=actual, expert_label_action=labels,
                        bc_raw_action=np.asarray(bc_raw).reshape(-1, 29), bc_target_action=np.asarray(bc_target).reshape(-1, 29),
                        gaussian_noise=np.asarray(noises).reshape(-1, 29),
                        expert_counterfactual_next_state=np.asarray(counterfactual).reshape(-1, 92),
                        qpos=poses, qvel=velocities, reference_goal_indices=goal_array,
                        future_goals=c.reference.state[goal_array].astype(np.float32),
                        episode_id=np.zeros(len(actual), dtype=np.int32), time_s=np.arange(len(actual)) * .02,
                        control_dt=np.array(.02), joint_names=np.asarray(G1_JOINT_NAMES),
                        board_joint_names=np.asarray(BOARD_JOINT_NAMES), residual_limits_rad=residual_limits,
                        verified_teacher=np.array(False), metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
    for name, value in (("result", metadata), ("choices", choices), ("contacts", records)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    inputs_valid()
    print(json.dumps({"stage": "collection_complete", "cpu_MPC_released": True,
                      "mixed_preparation_quality_passed": passed, "failure": failure,
                      "failed_gates": [k for k, v in audit["gates"].items() if not v],
                      "wall_seconds": collection_wall, "samples": len(actual)}), flush=True)


def fit(collection, output):
    if output.exists():
        raise ValueError("fresh fitting output required")
    inputs_valid()
    report = json.loads((collection / "result.json").read_text())
    if not report["mixed_preparation_quality_passed"] or not all(report["audit"]["gates"].values()):
        raise ValueError("failed mixed collection cannot be used for training")
    torch.manual_seed(58)
    torch.set_num_threads(2)
    reference = JointReference.load(REFERENCE)
    policy, checkpoint = load_policy(CHECKPOINT, reference.layout, "cuda:0")
    with np.load(collection / "mixed_transitions.npz", allow_pickle=False) as data, np.load(TEACHER, allow_pickle=False) as old:
        if data["state"].shape != (138, 92) or data["expert_label_action"].shape != (138, 29):
            raise ValueError("one complete 138-sample mixed episode is required")
        if not np.array_equal(data["residual_limits_rad"], policy.residual_limits.cpu().numpy().reshape(29)):
            raise ValueError("residual/head contract changed")
        combined = [np.concatenate((old[old_key], data[new_key]), axis=0) for old_key, new_key in
                    (("state", "state"), ("action", "expert_label_action"), ("future_goals", "future_goals"))]
    s, labels, goals = [torch.tensor(a, dtype=torch.float32, device="cuda:0") for a in combined]
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    output.mkdir(parents=True)
    history = []
    started = time.monotonic()
    for update in range(1, 1001):
        if time.monotonic() - started > 55:
            raise RuntimeError("GPU fitting budget exhausted; no additional training")
        policy.train()
        loss = (policy(s, goals) - labels).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("nonfinite supervised fitting")
        optimizer.step()
        if update % 250:
            continue
        policy.eval()
        with torch.inference_mode():
            fit_rmse = float((policy(s, goals) - labels).square().mean().sqrt())
        candidate = copy.deepcopy(checkpoint)
        candidate.update(policy=policy.state_dict(), behavior_cloning_updates_completed=3000 + update,
                         policy_initialization="native_MPC_DAgger_supervised", seed=58,
                         mpc_dagger_updates_completed=update, mpc_dagger_collection_sha256=digest(collection / "mixed_transitions.npz"),
                         mpc_dagger_collection_metadata=report, world_model_trained=False,
                         policy_updates_completed=0, ollie_success_certified=False)
        path = output / f"mpc_dagger_{update:05d}.pt"
        torch.save(candidate, path)
        native_dir = output / f"native_{update:05d}"
        native = evaluate(path, REFERENCE, native_dir, seconds=2.76)
        with np.load(native_dir / "rollout.npz", allow_pickle=False) as saved:
            screening_xy = float(np.linalg.norm(saved["state"][:, :2] - saved["state"][:, 71:73], axis=-1).max())
            qa = replay(saved["qpos"], saved["qvel"], saved["state"], saved["control"])
        (native_dir / "independent_preparation_audit.json").write_text(json.dumps(qa, indent=2, allow_nan=False) + "\n")
        cost = (native["ranking_cost"] + 10 * max(0., .85 - native["minimum_robot_upright_cosine"])
                + 10 * max(0., screening_xy - .2))
        row = {"additional_update": update, "combined_action_rmse_rad": fit_rmse, "legacy_prefix_cost": cost,
               "preparation_quality_passed": qa["preparation_quality_passed"],
               "failed_gates": [k for k, v in qa["gates"].items() if not v], "checkpoint_sha256": digest(path),
               "checkpoint": str(path), "screening_boundary_maximum_xy_m": screening_xy, "audit": qa}
        history.append(row)
        print(json.dumps({key: row[key] for key in ("additional_update", "combined_action_rmse_rad", "preparation_quality_passed", "failed_gates")}), flush=True)
    passing = [r for r in history if r["preparation_quality_passed"]]
    chosen = min(passing or history, key=lambda r: r["legacy_prefix_cost"])
    summary = {"schema": "supermimic.ollie.mpc_dagger_supervised_repair.v1", "seed": 58, "updates": 1000,
               "learning_rate": 1e-4, "schedule": "constant", "samples": 276,
               "selected_update": chosen["additional_update"], "selected_passing": bool(passing),
               "selection_rule": "minimum legacy cost among gate-passing set; empty set selects FAILED diagnostic only",
               "history": history, "wall_seconds": time.monotonic() - started,
               "world_model_trained": False, "bptt_updates": 0, "ollie_success_certified": False,
               "collection_sha256": digest(collection / "mixed_transitions.npz")}
    # Exactly one full diagnostic after fixed 4-candidate selection.
    summary["full_diagnostic"] = evaluate(chosen["checkpoint"], REFERENCE, output / "full_diagnostic", seconds=5.14)
    (output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    inputs_valid()
    print(json.dumps({"stage": "fitting_complete", "selected_update": summary["selected_update"],
                      "selected_passing": summary["selected_passing"], "full_outcome": summary["full_diagnostic"]["outcome"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "fit"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "collect":
        collect(args.out_dir)
    else:
        if args.collection is None:
            parser.error("fit requires --collection")
        fit(args.collection, args.out_dir)


if __name__ == "__main__":
    main()
