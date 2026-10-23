"""Replay saved robot controls and audit the entire collision envelope on CPU.

Defaults match PD best_rollout.npz: actuator-order controls held for its timestep,
saved qpos/qvel, and zero initial ctrl. Unknown archives need explicit CLI choices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local.evaluate_ollie_policy import ContactObserver, continuous_windows, outcome
from scripts.video.audit_ollie_timing import geometry_minimum_z
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_report(mask, times):
    windows = continuous_windows(mask, times)
    return {
        "total_seconds": float(np.count_nonzero(mask) * (times[1] - times[0])),
        "longest_continuous_seconds": max((row["duration_s"] for row in windows), default=0.0),
        "windows": windows,
    }


def audit(args):
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.rollout, allow_pickle=False) as source:
        qpos = source["qpos"].copy()
        qvel = source["qvel"].copy()
        controls = source["control" if "control" in source else "action"].copy()
        archive_dt = float(source["timestep"]) if "timestep" in source else None
    control_dt = args.control_dt if args.control_dt is not None else archive_dt
    if control_dt is None or not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("Provide --control-dt for an archive without a timestep")
    physics_dt = args.physics_dt if args.physics_dt is not None else min(control_dt, 0.0025)
    if not np.isclose(physics_dt, 0.0025):
        raise ValueError("Current shared continuous-window contract requires 0.0025s physics steps")
    substeps = round(control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
        raise ValueError("Control interval must be a positive multiple of the physics step")
    model, _, index = compile_scene(timestep=physics_dt)
    if qpos.shape != (len(controls) + 1, model.nq) or qvel.shape != (len(qpos), model.nv):
        raise ValueError(
            "Saved qpos/qvel must cover each held control boundary, including initialization"
        )
    if controls.shape != (len(qpos) - 1, 29) or not all(
        np.isfinite(array).all() for array in (qpos, qvel, controls)
    ):
        raise ValueError("Expected finite qpos/qvel and 29-dimensional actual controls")
    if model.nu != 29 or model.neq != 0 or mujoco.get_mjcb_control() is not None:
        raise ValueError(
            "Requires only 29 robot actuators, no equality constraints or global callback"
        )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
    if args.initial_control == "initial-pose":
        data.ctrl[index.robot_actuators] = qpos[0, index.robot_joint_qpos]
    elif args.initial_control == "first-action":
        if args.control_order == "actuator":
            data.ctrl[:] = controls[0]
        else:
            data.ctrl[index.robot_actuators] = controls[0]
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, index, args.force_threshold)
    ground = model.geom("ground").id
    active = [
        gid
        for gid in range(model.ngeom)
        if model.geom_bodyid[gid] != 0
        and (
            (model.geom_contype[gid] & model.geom_conaffinity[ground])
            or (model.geom_contype[ground] & model.geom_conaffinity[gid])
        )
    ]
    board = [gid for gid in active if gid in observer.board]
    if set(board) != observer.board:
        raise ValueError("Every board geom must participate in ground collision")
    board_columns = [active.index(gid) for gid in board]
    records, minimums, forces, replay_qpos, replay_qvel = [], [], [], [], []
    board_rotations = []

    def observe():
        row = observer.observe(data)
        minimum = np.array([geometry_minimum_z(model, data, [gid]) for gid in active])
        if not np.isclose(
            minimum[board_columns].min(), row["minimum_board_clearance_m"], atol=1e-12
        ):
            raise RuntimeError("Shared board-envelope implementations disagree")
        normal = np.zeros(len(active))
        for contact_id, contact in enumerate(data.contact[: data.ncon]):
            if ground not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == ground else contact.geom[0])
            if other not in active:
                continue
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            normal[active.index(other)] += max(0.0, float(force[0]))
        row["board_ground_normal_force_n"] = float(normal[board_columns].sum())
        records.append(row)
        minimums.append(minimum)
        forces.append(normal)
        replay_qpos.append(data.qpos.copy())
        replay_qvel.append(data.qvel.copy())
        board_rotations.append(data.xmat[index.board_body].reshape(3, 3).copy())

    observe()
    error_qpos = error_qvel = 0.0
    for action_index, control in enumerate(controls):
        if args.control_order == "actuator":
            data.ctrl[:] = control
        else:
            data.ctrl[index.robot_actuators] = control
        for _ in range(substeps):
            previous_time = float(data.time)
            mujoco.mj_step(model, data)
            if not np.isclose(data.time, previous_time + physics_dt, atol=1e-10, rtol=0):
                raise RuntimeError("Native time discontinuity; no automatic restart is allowed")
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Nonfinite native replay")
            observe()
        error_qpos = max(error_qpos, float(np.max(np.abs(data.qpos - qpos[action_index + 1]))))
        error_qvel = max(error_qvel, float(np.max(np.abs(data.qvel - qvel[action_index + 1]))))
    times = np.array([row["time_physics"] for row in records])
    minimums, forces = np.asarray(minimums), np.asarray(forces)
    board_lowest = minimums[:, board_columns].min(axis=1)
    board_force = forces[:, board_columns].sum(axis=1)
    force_free = board_force < args.force_threshold
    tiny_gap = force_free & (board_lowest > 1e-6)
    qualified = force_free & (board_lowest > args.clearance)
    qualified_windows = [
        row
        for row in continuous_windows(qualified, times)
        if row["duration_s"] >= args.minimum_flight - 1e-9
    ]
    rotations = np.asarray(board_rotations)
    report = {
        "schema": "supermimic.ollie.full_envelope_audit.v1",
        "input": str(args.rollout.resolve()),
        "input_sha256": file_hash(args.rollout),
        "audit_script_sha256": file_hash(Path(__file__)),
        "physics": "native CPU MuJoCo",
        "mujoco_version": mujoco.__version__,
        "physics_dt": physics_dt,
        "control_dt": control_dt,
        "held_substeps": substeps,
        "control_order": args.control_order,
        "initial_control": args.initial_control,
        "qpos_boundary_replay_max_error": error_qpos,
        "qvel_boundary_replay_max_error": error_qvel,
        "exact_replay_tolerance": args.tolerance,
        "exact_replay_passed": error_qpos <= args.tolerance and error_qvel <= args.tolerance,
        "board_collision_geom_names": [model.geom(gid).name for gid in board],
        "g1_order_robot_actuator_ids": index.robot_actuators.tolist(),
        "ground_plane_z": 0.0,
        "board_envelope_max_clearance_m": float(board_lowest.max()),
        "all_wheel_cylinder_minimum_max_clearance_m": max(
            row["minimum_wheel_clearance_m"] for row in records
        ),
        "force_threshold_n": args.force_threshold,
        "positive_gap_threshold_m": 1e-6,
        "force_free_positive_gap": window_report(tiny_gap, times),
        "qualified_board_flight_clearance_m": args.clearance,
        "qualified_board_flight_minimum_duration_s": args.minimum_flight,
        "qualified_board_flight_windows": qualified_windows,
        "maximum_board_tilt_deg": float(
            np.rad2deg(np.arccos(np.clip(rotations[:, 2, 2], -1, 1))).max()
        ),
        "minimum_board_up_cosine": float(rotations[:, 2, 2].min()),
        "final_board_up_cosine": float(rotations[-1, 2, 2]),
        "contact_samples": len(records),
        "no_pose_writes_after_initialization": True,
        "outcome": outcome(records, full_reference=True, numerical_failure=None),
        "scope": "A complete saved control episode, not a full original-video reproduction or learned policy claim.",
    }
    if args.legacy_contacts:
        legacy_rows = list(csv.DictReader(args.legacy_contacts.open()))
        if len(legacy_rows) != len(records) - 1 or not np.allclose(
            [float(row["time_s"]) for row in legacy_rows], times[1:], atol=1e-9
        ):
            raise ValueError("Legacy contacts must have one row per replayed physics substep")
        proxy = np.r_[
            False,
            [
                float(row["minimum_wheel_clearance_m"]) > 0.005 and int(row["board_ground"]) == 0
                for row in legacy_rows
            ],
        ]
        report["legacy_contacts_sha256"] = file_hash(args.legacy_contacts)
        report["legacy_wheel_no_contact_proxy"] = window_report(proxy, times)
        if proxy.any():
            report["legacy_proxy_whole_board_clearance_range_m"] = [
                float(board_lowest[proxy].min()),
                float(board_lowest[proxy].max()),
            ]
            report["legacy_proxy_robot_ground_fraction"] = float(
                np.mean(
                    [
                        records[slot]["unique_robot_geoms_ground"] > 0
                        for slot in np.flatnonzero(proxy)
                    ]
                )
            )
    with (args.out_dir / "contacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    names = [model.geom(gid).name for gid in active]
    with (args.out_dir / "per_geom_support.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_physics",
                *[f"{name}.minimum_world_z_m" for name in names],
                *[f"{name}.ground_normal_force_n" for name in names],
            ]
        )
        writer.writerows(np.c_[times, minimums, forces])
    np.savez_compressed(
        args.out_dir / "geometry_and_forces.npz",
        time_physics=times,
        geom_names=np.asarray([model.geom(gid).name for gid in active]),
        geom_ids=np.asarray(active),
        board_columns=np.asarray(board_columns),
        geom_minimum_world_z_m=minimums,
        geom_ground_normal_force_n=forces,
        board_rotation_world=rotations,
    )
    np.savez_compressed(
        args.out_dir / "actual_replay.npz",
        qpos=np.asarray(replay_qpos),
        qvel=np.asarray(replay_qvel),
        control=controls,
        control_order=np.asarray(args.control_order),
        control_dt=np.asarray(control_dt),
        physics_dt=np.asarray(physics_dt),
        time_physics=times,
        board_joint_names=np.asarray(BOARD_JOINT_NAMES),
        robot_joint_names=np.asarray(G1_JOINT_NAMES),
    )
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--legacy-contacts", type=Path)
    parser.add_argument("--control-dt", type=float)
    parser.add_argument("--physics-dt", type=float)
    parser.add_argument("--control-order", choices=("actuator", "g1"), default="actuator")
    parser.add_argument(
        "--initial-control", choices=("zero", "initial-pose", "first-action"), default="zero"
    )
    parser.add_argument("--force-threshold", type=float, default=0.1)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--minimum-flight", type=float, default=0.04)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if min(args.force_threshold, args.clearance, args.minimum_flight, args.tolerance) <= 0:
        parser.error("Thresholds must be positive")
    report = audit(args)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "input",
                    "exact_replay_passed",
                    "qpos_boundary_replay_max_error",
                    "qvel_boundary_replay_max_error",
                    "board_envelope_max_clearance_m",
                    "qualified_board_flight_windows",
                    "contact_samples",
                )
            },
            indent=2,
        )
    )
    if not report["exact_replay_passed"]:
        raise SystemExit(
            "Replay mismatch: do not certify this source using the reconstructed episode"
        )


if __name__ == "__main__":
    main()
