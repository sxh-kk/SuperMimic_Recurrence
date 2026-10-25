"""One bounded native response/linear prior experiment, not policy training."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from scripts.local import optimize_ollie_top_support_landing as landing
from scripts.local.continue_ollie_sampling_mpc import fingerprint
from scripts.local.evaluate_ollie_policy import outcome
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES

protocol = landing.protocol
SCHEMA = "supermimic.ollie.prelaunch_response_prior.v1"
WARM_SHA = "fc0a113a3003f2ff669510cbfe3ad4d6f0ff31f05ace9738724707d91167a9d6"
FEATURES = (
    "heading_lateral_capture_point_m",
    "relative_com_world_vz_m_s",
    "root_minus_board_world_z_m",
    "robot_upright",
    "board_upright",
    "board_world_z_m",
    "left_site_board_x_m",
    "left_site_board_y_m",
    "right_site_board_x_m",
    "right_site_board_y_m",
)
SCALES = np.array([0.04, 0.25, 0.05, 0.1, 0.1, 0.02, 0.05, 0.05, 0.05, 0.05])


@dataclass(frozen=True)
class Config:
    prefix_controls: int = 145
    horizon: int = 112
    short_controls: int = 8
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    pulse_amplitude_rad: float = 0.03
    lateral_cp_target_m: float = -0.03
    leg_bound_rad: float = 0.15
    other_bound_rad: float = 0.30
    regularizer_weight: float = 0.1
    regularizer_scale_rad: float = 0.15
    maximum_branches: int = 61
    wall_budget_s: float = 120.0


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def features(controller, actual):
    """True endpoint pose/Jacobian velocity; no force solve or integration."""
    model, ix = controller.model, controller.index
    position = copy.copy(actual)
    mujoco.mj_fwdPosition(model, position)
    robot, board = model.body("pelvis").id, ix.board_body
    jacobian = np.empty((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, position, jacobian, robot)
    robot_v = jacobian @ actual.qvel
    mujoco.mj_jacSubtreeCom(model, position, jacobian, board)
    board_v = jacobian @ actual.qvel
    rc, bc = position.subtree_com[robot].copy(), position.subtree_com[board].copy()
    rotation = position.xmat[board].reshape(3, 3)
    heading = rotation[:2, 0].copy()
    heading /= max(np.linalg.norm(heading), 1e-8)
    lateral = np.array([-heading[1], heading[0]])
    omega = np.sqrt(9.81 / max(rc[2] - position.xpos[board, 2], 0.2))
    cp = rc[:2] - bc[:2] + (robot_v[:2] - board_v[:2]) / omega
    feet = [
        rotation.T @ (position.site_xpos[model.site(side + "_foot").id] - position.xpos[board])
        for side in ("left", "right")
    ]
    vector = np.array(
        [
            cp @ lateral,
            robot_v[2] - board_v[2],
            position.xpos[robot, 2] - position.xpos[board, 2],
            position.xmat[robot, 8],
            position.xmat[board, 8],
            position.xpos[board, 2],
            *feet[0][:2],
            *feet[1][:2],
        ]
    )
    details = {
        "time_native_s": float(actual.time),
        "stage": "actual endpoint position/Jacobian; no cold force",
        "features": dict(zip(FEATURES, vector.tolist(), strict=True)),
        "robot_com_world_xyz_m": rc.tolist(),
        "board_com_world_xyz_m": bc.tolist(),
        "robot_com_world_velocity_m_s": robot_v.tolist(),
        "board_com_world_velocity_m_s": board_v.tolist(),
        "relative_com_world_velocity_m_s": (robot_v - board_v).tolist(),
        "capture_point_world_xy_m": cp.tolist(),
        "capture_point_is_lip_soft_hint": True,
    }
    return vector, details


def branch(controller, controls, deadline, budget, label):
    """Only canonical scratch29PD steps, sharing the exact warm solver state."""
    if budget[0] >= Config.maximum_branches:
        raise RuntimeError("Fixed61-branch budget exhausted")
    budget[0] += 1
    model, ix = controller.model, controller.index
    data = copy.copy(controller.data)
    arrays = {
        "qpos": [data.qpos.copy()],
        "qvel": [data.qvel.copy()],
        "state": [controller.indices.pack(data.qpos, data.qvel).numpy()],
        "time_physics": [float(data.time)],
    }
    records = [controller.observer.observe(data)]
    physical, force_rows = [], []
    for target in controls:
        data.ctrl[ix.robot_actuators] = target
        for _ in range(8):
            if time.monotonic() > deadline:
                raise TimeoutError("Fixed120s wall budget exhausted")
            mujoco.mj_step(model, data)
            force_rows.append(
                {
                    "interval_start_s": float(data.time - model.opt.timestep),
                    "interval_end_s": float(data.time),
                    **protocol.foot_forces(controller, data),
                }
            )
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"Nonfinite native branch: {label}")
            row = controller.observer.observe(data)
            if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                raise RuntimeError("Unexpected external force")
            records.append(row)
            physical.append(controller.state(data))
        arrays["qpos"].append(data.qpos.copy())
        arrays["qvel"].append(data.qvel.copy())
        arrays["state"].append(controller.indices.pack(data.qpos, data.qvel).numpy())
        arrays["time_physics"].append(float(data.time))
    vector, details = features(controller, data)
    return {
        **{key: np.asarray(value) for key, value in arrays.items()},
        "action": np.asarray(controls).copy(),
        "fullphysics": np.asarray(physical),
        "records": records,
        "execution_force_rows": force_rows,
        "features": vector,
        "feature_details": details,
        "label": label,
    }


def save_branch(path, result):
    np.savez_compressed(
        path, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("logs/ollie_recoverable_verified_candidate_01/actual_rollout.npz"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("logs/ollie_survival_priority_landing_01/actual_rollout.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("A fresh output directory is required")
    started, cfg = time.monotonic(), Config()
    deadline, count = started + cfg.wall_budget_s, [0]
    if (
        protocol.digest(args.source) != landing.SOURCE_SHA
        or protocol.digest(args.baseline) != WARM_SHA
    ):
        raise ValueError("Frozen actual82/85 source hash mismatch")
    full, warm = load(args.source), load(args.baseline)
    if not np.array_equal(full["action"][:145], warm["action"][:145]):
        raise ValueError("Actual82/85 do not share the exact original prefix")
    if warm["action"].shape != (257, 29) or bool(warm["verified_teacher"]):
        raise ValueError("Expected complete actual85, not a teacher")
    baseline = warm["action"][145:].copy()
    wave = np.sin(np.pi * ((np.arange(8) + 0.5) * cfg.control_dt) / 0.16)
    physics = protocol.model_sources()
    frozen85 = json.loads((args.baseline.parent / "frozen_manifest.json").read_text())
    if physics != frozen85["canonical_assets"]:
        raise ValueError("Canonical physical assets differ from actual85")
    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "config": asdict(cfg),
        "feature_names": FEATURES,
        "feature_scales": SCALES.tolist(),
        "wave_control_midpoint_samples": wave.tolist(),
        "source82": {"path": str(args.source.resolve()), "sha256": landing.SOURCE_SHA},
        "baseline85": {"path": str(args.baseline.resolve()), "sha256": WARM_SHA},
        "script_sha256": protocol.digest(__file__),
        "mujoco_version": mujoco.__version__,
        "material": "canonical12deg/defaultmu.6",
        "is_teacher": False,
        "physics": physics,
        "algorithm": "one native response linear prior; no learning",
        "baseline_fullphysics_archive_sha256": protocol.digest(
            args.baseline.parent / "winner_evaluated_branch.npz"
        ),
        "baseline_execution_force_archive_sha256": protocol.digest(
            args.baseline.parent / "actual_execution_forces.json"
        ),
    }
    protocol.write_json(args.out_dir / "frozen_manifest.json", manifest)
    adapter, _, _ = landing.prepare_source(args.source, args.out_dir / "input_view")
    source = load(adapter)
    controller = NativeSamplingMPC(
        Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
        Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"),
        seed=87,
        foot_deck_friction=None,
    )
    controller.prefix_path = adapter
    results = []
    try:
        if (
            controller.model.npair
            or controller.model.neq
            or controller.model.nmocap
            or controller.model.na
        ):
            raise RuntimeError("Expected canonical passive unconstrained model")
        if not np.array_equal(baseline, np.clip(baseline, *controller.target_limits.T)):
            raise ValueError("Baseline target limits differ")
        prefix_records, prefix_proof = protocol.replay_prefix(controller, source, cfg)
        protocol.write_json(args.out_dir / "prefix_proof.json", prefix_proof)
        hot = fingerprint(controller.data, controller.model)
        hot_warmstart = controller.data.qacc_warmstart.copy()
        first = branch(controller, baseline[:8], deadline, count, "actual85_baseline8")
        results.append(first)
        qerr = float(abs(first["qpos"] - warm["qpos"][145:154]).max())
        verr = float(abs(first["qvel"] - warm["qvel"][145:154]).max())
        serr = float(abs(first["state"] - warm["state"][145:154]).max())
        old_branch = load(args.baseline.parent / "winner_evaluated_branch.npz")
        physics_error = float(abs(first["fullphysics"] - old_branch["fullphysics"][:64]).max())
        old_force = json.loads((args.baseline.parent / "actual_execution_forces.json").read_text())[
            :64
        ]
        force_keys = tuple(protocol.foot_forces(controller, controller.data))
        force_error = max(
            abs(a[key] - b[key])
            for a, b in zip(first["execution_force_rows"], old_force, strict=True)
            for key in force_keys
        )
        proof = {
            "qpos_error": qerr,
            "qvel_error": verr,
            "state92_error": serr,
            "fullphysics_all64_substeps_error": physics_error,
            "same_stage_execution_force_error_n": force_error,
            "endpoint_native_s": first["feature_details"]["time_native_s"],
        }
        protocol.write_json(args.out_dir / "baseline85_exact_proof.json", proof)
        if max(qerr, verr, serr, physics_error) != 0 or force_error > 1e-7:
            raise RuntimeError("Actual85 baseline differs; no response solve allowed")
        print(json.dumps({"stage": "baseline_exact", **proof}), flush=True)
        base8 = baseline[:8]
        clip_rows, effective = [], np.zeros((29, 2))
        valid_columns = np.ones(29, bool)
        for joint in range(29):
            for sign_index, sign in enumerate((1, -1)):
                requested = base8.copy()
                requested[:, joint] += sign * cfg.pulse_amplitude_rad * wave
                controls = np.clip(requested, *controller.target_limits.T)
                amplitude = float(wave @ (controls[:, joint] - base8[:, joint]) / (wave @ wave))
                shape_error = float(
                    abs(controls[:, joint] - base8[:, joint] - amplitude * wave).max()
                )
                clipped = bool(np.any(controls != requested))
                effective[joint, sign_index] = amplitude
                valid_columns[joint] &= not clipped and shape_error < 1e-12
                clip_rows.append(
                    {
                        "joint": G1_JOINT_NAMES[joint],
                        "sign": sign,
                        "effective_amplitude_rad": amplitude,
                        "clipped": clipped,
                        "profile_shape_error_rad": shape_error,
                    }
                )
                results.append(
                    branch(
                        controller, controls, deadline, count, f"{G1_JOINT_NAMES[joint]}_{sign:+d}"
                    )
                )
        response = np.stack([r["features"] for r in results])
        jacobian = np.zeros((len(FEATURES), 29))
        for joint in np.flatnonzero(valid_columns):
            denominator = effective[joint, 0] - effective[joint, 1]
            jacobian[:, joint] = (response[1 + 2 * joint] - response[2 + 2 * joint]) / denominator
        np.savez_compressed(
            args.out_dir / "native_responses.npz",
            features=response,
            jacobian=jacobian,
            effective_amplitudes=effective,
            valid_columns=valid_columns,
            action=np.stack([r["action"] for r in results]),
            fullphysics=np.stack([r["fullphysics"] for r in results]),
            hot_qacc_warmstart=hot_warmstart,
        )
        protocol.write_json(args.out_dir / "response_clipping.json", clip_rows)
        valid = np.flatnonzero(valid_columns)
        if not len(valid):
            raise RuntimeError("No valid un-clipped response column")
        target = first["features"].copy()
        target[0] = cfg.lateral_cp_target_m
        amplitude_bound = np.array([cfg.leg_bound_rad] * 12 + [cfg.other_bound_rad] * 17)
        low = np.maximum(
            -amplitude_bound, ((controller.target_limits[:, 0] - base8) / wave[:, None]).max(0)
        )
        high = np.minimum(
            amplitude_bound, ((controller.target_limits[:, 1] - base8) / wave[:, None]).min(0)
        )
        matrix = np.vstack(
            (
                jacobian[:, valid] / SCALES[:, None],
                np.sqrt(cfg.regularizer_weight) / cfg.regularizer_scale_rad * np.eye(len(valid)),
            )
        )
        rhs = np.r_[(target - first["features"]) / SCALES, np.zeros(len(valid))]
        solved = lsq_linear(matrix, rhs, bounds=(low[valid], high[valid]), tol=1e-10, max_iter=100)
        delta = np.zeros(29)
        delta[valid] = solved.x
        corrected = baseline.copy()
        corrected[:8] = np.clip(base8 + wave[:, None] * delta, *controller.target_limits.T)
        predicted = first["features"] + jacobian @ delta
        solve_record = {
            "lsq_calls": 1,
            "success": bool(solved.success),
            "status": int(solved.status),
            "message": solved.message,
            "cost": float(solved.cost),
            "iterations": int(solved.nit),
            "valid_columns": valid.tolist(),
            "delta_rad": delta.tolist(),
            "delta_lower": low.tolist(),
            "delta_upper": high.tolist(),
            "baseline": first["feature_details"],
            "target_features": target.tolist(),
            "predicted_features": predicted.tolist(),
        }
        protocol.write_json(args.out_dir / "one_linear_solve.json", solve_record)
        short = branch(controller, corrected[:8], deadline, count, "corrected8_once")
        save_branch(args.out_dir / "corrected8_actual.npz", short)
        protocol.write_json(args.out_dir / "corrected8_features.json", short["feature_details"])
        print(
            json.dumps(
                {
                    "stage": "response_and_one_correction",
                    "branches": count[0],
                    "baseline_cp_m": float(first["features"][0]),
                    "linear_predicted_cp_m": float(predicted[0]),
                    "actual_corrected_cp_m": float(short["features"][0]),
                    "valid_columns": len(valid),
                }
            ),
            flush=True,
        )
        complete = branch(controller, corrected, deadline, count, "corrected_full112_once")
        records = prefix_records + complete["records"][1:]
        result = outcome(records, full_reference=True, numerical_failure=None)
        old_outcome = json.loads((args.baseline.parent / "result.json").read_text())
        earlier = result["robot_ground_first_time_s"] is not None and (
            old_outcome["robot_ground_first_time_s"] is None
            or result["robot_ground_first_time_s"] < old_outcome["robot_ground_first_time_s"] - 1e-9
        )
        retained_flight = (
            result["criteria"]["tail_contact_precedes_40ms_board_flight"]
            and result["criteria"]["board_flight_clearance_at_least_15mm"]
        )
        result.update(
            schema=SCHEMA,
            branch_count=count[0],
            lsq_calls=1,
            verified_teacher=False,
            eligible_reference=False,
            training_performed=False,
            initializations=1,
            phase_resets=0,
            root_board_writes_after_initialization=0,
            retained_qualifying_flight=retained_flight,
            floor_earlier_than_actual85=earlier,
            baseline85_floor_s=old_outcome["robot_ground_first_time_s"],
            main_hot_fingerprint_unchanged=fingerprint(controller.data, controller.model) == hot,
            baseline_cp_m=float(first["features"][0]),
            corrected8_cp_m=float(short["features"][0]),
            wall_seconds=time.monotonic() - started,
        )
        if count[0] != cfg.maximum_branches or not result["main_hot_fingerprint_unchanged"]:
            raise RuntimeError("Branch budget or main-state invariant differs")
        arrays = {
            key: np.concatenate((full[key][:145], complete[key]))
            for key in ("qpos", "qvel", "state")
        }
        native_prefix_times = np.array([r["time_physics"] for r in prefix_records[::8]])
        times = np.r_[native_prefix_times[:145], complete["time_physics"]]
        np.savez_compressed(
            args.out_dir / "actual_rollout.npz",
            **arrays,
            action=np.concatenate((full["action"][:145], corrected)),
            time_physics=times,
            time_video=times * 1.5,
            control_dt=np.asarray(0.02),
            robot_joint_names=np.asarray(G1_JOINT_NAMES),
            board_joint_names=np.asarray(BOARD_JOINT_NAMES),
            verified_teacher=np.asarray(False),
        )
        np.savez_compressed(
            args.out_dir / "prior_targets.npz",
            action_targets=corrected,
            robot_joint_names=np.asarray(G1_JOINT_NAMES),
            delta_rad=delta,
            schema=np.asarray(SCHEMA),
            verified_teacher=np.asarray(False),
            eligible_reference=np.asarray(False),
        )
        protocol.write_json(args.out_dir / "contacts.json", records)
        protocol.write_json(
            args.out_dir / "execution_forces_suffix.json", complete["execution_force_rows"]
        )
        protocol.write_json(args.out_dir / "result.json", result)
        if (
            protocol.digest(args.source) != landing.SOURCE_SHA
            or protocol.digest(args.baseline) != WARM_SHA
        ):
            raise RuntimeError("Source changed")
        protocol.write_json(
            args.out_dir / "output_hashes.json",
            {p.name: protocol.digest(p) for p in args.out_dir.iterdir() if p.is_file()},
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    except (RuntimeError, ValueError, TimeoutError, KeyboardInterrupt) as error:
        protocol.write_json(
            args.out_dir / "failure.json",
            {
                "error": str(error) or type(error).__name__,
                "started_branch_count": count[0],
                "completed_short_branches": len(results),
                "main_time_s": float(controller.data.time),
                "wall_seconds": time.monotonic() - started,
            },
        )
        if results:
            np.savez_compressed(
                args.out_dir / "partial_native_features.npz",
                features=np.stack([r["features"] for r in results]),
            )
        raise
    finally:
        controller.pool.close()


if __name__ == "__main__":
    main()
