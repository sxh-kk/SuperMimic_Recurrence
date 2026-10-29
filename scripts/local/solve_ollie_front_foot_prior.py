"""One small native front-foot response prior; original physics and strict v3.

No CEM, line search, policy training, external force, or hot-state replacement.
Only the saved actual90 PD targets receive a bounded six-parameter pulse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from scripts.local import solve_ollie_prelaunch_balance_prior as previous
from scripts.local.evaluate_ollie_policy import outcome
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, SceneIndex, make_scene_spec

protocol = previous.protocol
SCHEMA = "supermimic.ollie.front_foot_small_response.v1"
SOURCE_SHA = "ac6effe28909ca875149e6f94cdb17b3b59b0152a4960dfff4ab7e38beb35526"
PARAMETER_IDS = np.array([0, 1, 2, 3, 6, 9])
FEATURES = (
    "left_site_board_x_m",
    "left_site_board_y_m",
    "left_site_board_z_m",
    "left_site_up_dot_board_up",
    "right_site_board_x_m",
    "right_site_board_y_m",
    "right_site_board_z_m",
    "right_site_up_dot_board_up",
    "root_minus_board_world_z_m",
    "robot_upright",
    "board_upright",
    "board_com_world_z_m",
    "robot_com_world_vx_m_s",
    "robot_com_world_vy_m_s",
    "robot_com_world_vz_m_s",
    "board_com_world_vx_m_s",
    "board_com_world_vy_m_s",
    "board_com_world_vz_m_s",
    "whole_board_clearance_m",
)
SCALES = np.array(
    [
        0.03,
        0.02,
        0.015,
        0.1,
        0.03,
        0.025,
        0.015,
        0.1,
        0.035,
        0.05,
        0.05,
        0.015,
        0.25,
        0.15,
        0.25,
        0.25,
        0.15,
        0.25,
        0.01,
    ]
)


@dataclass(frozen=True)
class Config:
    prefix_controls: int = 130
    horizon: int = 127
    short_controls: int = 38
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    sample_amplitude_rad: float = 0.04
    correction_bound_rad: float = 0.015
    maximum_branches: int = 15
    wall_budget_s: float = 120.0
    rise_start_s: float = 2.90
    rise_end_s: float = 3.06
    hold_end_s: float = 3.30
    return_end_s: float = 3.46
    observation_times_s: tuple = (3.26, 3.275, 3.29, 3.305, 3.32)
    left_y_target_m: float = 0.07
    left_z_target_m: float = 0.012
    regularizer_weight: float = 0.1


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value**3 * (10 + value * (-15 + 6 * value))


def wave(config):
    times = (
        config.prefix_controls * config.control_dt
        + (np.arange(config.horizon) + 0.5) * config.control_dt
    )
    return smoothstep((times - config.rise_start_s) / (config.rise_end_s - config.rise_start_s)) * (
        1 - smoothstep((times - config.hold_end_s) / (config.return_end_s - config.hold_end_s))
    )


def observe_features(controller, branch, config):
    """Saved true postforward states plus genuine recorded contacts; FK only."""
    model, ix = controller.model, controller.index
    robot, board = model.body("pelvis").id, ix.board_body
    scratch = mujoco.MjData(model)
    values, details = [], []
    for stamp in config.observation_times_s:
        sample = round((stamp - config.prefix_controls * config.control_dt) / config.physics_dt) - 1
        state = branch["fullphysics"][sample]
        if abs(float(state[0]) - stamp) > 1e-10:
            raise ValueError("Actual native feature clock differs")
        scratch.qpos[:], scratch.qvel[:] = state[1 : 1 + model.nq], state[1 + model.nq :]
        mujoco.mj_fwdPosition(model, scratch)
        rotation, position = scratch.xmat[board].reshape(3, 3), scratch.xpos[board]
        feet, align = [], []
        for side in ("left", "right"):
            site = model.site(side + "_foot").id
            feet.append(rotation.T @ (scratch.site_xpos[site] - position))
            align.append(float(scratch.site_xmat[site].reshape(3, 3)[:, 2] @ rotation[:, 2]))
        velocities = []
        for body in (robot, board):
            jac = np.empty((3, model.nv))
            mujoco.mj_jacSubtreeCom(model, scratch, jac, body)
            velocities.append(jac @ scratch.qvel)
        record = branch["records"][sample + 1]
        vector = np.r_[
            feet[0],
            align[0],
            feet[1],
            align[1],
            scratch.xpos[robot, 2] - position[2],
            scratch.xmat[robot, 8],
            scratch.xmat[board, 8],
            scratch.subtree_com[board, 2],
            velocities[0],
            velocities[1],
            record["minimum_board_clearance_m"],
        ]
        values.append(vector)
        details.append(
            {
                "time_s": stamp,
                "features": dict(zip(FEATURES, vector.tolist(), strict=True)),
                "left_true_top_n": record["left_foot_deck_upward_support_n"],
                "right_true_top_n": record["right_foot_deck_upward_support_n"],
                "real_board_ground_count": record["unique_board_geoms_ground"],
                "robot_ground_count": record["unique_robot_geoms_ground"],
                "maximum_penetration_m": record["maximum_penetration_m"],
            }
        )
    return np.asarray(values), details


def static_diagnosis(source):
    """Local fixed-root geometry is not dynamic reachability certification."""
    model = make_scene_spec(timestep=0.0025, foot_deck_friction=None).compile()
    ix, data = SceneIndex.from_model(model), mujoco.MjData(model)
    site = model.site("left_foot").id
    rows = []
    for frame in (145, 150, 155, 160, 163, 166):
        data.qpos[:], data.qvel[:] = source["qpos"][frame], source["qvel"][frame]
        mujoco.mj_fwdPosition(model, data)
        rotation = data.xmat[ix.board_body].reshape(3, 3)
        jac = np.empty((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site)
        local = rotation.T @ jac[:, ix.robot_joint_qvel[:6]]
        position = rotation.T @ (data.site_xpos[site] - data.xpos[ix.board_body])
        rows.append(
            {
                "time_s": frame * 0.02,
                "left_site_board_m": position.tolist(),
                "left6_site_jacobian_board_xyz_m_per_rad": local.tolist(),
                "left_y_target_gap_m": float(position[1] - 0.07),
                "left4_linear_maximum_y_change_with15mrad_box_m": float(
                    abs(local[1, :4]).sum() * 0.015
                ),
            }
        )
    return {
        "integration_steps": 0,
        "rows": rows,
        "limit": "Local fixed-root Jacobian directions only; no IK or proof that the small PD pulse can fully recover the foot.",
    }


def save_branch(path, branch):
    previous.save_branch(path.with_suffix(".npz"), branch)
    protocol.write_json(path.with_name(path.name + "_contacts.json"), branch["records"])
    protocol.write_json(
        path.with_name(path.name + "_execution_forces.json"), branch["execution_force_rows"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("logs/ollie_air_catch_landing_01/actual_rollout.npz")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("inspect", "run"), default="inspect")
    args, config = parser.parse_args(), Config()
    if args.out_dir.exists():
        parser.error("Use a fresh output directory; preserve previous evidence")
    if protocol.digest(args.source) != SOURCE_SHA:
        raise ValueError("Frozen complete actual90 changed")
    source = previous.load(args.source)
    if source["action"].shape != (257, 29) or bool(source["verified_teacher"]):
        raise ValueError("Expected original complete actual90 exploration")
    source_result = json.loads((args.source.parent / "result.json").read_text())
    physics = protocol.model_sources()
    if physics != source_result["frozen_manifest"]["canonical_assets"]:
        raise ValueError("Original physical model differs from actual90")
    args.out_dir.mkdir(parents=True)
    pulse, baseline = wave(config), source["action"][config.prefix_controls :].copy()
    manifest = {
        "schema": SCHEMA,
        "mode": args.mode,
        "config": asdict(config),
        "source": {"path": str(args.source.resolve()), "sha256": SOURCE_SHA},
        "script_sha256": protocol.digest(__file__),
        "physics": physics,
        "native_branch_helper_sha256": protocol.digest(previous.__file__),
        "parameter_joint_names": [G1_JOINT_NAMES[i] for i in PARAMETER_IDS],
        "parameter_joint_indices": PARAMETER_IDS.tolist(),
        "pulse_control_midpoints": pulse.tolist(),
        "feature_names": FEATURES,
        "feature_scales": SCALES.tolist(),
        "is_teacher": False,
        "trust_region": "six amplitudes within±.015rad; L2<=sqrt(6)*.015=.036743<sample radius.04rad; no external iteration",
    }
    protocol.write_json(args.out_dir / "frozen_manifest.json", manifest)
    protocol.write_json(args.out_dir / "mechanical_fk_diagnosis.json", static_diagnosis(source))
    if args.mode == "inspect":
        print(
            json.dumps({"mode": "inspect", "native_integration_steps": 0, "config": asdict(config)})
        )
        return
    started = time.monotonic()
    deadline, count = started + config.wall_budget_s, [0]
    controller = NativeSamplingMPC(
        Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
        Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"),
        seed=91,
        foot_deck_friction=None,
    )
    controller.prefix_path = args.source
    collected = []

    def branch(targets, label):
        if count[0] >= config.maximum_branches:
            raise RuntimeError("Fixed15-branch budget exhausted")
        value = previous.branch(controller, targets, deadline, count, label)
        values, details = observe_features(controller, value, config)
        value["window_features"], value["window_details"] = values, details
        collected.append(value)
        return value

    try:
        prefix_records, prefix_proof = protocol.replay_prefix(controller, source, config)
        protocol.write_json(args.out_dir / "prefix_proof.json", prefix_proof)
        hot = previous.fingerprint(controller.data, controller.model)
        hot_state, hot_warmstart = controller.state(), controller.data.qacc_warmstart.copy()
        first = branch(baseline[: config.short_controls], "actual90_baseline38")
        state_errors = {
            key: float(abs(first[key] - source[key][130:169]).max())
            for key in ("qpos", "qvel", "state")
        }
        prior_branch = previous.load(args.source.parent / "winner_evaluated_branch.npz")
        full_error = float(abs(first["fullphysics"] - prior_branch["fullphysics"][:304]).max())
        source_forces = json.loads(
            (args.source.parent / "actual_execution_forces.json").read_text()
        )[:304]
        force_error = max(
            abs(a[key] - b[key])
            for a, b in zip(first["execution_force_rows"], source_forces, strict=True)
            for key in a
        )
        protocol.write_json(
            args.out_dir / "baseline90_exact_proof.json",
            {**state_errors, "fullphysics_error": full_error, "execution_force_error": force_error},
        )
        if max(*state_errors.values(), full_error, force_error) != 0:
            raise RuntimeError("Baseline90 differs; response solve forbidden")
        limits, effective, valid = controller.target_limits, np.zeros((6, 2)), np.ones(6, bool)
        clipping = []
        for parameter, joint in enumerate(PARAMETER_IDS):
            for side, sign in enumerate((1, -1)):
                requested = baseline.copy()
                requested[:, joint] += sign * config.sample_amplitude_rad * pulse
                target = np.clip(requested, *limits.T)
                actual_amplitude = float(
                    pulse @ (target[:, joint] - baseline[:, joint]) / (pulse @ pulse)
                )
                shape_error = float(
                    abs(target[:, joint] - baseline[:, joint] - actual_amplitude * pulse).max()
                )
                clipped = bool(np.any(target != requested))
                effective[parameter, side] = actual_amplitude
                valid[parameter] &= not clipped and shape_error < 1e-12
                clipping.append(
                    {
                        "joint": G1_JOINT_NAMES[joint],
                        "sign": sign,
                        "clipped": clipped,
                        "effective_amplitude_rad": actual_amplitude,
                        "profile_shape_error_rad": shape_error,
                    }
                )
                branch(target[: config.short_controls], f"{G1_JOINT_NAMES[joint]}_{sign:+d}")
        responses = np.stack([value["window_features"] for value in collected])
        jacobian = np.zeros((*responses.shape[1:], 6))
        for parameter in np.flatnonzero(valid):
            jacobian[..., parameter] = (
                responses[1 + 2 * parameter] - responses[2 + 2 * parameter]
            ) / (effective[parameter, 0] - effective[parameter, 1])
        np.savez_compressed(
            args.out_dir / "native_responses.npz",
            features=responses,
            jacobian=jacobian,
            effective_amplitudes=effective,
            valid_columns=valid,
            action=np.stack([value["action"] for value in collected]),
            fullphysics=np.stack([value["fullphysics"] for value in collected]),
            hot_fullphysics=hot_state,
            hot_qacc_warmstart=hot_warmstart,
            observation_times_s=np.array(config.observation_times_s),
            parameter_joint_indices=PARAMETER_IDS,
        )
        protocol.write_json(args.out_dir / "response_clipping.json", clipping)
        protocol.write_json(
            args.out_dir / "response_window_details.json",
            [value["window_details"] for value in collected],
        )
        protocol.write_json(
            args.out_dir / "response_contacts.json", [value["records"] for value in collected]
        )
        protocol.write_json(
            args.out_dir / "response_execution_forces.json",
            [value["execution_force_rows"] for value in collected],
        )
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise RuntimeError("No valid native response columns")
        low, high = (
            np.full(6, -config.correction_bound_rad),
            np.full(6, config.correction_bound_rad),
        )
        active = pulse > 0
        for parameter, joint in enumerate(PARAMETER_IDS):
            low[parameter] = max(
                low[parameter],
                float(((limits[joint, 0] - baseline[active, joint]) / pulse[active]).max()),
            )
            high[parameter] = min(
                high[parameter],
                float(((limits[joint, 1] - baseline[active, joint]) / pulse[active]).min()),
            )
        target = responses[0].copy()
        target[:, 1] = config.left_y_target_m
        target[:, 2] = config.left_z_target_m
        matrix = np.vstack(
            (
                (jacobian[..., indices] / SCALES[None, :, None]).reshape(-1, len(indices)),
                np.sqrt(config.regularizer_weight)
                / config.correction_bound_rad
                * np.eye(len(indices)),
            )
        )
        rhs = np.r_[((target - responses[0]) / SCALES).ravel(), np.zeros(len(indices))]
        solved = lsq_linear(
            matrix, rhs, bounds=(low[indices], high[indices]), tol=1e-10, max_iter=100
        )
        delta = np.zeros(6)
        delta[indices] = solved.x
        if np.linalg.norm(delta) > config.sample_amplitude_rad + 1e-12:
            raise RuntimeError("Correction left the predeclared response trust radius")
        corrected = baseline.copy()
        corrected[:, PARAMETER_IDS] += pulse[:, None] * delta
        if not np.array_equal(corrected, np.clip(corrected, *limits.T)):
            raise RuntimeError("Feasible correction unexpectedly clips targets")
        predicted = responses[0] + jacobian @ delta
        protocol.write_json(
            args.out_dir / "one_linear_solve.json",
            {
                "lsq_calls": 1,
                "success": bool(solved.success),
                "iterations": int(solved.nit),
                "cost": float(solved.cost),
                "delta_rad": delta.tolist(),
                "delta_l2_rad": float(np.linalg.norm(delta)),
                "lower": low.tolist(),
                "upper": high.tolist(),
                "valid_columns": indices.tolist(),
                "baseline_features": responses[0].tolist(),
                "target_features": target.tolist(),
                "predicted_features": predicted.tolist(),
                "limitation": "Conditional finite response secants, not a smooth-contact derivative or complete foot recovery certificate.",
            },
        )
        short = branch(corrected[: config.short_controls], "corrected38_once")
        save_branch(args.out_dir / "corrected38_actual", short)
        protocol.write_json(args.out_dir / "corrected_window_details.json", short["window_details"])
        complete = branch(corrected, "corrected_full127_once")
        save_branch(args.out_dir / "complete_suffix_actual", complete)
        for key in ("qpos", "qvel", "state", "fullphysics"):
            if not np.array_equal(short[key], complete[key][: len(short[key])]):
                raise RuntimeError("Short and full corrected histories differ")
        records = prefix_records + complete["records"][1:]
        result = outcome(records, full_reference=True, numerical_failure=None)
        result.update(
            schema=SCHEMA,
            branch_count=count[0],
            lsq_calls=1,
            verified_teacher=False,
            eligible_reference=bool(result["success"]),
            training_performed=False,
            actor_updates=0,
            initializations=1,
            phase_resets=0,
            root_board_writes_after_initialization=0,
            main_hot_fingerprint_unchanged=previous.fingerprint(controller.data, controller.model)
            == hot,
            wall_seconds=time.monotonic() - started,
            baseline_left_y_m=responses[0, :, 1].tolist(),
            corrected_left_y_m=short["window_features"][:, 1].tolist(),
            delta_l2_rad=float(np.linalg.norm(delta)),
            source90_sha256=SOURCE_SHA,
        )
        if count[0] != 15 or not result["main_hot_fingerprint_unchanged"]:
            raise RuntimeError("Branch budget or hot-main invariant differs")
        arrays = {
            key: np.concatenate((source[key][:130], complete[key]))
            for key in ("qpos", "qvel", "state")
        }
        times = np.r_[
            np.array([r["time_physics"] for r in prefix_records[::8]])[:130],
            complete["time_physics"],
        ]
        np.savez_compressed(
            args.out_dir / "actual_rollout.npz",
            **arrays,
            action=np.concatenate((source["action"][:130], corrected)),
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
            parameter_joint_indices=PARAMETER_IDS,
            delta_rad=delta,
            verified_teacher=np.asarray(False),
            eligible_reference=np.asarray(result["success"]),
            schema=np.asarray(SCHEMA),
        )
        protocol.write_json(args.out_dir / "contacts.json", records)
        protocol.write_json(
            args.out_dir / "execution_forces_suffix.json", complete["execution_force_rows"]
        )
        protocol.write_json(args.out_dir / "result.json", result)
        if protocol.digest(args.source) != SOURCE_SHA:
            raise RuntimeError("Actual90 source changed")
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
                "started_branches": count[0],
                "completed_branches": len(collected),
                "wall_seconds": time.monotonic() - started,
                "main_time_s": float(controller.data.time),
                "eligible_reference": False,
            },
        )
        if collected:
            np.savez_compressed(
                args.out_dir / "partial_response_features.npz",
                features=np.stack([value["window_features"] for value in collected]),
            )
        raise
    finally:
        controller.pool.close()


if __name__ == "__main__":
    main()
