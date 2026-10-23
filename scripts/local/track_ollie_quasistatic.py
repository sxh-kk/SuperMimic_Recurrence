"""Native CPU contact-feedforward tracking, not a learned SuperMimic policy.

The robot's 29 PD targets are the only control inputs. Solved point forces are
never applied, the board is passive, and a rollout has exactly one initialization.
"""

from __future__ import annotations

import argparse
import csv
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
from supermimic.ollie.control import FootBoardSupport
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.quasistatic import ContactStatics
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


class TrackingExperiment:
    def __init__(self, reference, seconds=2.76, inertial=False, absolute_upright=False):
        self.reference, self.seconds = reference, seconds
        self.absolute_upright = absolute_upright
        self.model, self.data, self.index = compile_scene(timestep=0.0025)
        self.indices = JointStateIndices.from_model(self.model, board_joint_names=BOARD_JOINT_NAMES)
        self.solver = ContactStatics(self.model, self.index)
        self.steps = round(seconds / 0.02)
        self.frames = np.arange(self.steps + 1)
        self.times = self.frames * 0.02
        self.qpos, self.qvel = [
            v.numpy() for v in self.indices.unpack(torch.as_tensor(reference.state))
        ]
        self.root_body = self.model.body("pelvis").id
        self.feet = [self.model.site(side + "_foot").id for side in ("left", "right")]
        self.support = FootBoardSupport(self.model)
        self.ref_feet, self.ref_com, self.ref_rotation, self.solutions = [], [], [], []
        velocity_derivative = np.gradient(self.qvel, reference.time_physics, axis=0)
        previous = None
        for frame in self.frames:
            self.data.qpos[:] = self.qpos[frame]
            self.data.qvel[:] = self.qvel[frame]
            mujoco.mj_forward(self.model, self.data)
            rotation = self.data.xmat[self.index.board_body].reshape(3, 3)
            center = self.data.xpos[self.index.board_body]
            self.ref_feet.append((self.data.site_xpos[self.feet] - center) @ rotation)
            self.ref_com.append(rotation.T @ (self.data.subtree_com[self.root_body] - center))
            self.ref_rotation.append(self.data.xmat[self.root_body].reshape(3, 3).copy())
            previous = self.solver.solve(
                self.data,
                initial=previous,
                acceleration=velocity_derivative[frame] if inertial else None,
                velocity=self.qvel[frame] if inertial else None,
            )
            self.solutions.append(previous)
            if frame % 25 == 0:
                print(
                    json.dumps({"solved_reference_frame": int(frame), **previous.diagnostics()}),
                    flush=True,
                )
        self.ref_feet, self.ref_com, self.ref_rotation = map(
            np.asarray, (self.ref_feet, self.ref_com, self.ref_rotation)
        )
        self.ref_com_velocity = np.gradient(self.ref_com, self.times, axis=0)
        self.jac = np.zeros((3, self.model.nv))
        self.board_jac = np.zeros_like(self.jac)
        self.rot_jac = np.zeros_like(self.jac)
        yaw = np.arctan2(self.ref_rotation[0, 1, 0], self.ref_rotation[0, 0, 0])
        self.heading = Rotation.from_euler("z", yaw).as_matrix()

    def feedback(self, frame, scale):
        data, model, index = self.data, self.model, self.index
        loads = self.support.vertical_loads(data)
        if not scale or not (loads > 1.0).any():
            return np.zeros(29), loads
        actual_rotation = data.xmat[self.root_body].reshape(3, 3)
        desired_rotation = self.ref_rotation[frame]
        if self.absolute_upright:
            yaw = np.arctan2(desired_rotation[1, 0], desired_rotation[0, 0])
            desired_rotation = Rotation.from_euler("z", yaw).as_matrix()
        tilt = (
            desired_rotation
            @ Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec()
        )
        omega = actual_rotation @ data.qvel[index.robot_qvel + 3 : index.robot_qvel + 6]
        desired_omega = (
            desired_rotation @ self.qvel[frame, index.robot_qvel + 3 : index.robot_qvel + 6]
        )
        if self.absolute_upright:
            desired_omega[:2] = 0
        board_rotation = data.xmat[index.board_body].reshape(3, 3)
        desired_com = data.xpos[index.board_body] + board_rotation @ self.ref_com[frame]
        com_error = data.subtree_com[self.root_body] - desired_com
        mujoco.mj_jacSubtreeCom(model, data, self.jac, self.root_body)
        mujoco.mj_jac(model, data, self.board_jac, self.rot_jac, desired_com, index.board_body)
        relative_velocity = (
            self.jac - self.board_jac
        ) @ data.qvel - board_rotation @ self.ref_com_velocity[frame]
        forward, lateral = self.heading[:, 0], self.heading[:, 1]
        rate_error = omega - desired_omega
        pitch = scale * (
            1.4 * (tilt @ lateral)
            + 0.14 * (rate_error @ lateral)
            + 1.2 * (com_error @ forward)
            + 0.5 * (relative_velocity @ forward)
        )
        roll = scale * (
            0.8 * (tilt @ forward)
            + 0.10 * (rate_error @ forward)
            - 0.7 * (com_error @ lateral)
            - 0.5 * (relative_velocity @ lateral)
        )
        tilt_world = forward * roll + lateral * pitch
        ankle = np.array([4, 5, 10, 11])
        joints = model.dof_jntid[index.robot_joint_qvel[ankle]]
        correction = np.zeros(29)
        # Positive projected ankle target sign; only force-supported feet engage.
        correction[ankle] = (data.xaxis[joints] @ tilt_world) * np.repeat(loads > 1.0, 2)
        return correction, loads

    def run(self, *, scale=0.0, constant_frame=None):
        model, data, index = self.model, self.data, self.index
        start = 0 if constant_frame is None else constant_frame
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self.qpos[start]
        data.qvel[:] = 0
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        data.ctrl[index.robot_actuators] = self.reference.robot_state[start, 7:36]
        mujoco.mj_forward(model, data)
        observer = ContactObserver(model, index, 0.1)
        steps = self.steps if constant_frame is None else 50
        boundary_state = [self.indices.pack(data.qpos, data.qvel).numpy()]
        qpos, qvel = [data.qpos.copy()], [data.qvel.copy()]
        records, extra, actions, raw_actions, feedforward, feedback, loads = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        def observe(goal_coordinate):
            record = observer.observe(data)
            rotation = data.xmat[index.board_body].reshape(3, 3)
            feet = (data.site_xpos[self.feet] - data.xpos[index.board_body]) @ rotation
            desired_feet = np.array(
                [
                    [
                        np.interp(goal_coordinate, self.frames, self.ref_feet[:, side, axis])
                        for axis in range(3)
                    ]
                    for side in range(2)
                ]
            )
            expected_height = np.interp(
                goal_coordinate, self.frames, self.reference.robot_state[self.frames, 2]
            )
            extra.append(
                [
                    float(np.abs(feet[:, 2] - desired_feet[:, 2]).max()),
                    float(np.linalg.norm(feet[:, :2] - desired_feet[:, :2], axis=-1).max()),
                    float(data.qpos[index.robot_qpos + 2] - expected_height),
                    float(
                        np.linalg.norm(
                            data.qpos[index.robot_qpos : index.robot_qpos + 2]
                            - data.qpos[index.board_qpos : index.board_qpos + 2]
                        )
                    ),
                ]
            )
            records.append(record)

        observe(start)
        substep_states, controls = [], []
        for step in range(steps):
            goal = step + 1 if constant_frame is None else constant_frame
            reference_velocity = (
                self.reference.robot_state[goal, 42:71] if constant_frame is None else np.zeros(29)
            )
            _, raw = self.solver.pd_target(
                self.reference.robot_state[goal, 7:36], reference_velocity, self.solutions[goal]
            )
            correction, support_load = (
                self.feedback(goal, scale)
                if constant_frame is None
                else (np.zeros(29), np.zeros(2))
            )
            target = np.clip(
                raw + correction,
                self.solver.target_limits[:, 0] + 0.005,
                self.solver.target_limits[:, 1] - 0.005,
            )
            actions.append(target.copy())
            raw_actions.append((raw + correction).copy())
            feedforward.append(raw - self.reference.robot_state[goal, 7:36])
            feedback.append(correction)
            loads.append(support_load)
            data.ctrl[index.robot_actuators] = target
            for substep in range(8):
                before = data.time
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if (
                    not np.isclose(data.time, before + 0.0025, rtol=0, atol=1e-10)
                    or not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                ):
                    raise RuntimeError("Nonfinite or discontinuous physical rollout")
                observe(step + (substep + 1) / 8 if constant_frame is None else constant_frame)
                substep_states.append(self.indices.pack(data.qpos, data.qvel).numpy())
                controls.append(data.ctrl[index.robot_actuators].copy())
            boundary_state.append(self.indices.pack(data.qpos, data.qvel).numpy())
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
        extra, actions = np.asarray(extra), np.asarray(actions)
        fractions = {
            side: float(np.mean([row[f"{side}_foot_board"] for row in records]))
            for side in ("left", "right")
        }
        top_fraction = {
            side: float(
                np.mean([row[f"{side}_foot_deck_upward_support_n"] > 1.0 for row in records])
            )
            for side in ("left", "right")
        }
        final_records = [row for row in records if row["time_physics"] >= data.time - 0.3 + 1e-8]
        final_top_fraction = {
            side: float(
                np.mean([row[f"{side}_foot_deck_upward_support_n"] > 1.0 for row in final_records])
            )
            for side in ("left", "right")
        }
        metrics = {
            "minimum_upright": min(row["robot_upright_cosine"] for row in records),
            "robot_ground": any(row["unique_robot_geoms_ground"] for row in records),
            "first_robot_ground_s": next(
                (row["time_physics"] for row in records if row["unique_robot_geoms_ground"]), None
            ),
            "maximum_penetration_m": max(row["maximum_penetration_m"] for row in records),
            "foot_board_force_contact_fraction": fractions,
            "foot_deck_upward_support_above_1n_fraction": top_fraction,
            "final_300ms_foot_deck_upward_support_above_1n_fraction": final_top_fraction,
            "final_robot_linear_speed_m_s": records[-1]["robot_linear_speed_m_s"],
            "final_robot_angular_speed_rad_s": records[-1]["robot_angular_speed_rad_s"],
            "final_board_linear_speed_m_s": records[-1]["board_linear_speed_m_s"],
            "four_wheels_ground_fraction": float(
                np.mean([row["unique_wheels_ground"] == 4 for row in records])
            ),
            "foot_gap_max_m": float(extra[:, 0].max()),
            "foot_planar_slip_max_m": float(extra[:, 1].max()),
            "root_height_rmse_m": float(np.sqrt(np.mean(extra[:, 2] ** 2))),
            "root_height_max_error_m": float(np.abs(extra[:, 2]).max()),
            "root_board_xy_max_m": float(extra[:, 3].max()),
            "no_external_forces": all(
                row["external_generalized_force_norm"] == row["external_body_force_norm"] == 0
                for row in records
            ),
            "initial_penetration_m": records[0]["maximum_penetration_m"],
            "seconds": float(data.time),
            "feedback_scale": scale,
            "constant_frame": constant_frame,
        }
        gates = {
            "upright": metrics["minimum_upright"] >= 0.85,
            "no_robot_ground": not metrics["robot_ground"],
            "penetration": metrics["maximum_penetration_m"] <= 0.01,
            "force_supported_feet": min(fractions.values()) >= 0.8,
            "true_upper_deck_support": min(top_fraction.values()) >= 0.8
            and min(final_top_fraction.values()) >= 0.8,
            "wheel_support": metrics["four_wheels_ground_fraction"] >= 0.8,
            "foot_gap": metrics["foot_gap_max_m"] <= 0.025,
            "foot_slip": metrics["foot_planar_slip_max_m"] <= 0.04,
            "root_height": metrics["root_height_rmse_m"] <= 0.08
            and metrics["root_height_max_error_m"] <= 0.16,
            "root_over_board": metrics["root_board_xy_max_m"] <= 0.2,
            "initial_geometry": metrics["initial_penetration_m"] <= 0.002,
            "no_external_forces": metrics["no_external_forces"],
            "static_solutions_feasible": all(sol.success for sol in self.solutions)
            if constant_frame is None
            else self.solutions[constant_frame].success,
        }
        metrics["gates"], metrics["verified_teacher"] = (
            gates,
            bool(all(gates.values()) and constant_frame is None),
        )
        metrics["constant_posture_gate_passed"] = (
            bool(all(gates.values())) if constant_frame is not None else None
        )
        metrics["teacher_criteria_version"] = "stance_upper_deck_support_v2"
        return (
            metrics,
            records,
            {
                "all_boundary_states": np.asarray(boundary_state),
                "qpos": np.asarray(qpos),
                "qvel": np.asarray(qvel),
                "action": actions,
                "raw_action": np.asarray(raw_actions),
                "feedforward_bias_rad": np.asarray(feedforward),
                "feedback_bias_rad": np.asarray(feedback),
                "actual_support_loads_n": np.asarray(loads),
                "substep_states": np.asarray(substep_states),
                "substep_controls": np.asarray(controls),
                "geometric_errors": extra,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/ollie/self_collision_corrected/joint_reference_50hz.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.76)
    parser.add_argument(
        "--feedback-scales", type=float, nargs="+", default=[0.0, 0.25, 0.45, 0.5, 1.0]
    )
    parser.add_argument("--inertial", action="store_true")
    parser.add_argument(
        "--absolute-upright",
        action="store_true",
        help="Balance feedback targets level pelvis roll/pitch; reference qpos is unchanged",
    )
    parser.add_argument(
        "--residual-limits", type=Path, default=Path("data/ollie/residual_limits.json")
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Choose a fresh output directory")
    if (
        not np.isfinite(args.feedback_scales).all()
        or min(args.feedback_scales) < 0
        or len(set(args.feedback_scales)) != len(args.feedback_scales)
    ):
        parser.error("feedback-scales must be distinct, finite and nonnegative")
    if not 0 < args.seconds <= 2.76 or not np.isclose(
        args.seconds / 0.02, round(args.seconds / 0.02)
    ):
        parser.error("seconds must be a multiple of 0.02 within the verified stance prior, <=2.76")
    torch.set_num_threads(2)
    reference = JointReference.load(args.reference)
    if (
        not np.isclose(reference.metadata["physics_fps"], 50)
        or reference.time_physics[-1] < args.seconds
    ):
        parser.error("Need a 50Hz reference covering the requested interval")
    if not np.all(reference.contacts[: round(args.seconds / 0.02) + 1, :2] == 1):
        parser.error(
            "Requested support interval must have explicit bilateral foot-board stance priors"
        )
    experiment = TrackingExperiment(reference, args.seconds, args.inertial, args.absolute_upright)
    args.out_dir.mkdir(parents=True)
    limits_doc = json.loads(args.residual_limits.read_text())
    if tuple(limits_doc["joint_names"]) != G1_JOINT_NAMES:
        raise ValueError("Residual-limit joint order mismatch")
    residual_limits = np.asarray(limits_doc["limits_rad"])
    if (
        residual_limits.shape != (29,)
        or not np.isfinite(residual_limits).all()
        or (residual_limits <= 0).any()
    ):
        raise ValueError("Need 29 positive finite residual limits for diagnostic comparison")
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    solver_report = [
        dict(frame=k, **sol.diagnostics()) for k, sol in enumerate(experiment.solutions)
    ]
    np.savez_compressed(
        args.out_dir / "contact_feedforward.npz",
        tau_nm=np.array([s.tau_nm for s in experiment.solutions]),
        forces_local=np.array([s.forces_local for s in experiment.solutions]),
        time_s=experiment.times,
        qpos=experiment.qpos[experiment.frames],
        qvel=experiment.qvel[experiment.frames],
        reference_sha256=np.asarray(reference_hash),
        metadata_json=np.asarray(json.dumps(solver_report)),
    )
    reports = []
    cases = [(f"feedback_{scale:g}", scale, None) for scale in args.feedback_scales]
    deepest = int(np.argmin(reference.robot_state[experiment.frames, 2]))
    cases += [("constant_initial", 0.0, 0), ("constant_deepest", 0.0, deepest)]
    for name, scale, constant in cases:
        metrics, records, arrays = experiment.run(scale=scale, constant_frame=constant)
        out = args.out_dir / name
        out.mkdir()
        goal_indices = np.minimum(
            np.arange(1, len(arrays["action"]) + 1)[:, None] + [0, 5, 10], len(reference.state) - 1
        )
        if constant is not None:
            goal_indices[:] = constant
        qreference = (
            reference.robot_state[goal_indices[:, 0], 7:36]
            if constant is None
            else np.broadcast_to(reference.robot_state[constant, 7:36], arrays["action"].shape)
        )
        action_delta = np.abs(arrays["action"] - qreference).max(axis=0)
        metrics.update(
            joint_names=list(G1_JOINT_NAMES),
            action_reference_max_abs_by_joint_rad=action_delta.tolist(),
            action_reference_max_abs_rad=float(action_delta.max()),
            action_reference_signed_min_rad=(arrays["action"] - qreference).min(axis=0).tolist(),
            action_reference_signed_max_rad=(arrays["action"] - qreference).max(axis=0).tolist(),
            raw_action_reference_max_abs_by_joint_rad=np.abs(arrays["raw_action"] - qreference)
            .max(axis=0)
            .tolist(),
            residual_limit_exceeded_joint_names=[
                name
                for name, value, limit in zip(
                    G1_JOINT_NAMES, action_delta, residual_limits, strict=True
                )
                if value > limit + 1e-8
            ],
            clipped_target_count=int(
                np.count_nonzero(np.abs(arrays["action"] - arrays["raw_action"]) > 1e-8)
            ),
            method="Contact feedforward PD + force-gated ankle feedback; not a learned policy",
            reference_sha256=reference_hash,
            reference=str(args.reference.resolve()),
            reference_minimum_upright_in_stance=float(
                (
                    1
                    - 2
                    * (
                        reference.robot_state[experiment.frames, 4] ** 2
                        + reference.robot_state[experiment.frames, 5] ** 2
                    )
                ).min()
            ),
            physics_dt=0.0025,
            control_dt=0.02,
            inertial_reference_term=args.inertial,
            absolute_upright_feedback=args.absolute_upright,
            initializations=1,
            intermediate_resets=0,
            board_actuators=0,
            forces_applied=False,
        )
        dataset_name = "teacher.npz" if metrics["verified_teacher"] else "unverified_candidate.npz"
        boundaries = arrays["all_boundary_states"]
        np.savez_compressed(
            out / dataset_name,
            **arrays,
            state=boundaries[:-1].astype(np.float32),
            next_state=boundaries[1:].astype(np.float32),
            episode_id=np.zeros(len(arrays["action"]), dtype=np.int32),
            time_s=np.arange(len(arrays["action"])) * 0.02,
            reference_sha256=np.asarray(reference_hash),
            future_goals=reference.state[goal_indices].astype(np.float32),
            reference_goal_indices=goal_indices,
            reference_offsets=np.array([0, 5, 10]),
            board_joint_names=np.asarray(BOARD_JOINT_NAMES),
            joint_names=np.asarray(G1_JOINT_NAMES),
            verified_teacher=np.asarray(metrics["verified_teacher"]),
            metadata_json=np.asarray(json.dumps(metrics)),
        )
        with (out / "native_contacts.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (out / "result.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        reports.append({"case": name, **metrics})
        print(json.dumps({"case": name, **metrics}), flush=True)
    (args.out_dir / "result.json").write_text(
        json.dumps({"solver": solver_report, "trials": reports}, ensure_ascii=False, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
