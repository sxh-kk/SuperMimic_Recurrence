"""Repair only 14 G1 arm joints in a kinematic reference; preserve root/legs/board.

Uses actual MuJoCo contacts and relative point Jacobians without disabling any
collision. Geometry validity does not certify dynamic feasibility or a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear

from supermimic.ollie.physics import compile_scene
from supermimic.ollie.reference import JointReference


def category(model, index, first, second):
    bodies = model.geom_bodyid[[first, second]]
    roots = model.body_rootid[bodies]
    robot = model.body_rootid[model.jnt_bodyid[model.joint("floating_base_joint").id]]
    board = model.body_rootid[index.board_body]
    if model.geom("ground").id in (first, second):
        return "ground"
    if (roots == robot).all():
        return "self"
    if robot in roots and board in roots:
        return "robot_board"
    if (roots == board).all():
        return "board_internal"
    return "other"


class ArmRepair:
    def __init__(self, reference):
        self.reference = reference
        self.model, self.data, self.index = compile_scene(include_robot=True)
        self.arm_qpos = self.index.robot_joint_qpos[15:]
        self.arm_dofs = self.index.robot_joint_qvel[15:]
        ids = self.model.dof_jntid[self.arm_dofs]
        self.lower, self.upper = self.model.jnt_range[ids, 0], self.model.jnt_range[ids, 1]
        self.jac1 = np.zeros((3, self.model.nv))
        self.jac2 = np.zeros_like(self.jac1)
        self.rot_jac = np.zeros_like(self.jac1)

    def pose(self, frame, arm):
        robot, board = self.reference.robot_state[frame], self.reference.board_state[frame]
        qpos = self.model.qpos0.copy()
        qpos[self.index.robot_qpos : self.index.robot_qpos + 7] = robot[:7]
        qpos[self.index.robot_joint_qpos] = robot[7:36]
        qpos[self.arm_qpos] = arm
        qpos[self.index.board_qpos : self.index.board_qpos + 7] = board[:7]
        qpos[self.index.wheel_qpos] = board[7:11]
        return qpos

    def assign(self, frame, arm):
        self.data.qpos[:] = self.pose(frame, arm)
        mujoco.mj_forward(self.model, self.data)

    def constraints(self, margin):
        gradients, errors = [], []
        for contact in self.data.contact[: self.data.ncon]:
            if float(contact.dist) >= margin:
                continue
            first, second = map(int, contact.geom)
            mujoco.mj_jac(
                self.model,
                self.data,
                self.jac1,
                self.rot_jac,
                contact.pos,
                int(self.model.geom_bodyid[first]),
            )
            mujoco.mj_jac(
                self.model,
                self.data,
                self.jac2,
                self.rot_jac,
                contact.pos,
                int(self.model.geom_bodyid[second]),
            )
            gradient = contact.frame[:3] @ (self.jac2 - self.jac1)[:, self.arm_dofs]
            if np.linalg.norm(gradient) > 1e-8:
                gradients.append(gradient.copy())
                errors.append(margin - float(contact.dist))
        return np.asarray(gradients), np.asarray(errors)

    def movable_penetration(self):
        gradients, errors = self.constraints(0.0)
        return float(errors.max()) if len(gradients) else 0.0

    def temporal_constraints(self, frame, arm, arms, margin, substeps=8):
        """Linearize the current sample and both adjacent interpolation intervals."""
        self.assign(frame, arm)
        gradients, errors = self.constraints(margin)
        gradient_parts, error_parts = [gradients], [errors]
        start = self.data.qpos.copy()
        velocity = np.zeros(self.model.nv)
        for neighbour in (frame - 1, frame + 1):
            if not 0 <= neighbour < len(arms):
                continue
            end = self.pose(neighbour, arms[neighbour])
            mujoco.mj_differentiatePos(self.model, velocity, 1.0, start, end)
            for fraction in np.arange(1, substeps) / substeps:
                self.data.qpos[:] = start
                mujoco.mj_integratePos(self.model, self.data.qpos, velocity, fraction)
                mujoco.mj_forward(self.model, self.data)
                gradients, errors = self.constraints(margin)
                if len(gradients):
                    # Hinge interpolation gives this exact endpoint weight.
                    gradient_parts.append(gradients * (1 - fraction))
                    error_parts.append(errors)
        nonempty = [part for part in gradient_parts if len(part)]
        gradients = np.concatenate(nonempty) if nonempty else np.empty((0, 14))
        errors = (
            np.concatenate([part for part in error_parts if len(part)]) if nonempty else np.empty(0)
        )
        penetration = max(0.0, float(errors.max()) - margin) if len(errors) else 0.0
        return gradients, errors, penetration

    def audit(self, arms):
        maxima = {key: 0.0 for key in ("self", "robot_board", "ground", "board_internal", "other")}
        worst, fixed = [], []
        feet, com = [], []
        for frame, arm in enumerate(arms):
            self.assign(frame, arm)
            feet.append(
                self.data.site_xpos[
                    [self.model.site("left_foot").id, self.model.site("right_foot").id]
                ].copy()
            )
            com.append(
                (self.model.body_mass[:, None] * self.data.xipos).sum(axis=0)
                / self.model.body_mass.sum()
            )
            for contact in self.data.contact[: self.data.ncon]:
                first, second = map(int, contact.geom)
                penetration = max(0.0, -float(contact.dist))
                kind = category(self.model, self.index, first, second)
                maxima[kind] = max(maxima[kind], penetration)
                if penetration > 0.001:
                    row = {
                        "frame": frame,
                        "time_video": float(self.reference.time_video[frame]),
                        "penetration_m": penetration,
                        "category": kind,
                        "geoms": [self.model.geom(first).name, self.model.geom(second).name],
                    }
                    worst.append(row)
                    mujoco.mj_jac(
                        self.model,
                        self.data,
                        self.jac1,
                        self.rot_jac,
                        contact.pos,
                        int(self.model.geom_bodyid[first]),
                    )
                    mujoco.mj_jac(
                        self.model,
                        self.data,
                        self.jac2,
                        self.rot_jac,
                        contact.pos,
                        int(self.model.geom_bodyid[second]),
                    )
                    if (
                        np.linalg.norm(
                            contact.frame[:3] @ (self.jac2 - self.jac1)[:, self.arm_dofs]
                        )
                        < 1e-8
                    ):
                        fixed.append(row)
        worst.sort(key=lambda row: -row["penetration_m"])
        return (
            {
                "maximum_penetration_by_category_m": maxima,
                "contacts_over_2mm": sum(row["penetration_m"] > 0.002 for row in worst),
                "worst_contacts": worst[:30],
                "unmovable_contacts_over_2mm": [
                    row for row in fixed if row["penetration_m"] > 0.002
                ][:30],
            },
            np.asarray(feet),
            np.asarray(com),
        )

    def audit_interpolated(self, arms, substeps=8):
        """Check the active collision model between samples, without simulation.

        MuJoCo position integration gives linear hinge interpolation and shortest
        quaternion interpolation. This audits one specified kinematic interpolant;
        it does not constrain a physical rollout to follow those poses.
        """
        poses = []
        for frame, arm in enumerate(arms):
            self.assign(frame, arm)
            poses.append(self.data.qpos.copy())
        maxima = {key: 0.0 for key in ("self", "robot_board", "ground", "board_internal", "other")}
        counts = dict.fromkeys(maxima, 0)
        over_limit, samples, worst = 0, 0, None
        velocity = np.zeros(self.model.nv)
        for frame, start in enumerate(poses):
            fractions = range(substeps) if frame + 1 < len(poses) else [0]
            if frame + 1 < len(poses):
                mujoco.mj_differentiatePos(self.model, velocity, 1.0, start, poses[frame + 1])
            for fraction in fractions:
                self.data.qpos[:] = start
                if fraction:
                    mujoco.mj_integratePos(
                        self.model, self.data.qpos, velocity, fraction / substeps
                    )
                mujoco.mj_forward(self.model, self.data)
                samples += 1
                for contact in self.data.contact[: self.data.ncon]:
                    first, second = map(int, contact.geom)
                    penetration = max(0.0, -float(contact.dist))
                    kind = category(self.model, self.index, first, second)
                    maxima[kind] = max(maxima[kind], penetration)
                    over_limit += int(penetration > 0.002)
                    counts[kind] += int(penetration > 0.002)
                    if worst is None or penetration > worst["penetration_m"]:
                        worst = {
                            "frame_coordinate": frame + fraction / substeps,
                            "penetration_m": penetration,
                            "category": kind,
                            "geoms": [self.model.geom(first).name, self.model.geom(second).name],
                        }
        return {
            "substeps_per_control_interval": substeps,
            "samples": samples,
            "maximum_penetration_by_category_m": maxima,
            "contacts_over_2mm": over_limit,
            "contacts_over_2mm_by_category": counts,
            "worst_contact": worst,
            "scope": "Prescribed kinematic interpolation; active collision geometry only, not a dynamics rollout",
        }

    def repair(self, passes=4, iterations=24, velocity=12.0, margin=0.003):
        original = self.reference.robot_state[:, 22:36]
        target = np.clip(
            gaussian_filter1d(original, 1.0, axis=0, mode="nearest"), self.lower, self.upper
        )
        arms = target.copy()
        dt = float(self.reference.time_physics[1] - self.reference.time_physics[0])
        max_change = velocity * dt
        logs = []
        for outer in range(passes):
            order = range(len(arms)) if outer % 2 == 0 else range(len(arms) - 1, -1, -1)
            for frame in order:
                lower, upper = self.lower.copy(), self.upper.copy()
                neighbours = []
                for neighbour in (frame - 1, frame + 1):
                    if 0 <= neighbour < len(arms):
                        lower = np.maximum(lower, arms[neighbour] - max_change)
                        upper = np.minimum(upper, arms[neighbour] + max_change)
                        neighbours.append(arms[neighbour])
                if (lower > upper + 1e-9).any():
                    raise ValueError("Infeasible arm temporal bounds")
                q = np.clip(arms[frame], lower, upper)
                smooth = np.mean(neighbours, axis=0)
                for _ in range(iterations):
                    gradients, errors, current_penetration = self.temporal_constraints(
                        frame, q, arms, margin
                    )
                    a = [np.eye(14) * 0.08, np.eye(14) * 0.03]
                    b = [(target[frame] - q) * 0.08, (smooth - q) * 0.03]
                    if len(gradients):
                        a.append(gradients * 80)
                        b.append(errors * 80)
                    a, b = np.concatenate(a), np.concatenate(b)
                    lo, hi = np.maximum(lower - q, -0.15), np.minimum(upper - q, 0.15)
                    free = hi - lo > 1e-12
                    step = (lo + hi) / 2
                    if free.any():
                        step[free] = lsq_linear(
                            a[:, free],
                            b - a[:, ~free] @ step[~free],
                            bounds=(lo[free], hi[free]),
                            method="bvls",
                            tol=1e-7,
                            max_iter=40,
                        ).x
                    accepted = False
                    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                        candidate = np.clip(q + alpha * step, lower, upper)
                        _, _, penetration = self.temporal_constraints(frame, candidate, arms, 0.0)
                        if penetration <= max(0.0015, current_penetration - 1e-7):
                            q = candidate
                            accepted = True
                            break
                    if not accepted or np.linalg.norm(alpha * step) < 1e-5:
                        break
                arms[frame] = q
            audit, _, _ = self.audit(arms)
            interpolated = self.audit_interpolated(arms)
            row = {
                "pass": outer + 1,
                **audit["maximum_penetration_by_category_m"],
                "contacts_over_2mm": audit["contacts_over_2mm"],
                "interpolated_self_max_m": interpolated["maximum_penetration_by_category_m"][
                    "self"
                ],
            }
            logs.append(row)
            print(json.dumps(row), flush=True)
            if audit["contacts_over_2mm"] == 0 and row["interpolated_self_max_m"] <= 0.002:
                break
        return arms, logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/ollie/contact_extended/joint_reference_50hz.npz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/ollie/self_collision_corrected/joint_reference_50hz.npz"),
    )
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-velocity", type=float, default=12.0)
    parser.add_argument("--audit-substeps", type=int, default=8)
    args = parser.parse_args()
    if args.reference.resolve() == args.out.resolve():
        parser.error("--out must be a new reference branch, not the source file")
    if min(args.passes, args.iterations, args.audit_substeps) < 1 or args.max_velocity <= 0:
        parser.error("passes, iterations, audit-substeps and max-velocity must be positive")
    source = JointReference.load(args.reference)
    problem = ArmRepair(source)
    before, before_feet, before_com = problem.audit(source.robot_state[:, 22:36])
    before_interpolation = problem.audit_interpolated(
        source.robot_state[:, 22:36], args.audit_substeps
    )
    arms, history = problem.repair(args.passes, args.iterations, args.max_velocity)
    after, after_feet, after_com = problem.audit(arms)
    interpolation_audit = problem.audit_interpolated(arms, args.audit_substeps)
    result = JointReference.load(args.reference)
    result.robot_state[:, 22:36] = arms
    result.robot_state[:, 57:71] = np.gradient(arms, result.time_physics, axis=0)
    limits = np.maximum(problem.lower - arms, 0) + np.maximum(arms - problem.upper, 0)
    report = {
        "source": str(args.reference.resolve()),
        "source_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "method": "14-arm-only bounded contact-Jacobian kinematic correction; no collision disabled",
        "before": before,
        "before_interpolation_audit": before_interpolation,
        "after": after,
        "interpolation_audit": interpolation_audit,
        "history": history,
        "configuration": {
            "passes": args.passes,
            "iterations_per_frame": args.iterations,
            "max_velocity_rad_s": args.max_velocity,
            "collision_margin_m": 0.003,
            "optimization_substeps": 8,
            "audit_substeps": args.audit_substeps,
        },
        "max_arm_position_change_rad": float(np.abs(arms - source.robot_state[:, 22:36]).max()),
        "arm_rms_position_change_rad": float(
            np.sqrt(np.mean((arms - source.robot_state[:, 22:36]) ** 2))
        ),
        "max_arm_velocity_rad_s": float(
            np.abs(np.diff(arms, axis=0) / np.diff(source.time_physics)[:, None]).max()
        ),
        "arm_joint_limit_violation_rad": float(limits.max()),
        "feet_world_position_max_change_m": float(np.abs(after_feet - before_feet).max()),
        "total_com_max_position_change_m": float(
            np.linalg.norm(after_com - before_com, axis=-1).max()
        ),
        "all_geometry_under_2mm": after["contacts_over_2mm"] == 0
        and interpolation_audit["contacts_over_2mm"] == 0,
        "sampled_geometry_under_2mm": after["contacts_over_2mm"] == 0,
        "self_geometry_under_2mm": max(
            after["maximum_penetration_by_category_m"]["self"],
            interpolation_audit["maximum_penetration_by_category_m"]["self"],
        )
        <= 0.002,
        "root_legs_board_preserved": bool(
            np.array_equal(source.robot_state[:, :22], result.robot_state[:, :22])
            and np.array_equal(source.robot_state[:, 36:57], result.robot_state[:, 36:57])
            and np.array_equal(source.board_state, result.board_state)
        ),
        "scope": "Kinematic reference repair only; no force/dynamics or learned-policy success claim",
    }
    result.metadata["self_collision_repair"] = report
    result.metadata["provenance"]["arm_repair_source"] = str(args.reference.resolve())
    result.save(args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    np.savez_compressed(
        args.out.with_name(args.out.stem + "_arm_diagnostics.npz"),
        original_arms=source.robot_state[:, 22:36],
        corrected_arms=arms,
        time_physics=source.time_physics,
        time_video=source.time_video,
        before_com=before_com,
        after_com=after_com,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("before", "after", "history")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
