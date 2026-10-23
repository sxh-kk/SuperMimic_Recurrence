"""Video-supported sliding/catch contact IK, with fixed board and COM nodes.

No forces, welds, simulation episode, or measured3D contact are synthesized.
A failed geometric/temporal candidate remains explicitly FAILED for inspection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from scripts.video.repair_ollie_air_reference import Geometry
from scripts.video.repair_ollie_ballistic_reference import Audit, component_audit
from scripts.video.repair_ollie_self_collisions import category
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.contact_ik import entity_state_from_pose
from supermimic.ollie.reference import JointReference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quintic(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10 + value * (-15 + 6 * value))


class ContactRepair:
    def __init__(self, reference, args):
        self.reference, self.args = reference, args
        self.geometry = Geometry(reference)
        self.model, self.data, self.index = (
            self.geometry.model,
            self.geometry.data,
            self.geometry.index,
        )
        self.feet = self.geometry.feet
        self.source = self.geometry.poses()
        self.robot_fraction = self.geometry.robot_mass / self.geometry.total_mass
        self.mass = self.model.body_mass.copy()
        ix = self.index
        self.full = np.r_[np.arange(ix.robot_qvel, ix.robot_qvel + 6), ix.robot_joint_qvel[:15]]
        self.qj = ix.robot_joint_qpos[:15]
        ranges = self.model.jnt_range[self.model.dof_jntid[ix.robot_joint_qvel[:15]]]
        self.lower, self.upper = ranges.T
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros_like(self.jp)
        self.jp2 = np.zeros_like(self.jp)
        self.jr2 = np.zeros_like(self.jp)
        self.jcom = np.zeros_like(self.jp)
        self.working_frame = 0
        self.target_com = np.empty((len(self.source), 3))
        for f, p in enumerate(self.source):
            self.assign(p)
            self.target_com[f] = self.com()
        self.top = {
            name: self.model.geom("board_" + name).id for name in ("center", "tail", "nose")
        }

    def assign(self, q):
        self.data.qpos[:] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def com(self):
        return (self.mass[:, None] * self.data.xipos).sum(0) / self.mass.sum()

    def fixed_com(self, frame):
        return self.args.com_start - 1e-9 <= frame / 50.0 <= self.args.com_end + 1e-9

    def project_com(self, q, frame):
        if not self.fixed_com(frame):
            return q
        self.assign(q)
        q = q.copy()
        q[self.index.robot_qpos : self.index.robot_qpos + 3] += (
            self.target_com[frame] - self.com()
        ) / self.robot_fraction
        self.assign(q)
        return q

    def basis(self, frame):
        """Eliminate root translation via exact whole-system COM preservation."""
        columns = self.full[3:] if self.fixed_com(frame) else self.full
        t = np.zeros((self.model.nv, len(columns)))
        t[columns, np.arange(len(columns))] = 1.0
        if self.fixed_com(frame):
            mujoco.mj_jacSubtreeCom(self.model, self.data, self.jcom, 0)
            t[self.index.robot_qvel : self.index.robot_qvel + 3] = (
                -self.jcom[:, columns] / self.robot_fraction
            )
        return t

    def phase(self, frame):
        time = frame / 50.0
        # These are engineering contact/approach priors, never force labels.
        left_x = float(
            np.interp(
                time,
                [2.8, 2.90, 3.14, 3.28, 3.56, 4.0],
                [0.205, 0.205, self.args.nose_x, self.args.nose_x, 0.26, 0.20780671638984932],
            )
        )
        left_y = float(np.interp(time, [2.8, 3.10, 3.56, 4.0], [-0.0249234, 0.0, 0.0, 0.025]))
        phases = [
            {
                "side": 0,
                "name": "left_front_slide" if time <= 3.28 else "left_landing",
                "x": left_x,
                "y": left_y,
                "surface": "center" if left_x < 0.278 else "nose",
                "gap": self.args.contact_gap,
                "contact": True,
            }
        ]
        if time <= 2.90 + 1e-9:
            phases.append(
                {
                    "side": 1,
                    "name": "right_tail_until_release",
                    "x": -0.32,
                    "y": 0.02245289,
                    "surface": "tail",
                    "gap": self.args.contact_gap,
                    "contact": True,
                }
            )
        elif time >= self.args.rear_approach_start - 1e-9:
            weight = quintic(
                (time - self.args.rear_approach_start)
                / (self.args.rear_catch - self.args.rear_approach_start)
            )
            gap = self.args.contact_gap + (1 - weight) * self.args.rear_approach_gap
            phases.append(
                {
                    "side": 1,
                    "name": "right_approach"
                    if time < self.args.rear_catch - 1e-9
                    else "right_catch",
                    "x": -0.205,
                    "y": float(np.interp(time, [3.32, 4.0], [0.0, -0.025])),
                    "surface": "center",
                    "gap": gap,
                    "contact": time >= self.args.rear_catch - 1e-9,
                }
            )
        return phases

    def contact_rows(self, phase, basis):
        """Point-to-real-capsule distance at a finite upper/edge box witness.

        Foot orientation is unconstrained. Selecting the closest capsule permits
        the toe/side to meet the authored finite surface without a flat-sole weld.
        """
        m, d, ix = self.model, self.data, self.index
        gid = self.top[phase["surface"]]
        sr = d.geom_xmat[gid].reshape(3, 3)
        board_r = d.xmat[ix.board_body].reshape(3, 3)
        board_p = d.xpos[ix.board_body]
        raw = board_p + board_r @ np.array([phase["x"], phase["y"], 0.0])
        local = sr.T @ (raw - d.geom_xpos[gid])
        local[:2] = np.clip(local[:2], -m.geom_size[gid, :2], m.geom_size[gid, :2])
        local[2] = m.geom_size[gid, 2]
        target = d.geom_xpos[gid] + sr @ local
        candidates = []
        for foot in self.feet.geoms[phase["side"]]:
            axis = d.geom_xmat[foot].reshape(3, 3)[:, 2]
            c = d.geom_xpos[foot]
            radius, half = m.geom_size[foot, :2]
            parameter = np.clip((target - c) @ axis, -half, half)
            segment = c + parameter * axis
            vector = segment - target
            length = np.linalg.norm(vector)
            direction = vector / max(length, 1e-12)
            foot_point = segment - radius * direction
            candidates.append((length - radius, int(foot), foot_point, direction))
        distance, foot, point, normal = min(candidates, key=lambda row: row[0])
        mujoco.mj_jac(m, d, self.jp, self.jr, point, int(m.geom_bodyid[foot]))
        distance_gradient = normal @ self.jp @ basis
        site = int(self.feet.sites[phase["side"]])
        site_local = board_r.T @ (d.site_xpos[site] - board_p)
        mujoco.mj_jacSite(m, d, self.jp, self.jr, site)
        xy = board_r[:, :2].T @ self.jp @ basis
        # XY is a weak sliding guide, not a fixed sole pose.
        a = np.vstack([distance_gradient * self.args.contact_weight, xy * self.args.xy_weight])
        b = np.r_[
            (phase["gap"] - distance) * self.args.contact_weight,
            (np.array([phase["x"], phase["y"]]) - site_local[:2]) * self.args.xy_weight,
        ]
        return (
            a,
            b,
            {
                "distance_to_finite_top_witness_m": float(distance),
                "selected_capsule": m.geom(foot).name,
                "site_board_xyz": site_local.tolist(),
                "surface": phase["surface"],
            },
        )

    def collision_rows(self, basis, margin):
        rows = []
        errors = []
        peak = 0.0
        for c in self.data.contact[: self.data.ncon]:
            first, second = map(int, c.geom)
            kind = category(self.model, self.index, first, second)
            if kind in ("board_internal", "other"):
                continue
            peak = max(peak, -float(c.dist))
            if c.dist >= margin:
                continue
            mujoco.mj_jac(
                self.model, self.data, self.jp, self.jr, c.pos, int(self.model.geom_bodyid[first])
            )
            mujoco.mj_jac(
                self.model,
                self.data,
                self.jp2,
                self.jr2,
                c.pos,
                int(self.model.geom_bodyid[second]),
            )
            grad = c.frame[:3] @ (self.jp2 - self.jp) @ basis
            if np.linalg.norm(grad) > 1e-9:
                rows.append(grad.copy())
                errors.append(margin - float(c.dist))
        return rows, errors, peak

    def temporal_collision_rows(self, q, previous, basis):
        self.assign(q)
        rows, errors, peak = self.collision_rows(basis, self.args.collision_margin)
        if previous is not None:
            velocity = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, velocity, 1.0, previous, q)
            for alpha in np.arange(1, 8) / 8:
                mid = previous.copy()
                mujoco.mj_integratePos(self.model, mid, velocity, alpha)
                self.assign(mid)
                rr, ee, pp = self.collision_rows(basis, self.args.collision_margin)
                rows.extend([row * alpha for row in rr])
                errors.extend(ee)
                peak = max(peak, pp)
        self.assign(q)
        return rows, errors, peak

    def pose_weights(self, frame):
        late = quintic((frame / 50.0 - 3.6) / 0.38)
        return np.r_[
            np.full(3, 0.6 + 20 * late), np.full(3, 2.0 + 20 * late), np.full(15, 0.5 + 20 * late)
        ]

    def merit(self, frame, q, previous, basis):
        """Exact nonlinear residual cost for every row in the local LS system."""
        self.assign(q)
        tangent = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, tangent, 1.0, q, self.source[frame])
        pose_residual = self.pose_weights(frame) * tangent[self.full]
        pose_cost = float(pose_residual @ pose_residual)
        contact_cost = sum(
            float(b @ b)
            for phase in self.phase(frame)
            for _, b, _ in [self.contact_rows(phase, basis)]
        )
        _, errors, peak = self.temporal_collision_rows(q, previous, basis)
        collision_cost = float(np.dot(errors, errors)) * self.args.collision_weight**2
        return (
            pose_cost + contact_cost + collision_cost,
            peak,
            {
                "pose_cost": pose_cost,
                "contact_cost": contact_cost,
                "collision_cost": collision_cost,
            },
        )

    def gap_metrics(self, frame):
        result = []
        for phase in self.phase(frame):
            minimum = 2.0
            for foot in self.feet.geoms[phase["side"]]:
                for surface in self.top.values():
                    minimum = min(
                        minimum,
                        float(
                            mujoco.mj_geomDistance(
                                self.model, self.data, int(foot), surface, 2.0, None
                            )
                        ),
                    )
            result.append(
                {"side": phase["side"], "phase": phase["name"], "actual_finite_gap_m": minimum}
            )
        return result

    def solve(self):
        m = self.model
        poses = self.source.copy()
        records = []
        trace = []
        start = round(self.args.modify_start * 50)
        end = round(self.args.modify_end * 50)
        for frame in range(start, end + 1):
            source = self.source[frame]
            q = source.copy()
            if frame > start:
                change = np.zeros(m.nv)
                mujoco.mj_differentiatePos(m, change, 1.0, self.source[frame - 1], poses[frame - 1])
                mujoco.mj_integratePos(m, q, change, 1.0)
            previous = poses[frame - 1]
            low = np.maximum(self.lower, previous[self.qj] - self.args.max_joint_velocity / 50.0)
            high = np.minimum(self.upper, previous[self.qj] + self.args.max_joint_velocity / 50.0)
            # Reach the untouched4.0s recovery pose without a terminal joint jump.
            remaining = (200 - frame) * self.args.max_joint_velocity / 50.0
            low = np.maximum(low, self.source[200, self.qj] - remaining)
            high = np.minimum(high, self.source[200, self.qj] + remaining)
            if np.any(low > high + 1e-9):
                raise RuntimeError(
                    "Sequential IK left no true-limit/velocity path to fixed recovery"
                )
            q[self.qj] = np.clip(q[self.qj], low, high)
            q = self.project_com(q, frame)
            for iteration in range(self.args.iterations):
                self.assign(q)
                basis = self.basis(frame)
                tangent = np.zeros(m.nv)
                mujoco.mj_differentiatePos(m, tangent, 1.0, q, source)
                # Same pose residual and weights are used by merit().
                weights = self.pose_weights(frame)
                a = [weights[:, None] * basis[self.full]]
                b = [weights * tangent[self.full]]
                for phase in self.phase(frame):
                    aa, bb, _ = self.contact_rows(phase, basis)
                    a.append(aa)
                    b.append(bb)
                cr, ce, old_peak = self.temporal_collision_rows(q, previous, basis)
                if cr:
                    a.append(np.asarray(cr) * self.args.collision_weight)
                    b.append(np.asarray(ce) * self.args.collision_weight)
                n = basis.shape[1]
                lo = np.full(n, -0.06)
                hi = np.full(n, 0.06)
                lo[-15:] = np.maximum(lo[-15:], low - q[self.qj])
                hi[-15:] = np.minimum(hi[-15:], high - q[self.qj])
                if np.any(lo >= hi):
                    q[self.qj] = np.clip(q[self.qj], low + 1e-10, high - 1e-10)
                    q = self.project_com(q, frame)
                    continue
                vector_b = np.concatenate(b)
                step = lsq_linear(np.vstack(a), vector_b, bounds=(lo, hi), tol=1e-8).x
                old_merit, old_peak, old_costs = self.merit(frame, q, previous, basis)
                zero_cost_error = abs(old_merit - float(vector_b @ vector_b))
                before_gaps = self.gap_metrics(frame)
                accepted = False
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    trial = q.copy()
                    mujoco.mj_integratePos(m, trial, basis @ step, alpha)
                    trial = self.project_com(trial, frame)
                    score, peak, costs = self.merit(frame, trial, previous, basis)
                    if (
                        peak <= max(old_peak, self.args.collision_margin) + 1e-7
                        and score <= old_merit + 1e-5
                    ):
                        q = trial
                        accepted = True
                        break
                self.assign(q)
                trace.append(
                    {
                        "frame": frame,
                        "iteration": iteration,
                        "accepted": accepted,
                        "ls_zero_step_cost_abs_error": zero_cost_error,
                        "merit_before": old_merit,
                        "merit_after": score if accepted else old_merit,
                        "peak_before_m": old_peak,
                        "peak_after_m": peak if accepted else old_peak,
                        "costs_before": old_costs,
                        "costs_after": costs if accepted else old_costs,
                        "gaps_before": before_gaps,
                        "gaps_after": self.gap_metrics(frame),
                    }
                )
                if not accepted or np.linalg.norm(step) < 1e-6:
                    break
            poses[frame] = q
            self.assign(q)
            basis = self.basis(frame)
            records.append(
                {
                    "frame": frame,
                    "time_physics": frame / 50.0,
                    "iterations": iteration + 1,
                    "com_error_m": float(np.linalg.norm(self.com() - self.target_com[frame]))
                    if self.fixed_com(frame)
                    else None,
                    "contact_targets": [
                        {**phase, **self.contact_rows(phase, basis)[2]}
                        for phase in self.phase(frame)
                    ],
                }
            )
        return poses, records, trace

    def safety_backoff(self, poses):
        records = []
        tangent = np.zeros((len(poses), self.model.nv))
        for frame in range(len(poses)):
            mujoco.mj_differentiatePos(
                self.model, tangent[frame], 1.0, self.source[frame], poses[frame]
            )
        for weight in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0):
            trial = self.source.copy()
            for frame in range(len(trial)):
                if weight:
                    mujoco.mj_integratePos(self.model, trial[frame], tangent[frame], weight)
                trial[frame] = self.project_com(trial[frame], frame)
            trial[:140] = self.source[:140]
            _, report = self.geometry.sample(trial)
            speed = float(
                np.abs(np.diff(trial[:, self.index.robot_joint_qpos], axis=0) * 50.0).max()
            )
            passed = bool(
                all(
                    report["maximum_penetration_by_category_m"][k] <= 0.002 + 1e-9
                    for k in ("self", "robot_board", "ground")
                )
                and speed <= self.args.max_joint_velocity + 1e-6
            )
            records.append(
                {
                    "correction_weight": weight,
                    "geometry": report,
                    "max_joint_speed_rad_s": speed,
                    "safe_geometry_velocity": passed,
                }
            )
            if passed:
                return trial, records
        raise RuntimeError("Even exact source failed inherited geometry/velocity gates")

    def surface_sample(self, poses):
        rows = []
        velocity = np.zeros(self.model.nv)
        for frame in range(
            round(self.args.modify_start * 50), round(self.args.modify_end * 50) + 1
        ):
            if frame + 1 < len(poses):
                mujoco.mj_differentiatePos(
                    self.model, velocity, 1.0, poses[frame], poses[frame + 1]
                )
            for sub in range(8):
                alpha = sub / 8.0
                time = frame / 50.0 + alpha / 50.0
                q = poses[frame].copy()
                mujoco.mj_integratePos(self.model, q, velocity, alpha)
                self.assign(q)
                # Continuous phase only for audit: categorical boundaries use exact time.
                nearest_frame = time * 50.0
                phases = self.phase(nearest_frame)
                for phase in phases:
                    if not phase["contact"]:
                        continue
                    side = phase["side"]
                    nearest = []
                    for foot in self.feet.geoms[side]:
                        for name, gid in self.top.items():
                            points = np.zeros(6)
                            distance = mujoco.mj_geomDistance(
                                self.model, self.data, int(foot), gid, 2.0, points
                            )
                            rr = self.data.geom_xmat[gid].reshape(3, 3)
                            bp = rr.T @ (points[3:] - self.data.geom_xpos[gid])
                            # Upper half includes an upper edge/side witness; never underside.
                            upper = bp[2] >= -1e-7
                            if upper:
                                nearest.append((float(distance), name, points.copy()))
                    best = (
                        min(nearest, key=lambda v: v[0]) if nearest else (2.0, "none", np.zeros(6))
                    )
                    rows.append(
                        {
                            "time": time,
                            "side": side,
                            "phase": phase["name"],
                            "geometry_gap_m": best[0],
                            "nearest_box": best[1],
                            "upper_or_upper_edge_witness": best[1] != "none",
                            "within_2mm": best[0] <= 0.002,
                            "points_world_m": best[2].reshape(2, 3).tolist(),
                        }
                    )
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/ollie/ballistic_tail_upright_candidate_v3/joint_reference_50hz.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--modify-start", type=float, default=2.80)
    parser.add_argument("--modify-end", type=float, default=3.98)
    parser.add_argument("--com-start", type=float, default=3.10)
    parser.add_argument("--com-end", type=float, default=3.58)
    parser.add_argument("--nose-x", type=float, default=0.33)
    parser.add_argument("--rear-approach-start", type=float, default=3.14)
    parser.add_argument("--rear-catch", type=float, default=3.32)
    parser.add_argument("--rear-approach-gap", type=float, default=0.15)
    parser.add_argument("--contact-gap", type=float, default=0.0008)
    parser.add_argument("--contact-weight", type=float, default=180.0)
    parser.add_argument("--xy-weight", type=float, default=30.0)
    parser.add_argument("--collision-weight", type=float, default=350.0)
    parser.add_argument("--collision-margin", type=float, default=0.0005)
    parser.add_argument("--max-joint-velocity", type=float, default=12.0)
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    if args.out_dir.exists() or args.diagnostics.exists():
        raise FileExistsError("Use new directories; no bound reference is overwritten")
    if not (2.80 <= args.modify_start < args.com_start < args.com_end < args.modify_end <= 3.98):
        raise ValueError("Reviewed window ordering required")
    if not (args.rear_approach_start < args.rear_catch <= 3.56):
        raise ValueError("Rear approach precedes conservative catch by landing")
    source = JointReference.load(args.source)
    if source.metadata["physics_fps"] != 50.0 or source.metadata["time_scale"] != 1.5:
        raise ValueError("Requires reviewed50Hz scale1.5 branch")
    args.out_dir.mkdir(parents=True)
    args.diagnostics.mkdir(parents=True)
    repair = ContactRepair(source, args)
    poses, records, iteration_trace = repair.solve()
    np.savez_compressed(args.diagnostics / "raw_local_ik_qpos.npz", qpos=poses)
    poses, backoff_records = repair.safety_backoff(poses)
    result = copy.deepcopy(source)
    ix = repair.index
    robot = np.c_[poses[:, ix.robot_qpos : ix.robot_qpos + 7], poses[:, ix.robot_joint_qpos]]
    state = entity_state_from_pose(robot, 50.0)
    result.robot_state[:, :22] = state[:, :22]
    result.robot_state[:, 36:57] = state[:, 36:57]
    # Preserve fullstate prefix including velocities; its next pose2.78 is untouched.
    regenerated_prefix_error = float(
        np.max(abs(result.robot_state[:139] - source.robot_state[:139]))
    )
    result.robot_state[:139] = source.robot_state[:139]
    result.robot_state[139, :36] = source.robot_state[139, :36]
    result.robot_state[:, 22:36] = source.robot_state[:, 22:36]
    result.robot_state[:, 57:] = source.robot_state[:, 57:]
    result.robot_state[201:] = source.robot_state[201:]
    audit = Audit(result, 0.002)
    samples, dense, _ = audit.sample(result)
    components = component_audit(samples, audit)
    surfaces = repair.surface_sample(repair.geometry.poses(result.robot_state, result.board_state))
    surface_summary = {}
    for name in sorted({row["phase"] for row in surfaces}):
        selected = [row for row in surfaces if row["phase"] == name]
        surface_summary[name] = {
            "samples": len(selected),
            "maximum_gap_m": max(row["geometry_gap_m"] for row in selected),
            "within_2mm_fraction": float(np.mean([row["within_2mm"] for row in selected])),
        }
    qvel = np.diff(result.robot_state[:, 7:36], axis=0) * 50.0
    ranges = repair.model.jnt_range[repair.model.dof_jntid[ix.robot_joint_qvel]]
    violation = float(
        max(
            0.0,
            np.max(
                np.maximum(
                    ranges[:, 0] - result.robot_state[:, 7:36],
                    result.robot_state[:, 7:36] - ranges[:, 1],
                )
            ),
        )
    )
    lo, hi = round(args.com_start * 50), round(args.com_end * 50)
    com_error = np.linalg.norm(
        samples["com"][::8][lo : hi + 1] - repair.target_com[lo : hi + 1], axis=1
    )
    invariants = {
        "protected_complete_state_through_2_76_exact": bool(
            np.array_equal(source.state[:139], result.state[:139])
        ),
        "pose_2_78_exact": bool(
            np.array_equal(source.robot_state[139, :36], result.robot_state[139, :36])
        ),
        "board_all_state_exact": bool(np.array_equal(source.board_state, result.board_state)),
        "arms_q_qd_exact": bool(
            np.array_equal(source.robot_state[:, 22:36], result.robot_state[:, 22:36])
            and np.array_equal(source.robot_state[:, 57:], result.robot_state[:, 57:])
        ),
        "com25_node_max_error_m": float(com_error.max()),
        "unmodified_suffix_from_4_02_exact": bool(
            np.array_equal(source.state[201:], result.state[201:])
        ),
    }
    geometry_pass = all(
        dense["maximum_penetration_by_category_m"][k] <= 0.002 + 1e-9
        for k in ("self", "robot_board", "ground")
    )
    contacts_pass = all(s["within_2mm_fraction"] == 1.0 for s in surface_summary.values())
    velocity_pass = float(abs(qvel).max()) <= args.max_joint_velocity + 1e-6
    passed = bool(
        geometry_pass
        and contacts_pass
        and velocity_pass
        and violation <= 1e-8
        and all(v for k, v in invariants.items() if isinstance(v, bool))
        and com_error.max() < 1e-8
    )
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": sha(args.source),
        "script_sha256": sha(__file__),
        "meaning": "Engineering kinematic contact candidate, no measured forces or physics-success claim",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "invariants": invariants,
        "surface_summary": surface_summary,
        "safety_backoff": backoff_records,
        "accepted_correction_weight": backoff_records[-1]["correction_weight"],
        "dense_geometry": dense,
        "component_necessary_conditions": components,
        "regenerated_prefix_velocity_roundoff_max": regenerated_prefix_error,
        "merit_scope": "Same source-pose, contact and weighted temporal-collision squared residuals as LS; independent peak non-worsening gate retained",
        "max_ls_merit_zero_step_cost_error": max(
            (row["ls_zero_step_cost_abs_error"] for row in iteration_trace), default=0.0
        ),
        "max_joint_speed_rad_s": float(abs(qvel).max()),
        "max_joint_limit_violation_rad": violation,
        "max_per_joint_change_rad": dict(
            zip(
                G1_JOINT_NAMES,
                np.max(
                    abs(result.robot_state[:, 7:36] - source.robot_state[:, 7:36]), axis=0
                ).tolist(),
                strict=True,
            )
        ),
        "geometry_pass_2mm": geometry_pass,
        "contact_geometry_pass": contacts_pass,
        "joint_velocity_pass": velocity_pass,
        "kinematic_candidate_passed": passed,
        "candidate_status": "KINEMATIC_CHECKS_PASSED_DYNAMICS_UNVERIFIED"
        if passed
        else "FAILED_KINEMATIC_OR_TEMPORAL_CHECKS",
        "physics_success": False,
        "unresolved": [
            "Contact timing/point/path are explicit priors from qualitative2D video, never measured3D forces",
            "Global node COM preservation is necessary only; dense interpolation, contact force and inverse dynamics remain separate checks",
            "No simulation weld, state-following controller, or external force added",
            "Failed output is saved for inspection and must not be promoted as a feasible reference",
        ],
    }
    result.metadata["air_contact_repair"] = report
    result.metadata["provenance"]["air_contact_source_sha256"] = report["source_sha256"]
    result.metadata["upstream_qa_before_air_contact"] = copy.deepcopy(result.metadata.get("qa", {}))
    result.metadata["qa"] = {
        "passed_geometric_thresholds": passed,
        "meaning": report["meaning"],
        "joint_velocity_max_rad_s": report["max_joint_speed_rad_s"],
        "joint_limit_violation_max_rad": violation,
        "dense_geometry": dense,
    }
    # Requested targets never become achieved/observed contact labels.
    # Preserve source unknowns and already explicit priors; export targets separately.
    report["contact_label_policy"] = (
        "Source labels exact; requested targets and geometric achievements are separate diagnostics, never promoted to contact=1"
    )
    report["invariants"]["source_contact_labels_exact"] = bool(
        np.array_equal(result.contacts, source.contacts)
    )
    requested = np.full((len(source.time_physics), 2), -1, dtype=np.int8)
    for f in range(round(args.modify_start * 50), round(args.modify_end * 50) + 1):
        for phase in repair.phase(f):
            requested[f, phase["side"]] = 1 if phase["contact"] else -1
    np.savez_compressed(
        args.diagnostics / "requested_contact_targets.npz",
        requested_contact_prior=requested,
        meaning=np.array("Requested engineering targets, not achieved geometry or measured forces"),
    )
    result.save(args.out_dir / "joint_reference_50hz.npz")
    report["output_sha256"] = sha(args.out_dir / "joint_reference_50hz.npz")
    (args.diagnostics / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.diagnostics / "solver_frames.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.diagnostics / "iteration_trace.json").write_text(
        json.dumps(iteration_trace, indent=2) + "\n"
    )
    (args.diagnostics / "surface_witnesses.json").write_text(json.dumps(surfaces, indent=2) + "\n")
    np.savez_compressed(args.diagnostics / "after_samples.npz", **samples)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "output_sha256",
                    "candidate_status",
                    "invariants",
                    "surface_summary",
                    "dense_geometry",
                    "max_joint_speed_rad_s",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
