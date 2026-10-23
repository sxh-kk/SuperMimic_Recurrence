"""Explicit tail-surface kinematic branch; never inject contact forces or edit inputs.

The source video supports a qualitative rear-foot-on-tail prior. Its exact metric
anchor/contact force is unknown. Only root/legs/waist are optimized; all arms and
all board state values remain exact. This does not certify dynamic feasibility.
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
from scipy.spatial.transform import Rotation, Slerp

from scripts.video.repair_ollie_air_reference import Geometry
from scripts.video.repair_ollie_self_collisions import category
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.contact_ik import entity_state_from_pose
from supermimic.ollie.reference import JointReference


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10 + value * (-15 + 6 * value))


class TailRepair:
    """Small damped-Jacobian IK against actual inclined box support planes."""

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
        ix = self.index
        self.qids = np.r_[np.arange(ix.robot_qpos, ix.robot_qpos + 7), ix.robot_joint_qpos[:15]]
        self.dofs = np.r_[np.arange(ix.robot_qvel, ix.robot_qvel + 6), ix.robot_joint_qvel[:15]]
        ids = self.model.dof_jntid[ix.robot_joint_qvel[:15]]
        self.lower, self.upper = self.model.jnt_range[ids].T
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros_like(self.jp)
        self.jp2 = np.zeros_like(self.jp)
        self.jr2 = np.zeros_like(self.jp)
        self.tangent = np.zeros(self.model.nv)
        self.pelvis = self.model.body("pelvis").id
        self.torso = self.model.body("torso_link").id
        self.stance_end = round(args.left_contact_end * reference.metadata["physics_fps"])
        self.right_end = round(args.right_contact_end * reference.metadata["physics_fps"])
        self.release_end = round(args.release_end * reference.metadata["physics_fps"])
        self.surface = [self.model.geom("board_center").id, self.model.geom("board_tail").id]
        self.anchors = np.array(
            [[args.left_anchor, -0.024923426763277894], [args.right_anchor, 0.022452889976235117]]
        )
        # Stance yaw is inherited, while normals follow the actual selected box.
        self.yaws = []
        for side, site in enumerate(self.feet.sites):
            yaw = []
            for frame in range(self.stance_end + 1):
                self.assign(self.source[frame])
                rel = self.data.xmat[ix.board_body].reshape(3, 3).T @ self.data.site_xmat[
                    site
                ].reshape(3, 3)
                yaw.append(np.arctan2(rel[1, 0], rel[0, 0]))
            self.yaws.append(float(np.arctan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw)))))

    def assign(self, pose):
        self.data.qpos[:] = pose
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def contact_target(self, side):
        """Site XY and exact lowest capsule support gap on the selected box top."""
        m, d, ix = self.model, self.data, self.index
        board_r = d.xmat[ix.board_body].reshape(3, 3)
        board_p = d.xpos[ix.board_body]
        gid = self.surface[side]
        surface_r = d.geom_xmat[gid].reshape(3, 3)
        normal = surface_r[:, 2]
        top = d.geom_xpos[gid] + normal * m.geom_size[gid, 2]
        support = self.feet.support(m, d, side, normal)
        gap = float((support - top) @ normal)
        site = d.site_xpos[self.feet.sites[side]]
        local = board_r.T @ (site - board_p)
        # The tail orientation is authored as local Ry(12deg); retain stance yaw.
        desired_r = surface_r @ Rotation.from_euler("z", self.yaws[side]).as_matrix()
        current_r = d.site_xmat[self.feet.sites[side]].reshape(3, 3)
        rot_error = Rotation.from_matrix(desired_r @ current_r.T).as_rotvec()
        mujoco.mj_jacSite(m, d, self.jp, self.jr, int(self.feet.sites[side]))
        xy_jac = board_r[:, :2].T @ self.jp[:, self.dofs]
        angular_jac = self.jr[:, self.dofs].copy()
        mujoco.mj_jac(m, d, self.jp, self.jr, support, int(m.site_bodyid[self.feet.sites[side]]))
        jac = np.vstack([xy_jac, normal @ self.jp[:, self.dofs]])
        err = np.r_[self.anchors[side] - local[:2], self.args.support_gap - gap]
        return jac, err, angular_jac, rot_error

    def constraints(self, margin):
        gradients, errors = [], []
        for contact in self.data.contact[: self.data.ncon]:
            if contact.dist >= margin:
                continue
            first, second = map(int, contact.geom)
            kind = category(self.model, self.index, first, second)
            if kind in ("board_internal", "other"):
                continue
            mujoco.mj_jac(
                self.model,
                self.data,
                self.jp,
                self.jr,
                contact.pos,
                int(self.model.geom_bodyid[first]),
            )
            mujoco.mj_jac(
                self.model,
                self.data,
                self.jp2,
                self.jr2,
                contact.pos,
                int(self.model.geom_bodyid[second]),
            )
            gradient = contact.frame[:3] @ (self.jp2 - self.jp)[:, self.dofs]
            if np.linalg.norm(gradient) > 1e-9:
                gradients.append(gradient.copy())
                errors.append(margin - float(contact.dist))
        return gradients, errors

    def temporal_constraints(self, q, previous, margin):
        self.assign(q)
        gradients, errors = self.constraints(margin)
        peak = self.penetration()
        if previous is not None:
            velocity = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, velocity, 1.0, previous, q)
            for alpha in np.arange(1, 8) / 8:
                mid = previous.copy()
                mujoco.mj_integratePos(self.model, mid, velocity, alpha)
                self.assign(mid)
                gs, es = self.constraints(margin)
                gradients.extend([g * alpha for g in gs])
                errors.extend(es)
                peak = max(peak, self.penetration())
        self.assign(q)
        return gradients, errors, peak

    def penetration(self):
        return max(
            (
                max(0.0, -float(c.dist))
                for c in self.data.contact[: self.data.ncon]
                if category(self.model, self.index, *map(int, c.geom))
                not in ("other", "board_internal")
            ),
            default=0.0,
        )

    def solve(self):
        m, d, args = self.model, self.data, self.args
        poses = self.source.copy()
        records = []
        for frame in range(self.right_end + 1):
            source = self.source[frame]
            q = source.copy()
            if frame:
                # Follow source motion while carrying the last solved adjustment.
                correction = np.zeros(m.nv)
                mujoco.mj_differentiatePos(
                    m, correction, 1.0, self.source[frame - 1], poses[frame - 1]
                )
                mujoco.mj_integratePos(m, q, correction, 1.0)
            q[self.index.robot_joint_qpos[:15]] = np.clip(
                q[self.index.robot_joint_qpos[:15]], self.lower, self.upper
            )
            lo, hi = np.full(len(self.dofs), -0.08), np.full(len(self.dofs), 0.08)
            for iteration in range(args.iterations):
                self.assign(q)
                a, b = [], []
                mujoco.mj_differentiatePos(m, self.tangent, 1.0, q, source)
                prior_weight = np.r_[np.full(3, 0.8), np.full(3, 1.5), np.full(15, 0.5)]
                a.append(np.diag(prior_weight))
                b.append(self.tangent[self.dofs] * prior_weight)
                if args.minimum_pelvis_upright is not None:
                    # Positive d(upright)/d(angular velocity), in world coordinates.
                    normal = d.xmat[self.pelvis].reshape(3, 3)[:, 2]
                    upright = normal[2]
                    requested = args.minimum_pelvis_upright + args.upright_margin
                    if upright < requested:
                        mujoco.mj_jacBody(m, d, self.jp, self.jr, self.pelvis)
                        gradient = (
                            np.cross(normal, np.array([0.0, 0.0, 1.0])) @ self.jr[:, self.dofs]
                        )
                        a.append(gradient[None] * args.upright_weight)
                        b.append(np.array([requested - upright]) * args.upright_weight)
                    # Retain source torso orientation using available waist DoFs.
                    current_torso = d.xmat[self.torso].reshape(3, 3).copy()
                    mujoco.mj_jacBody(m, d, self.jp, self.jr, self.torso)
                    torso_jac = self.jr[:, self.dofs].copy()
                    self.assign(source)
                    desired_torso = d.xmat[self.torso].reshape(3, 3).copy()
                    self.assign(q)
                    torso_error = Rotation.from_matrix(desired_torso @ current_torso.T).as_rotvec()
                    a.append(torso_jac * args.torso_orientation_weight)
                    b.append(torso_error * args.torso_orientation_weight)
                active = [0, 1] if frame <= self.stance_end else [1]
                for side in active:
                    jac, err, jrot, errot = self.contact_target(side)
                    a += [jac * args.position_weight, jrot * args.orientation_weight]
                    b += [err * args.position_weight, errot * args.orientation_weight]
                if frame > self.stance_end:
                    # Released front foot follows its source world trajectory;
                    # there is no constraint attaching it to the board.
                    site = int(self.feet.sites[0])
                    current_site = d.site_xpos[site].copy()
                    mujoco.mj_jacSite(m, d, self.jp, self.jr, site)
                    jac = self.jp[:, self.dofs].copy()
                    self.assign(source)
                    target = d.site_xpos[site].copy()
                    self.assign(q)
                    a.append(jac * 30.0)
                    b.append((target - current_site) * 30.0)
                gradients, errors, old_pen = self.temporal_constraints(
                    q, poses[frame - 1] if frame else None, args.collision_margin
                )
                if gradients:
                    a.append(np.array(gradients) * 250.0)
                    b.append(np.array(errors) * 250.0)
                # Exact true joint limits, plus per-frame velocity constraint.
                lo[:], hi[:] = -0.08, 0.08
                joint = q[self.index.robot_joint_qpos[:15]]
                low, high = self.lower.copy(), self.upper.copy()
                if frame:
                    previous = poses[frame - 1, self.index.robot_joint_qpos[:15]]
                    low = np.maximum(low, previous - args.max_joint_velocity / 50.0)
                    high = np.minimum(high, previous + args.max_joint_velocity / 50.0)
                lo[6:] = np.maximum(lo[6:], low - joint)
                hi[6:] = np.minimum(hi[6:], high - joint)
                # q may carry a prior-frame correction across a sharp source change.
                if np.any(lo >= hi):
                    q[self.index.robot_joint_qpos[:15]] = np.clip(joint, low + 1e-10, high - 1e-10)
                    continue
                step = lsq_linear(np.vstack(a), np.concatenate(b), bounds=(lo, hi), tol=1e-9).x
                old_err = sum(
                    np.dot(self.contact_target(side)[1], self.contact_target(side)[1])
                    for side in active
                )
                if args.minimum_pelvis_upright is not None:
                    u_old = d.xmat[self.pelvis].reshape(3, 3)[2, 2]
                    deficit = max(args.minimum_pelvis_upright + args.upright_margin - u_old, 0.0)
                    old_err += (deficit * args.upright_weight / args.position_weight) ** 2
                accepted = False
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    trial = q.copy()
                    velocity = np.zeros(m.nv)
                    velocity[self.dofs] = step
                    mujoco.mj_integratePos(m, trial, velocity, alpha)
                    self.assign(trial)
                    _, _, pen = self.temporal_constraints(
                        trial, poses[frame - 1] if frame else None, args.collision_margin
                    )
                    err = sum(
                        np.dot(self.contact_target(side)[1], self.contact_target(side)[1])
                        for side in active
                    )
                    if args.minimum_pelvis_upright is not None:
                        u_new = d.xmat[self.pelvis].reshape(3, 3)[2, 2]
                        deficit = max(
                            args.minimum_pelvis_upright + args.upright_margin - u_new, 0.0
                        )
                        err += (deficit * args.upright_weight / args.position_weight) ** 2
                    # Permit small existing self overlap while recovering geometry,
                    # never trade a deep body/deck collision for smaller foot error.
                    if pen <= max(old_pen, args.collision_margin) + 1e-6 and err <= old_err + 1e-7:
                        q = trial
                        accepted = True
                        break
                if not accepted or np.linalg.norm(step) < 1e-6:
                    break
            poses[frame] = q
            self.assign(q)
            records.append(
                {
                    "frame": frame,
                    "time_physics": frame / 50.0,
                    "iterations": iteration + 1,
                    "errors_m": [self.contact_target(s)[1].tolist() for s in active],
                    "penetration_m": self.penetration(),
                }
            )
        # Release: fade a frozen correction to zero. No air-foot board target.
        delta = np.zeros(m.nv)
        mujoco.mj_differentiatePos(
            m, delta, 1.0, self.source[self.right_end], poses[self.right_end]
        )
        for frame in range(self.right_end + 1, self.release_end):
            alpha = (frame - self.right_end) / (self.release_end - self.right_end)
            weight = 1 - smoothstep(alpha)
            mujoco.mj_integratePos(m, poses[frame], delta, weight)
        return poses, records

    def surface_audit(self, poses):
        rows = []
        velocity = np.zeros(self.model.nv)
        for frame, start in enumerate(poses):
            if frame + 1 < len(poses):
                mujoco.mj_differentiatePos(self.model, velocity, 1.0, start, poses[frame + 1])
            for sub in range(8) if frame + 1 < len(poses) else [0]:
                alpha = sub / 8.0
                q = start.copy()
                if sub:
                    mujoco.mj_integratePos(self.model, q, velocity, alpha)
                self.assign(q)
                time = frame / 50.0 + alpha / 50.0
                for side, end in [(0, self.stance_end), (1, self.right_end)]:
                    if time > end / 50.0 + 1e-9:
                        continue
                    gid = self.surface[side]
                    rr = self.data.geom_xmat[gid].reshape(3, 3)
                    support = self.feet.support(self.model, self.data, side, rr[:, 2])
                    local = rr.T @ (support - self.data.geom_xpos[gid])
                    gap = local[2] - self.model.geom_size[gid, 2]
                    board_r = self.data.xmat[self.index.board_body].reshape(3, 3)
                    site = board_r.T @ (
                        self.data.site_xpos[self.feet.sites[side]]
                        - self.data.xpos[self.index.board_body]
                    )
                    # Require a true nearest capsule-to-box pair on the box top.
                    # A capsule may overhang the edge; an arbitrary lowest endpoint
                    # outside the patch is not evidence that all support is absent.
                    distances = []
                    valid_top = []
                    for foot in self.feet.geoms[side]:
                        points = np.zeros(6)
                        distance = mujoco.mj_geomDistance(
                            self.model, self.data, int(foot), gid, 1.0, points
                        )
                        board_point = rr.T @ (points[3:] - self.data.geom_xpos[gid])
                        top_point = board_point[2] >= self.model.geom_size[
                            gid, 2
                        ] - 1e-7 and np.all(
                            np.abs(board_point[:2]) <= self.model.geom_size[gid, :2] + 1e-7
                        )
                        distances.append(distance)
                        if top_point:
                            valid_top.append(distance)
                    nearest_top = min(valid_top, default=1.0)
                    rows.append(
                        [
                            time,
                            side,
                            gap,
                            *local,
                            *site,
                            float(
                                np.all(np.abs(local[:2]) <= self.model.geom_size[gid, :2] + 1e-8)
                            ),
                            min(distances),
                            nearest_top,
                            float(nearest_top < 0.003),
                        ]
                    )
        arr = np.asarray(rows)
        report = {}
        for side in range(2):
            ss = arr[arr[:, 1] == side]
            report[("left_center", "right_tail")[side]] = {
                "samples": len(ss),
                "minimum_gap_m": float(ss[:, 2].min()),
                "maximum_gap_m": float(ss[:, 2].max()),
                "max_site_xy_error_m": float(
                    np.linalg.norm(ss[:, 6:8] - self.anchors[side], axis=1).max()
                ),
                "lowest_endpoint_inside_selected_box_top_fraction": float(ss[:, 9].mean()),
                "nearest_capsule_box_top_within_3mm_fraction": float(ss[:, 12].mean()),
                "maximum_nearest_actual_top_distance_m": float(ss[:, 11].max()),
                "minimum_capsule_box_signed_distance_m": float(ss[:, 10].min()),
                "site_x_range_m": [float(ss[:, 6].min()), float(ss[:, 6].max())],
            }
        return arr, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/ollie/air_relative_video_timed_v1/joint_reference_50hz.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--right-anchor", type=float, default=-0.32)
    parser.add_argument("--left-anchor", type=float, default=0.20502073041326405)
    parser.add_argument("--left-contact-end", type=float, default=2.76)
    parser.add_argument("--right-contact-end", type=float, default=2.90)
    parser.add_argument("--release-end", type=float, default=3.14)
    parser.add_argument("--support-gap", type=float, default=0.0008)
    parser.add_argument("--max-joint-velocity", type=float, default=12.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--position-weight", type=float, default=120.0)
    parser.add_argument("--orientation-weight", type=float, default=12.0)
    parser.add_argument("--collision-margin", type=float, default=0.0005)
    parser.add_argument(
        "--minimum-pelvis-upright",
        type=float,
        default=None,
        help="Opt-in lower bound during solved contact phase; default preserves v3 algorithm",
    )
    parser.add_argument("--upright-margin", type=float, default=0.005)
    parser.add_argument("--upright-weight", type=float, default=300.0)
    parser.add_argument("--torso-orientation-weight", type=float, default=15.0)
    args = parser.parse_args()
    if args.out_dir.exists() or args.diagnostics.exists():
        raise FileExistsError(
            "Use new output/diagnostic directories; bound inputs are never replaced"
        )
    if not (0 <= args.left_contact_end <= args.right_contact_end < args.release_end):
        raise ValueError("Require left-end <= right-end < release-end")
    if args.max_joint_velocity <= 0 or args.iterations < 1 or args.support_gap < 0:
        raise ValueError(
            "Positive velocity/iteration settings and nonnegative support gap required"
        )
    if args.minimum_pelvis_upright is not None and not (0.0 < args.minimum_pelvis_upright < 1.0):
        raise ValueError("Requested upright must lie between0 and1")
    if args.upright_margin < 0 or args.upright_weight <= 0 or args.torso_orientation_weight < 0:
        raise ValueError(
            "Upright margin/torso weight nonnegative, upright weight positive required"
        )
    source = JointReference.load(args.source)
    if source.metadata["physics_fps"] != 50.0 or source.metadata["time_scale"] != 1.5:
        raise ValueError("Reviewed contact time priors require the50Hz scale1.5 branch")
    if args.release_end >= source.time_physics[-1]:
        raise ValueError("Release must finish before the final source sample")
    if not all(
        np.isclose(time * 50.0, round(time * 50.0))
        for time in (args.left_contact_end, args.right_contact_end, args.release_end)
    ):
        raise ValueError("Contact and release endpoints must align with50Hz source samples")
    if not (-0.35 <= args.right_anchor <= -0.29):
        raise ValueError("Reviewed right anchor must remain within the authored kicktail region")
    args.out_dir.mkdir(parents=True)
    args.diagnostics.mkdir(parents=True)
    repair = TailRepair(source, args)
    poses, records = repair.solve()
    result = copy.deepcopy(source)
    ix = repair.index
    robot = np.concatenate(
        [poses[:, ix.robot_qpos : ix.robot_qpos + 7], poses[:, ix.robot_joint_qpos]], -1
    )
    regenerated = entity_state_from_pose(robot, 50.0)
    # Preserve arm positions AND velocity values exactly, including prior smoothing.
    result.robot_state[:, :22] = regenerated[:, :22]
    result.robot_state[:, 36:57] = regenerated[:, 36:57]
    result.robot_state[:, 22:36] = source.robot_state[:, 22:36]
    result.robot_state[:, 57:71] = source.robot_state[:, 57:71]
    # Exact old suffix including velocities after the correction fully vanishes.
    result.robot_state[repair.release_end + 1 :] = source.robot_state[repair.release_end + 1 :]
    _, dense = repair.geometry.sample(repair.geometry.poses(result.robot_state, result.board_state))
    surface_values, surface = repair.surface_audit(
        repair.geometry.poses(result.robot_state, result.board_state)
    )
    velocity = np.diff(result.robot_state[:, 7:36], axis=0) * 50.0
    ranges = repair.model.jnt_range[repair.model.dof_jntid[ix.robot_joint_qvel]]
    violations = np.maximum(
        ranges[:, 0] - result.robot_state[:, 7:36], result.robot_state[:, 7:36] - ranges[:, 1]
    )
    robot_shift = np.linalg.norm(result.robot_state[:, :3] - source.robot_state[:, :3], axis=1)
    joint_change = np.max(abs(result.robot_state[:, 7:36] - source.robot_state[:, 7:36]), axis=0)
    geometry_pass = all(
        dense["maximum_penetration_by_category_m"][key] <= 0.002 + 1e-9
        for key in ("self", "robot_board", "ground")
    )
    surface_pass = all(
        row["max_site_xy_error_m"] < 0.003
        and row["minimum_gap_m"] >= -0.0015
        and row["maximum_gap_m"] < 0.003
        and row["nearest_capsule_box_top_within_3mm_fraction"] == 1.0
        and row["minimum_capsule_box_signed_distance_m"] >= -0.0015
        for row in surface.values()
    )
    velocity_pass = bool(np.max(abs(velocity)) <= args.max_joint_velocity + 1e-6)
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "script_sha256": sha256(__file__),
        "method": "Explicit engineering rear-foot-tail contact IK; no dynamics/force injection or inferred measured contact",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "board_full_state_exact": bool(np.array_equal(source.board_state, result.board_state)),
        "arms_position_velocity_exact": bool(
            np.array_equal(source.robot_state[:, 22:36], result.robot_state[:, 22:36])
            and np.array_equal(source.robot_state[:, 57:], result.robot_state[:, 57:])
        ),
        "unknown_air_not_attached": True,
        "surface_audit": surface,
        "dense_full_geometry": dense,
        "max_joint_limit_violation_rad": float(max(0.0, violations.max())),
        "max_adjacent_joint_speed_rad_s": float(abs(velocity).max()),
        "max_root_shift_m": float(robot_shift.max()),
        "max_root_speed_m_s": float(np.linalg.norm(result.robot_state[:, 36:39], axis=1).max()),
        "max_root_acceleration_m_s2": float(
            np.linalg.norm(np.gradient(result.robot_state[:, 36:39], 0.02, axis=0), axis=1).max()
        ),
        "max_per_joint_change_rad": dict(zip(G1_JOINT_NAMES, joint_change.tolist(), strict=True)),
        "geometry_pass_2mm": geometry_pass,
        "surface_geometry_pass": surface_pass,
        "velocity_pass": velocity_pass,
        "kinematic_candidate_passed": bool(
            geometry_pass and surface_pass and velocity_pass and violations.max() <= 1e-8
        ),
        "physics_success": False,
        "unresolved": [
            "Metric right-tail anchor is a qualitative-video engineering assumption, not measured3D contact",
            "Right contact ends2.90s; source near-touch4.421video is uncertain; compare explicit2.94s variant",
            "No contact force, torque feasibility, stability or Ollie success certified",
            "Preparation prefix changed; old teacher/BC must not be reused as hash-compatible",
        ],
    }
    # Check the requested pelvis requirement over the same2.5ms pose interpolation.
    dense_time = np.arange((len(source.time_physics) - 1) * 8 + 1) * 0.0025
    rotations = Rotation.from_quat(result.robot_state[:, [4, 5, 6, 3]])
    dense_upright = Slerp(source.time_physics, rotations)(dense_time).as_matrix()[:, 2, 2]
    prepare = dense_time <= args.left_contact_end + 1e-9
    report["minimum_prepare_pelvis_upright_dense"] = float(dense_upright[prepare].min())
    report["minimum_pelvis_upright_requested"] = args.minimum_pelvis_upright
    report["upright_pass"] = bool(
        args.minimum_pelvis_upright is None
        or report["minimum_prepare_pelvis_upright_dense"] >= args.minimum_pelvis_upright
    )
    report["kinematic_candidate_passed"] &= report["upright_pass"]
    np.savez_compressed(
        args.diagnostics / "pelvis_upright_dense.npz",
        time_physics=dense_time,
        upright=dense_upright,
    )
    torso_errors = []
    for old, new in zip(
        repair.source, repair.geometry.poses(result.robot_state, result.board_state), strict=True
    ):
        repair.assign(old)
        original = repair.data.xmat[repair.torso].reshape(3, 3).copy()
        repair.assign(new)
        changed = repair.data.xmat[repair.torso].reshape(3, 3).copy()
        torso_errors.append(np.linalg.norm(Rotation.from_matrix(changed @ original.T).as_rotvec()))
    report["max_torso_orientation_change_rad"] = float(max(torso_errors))
    report["surface_contact_label_meaning"] = (
        "Engineering prior, not observed3D points or measured forces"
    )
    report["upstream_metadata_scope"] = (
        "Earlier contact/air IK diagnostics describe source-stage provenance; current geometry is audited here"
    )
    report["unchanged_suffix_from_physical_s"] = float(source.time_physics[repair.release_end + 1])
    report["unchanged_suffix_state_exact"] = bool(
        np.array_equal(
            result.state[repair.release_end + 1 :], source.state[repair.release_end + 1 :]
        )
    )
    # Expose CURRENT branch QA, retaining the old report only as stage provenance.
    result.metadata["upstream_contact_qa"] = copy.deepcopy(result.metadata.get("qa", {}))
    result.metadata["qa"] = {
        "passed_geometric_thresholds": report["kinematic_candidate_passed"],
        "meaning": "Current tail branch: prescribed2.5ms collision and actual box-top distance QA; no dynamics certification",
        "joint_limit_violation_max_rad": report["max_joint_limit_violation_rad"],
        "joint_velocity_max_rad_s": float(np.max(np.abs(result.robot_state[:, 42:]))),
        "joint_velocity_bound_rad_s": args.max_joint_velocity,
        "maximum_penetration_by_category_m": dense["maximum_penetration_by_category_m"],
        "contact_labels_are_engineering_priors": True,
    }
    result.metadata["tail_contact_repair"] = report
    result.metadata["provenance"]["tail_contact_source_sha256"] = report["source_sha256"]
    result.metadata["provenance"]["tail_contact_script_sha256"] = report["script_sha256"]
    # Contact labels are engineering priors; air remains unknown, not welded.
    for side, end in enumerate((repair.stance_end, repair.right_end)):
        col = result.contact_names.index(("left_foot_board", "right_foot_board")[side])
        result.contacts[: end + 1, col] = 1
    result.save(args.out_dir / "joint_reference_50hz.npz")
    report["output_sha256"] = sha256(args.out_dir / "joint_reference_50hz.npz")
    if sha256(args.source) != report["source_sha256"]:
        raise RuntimeError("Source changed during fitting; do not use this candidate")
    (args.diagnostics / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.diagnostics / "solver_frames.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(
        args.diagnostics / "surface_audit.npz",
        values=surface_values,
        columns=np.array(
            [
                "time_physics",
                "side",
                "gap",
                "surface_x",
                "surface_y",
                "surface_z",
                "site_board_x",
                "site_board_y",
                "site_board_z",
                "lowest_endpoint_inside_top_patch",
                "capsule_box_min_distance",
                "nearest_top_distance",
                "top_within_3mm",
            ]
        ),
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "output_sha256",
                    "surface_audit",
                    "dense_full_geometry",
                    "max_adjacent_joint_speed_rad_s",
                    "max_root_shift_m",
                    "kinematic_candidate_passed",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
