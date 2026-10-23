"""Soft contact-aware kinematic reference correction, never a physics controller.

Uses the real G1 foot sites and capsule support geometry. Only explicitly marked
stance frames constrain feet to the board; unknown/flight frames are not welded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import quat_rotate_inverse, resample_motion
from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_TOP_OFFSET


@dataclass(frozen=True)
class ContactIKConfig:
    max_joint_velocity: float = 12.0
    max_root_speed: float = 3.0
    max_root_angular_speed: float = 8.0
    smoothing_sigma_frames: float = 1.5
    iterations: int = 18
    board_half_length: float = 0.40065
    board_half_width: float = 0.105
    stance_half_length: float = 0.27
    stance_half_width: float = 0.025
    position_weight: float = 60.0
    orientation_weight: float = 3.0
    pose_weight: float = 0.6


def entity_state_from_pose(qpos: np.ndarray, fps: float) -> np.ndarray:
    """Root xyz/wxyz + any scalar joints -> SI state with body angular velocity."""
    if qpos.shape[1] == 36:
        csv = np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=-1)
        motion = resample_motion(csv, fps, fps)
        return np.concatenate([
            motion.root_pos, motion.root_quat, motion.joint_pos, motion.root_lin_vel_w,
            quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w), motion.joint_vel,
        ], -1)
    rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    world_angular = (rotation[2:] * rotation[:-2].inv()).as_rotvec() * (fps / 2)
    world_angular = np.concatenate([world_angular[:1], world_angular, world_angular[-1:]])
    return np.concatenate([
        qpos, np.gradient(qpos[:, :3], 1 / fps, axis=0), rotation.inv().apply(world_angular),
        np.gradient(qpos[:, 7:], 1 / fps, axis=0),
    ], -1)


class FootGeometry:
    def __init__(self, model):
        self.sites = np.array([model.site(f"{side}_foot").id for side in ("left", "right")])
        self.geoms = [np.array([
            i for i in range(model.ngeom)
            if re.fullmatch(fr"{side}_foot[1-7]_collision", model.geom(i).name or "")
        ]) for side in ("left", "right")]
        if any(len(indices) != 7 for indices in self.geoms):
            raise ValueError("Expected seven actual G1 foot collision capsules per foot")
        if any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE for group in self.geoms for i in group):
            raise ValueError("Foot support calculation requires capsule collision geometry")

    def support(self, model, data, side: int, normal: np.ndarray):
        """Lowest real capsule surface point along a world-space board normal."""
        points = []
        for gid in self.geoms[side]:
            axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            endpoint = data.geom_xpos[gid] - np.sign(np.dot(axis, normal)) * model.geom_size[gid, 1] * axis
            points.append(endpoint - model.geom_size[gid, 0] * normal)
        return np.asarray(points)[np.argmin(np.asarray(points) @ normal)]


def correct_contact_reference(
    model, index, robot_qpos: np.ndarray, board_qpos: np.ndarray,
    contact_mask: np.ndarray, time_physics: np.ndarray,
    config: ContactIKConfig | None = None,
):
    """Return corrected robot qpos plus honest geometric/temporal diagnostics.

    Fixed board dimensions are retained. A contact-conditioned root translation
    calibration removes the GMR global scale/translation mismatch; it is an
    engineering assumption recorded in diagnostics, not a recovered camera path.
    """
    cfg = config or ContactIKConfig()
    frames = len(robot_qpos)
    if robot_qpos.shape != (frames, 36) or board_qpos.shape != (frames, 7) or contact_mask.shape != (frames, 2):
        raise ValueError("Expected robot Nx36, board Nx7, and contact mask Nx2")
    if frames < 3 or not np.isfinite(robot_qpos).all() or not np.isfinite(board_qpos).all():
        raise ValueError("Need at least three finite poses")
    dt = float(time_physics[1] - time_physics[0])
    if dt <= 0 or not np.allclose(np.diff(time_physics), dt):
        raise ValueError("Physical time must be uniform and increasing")
    if not contact_mask.any():
        raise ValueError("No explicitly marked stance frames; provide contact priors or stance windows")
    if min(cfg.max_joint_velocity, cfg.max_root_speed, cfg.max_root_angular_speed, cfg.iterations) <= 0:
        raise ValueError("Velocity bounds and iteration count must be positive")
    data = mujoco.MjData(model)
    feet = FootGeometry(model)
    foot_geoms = set(np.concatenate(feet.geoms).tolist())
    board_geoms = {model.geom(name).id for name in BOARD_GEOM_NAMES}
    robot_v = np.r_[np.arange(index.robot_qvel, index.robot_qvel + 6), index.robot_joint_qvel]
    joint_ids = model.dof_jntid[index.robot_joint_qvel]
    ranges = model.jnt_range[joint_ids]
    target = robot_qpos.copy()
    target[:, 7:] = np.clip(gaussian_filter1d(
        target[:, 7:], cfg.smoothing_sigma_frames, axis=0, mode="nearest"
    ), ranges[:, 0], ranges[:, 1]) if cfg.smoothing_sigma_frames > 0 else np.clip(
        target[:, 7:], ranges[:, 0], ranges[:, 1]
    )

    def assign(robot, board):
        data.qpos[index.robot_qpos:index.robot_qpos + 7] = robot[:7]
        data.qpos[index.robot_joint_qpos] = robot[7:]
        data.qpos[index.board_qpos:index.board_qpos + 7] = board
        data.qpos[index.wheel_qpos] = 0
        mujoco.mj_forward(model, data)

    def overlap_score(rotation, board_position):
        normal = rotation[:, 2]
        score = 0.0
        for foot_side, foot_site in enumerate(feet.sites):
            local = rotation.T @ (data.site_xpos[foot_site] - board_position)
            if abs(local[0]) <= cfg.board_half_length + 0.03 and abs(local[1]) <= cfg.board_half_width + 0.1:
                gap_value = np.dot(feet.support(model, data, foot_side, normal) - board_position, normal) - BOARD_TOP_OFFSET
                score = max(score, -float(gap_value) - 0.0015)
        for item in data.contact[:data.ncon]:
            first, second = map(int, item.geom)
            if (first in foot_geoms and second in board_geoms) or (second in foot_geoms and first in board_geoms):
                score = max(score, -float(item.dist) - 0.002)
        return score

    # Calibrate only on explicit stance samples, interpolate correction in gaps.
    # Board geometry remains 0.8013 x 0.21 m, without applying the 0.866 human scale.
    offsets = np.full((frames, 3), np.nan)
    for frame in np.flatnonzero(contact_mask.any(axis=1)):
        assign(target[frame], board_qpos[frame])
        r = Rotation.from_quat(board_qpos[frame, [4, 5, 6, 3]]).as_matrix()
        active = np.flatnonzero(contact_mask[frame])
        centre = data.site_xpos[feet.sites[active]].mean(axis=0)
        support_height = np.mean([np.dot(feet.support(model, data, side, r[:, 2]) - board_qpos[frame, :3], r[:, 2]) for side in active])
        centre_local = r.T @ (centre - board_qpos[frame, :3])
        offsets[frame] = r @ np.array([-centre_local[0], -centre_local[1], BOARD_TOP_OFFSET - support_height])
    known = np.flatnonzero(np.isfinite(offsets[:, 0]))
    for axis in range(3):
        offsets[:, axis] = np.interp(time_physics, time_physics[known], offsets[known, axis])
        offsets[:, axis] = gaussian_filter1d(offsets[:, axis], 2.0, mode="nearest")
    target[:, :3] += offsets

    # One per-foot anchor/yaw per contiguous marked stance segment. Unknown gaps
    # release the foot; the next landing has its own explicitly estimated anchor.
    anchors = np.zeros((frames, 2, 3))
    yaw = np.zeros((frames, 2))
    for side in range(2):
        marked = np.flatnonzero(contact_mask[:, side])
        for segment in np.split(marked, np.flatnonzero(np.diff(marked) > 1) + 1):
            if not len(segment):
                continue
            positions, yaws = [], []
            for frame in segment:
                assign(target[frame], board_qpos[frame])
                r = Rotation.from_quat(board_qpos[frame, [4, 5, 6, 3]]).as_matrix()
                positions.append(r.T @ (data.site_xpos[feet.sites[side]] - board_qpos[frame, :3]))
                relative = r.T @ data.site_xmat[feet.sites[side]].reshape(3, 3)
                yaws.append(np.arctan2(relative[1, 0], relative[0, 0]))
            anchor = np.median(positions, axis=0)
            anchor[:2] = np.clip(anchor[:2], [-cfg.stance_half_length, -cfg.stance_half_width], [cfg.stance_half_length, cfg.stance_half_width])
            anchors[segment, side] = anchor
            yaw[segment, side] = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))

    corrected = target.copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    tangent = np.zeros(model.nv)
    velocity = np.zeros(model.nv)
    for frame in range(frames):
        q = target[frame].copy()
        previous = corrected[frame - 1] if frame else None
        lower, upper = ranges[:, 0].copy(), ranges[:, 1].copy()
        if previous is not None:
            lower = np.maximum(lower, previous[7:] - cfg.max_joint_velocity * dt)
            upper = np.minimum(upper, previous[7:] + cfg.max_joint_velocity * dt)
            q[:3] = previous[:3] + target[frame, :3] - target[frame - 1, :3]
            root_step_box = cfg.max_root_speed * dt / np.sqrt(3)
            q[:3] = np.clip(q[:3], previous[:3] - root_step_box, previous[:3] + root_step_box)
            q[7:] = previous[7:] + target[frame, 7:] - target[frame - 1, 7:]
            target_change = Rotation.from_quat(target[frame, [4, 5, 6, 3]]) * Rotation.from_quat(
                target[frame - 1, [4, 5, 6, 3]]
            ).inv()
            q[3:7] = (target_change * Rotation.from_quat(previous[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
        q[7:] = np.clip(q[7:], lower, upper)
        board_rotation = Rotation.from_quat(board_qpos[frame, [4, 5, 6, 3]]).as_matrix()
        normal = board_rotation[:, 2]
        for _ in range(cfg.iterations):
            assign(q, board_qpos[frame])
            current_overlap = overlap_score(board_rotation, board_qpos[frame, :3])
            current_q = q.copy()
            current_full = data.qpos.copy()
            assign(target[frame], board_qpos[frame])
            reference_full = data.qpos.copy()
            data.qpos[:] = current_full
            mujoco.mj_forward(model, data)
            mujoco.mj_differentiatePos(model, tangent, 1.0, current_full, reference_full)
            a = [np.eye(35) * cfg.pose_weight]
            b = [tangent[robot_v] * cfg.pose_weight]
            for side in np.flatnonzero(contact_mask[frame]):
                site = feet.sites[side]
                support = feet.support(model, data, side, normal)
                support_offset = np.dot(support - data.site_xpos[site], normal)
                local_target = anchors[frame, side].copy()
                local_target[2] = BOARD_TOP_OFFSET - support_offset
                desired = board_qpos[frame, :3] + board_rotation @ local_target
                desired_r = board_rotation @ Rotation.from_euler("z", yaw[frame, side]).as_matrix()
                current_r = data.site_xmat[site].reshape(3, 3)
                angle_error = Rotation.from_matrix(desired_r @ current_r.T).as_rotvec()
                mujoco.mj_jacSite(model, data, jacp, jacr, int(site))
                planar_jacobian = board_rotation[:, :2].T @ jacp[:, robot_v]
                angular_jacobian = jacr[:, robot_v].copy()
                mujoco.mj_jac(model, data, jacp, jacr, support, int(model.site_bodyid[site]))
                position_jacobian = np.concatenate([planar_jacobian, (normal @ jacp[:, robot_v])[None]])
                position_error = board_rotation.T @ (desired - data.site_xpos[site])
                a.extend([position_jacobian * cfg.position_weight, angular_jacobian * cfg.orientation_weight])
                b.extend([position_error * cfg.position_weight, angle_error * cfg.orientation_weight])
            # A unilateral geometric barrier can separate an unmarked foot from
            # the deck without attaching it. This does not infer contact forces.
            for side, site in enumerate(feet.sites):
                local_site = board_rotation.T @ (data.site_xpos[site] - board_qpos[frame, :3])
                if abs(local_site[0]) > cfg.board_half_length + 0.03 or abs(local_site[1]) > cfg.board_half_width + 0.1:
                    continue
                support = feet.support(model, data, side, normal)
                gap = np.dot(support - board_qpos[frame, :3], normal) - BOARD_TOP_OFFSET
                if gap < 0.001:
                    mujoco.mj_jac(model, data, jacp, jacr, support, int(model.site_bodyid[site]))
                    a.append((normal @ jacp[:, robot_v])[None] * cfg.position_weight * 2)
                    b.append(np.array([0.001 - gap]) * cfg.position_weight * 2)
            for contact in data.contact[:data.ncon]:
                g1, g2 = map(int, contact.geom)
                if float(contact.dist) >= 0:
                    continue
                if g1 in foot_geoms and g2 in board_geoms:
                    foot, direction = g1, -contact.frame[:3]
                elif g2 in foot_geoms and g1 in board_geoms:
                    foot, direction = g2, contact.frame[:3]
                else:
                    continue
                # Never solve a reference by moving a penetrated foot farther
                # under the board. The Ollie sole is on its upper side.
                if np.dot(direction, normal) <= 0:
                    continue
                mujoco.mj_jac(model, data, jacp, jacr, contact.pos, int(model.geom_bodyid[foot]))
                a.append((direction @ jacp[:, robot_v])[None] * cfg.position_weight)
                b.append(np.array([0.001 - float(contact.dist)]) * cfg.position_weight)
            # Solve the angle/velocity box inside IK. Clipping an unconstrained
            # solution afterwards can repeatedly request blocked ankle motion
            # instead of using the free root to resolve a penetration.
            step_lower, step_upper = np.full(35, -0.12), np.full(35, 0.12)
            if previous is not None:
                step_lower[:3] = np.maximum(step_lower[:3], previous[:3] - root_step_box - q[:3])
                step_upper[:3] = np.minimum(step_upper[:3], previous[:3] + root_step_box - q[:3])
            step_lower[6:] = np.maximum(step_lower[6:], lower - q[7:])
            step_upper[6:] = np.minimum(step_upper[6:], upper - q[7:])
            step = lsq_linear(
                np.concatenate(a), np.concatenate(b), bounds=(step_lower, step_upper),
                method="bvls", tol=1e-6, max_iter=40,
            ).x
            velocity[:] = 0
            velocity[robot_v] = step
            mujoco.mj_integratePos(model, data.qpos, velocity, 1.0)
            q[:7] = data.qpos[index.robot_qpos:index.robot_qpos + 7]
            q[7:] = np.clip(data.qpos[index.robot_joint_qpos], lower, upper)
            if previous is not None:
                difference = q[:3] - previous[:3]
                q[:3] = previous[:3] + difference * min(1.0, cfg.max_root_speed * dt / max(np.linalg.norm(difference), 1e-12))
                prev_rot = Rotation.from_quat(previous[[4, 5, 6, 3]])
                change = (Rotation.from_quat(q[[4, 5, 6, 3]]) * prev_rot.inv()).as_rotvec()
                change *= min(1.0, cfg.max_root_angular_speed * dt / max(np.linalg.norm(change), 1e-12))
                q[3:7] = (Rotation.from_rotvec(change) * prev_rot).as_quat()[[3, 0, 1, 2]]
            # A pose-only step can cross the thin deck before the unilateral
            # barrier activates. Backtrack on the actual nonlinear geometry so
            # the final iterate cannot simply be the penetrating half of a cycle.
            candidate = q.copy()
            change = (Rotation.from_quat(candidate[[4, 5, 6, 3]]) * Rotation.from_quat(current_q[[4, 5, 6, 3]]).inv()).as_rotvec()
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                q = current_q + alpha * (candidate - current_q)
                q[3:7] = (Rotation.from_rotvec(alpha * change) * Rotation.from_quat(current_q[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
                assign(q, board_qpos[frame])
                if overlap_score(board_rotation, board_qpos[frame, :3]) <= current_overlap + 1e-7:
                    accepted = True
                    break
            if not accepted:
                q = current_q
                break
            if np.linalg.norm(step) < 1e-5:
                break
        corrected[frame] = q

    gap = np.zeros((frames, 2))
    slip = np.full((frames, 2), np.nan)
    penetration = np.zeros(frames)
    foot_positions = np.empty((frames, 2, 3))
    for frame in range(frames):
        assign(corrected[frame], board_qpos[frame])
        r = Rotation.from_quat(board_qpos[frame, [4, 5, 6, 3]]).as_matrix()
        for side, site in enumerate(feet.sites):
            foot_positions[frame, side] = data.site_xpos[site]
            gap[frame, side] = np.dot(feet.support(model, data, side, r[:, 2]) - board_qpos[frame, :3], r[:, 2]) - BOARD_TOP_OFFSET
            if contact_mask[frame, side]:
                local = r.T @ (data.site_xpos[site] - board_qpos[frame, :3])
                slip[frame, side] = np.linalg.norm(local[:2] - anchors[frame, side, :2])
        for contact in data.contact[:data.ncon]:
            g1, g2 = map(int, contact.geom)
            if (g1 in foot_geoms and g2 in board_geoms) or (g2 in foot_geoms and g1 in board_geoms):
                penetration[frame] = max(penetration[frame], -float(contact.dist))
    if not np.isfinite(corrected).all():
        raise RuntimeError("Nonfinite contact IK result")
    return corrected, {
        "root_translation_calibration": offsets, "contact_mask": contact_mask,
        "foot_support_gap_m": gap, "foot_anchor_error_xy_m": slip,
        "foot_site_world": foot_positions, "foot_board_penetration_m": penetration,
        "anchors_board_local": anchors,
    }
