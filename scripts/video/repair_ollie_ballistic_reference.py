"""Robot-root-only COM consistency candidate; prescribed kinematics, no rollout.

The board, joint positions/velocities and root orientations/angular velocities
remain exact. A global ballistic COM is a necessary condition only: disconnected
robot/board components are audited independently, without inventing contact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.video.audit_ollie_timing import geometry_minimum_z, segments
from scripts.video.repair_ollie_air_reference import Geometry
from scripts.video.repair_ollie_self_collisions import category
from supermimic.ollie.reference import JointReference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self, reference, contact_band):
        self.geometry = Geometry(reference)
        self.model, self.data, self.index = (
            self.geometry.model,
            self.geometry.data,
            self.geometry.index,
        )
        self.contact_band = contact_band
        self.mass = self.model.body_mass.copy()
        self.robot = self.geometry.robot_bodies
        self.board = self.model.body_rootid == self.geometry.board_root
        self.robot_fraction = float(self.mass[self.robot].sum() / self.mass.sum())
        self.gravity = self.model.opt.gravity.copy()
        self.ground = self.model.geom("ground").id
        self.upper = max(
            float(
                self.model.geom(name).pos[2]
                + np.abs(
                    Rotation.from_quat(self.model.geom(name).quat[[1, 2, 3, 0]]).as_matrix()[2]
                )
                @ self.model.geom(name).size
            )
            for name in ("board_center", "board_tail", "board_nose")
        )
        self.robot_geoms = np.asarray(self.geometry.robot_geoms)
        self.board_geoms = np.asarray(self.geometry.board_geoms)

    def possible_robot_board_connection(self):
        """Conservative near contact, including positive gaps omitted by mj_contact."""
        model, data = self.model, self.data
        first, second = self.robot_geoms, self.board_geoms
        distance = np.linalg.norm(
            data.geom_xpos[first, None] - data.geom_xpos[None, second], axis=-1
        )
        bound = model.geom_rbound[first, None] + model.geom_rbound[None, second] + self.contact_band
        for r, b in np.argwhere(distance <= bound):
            if (
                mujoco.mj_geomDistance(
                    model, data, int(first[r]), int(second[b]), self.contact_band + 1e-6, None
                )
                <= self.contact_band
            ):
                return True
        return False

    def entity(self, geom):
        if geom == self.ground:
            return "ground"
        return "robot" if self.robot[self.model.geom_bodyid[geom]] else "board"

    def sample(self, reference, substeps=8):
        model, data, index = self.model, self.data, self.index
        poses = self.geometry.poses(reference.robot_state, reference.board_state)
        velocities = np.zeros((len(poses), model.nv))
        velocities[:, index.robot_qvel : index.robot_qvel + 6] = reference.robot_state[:, 36:42]
        velocities[:, index.robot_joint_qvel] = reference.robot_state[:, 42:71]
        velocities[:, index.board_qvel : index.board_qvel + 6] = reference.board_state[:, 11:17]
        velocities[:, index.wheel_qvel] = reference.board_state[:, 17:21]
        arrays = {
            key: []
            for key in (
                "time",
                "interval",
                "fraction",
                "com",
                "robot_com",
                "board_com",
                "com_velocity_jac",
                "robot_com_velocity_jac",
                "board_com_velocity_jac",
                "robot_floor",
                "board_floor",
                "foot_local",
                "sole_gap",
                "board_rotation",
                "robot_ground",
                "board_ground",
                "robot_board",
                "strict_robot_ground",
                "strict_board_ground",
                "strict_robot_board",
            )
        }
        maximum = dict.fromkeys(("self", "robot_board", "ground", "board_internal", "other"), 0.0)
        counts = dict.fromkeys(maximum, 0)
        cuts, worst = [], []
        velocity, jacobian = np.zeros(model.nv), np.zeros((3, model.nv))
        dt = 1 / reference.metadata["physics_fps"]
        for frame, pose in enumerate(poses):
            if frame + 1 < len(poses):
                mujoco.mj_differentiatePos(model, velocity, 1.0, pose, poses[frame + 1])
            for substep in range(substeps) if frame + 1 < len(poses) else [0]:
                fraction = substep / substeps
                data.qpos[:] = pose
                if substep:
                    mujoco.mj_integratePos(model, data.qpos, velocity, fraction)
                data.qvel[:] = velocities[frame]
                if fraction:
                    data.qvel[:] += fraction * (velocities[frame + 1] - velocities[frame])
                mujoco.mj_forward(model, data)
                moment = self.mass[:, None] * data.xipos
                arrays["com"].append(moment.sum(0) / self.mass.sum())
                arrays["robot_com"].append(moment[self.robot].sum(0) / self.mass[self.robot].sum())
                arrays["board_com"].append(moment[self.board].sum(0) / self.mass[self.board].sum())
                for name, body in (
                    ("com_velocity_jac", 0),
                    ("robot_com_velocity_jac", self.geometry.robot_root),
                    ("board_com_velocity_jac", self.geometry.board_root),
                ):
                    mujoco.mj_jacSubtreeCom(model, data, jacobian, body)
                    arrays[name].append(jacobian @ data.qvel)
                rotation = data.xmat[index.board_body].reshape(3, 3)
                position = data.qpos[index.board_qpos : index.board_qpos + 3]
                arrays["foot_local"].append(
                    (data.site_xpos[self.geometry.feet.sites] - position) @ rotation
                )
                arrays["sole_gap"].append(
                    [
                        float(
                            (
                                self.geometry.feet.support(model, data, side, rotation[:, 2])
                                - position
                            )
                            @ rotation[:, 2]
                            - 0.006
                        )
                        for side in range(2)
                    ]
                )
                arrays["board_rotation"].append(rotation.copy())
                arrays["robot_floor"].append(
                    geometry_minimum_z(model, data, self.geometry.robot_geoms)
                )
                arrays["board_floor"].append(
                    geometry_minimum_z(model, data, self.geometry.board_geoms)
                )
                time = float(reference.time_physics[frame] + fraction * dt)
                arrays["time"].append(time)
                arrays["interval"].append(frame)
                arrays["fraction"].append(fraction)
                edges = dict.fromkeys(("robot_ground", "board_ground", "robot_board"), False)
                strict = edges.copy()
                for contact in data.contact[: data.ncon]:
                    first, second = map(int, contact.geom)
                    entities = {self.entity(first), self.entity(second)}
                    edge = (
                        "robot_ground"
                        if entities == {"robot", "ground"}
                        else "board_ground"
                        if entities == {"board", "ground"}
                        else "robot_board"
                        if entities == {"robot", "board"}
                        else None
                    )
                    if edge:
                        edges[edge] |= contact.dist <= self.contact_band
                        strict[edge] |= contact.dist <= 1e-6
                    kind = category(model, index, first, second)
                    depth = max(0.0, -float(contact.dist))
                    maximum[kind] = max(maximum[kind], depth)
                    counts[kind] += int(depth > 0.002)
                    if depth > 0.002:
                        worst.append(
                            {
                                "time": time,
                                "category": kind,
                                "depth_m": depth,
                                "geoms": [model.geom(first).name, model.geom(second).name],
                            }
                        )
                        if edge in ("robot_board", "robot_ground"):
                            direction = np.array(contact.frame[:3]) * (
                                -1 if self.entity(first) == "robot" else 1
                            )
                            cuts.append((len(arrays["time"]) - 1, direction, float(contact.dist)))
                # A small positive gap is an uncertainty band, never a claim
                # of measured force. Include it conservatively in connectivity.
                edges["robot_ground"] |= arrays["robot_floor"][-1] <= self.contact_band
                edges["board_ground"] |= arrays["board_floor"][-1] <= self.contact_band
                edges["robot_board"] |= self.possible_robot_board_connection()
                for name in edges:
                    arrays[name].append(edges[name])
                    arrays["strict_" + name].append(strict[name])
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        report = {
            "samples": len(arrays["time"]),
            "sample_dt_s": dt / substeps,
            "maximum_penetration_by_category_m": maximum,
            "contacts_over_2mm_by_category": counts,
            "worst_contacts": sorted(worst, key=lambda row: -row["depth_m"])[:20],
        }
        return arrays, report, cuts


def air_mask(samples, threshold):
    return (
        (samples["robot_floor"] > threshold)
        & (samples["board_floor"] > threshold)
        & ~samples["robot_ground"]
        & ~samples["board_ground"]
    )


def windows(mask, time):
    return [
        {"start_s": float(time[group[0]]), "end_s": float(time[group[-1]]), "samples": len(group)}
        for group in segments(mask)
    ]


def correction(times, com, intervals, gravity, robot_fraction, parameters):
    """Affine root shifts: ballistic COM core, C1 Hermite bridges plus C1 bumps."""
    shift, derivative, weight = np.zeros_like(com), np.zeros_like(com), np.zeros(len(times))
    com_dot = np.gradient(com, times, axis=0)
    for number, interval in enumerate(intervals):
        left, lo, hi, right = interval["frames"]
        origin = interval["origin_s"]
        position, velocity, left_bump, right_bump = parameters[
            number * 12 : (number + 1) * 12
        ].reshape(4, 3)
        tau = times - origin
        delta = (
            position + tau[:, None] * velocity + 0.5 * tau[:, None] ** 2 * gravity - com
        ) / robot_fraction
        delta_dot = (velocity + tau[:, None] * gravity - com_dot) / robot_fraction
        shift[lo : hi + 1] += delta[lo : hi + 1]
        derivative[lo : hi + 1] += delta_dot[lo : hi + 1]
        weight[lo : hi + 1] = 1
        for start, end, is_left, bump in (
            (left, lo, True, left_bump),
            (hi, right, False, right_bump),
        ):
            ids = np.arange(start, end + 1)
            duration = times[end] - times[start]
            u = (times[ids] - times[start]) / duration
            h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
            h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
            d00, d10, d01, d11 = (
                (6 * u**2 - 6 * u) / duration,
                3 * u**2 - 4 * u + 1,
                (-6 * u**2 + 6 * u) / duration,
                3 * u**2 - 2 * u,
            )
            amplitude, slope = (
                (delta[end], delta_dot[end]) if is_left else (delta[start], delta_dot[start])
            )
            value = (h01 if is_left else h00)[:, None] * amplitude + (h11 if is_left else h10)[
                :, None
            ] * duration * slope
            speed = (d01 if is_left else d00)[:, None] * amplitude + (d11 if is_left else d10)[
                :, None
            ] * slope
            value += (16 * u**2 * (1 - u) ** 2)[:, None] * bump
            speed += (32 * u * (1 - u) * (1 - 2 * u) / duration)[:, None] * bump
            # Core endpoints were assigned above. Avoid adding their value twice.
            chosen = ids < lo if is_left else ids > hi
            shift[ids[chosen]] += value[chosen]
            derivative[ids[chosen]] += speed[chosen]
            weight[ids[chosen]] = (h01 if is_left else h00)[chosen]
    return shift, derivative, weight


def linear_rows(matrix, directions):
    return np.einsum("nij,ni->nj", matrix, directions)


def fit(reference, audit, before, intervals, iterations):
    times = reference.time_physics
    com = before["com"][::8]
    dimensions = len(intervals) * 12
    zero = np.zeros(dimensions)
    constant = correction(times, com, intervals, audit.gravity, audit.robot_fraction, zero)[0]
    matrix = np.stack(
        [
            correction(
                times, com, intervals, audit.gravity, audit.robot_fraction, np.eye(dimensions)[i]
            )[0]
            - constant
            for i in range(dimensions)
        ],
        axis=-1,
    )
    frame, alpha = before["interval"], before["fraction"]
    following = np.minimum(frame + 1, len(times) - 1)
    dense = matrix[frame] * (1 - alpha[:, None, None]) + matrix[following] * alpha[:, None, None]
    dense_constant = constant[frame] * (1 - alpha[:, None]) + constant[following] * alpha[:, None]
    objective = np.concatenate(
        [
            matrix.reshape(-1, dimensions) / 0.08,
            (np.diff(matrix, axis=0) / 0.02).reshape(-1, dimensions) / 3.0,
            (np.diff(matrix, n=2, axis=0) / 0.02**2).reshape(-1, dimensions) / 30.0,
        ]
    )
    offset = np.concatenate(
        [
            constant.reshape(-1) / 0.08,
            (np.diff(constant, axis=0) / 0.02).reshape(-1) / 3.0,
            (np.diff(reference.robot_state[:, :3] + constant, n=2, axis=0) / 0.02**2).reshape(-1)
            / 30.0,
        ]
    )
    hessian, linear = objective.T @ objective, -objective.T @ offset
    optimum = np.linalg.solve(hessian, linear)
    transform = solve_triangular(np.linalg.cholesky(hessian).T, np.eye(dimensions), lower=False)
    rows, lower, upper = [], [], []

    def append(directions, lo, hi):
        rows.append(linear_rows(dense, directions))
        displacement = np.sum(dense_constant * directions, axis=1)
        lower.extend(np.broadcast_to(lo, len(dense)) - displacement)
        upper.extend(np.broadcast_to(hi, len(dense)) - displacement)

    rotation, feet = before["board_rotation"], before["foot_local"]
    # Preserve the source upper halfspace (and raised kicktail envelope where
    # already clear), without imposing foot contact or an air-foot height target.
    gap_minimum = np.minimum(before["sole_gap"], audit.upper - 0.006) - 0.001
    for side in range(2):
        append(rotation[:, :, 2], gap_minimum[:, side] - before["sole_gap"][:, side], np.inf)
        for axis, limit in ((0, 0.39), (1, 0.16)):
            bound = np.maximum(limit, np.abs(feet[:, side, axis]))
            append(rotation[:, :, axis], -bound - feet[:, side, axis], bound - feet[:, side, axis])
    center = feet[:, :, 1].mean(1)
    bound = np.maximum(0.08, np.abs(center))
    append(rotation[:, :, 1], -bound - center, bound - center)
    append(np.tile([0.0, 0.0, 1.0], (len(dense), 1)), -0.002 - before["robot_floor"], np.inf)
    for axis in range(3):
        append(np.tile(np.eye(3)[axis], (len(dense), 1)), -0.5, 0.5)
    reports, parameters = [], optimum.copy()
    result = copy.deepcopy(reference)
    for outer in range(8):
        constraints = np.concatenate(rows)
        lo, hi = np.asarray(lower), np.asarray(upper)
        whitened = constraints @ transform
        origin = constraints @ optimum
        scale = np.maximum(np.linalg.norm(whitened, axis=1), 1e-12)
        solution = minimize(
            lambda z: float(0.5 * z @ z),
            np.zeros(dimensions),
            jac=lambda z: z,
            method="SLSQP",
            constraints=[
                LinearConstraint(
                    whitened / scale[:, None], (lo - origin) / scale, (hi - origin) / scale
                )
            ],
            options={"maxiter": iterations, "ftol": 1e-11},
        )
        parameters = optimum + transform @ solution.x
        shift = constant + np.einsum("nij,j->ni", matrix, parameters)
        result.robot_state[:, :3] = reference.robot_state[:, :3] + shift
        _after, geometry, cuts = audit.sample(result)
        violation = float(
            np.maximum(
                np.maximum(lo - constraints @ parameters, constraints @ parameters - hi), 0
            ).max()
        )
        reports.append(
            {
                "iteration": outer,
                "solver_success": bool(solution.success),
                "message": str(solution.message),
                "solver_iterations": int(solution.nit),
                "linear_constraint_violation_m": violation,
                "geometry": geometry,
            }
        )
        if not solution.success or violation > 1e-6 or not cuts:
            break
        for sample, direction, distance in cuts:
            row = direction @ dense[sample]
            rows.append(row[None])
            lower.append(-0.002 - distance + row @ parameters)
            upper.append(np.inf)
    shift, _derivative, _blend_weight = correction(
        times, com, intervals, audit.gravity, audit.robot_fraction, parameters
    )
    result.robot_state[:, :3] = reference.robot_state[:, :3] + shift
    # Keep root linear velocity the finite difference of its own position.
    # Do not hide fixed qd/omega discretization errors by adding a fictitious
    # root velocity solely to force a Jacobian-COM velocity identity.
    result.robot_state[:, 36:39] += np.gradient(shift, 0.02, axis=0)
    return result, parameters, reports, gap_minimum


def component_audit(samples, audit, dt=0.02, acceleration_tolerance=0.2):
    """No connection means no force path; candidate edges do not establish forces."""
    result = {}
    masks = {
        "system": ~samples["robot_ground"] & ~samples["board_ground"],
        "robot": ~samples["robot_ground"] & ~samples["robot_board"],
        "board": ~samples["board_ground"] & ~samples["robot_board"],
    }
    for entity, mask in masks.items():
        com = samples["com" if entity == "system" else entity + "_com"][::8]
        acceleration = np.diff(com, n=2, axis=0) / dt**2
        selected = np.array(
            [bool(mask[(i - 1) * 8 : (i + 1) * 8 + 1].all()) for i in range(1, len(com) - 1)]
        )
        error = acceleration[selected] - audit.gravity
        mass = float(
            audit.mass.sum() if entity == "system" else audit.mass[getattr(audit, entity)].sum()
        )
        result[entity] = {
            "potentially_disconnected_free_windows": windows(mask, samples["time"]),
            "whole_stencil_free_nodes": int(selected.sum()),
            "acceleration_tolerance_m_s2": acceleration_tolerance,
            "acceleration_error_rmse_xyz_m_s2": np.sqrt(np.mean(error**2, axis=0)).tolist()
            if len(error)
            else None,
            "acceleration_error_max_abs_xyz_m_s2": np.abs(error).max(0).tolist()
            if len(error)
            else None,
            "required_unexplained_external_force_peak_n": float(
                np.linalg.norm(error, axis=1).max() * mass
            )
            if len(error)
            else None,
            "passes_sampled_acceleration_necessary_condition": bool(
                not len(error) or np.abs(error).max() <= acceleration_tolerance
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--preserve-through", type=float, default=2.90)
    parser.add_argument("--contact-band-m", type=float, default=0.002)
    parser.add_argument("--blend-seconds", type=float, default=0.20)
    parser.add_argument("--solver-iterations", type=int, default=300)
    args = parser.parse_args()
    if args.out.exists() or args.log_dir.exists() or args.source.resolve() == args.out.resolve():
        parser.error("Only fresh independent output/log paths are allowed")
    if (
        not np.isfinite([args.preserve_through, args.contact_band_m, args.blend_seconds]).all()
        or args.contact_band_m <= 0
        or args.contact_band_m > 0.002
        or args.blend_seconds < 0.04
        or args.preserve_through < 0
        or args.solver_iterations < 1
    ):
        parser.error("Need finite protected time, a (0,2mm] uncertainty band, and >=40ms blending")
    source = JointReference.load(args.source)
    if not np.isclose(source.metadata["physics_fps"], 50):
        parser.error("This finite-difference/buffer contract requires a 50Hz source")
    args.log_dir.mkdir(parents=True)
    audit = Audit(source, args.contact_band_m)
    before, geometry_before, _ = audit.sample(source)
    mask = air_mask(before, args.contact_band_m)
    strict = (
        (before["robot_floor"] > 1e-6)
        & (before["board_floor"] > 1e-6)
        & ~before["strict_robot_ground"]
        & ~before["strict_board_ground"]
    )
    times, dt = source.time_physics, 0.02
    intervals, unrepaired = [], []
    for group in segments(mask):
        lo = max(0, int(np.floor((before["time"][group[0]] + 1e-10) / dt)) - 1)
        hi = min(len(times) - 1, int(np.ceil((before["time"][group[-1]] - 1e-10) / dt)) + 1)
        left = max(round((args.preserve_through + dt) / dt), lo - round(args.blend_seconds / dt))
        right = min(len(times) - 1, hi + round(args.blend_seconds / dt))
        if left >= lo or right <= hi:
            unrepaired.append(
                {
                    "air_window_s": [
                        float(before["time"][group[0]]),
                        float(before["time"][group[-1]]),
                    ],
                    "reason": "Insufficient supported space outside protected prefix for Hermite+buffer",
                }
            )
            continue
        if intervals and left <= intervals[-1]["frames"][-1]:
            raise ValueError(
                "Air windows have overlapping Hermite bridges; explicit joint fitting is required"
            )
        intervals.append(
            {
                "frames": [left, lo, hi, right],
                "origin_s": float((times[lo] + times[hi]) / 2),
                "air_window_s": [float(before["time"][group[0]]), float(before["time"][group[-1]])],
                "ballistic_buffer_window_s": [float(times[lo]), float(times[hi])],
                "bridge_outer_window_s": [float(times[left]), float(times[right])],
            }
        )
    if not intervals:
        raise ValueError("No geometrically separated repairable air window")
    result, parameters, solver, gap_minimum = fit(
        source, audit, before, intervals, args.solver_iterations
    )
    after, geometry_after, _ = audit.sample(result)
    protected = times <= args.preserve_through + 1e-8
    immutable = np.ones(71, dtype=bool)
    immutable[:3], immutable[36:39] = False, False
    invariants = {
        "protected_complete_state_exact": np.array_equal(
            source.state[protected], result.state[protected]
        ),
        "board_all_state_exact": np.array_equal(source.board_state, result.board_state),
        "all_robot_fields_except_root_xyz_and_linear_velocity_exact": np.array_equal(
            source.robot_state[:, immutable], result.robot_state[:, immutable]
        ),
        "times_labels_confidence_exact": all(
            np.array_equal(getattr(source, name), getattr(result, name))
            for name in ("time_video", "time_physics", "contacts", "phase", "confidence")
        ),
        "separated_global_air_mask_unchanged": np.array_equal(
            mask, air_mask(after, args.contact_band_m)
        ),
    }
    air_nodes = mask[::8]
    air_ids = np.flatnonzero(
        air_nodes & (np.arange(len(times)) > 0) & (np.arange(len(times)) < len(times) - 1)
    )
    com_before, com_after = before["com"][::8], after["com"][::8]

    def finite_difference_report(com, velocity):
        fd = np.gradient(com, dt, axis=0)
        acceleration = np.diff(com, n=2, axis=0) / dt**2
        return {
            "air_nodes": len(air_ids),
            "air_second_difference_acceleration_error_max_xyz_m_s2": np.abs(
                acceleration[air_ids - 1] - audit.gravity
            )
            .max(0)
            .tolist(),
            "air_saved_com_jac_velocity_vs_fd_max_xyz_m_s": np.abs(velocity[air_ids] - fd[air_ids])
            .max(0)
            .tolist(),
            "boundary_samples": [
                {
                    "time_s": float(times[i]),
                    "com_velocity_fd_m_s": fd[i].tolist(),
                    "com_velocity_jac_m_s": velocity[i].tolist(),
                }
                for i in air_ids
                if abs(times[i] - 3.14) < 1e-8
                or abs(times[i] - 3.16) < 1e-8
                or abs(times[i] - 3.18) < 1e-8
            ],
        }

    before_fd = finite_difference_report(com_before, before["com_velocity_jac"][::8])
    after_fd = finite_difference_report(com_after, after["com_velocity_jac"][::8])
    halfspace_violation = float(np.maximum(gap_minimum - after["sole_gap"], 0).max())
    full_geometry_pass = (
        max(geometry_after["maximum_penetration_by_category_m"].values()) <= 0.002 + 1e-8
    )
    global_pass = bool(
        all(invariants.values())
        and solver[-1]["solver_success"]
        and solver[-1]["linear_constraint_violation_m"] <= 1e-6
        and full_geometry_pass
        and halfspace_violation <= 1e-6
        and not unrepaired
        and max(after_fd["air_second_difference_acceleration_error_max_xyz_m_s2"]) < 1e-6
    )
    components_before, components_after = (
        component_audit(before, audit),
        component_audit(after, audit),
    )
    all_components = all(
        row["passes_sampled_acceleration_necessary_condition"] for row in components_after.values()
    )
    dense_residuals = []
    for interval, values in zip(intervals, parameters.reshape(-1, 4, 3), strict=True):
        tau = after["time"] - interval["origin_s"]
        expected = values[0] + tau[:, None] * values[1] + 0.5 * tau[:, None] ** 2 * audit.gravity
        chosen = (after["time"] >= interval["ballistic_buffer_window_s"][0]) & (
            after["time"] <= interval["ballistic_buffer_window_s"][1]
        )
        residual = after["com"][chosen] - expected[chosen]
        dense_residuals.append(
            {
                "ballistic_buffer_window_s": interval["ballistic_buffer_window_s"],
                "linear_interpolant_position_residual_max_xyz_m": np.abs(residual).max(0).tolist(),
                "linear_interpolant_position_residual_rmse_xyz_m": np.sqrt(
                    np.mean(residual**2, axis=0)
                ).tolist(),
            }
        )
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": sha(args.source),
        "method": "Robot root XYZ/linear velocity only; whole-system ballistic node positions with one-frame extension and support-side C1 Hermite bridges",
        "not_a_physical_rollout": True,
        "force_torque_or_policy_feasibility_certified": False,
        "global_com_correction_passed": global_pass,
        "global_com_pass_scope": "50Hz COM positions and their centered finite differences only; saved root linear velocity follows root-position FD. Jacobian-COM velocity retains input derivative discretization residual; the standard dense linear interpolant is not exactly ballistic",
        "all_sampled_component_necessary_conditions_passed": bool(global_pass and all_components),
        "candidate_status": "GLOBAL_NODE_CONDITION_ONLY_COMPONENT_DYNAMICS_FAILED"
        if global_pass and not all_components
        else "NODE_NECESSARY_CONDITIONS_ONLY"
        if global_pass
        else "FAILED_GLOBAL_CORRECTION",
        "recommended_as_physical_reference": False,
        "robot_mass_kg": float(audit.mass[audit.robot].sum()),
        "board_mass_kg": float(audit.mass[audit.board].sum()),
        "total_mass_kg": float(audit.mass.sum()),
        "gravity_m_s2": audit.gravity.tolist(),
        "com_method": "All body masses weighted by world inertial xipos; no body omitted",
        "contact_uncertainty_band_m": args.contact_band_m,
        "strict_geometric_microgap_windows_1um": windows(strict, before["time"]),
        "separated_global_air_windows": windows(mask, before["time"]),
        "near_ground_uncertainty_samples": int(np.count_nonzero(strict & ~mask)),
        "strict_microgaps_in_protected_prefix": int(
            np.count_nonzero(strict & (before["time"] <= args.preserve_through))
        ),
        "fitted_intervals": intervals,
        "unrepaired_separated_windows": unrepaired,
        "parameters_per_interval": parameters.reshape(-1, 4, 3).tolist(),
        "parameter_meaning": [
            "COM at interval origin",
            "COM velocity at origin",
            "left supported bridge bump XYZ",
            "right supported bridge bump XYZ",
        ],
        "solver": solver,
        "invariants": {k: bool(v) for k, v in invariants.items()},
        "before_geometry": geometry_before,
        "after_geometry": geometry_after,
        "halfspace_violation_m": halfspace_violation,
        "foot_constraint": "Sole support along board normal stays above min(source gap, raised-tail upper envelope minus center top) minus 1mm; source lateral/longitudinal envelopes retained; no foot weld/contact labels added",
        "root_translation_max_change_m": float(
            np.linalg.norm(result.robot_state[:, :3] - source.robot_state[:, :3], axis=1).max()
        ),
        "root_linear_velocity_max_change_m_s": float(
            np.linalg.norm(
                result.robot_state[:, 36:39] - source.robot_state[:, 36:39], axis=1
            ).max()
        ),
        "before_global_finite_difference": before_fd,
        "root_linear_velocity_vs_root_position_fd_max_xyz_m_s": np.abs(
            result.robot_state[:, 36:39] - np.gradient(result.robot_state[:, :3], 0.02, axis=0)
        )
        .max(0)
        .tolist(),
        "after_global_finite_difference": after_fd,
        "before_contact_components": components_before,
        "after_contact_components": components_after,
        "dense_com_discretization_residuals": dense_residuals,
        "discretization": "Exact ballistic COM at 50Hz nodes/finite differences; root linear velocity is its position finite difference. Fixed joint qd/root omega need not produce the same instantaneous Jacobian COM velocity. Dense 2.5ms collision QA uses standard linear root/joint and geodesic quaternion interpolation, not an exact continuous ballistic interpolant",
        "limitations": [
            "A <=2mm gap is geometrically uncertain, not measured supporting force; strict microgaps are reported and not repaired in protected states",
            "Disconnected free robot/board components must each be ballistic; fixed board inconsistency cannot be repaired by shifting only the robot",
            "Whole-system COM consistency does not establish contact forces, friction, momentum exchange, torque feasibility or policy success",
        ],
    }
    result.metadata["global_ballistic_candidate"] = report
    result.metadata["provenance"]["global_ballistic_source"] = str(args.source.resolve())
    result.metadata["provenance"]["global_ballistic_source_sha256"] = sha(args.source)
    result.save(args.out)
    report.update(output=str(args.out.resolve()), output_sha256=sha(args.out))
    (args.log_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.log_dir / "before_samples.npz", **before)
    np.savez_compressed(args.log_dir / "after_samples.npz", **after)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "output",
                    "output_sha256",
                    "global_com_correction_passed",
                    "all_sampled_component_necessary_conditions_passed",
                    "recommended_as_physical_reference",
                    "root_translation_max_change_m",
                    "halfspace_violation_m",
                    "after_geometry",
                    "after_global_finite_difference",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
