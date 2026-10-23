"""Repair relative air geometry in a new kinematic branch, never a rollout.

Robot root translation changes; optional board translation can also change.
The protected prefix, robot rotations/joints, board rotations/wheels and contact
labels remain exact. Loose envelopes and unilateral barriers do not attach a
foot to the board. Image and gravity constraints remain explicit assumptions.
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
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.video.audit_ollie_timing import geometry_minimum_z
from scripts.video.repair_ollie_self_collisions import category
from supermimic.ollie.contact_ik import FootGeometry
from supermimic.ollie.physics import compile_scene
from supermimic.ollie.reference import JointReference


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value**2 * (3 - 2 * value)


class Geometry:
    def __init__(self, reference):
        self.reference = reference
        self.model, self.data, self.index = compile_scene(timestep=0.0025)
        self.feet = FootGeometry(self.model)
        self.robot_root = int(
            self.model.body_rootid[
                self.model.jnt_bodyid[self.model.joint("floating_base_joint").id]
            ]
        )
        self.board_root = int(self.model.body_rootid[self.index.board_body])
        self.robot_bodies = self.model.body_rootid == self.robot_root
        self.robot_mass = float(self.model.body_mass[self.robot_bodies].sum())
        self.total_mass = float(self.model.body_mass.sum())
        ground = self.model.geom("ground").id
        active = [
            gid
            for gid in range(self.model.ngeom)
            if self.model.geom_bodyid[gid] != 0
            and (
                self.model.geom_contype[gid] & self.model.geom_conaffinity[ground]
                or self.model.geom_contype[ground] & self.model.geom_conaffinity[gid]
            )
        ]
        self.robot_geoms = [gid for gid in active if self.robot_bodies[self.model.geom_bodyid[gid]]]
        self.board_geoms = [
            gid
            for gid in active
            if self.model.body_rootid[self.model.geom_bodyid[gid]] == self.board_root
        ]

    def poses(self, robot_state=None, board_state=None):
        robot_state = self.reference.robot_state if robot_state is None else robot_state
        board_state = self.reference.board_state if board_state is None else board_state
        poses = np.repeat(self.model.qpos0[None], len(robot_state), axis=0)
        ix = self.index
        poses[:, ix.robot_qpos : ix.robot_qpos + 7] = robot_state[:, :7]
        poses[:, ix.robot_joint_qpos] = robot_state[:, 7:36]
        poses[:, ix.board_qpos : ix.board_qpos + 7] = board_state[:, :7]
        poses[:, ix.wheel_qpos] = board_state[:, 7:11]
        return poses

    def sample(self, poses, substeps=8, collect_contacts=True):
        model, data, ix = self.model, self.data, self.index
        times, intervals, fractions = [], [], []
        feet_local, gaps, board_rotation, robot_floor, board_floor, com = [], [], [], [], [], []
        maximums = dict.fromkeys(("self", "robot_board", "ground", "board_internal", "other"), 0.0)
        counts = dict.fromkeys(maximums, 0)
        worst = []
        velocity = np.zeros(model.nv)
        for i, start in enumerate(poses):
            if i + 1 < len(poses):
                mujoco.mj_differentiatePos(model, velocity, 1.0, start, poses[i + 1])
            for sub in range(substeps) if i + 1 < len(poses) else [0]:
                alpha = sub / substeps
                data.qpos[:] = start
                if sub:
                    mujoco.mj_integratePos(model, data.qpos, velocity, alpha)
                mujoco.mj_forward(model, data)
                rotation = data.xmat[ix.board_body].reshape(3, 3)
                position = data.qpos[ix.board_qpos : ix.board_qpos + 3]
                feet_local.append((data.site_xpos[self.feet.sites] - position) @ rotation)
                gaps.append(
                    [
                        float(
                            np.dot(
                                self.feet.support(model, data, side, rotation[:, 2]) - position,
                                rotation[:, 2],
                            )
                            - 0.006
                        )
                        for side in range(2)
                    ]
                )
                board_rotation.append(rotation.copy())
                robot_floor.append(geometry_minimum_z(model, data, self.robot_geoms))
                board_floor.append(geometry_minimum_z(model, data, self.board_geoms))
                com.append((model.body_mass[:, None] * data.xipos).sum(0) / self.total_mass)
                times.append(
                    float(
                        self.reference.time_physics[i]
                        + alpha / self.reference.metadata["physics_fps"]
                    )
                )
                intervals.append(i)
                fractions.append(alpha)
                if collect_contacts:
                    for contact in data.contact[: data.ncon]:
                        first, second = map(int, contact.geom)
                        depth = max(0.0, -float(contact.dist))
                        kind = category(model, ix, first, second)
                        maximums[kind] = max(maximums[kind], depth)
                        counts[kind] += int(depth > 0.002)
                        if depth > 0.002:
                            worst.append(
                                {
                                    "time_physics": times[-1],
                                    "category": kind,
                                    "depth_m": depth,
                                    "geoms": [model.geom(first).name, model.geom(second).name],
                                }
                            )
        arrays = {
            key: np.asarray(value)
            for key, value in {
                "time_physics": times,
                "interval": intervals,
                "fraction": fractions,
                "foot_site_board_local_m": feet_local,
                "sole_gap_m": gaps,
                "board_rotation": board_rotation,
                "robot_floor_m": robot_floor,
                "board_floor_m": board_floor,
                "system_com_world_m": com,
            }.items()
        }
        report = {
            "samples": len(times),
            "physics_sample_dt_s": 1 / self.reference.metadata["physics_fps"] / substeps,
            "maximum_penetration_by_category_m": maximums,
            "contacts_over_2mm_by_category": counts,
            "worst_contacts": sorted(worst, key=lambda row: -row["depth_m"])[:20],
            "meaning": "Prescribed full collision geometry at 2.5ms; no force/dynamics certification",
        }
        return arrays, report


def node_operator(samples, node_indices, frames):
    """Linear interpolation of 50Hz translation deltas, including fixed zero nodes."""
    lookup = {int(frame): j for j, frame in enumerate(node_indices)}
    operator = np.zeros((len(samples["interval"]), len(node_indices)))
    for row, (frame, alpha) in enumerate(
        zip(samples["interval"], samples["fraction"], strict=True)
    ):
        if int(frame) in lookup:
            operator[row, lookup[int(frame)]] += 1 - alpha
        if int(frame) + 1 in lookup:
            operator[row, lookup[int(frame) + 1]] += alpha
    expanded = np.zeros((len(samples["interval"]) * 3, len(node_indices) * 3))
    for axis in range(3):
        expanded[axis::3, axis::3] = operator
    nodes = np.zeros((frames, len(node_indices)))
    nodes[node_indices, np.arange(len(node_indices))] = 1
    return operator, expanded, nodes


def linear_vector_rows(operator, direction):
    return np.einsum("ij,ik->ijk", operator, direction).reshape(len(operator), -1)


def ballistic_fit(time, com, window):
    marked = (time >= window[0] - 1e-8) & (time <= window[1] + 1e-8)
    t = time[marked]
    centred = t - t.mean()
    design = np.column_stack([np.ones(len(t)), centred])
    expected = com[marked].copy()
    expected[:, 2] += 0.5 * 9.81 * centred**2
    line = design @ np.linalg.lstsq(design, expected, rcond=None)[0]
    residual = expected - line
    acceleration = [float(2 * np.polyfit(centred, com[marked, axis], 2)[0]) for axis in range(3)]
    return {
        "physical_window_s": list(window),
        "samples": len(t),
        "quadratic_acceleration_xyz_m_s2": acceleration,
        "free_flight_xyz_rmse_m": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "meaning": "Conditional kinematic COM audit; assumed unforced flight window and g=9.81, not measured forces",
    }


def board_image_terms(reference, node_indices, args):
    """Nine tracked graphic quads; dimensions/camera remain uncertain inputs."""
    body = np.load(args.body, allow_pickle=False)
    common = json.loads(args.common_world.read_text())
    camera_to_world = np.asarray(common["rotation_camera_to_world"])
    translation = np.asarray(common["translation_camera_to_world"])
    intrinsics = body["K_fullimg"][0]
    observations = json.loads(args.board_observations.read_text())["frames"]
    rows = [row for row in observations if 114 <= row["frame"] <= 122 and row["observed"]]
    video_time = np.array([row["t_video"] for row in rows])
    time = video_time / reference.metadata["time_scale"]
    positions = np.stack(
        [
            np.interp(time, reference.time_physics, reference.board_state[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    rotations = Slerp(
        reference.time_physics, Rotation.from_quat(reference.board_state[:, [4, 5, 6, 3]])
    )(time).as_matrix()
    graphic = np.array(
        [[-0.105, 0.09, 0.006], [0.105, 0.09, 0.006], [0.105, -0.09, 0.006], [-0.105, -0.09, 0.006]]
    )
    world = np.einsum("fij,kj->fki", rotations, graphic) + positions[:, None]
    camera = (world - translation) @ camera_to_world
    image = camera @ intrinsics.T
    image = image[..., :2] / image[..., 2:]
    target = np.array([row["quad"] for row in rows])
    basis = np.stack(
        [
            np.interp(
                time,
                reference.time_physics,
                (np.arange(len(reference.time_physics)) == node).astype(float),
            )
            for node in node_indices
        ],
        axis=1,
    )
    jacobians = np.zeros((len(rows), 4, 2, 3))
    for frame in range(len(rows)):
        for corner in range(4):
            x, y, z = camera[frame, corner]
            jacobians[frame, corner] = (
                np.array(
                    [
                        [intrinsics[0, 0] / z, 0, -intrinsics[0, 0] * x / z**2],
                        [0, intrinsics[1, 1] / z, -intrinsics[1, 1] * y / z**2],
                    ]
                )
                @ camera_to_world.T
            )
    matrix = np.einsum("fckd,fn->fcknd", jacobians, basis).reshape(-1, len(node_indices) * 3)
    return (
        matrix,
        (target - image).reshape(-1),
        {
            "source_frames": [row["frame"] for row in rows],
            "time_physics": time,
            "world_graphic_before": world,
            "old_image": image,
            "observed_image": target,
            "camera_to_world": camera_to_world,
            "translation": translation,
            "K": intrinsics,
            "basis": basis,
        },
    )


def image_report(terms, board_nodes):
    shift = terms["basis"] @ board_nodes
    world = terms["world_graphic_before"] + shift[:, None]
    camera = (world - terms["translation"]) @ terms["camera_to_world"]
    image = camera @ terms["K"].T
    image = image[..., :2] / image[..., 2:]
    old_error = np.linalg.norm(terms["old_image"] - terms["observed_image"], axis=-1)
    new_error = np.linalg.norm(image - terms["observed_image"], axis=-1)
    increment = np.linalg.norm(image - terms["old_image"], axis=-1)
    return {
        "source_frames": terms["source_frames"],
        "graphic_length_width_assumed_m": [0.21, 0.18],
        "old_corner_rmse_px": float(np.sqrt(np.mean(old_error**2))),
        "new_corner_rmse_px": float(np.sqrt(np.mean(new_error**2))),
        "increment_rmse_px": float(np.sqrt(np.mean(increment**2))),
        "increment_max_px": float(increment.max()),
        "old_per_frame_corner_rmse_px": np.sqrt(np.mean(old_error**2, axis=1)).tolist(),
        "new_per_frame_corner_rmse_px": np.sqrt(np.mean(new_error**2, axis=1)).tolist(),
        "new_graphic_pixels": image.tolist(),
        "observed_graphic_pixels": terms["observed_image"].tolist(),
        "meaning": "Nine original 2D tracked quads only; assumed graphic size, fixed K/extrinsics and fixed board rotations; not measured 3D positions",
    }


def fit_root(reference, geometry, samples, args):
    times = reference.time_physics
    dt = float(times[1] - times[0])
    # One unchanged pose beyond each protected side preserves the central
    # finite-difference velocity at the protected endpoint, rather than merely
    # copying an inconsistent velocity after altering its neighbour.
    selected = np.flatnonzero(
        (times > args.preserve_through + dt + 1e-8) & (times < args.fixed_suffix_from - dt - 1e-8)
    )
    if len(selected) < 8:
        raise ValueError("Repair window must leave at least eight free root-translation nodes")
    operator, _, nodes = node_operator(samples, selected, len(times))
    active = np.any(operator != 0, axis=1)
    t = samples["time_physics"][active]
    e = operator[active]
    rotation = samples["board_rotation"][active]
    feet = samples["foot_site_board_local_m"][active]
    gap = samples["sole_gap_m"][active]
    fade = smoothstep((t - args.preserve_through - dt) / 0.28) * smoothstep(
        (args.fixed_suffix_from - dt - t) / 0.28
    )
    # The actual raised nose/tail sit above the centre plane. A conservative
    # upper envelope prevents a centre-plane pass from hiding tip collisions.
    model = geometry.model
    upper = max(
        float(
            model.geom_pos[model.geom(name).id, 2]
            + np.abs(
                Rotation.from_quat(model.geom_quat[model.geom(name).id, [1, 2, 3, 0]]).as_matrix()[
                    2
                ]
            )
            @ model.geom_size[model.geom(name).id]
        )
        for name in ("board_center", "board_tail", "board_nose")
    )
    guard_fade = smoothstep((t - args.preserve_through - dt) / 0.06) * smoothstep(
        (args.fixed_suffix_from - dt - t) / 0.06
    )
    gap_minimum = -0.001 + 0.009 * fade + (upper - 0.006) * guard_fade
    constraints, lower, upper = [], [], []
    dimensions = len(selected) * 3

    def entity_matrix(matrix, entity="relative"):
        if not args.move_board:
            return matrix if entity != "board" else np.zeros_like(matrix)
        zero = np.zeros_like(matrix)
        return np.concatenate(
            [
                matrix if entity != "board" else zero,
                -matrix if entity == "relative" else matrix if entity == "board" else zero,
            ],
            axis=1,
        )

    def append(direction, lo, hi, entity="relative"):
        constraints.append(entity_matrix(linear_vector_rows(e, direction), entity))
        lower.extend(np.broadcast_to(lo, len(e)).tolist())
        upper.extend(np.broadcast_to(hi, len(e)).tolist())

    for side in range(2):
        append(rotation[:, :, 2], gap_minimum - gap[:, side], np.inf)
        append(
            rotation[:, :, 1],
            -args.foot_lateral_limit - feet[:, side, 1],
            args.foot_lateral_limit - feet[:, side, 1],
        )
        append(rotation[:, :, 0], -0.39 - feet[:, side, 0], 0.39 - feet[:, side, 0])
    append(
        rotation[:, :, 1],
        -args.centre_lateral_limit - feet[:, :, 1].mean(1),
        args.centre_lateral_limit - feet[:, :, 1].mean(1),
    )
    append(
        np.broadcast_to([0.0, 0.0, 1.0], (len(e), 3)),
        -0.001 - samples["robot_floor_m"][active],
        np.inf,
        "robot",
    )
    if args.move_board:
        append(
            np.broadcast_to([0.0, 0.0, 1.0], (len(e), 3)),
            -0.000001 - samples["board_floor_m"][active],
            np.inf,
            "board",
        )
    constraint_matrix = np.concatenate(constraints)
    lower, upper = np.asarray(lower), np.asarray(upper)

    # Loose preferred centre and a unilateral vertical correction; no foot
    # anchor, no height equality, and no measured/contact labels are created.
    local_target = np.zeros((len(e), 3))
    local_target[:, 0] = np.clip(feet[:, :, 0].mean(1), -0.08, 0.08) - feet[:, :, 0].mean(1)
    local_target[:, 1] = np.clip(feet[:, :, 1].mean(1), -0.02, 0.02) - feet[:, :, 1].mean(1)
    local_target[:, 2] = np.maximum(gap_minimum - gap.min(1), 0)
    target = np.einsum("fij,fj->fi", rotation, local_target)
    matrices, rhs = [], []

    def objective_term(matrix, wanted, sigma):
        matrices.append(matrix / sigma)
        rhs.append(np.asarray(wanted).reshape(-1) / sigma)

    def expanded(matrix):
        output = np.zeros((len(matrix) * 3, len(selected) * 3))
        for axis in range(3):
            output[axis::3, axis::3] = matrix
        return output

    objective_term(entity_matrix(expanded(e)), target, 0.10)
    objective_term(entity_matrix(np.eye(dimensions), "robot"), np.zeros(dimensions), 0.50)
    if args.move_board:
        objective_term(
            entity_matrix(np.eye(dimensions), "board"), np.zeros(dimensions), args.board_prior_sigma
        )
    difference = np.diff(np.eye(len(times)), axis=0) / dt
    curvature = np.diff(np.eye(len(times)), n=2, axis=0) / dt**2
    objective_term(
        entity_matrix(expanded(difference @ nodes), "robot"),
        -(difference @ reference.robot_state[:, :3]),
        3.0,
    )
    objective_term(
        entity_matrix(expanded(curvature @ nodes), "robot"),
        -(curvature @ reference.robot_state[:, :3]),
        args.acceleration_sigma,
    )
    if args.move_board:
        objective_term(
            entity_matrix(expanded(difference @ nodes), "board"),
            -(difference @ reference.board_state[:, :3]),
            3.0,
        )
        objective_term(
            entity_matrix(expanded(curvature @ nodes), "board"),
            -(curvature @ reference.board_state[:, :3]),
            args.acceleration_sigma,
        )
    com_marked = (samples["time_physics"] >= args.com_window[0] - 1e-8) & (
        samples["time_physics"] <= args.com_window[1] + 1e-8
    )
    tau = samples["time_physics"][com_marked]
    tau = tau - tau.mean()
    design = np.column_stack([np.ones(len(tau)), tau])
    projection = np.eye(len(tau)) - design @ np.linalg.pinv(design)
    corrected_com = samples["system_com_world_m"][com_marked].copy()
    corrected_com[:, 2] += 0.5 * 9.81 * tau**2
    com_matrix = expanded(projection @ operator[com_marked])
    robot_fraction = geometry.robot_mass / geometry.total_mass
    com_matrix = (
        np.concatenate([com_matrix * robot_fraction, com_matrix * (1 - robot_fraction)], axis=1)
        if args.move_board
        else com_matrix * robot_fraction
    )
    objective_term(
        com_matrix,
        -(projection @ corrected_com),
        args.com_sigma,
    )
    image_matrix, image_wanted, image_terms = board_image_terms(reference, selected, args)
    if args.move_board:
        objective_term(entity_matrix(image_matrix, "board"), image_wanted, args.reprojection_sigma)
    matrix, wanted = np.concatenate(matrices), np.concatenate(rhs)
    hessian, linear = matrix.T @ matrix, matrix.T @ wanted
    unconstrained = np.linalg.solve(hessian, linear)
    # Whiten the strictly convex quadratic: SLSQP's identity initial Hessian
    # is then exact, avoiding thousands of BFGS iterations across disparate
    # pixel/acceleration/metre units. This changes no objective or constraint.
    transform = solve_triangular(np.linalg.cholesky(hessian).T, np.eye(len(linear)), lower=False)
    whitened_matrix = constraint_matrix @ transform
    origin = constraint_matrix @ unconstrained
    row_scale = np.maximum(np.linalg.norm(whitened_matrix, axis=1), 1e-12)
    solution = minimize(
        lambda z: float(0.5 * z @ z),
        np.zeros(len(linear)),
        jac=lambda z: z,
        method="SLSQP",
        constraints=[
            LinearConstraint(
                whitened_matrix / row_scale[:, None],
                (lower - origin) / row_scale,
                (upper - origin) / row_scale,
            )
        ],
        options={"maxiter": args.solver_iterations, "ftol": 1e-12, "disp": False},
    )
    solution.x = unconstrained + transform @ solution.x
    violation = float(
        np.maximum(
            np.maximum(
                lower - constraint_matrix @ solution.x, constraint_matrix @ solution.x - upper
            ),
            0,
        ).max()
    )
    if not solution.success or violation > 1e-6:
        raise RuntimeError(
            f"Root fit failed: {solution.message}; inequality violation {violation}m"
        )
    shift = nodes @ solution.x[:dimensions].reshape(-1, 3)
    board_nodes = (
        solution.x[dimensions:].reshape(-1, 3) if args.move_board else np.zeros((len(selected), 3))
    )
    board_shift = nodes @ board_nodes
    return (
        shift,
        board_shift,
        {
            "success": bool(solution.success),
            "iterations": int(solution.nit),
            "message": str(solution.message),
            "maximum_constraint_violation_m": violation,
            "free_nodes": selected.tolist(),
            "root_translation_only": not args.move_board,
            "board_translation_also_optimized": args.move_board,
            "nine_observed_quads_reprojection": image_report(image_terms, board_nodes),
            "foot_barrier": "lowest of all seven sole capsules on upper board halfspace; unilateral inequality, not contact",
            "loose_envelope_limits_m": {
                "each_foot_lateral": args.foot_lateral_limit,
                "mean_foot_lateral": args.centre_lateral_limit,
                "each_foot_longitudinal": 0.39,
            },
            "assumed_free_flight_com_window_physical_s": args.com_window,
            "objective_sigmas": {
                "root_target_m": 0.1,
                "root_shift_m": 0.5,
                "root_speed_m_s": 3.0,
                "root_acceleration_m_s2": args.acceleration_sigma,
                "com_ballistic_residual_m": args.com_sigma,
            },
        },
    )


def migrate_teacher(source, output, old_reference, new_reference, new_reference_path, old_sha):
    """Make an explicit new copy; all applied actions and real physics stay exact."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(arrays["metadata_json"].item())
    if metadata.get("reference_sha256") != old_sha:
        raise ValueError("Teacher reference hash differs from the repair input")
    if "reference_sha256" in arrays and arrays["reference_sha256"].item() != old_sha:
        raise ValueError("Teacher top-level reference hash differs from its metadata/input")
    indices = arrays["reference_goal_indices"]
    prefix_frames = int(indices[:, 0].max()) + 1
    if not np.array_equal(old_reference.state[:prefix_frames], new_reference.state[:prefix_frames]):
        raise ValueError(
            "Actual teacher's entire current-goal reference prefix changed; migration requires a new physical replay"
        )
    if not np.array_equal(
        arrays["future_goals"], old_reference.state[indices].astype(arrays["future_goals"].dtype)
    ):
        raise ValueError("Teacher future goals do not match its declared source reference")
    old_goals = arrays["future_goals"].copy()
    arrays["future_goals"] = new_reference.state[indices].astype(old_goals.dtype)
    metadata["reference"] = str(new_reference_path.resolve())
    metadata["reference_sha256"] = sha256(new_reference_path)
    if "reference_sha256" in arrays:
        arrays["reference_sha256"] = np.array(metadata["reference_sha256"])
    migration = {
        "source_teacher": str(source.resolve()),
        "source_teacher_sha256": sha256(source),
        "old_reference_sha256": old_sha,
        "new_reference_sha256": metadata["reference_sha256"],
        "method": "New archive only: top-level/metadata reference hashes and explicit future_goals replaced after exact actual-state/action verification; physical teacher was not rerun or altered",
        "future_goal_vectors_changed": int(
            np.any(old_goals != arrays["future_goals"], axis=-1).sum()
        ),
        "all_physical_arrays_preserved": True,
        "protected_reference_prefix_frames": prefix_frames,
        "protected_reference_prefix_through_physical_s": float(
            old_reference.time_physics[prefix_frames - 1]
        ),
        "old_reference_prefix_sha256": array_hash(old_reference.state[:prefix_frames]),
        "new_reference_prefix_sha256": array_hash(new_reference.state[:prefix_frames]),
        "condition": "BC must be retrained; old actor latent/goals are not treated as compatible",
    }
    metadata["reference_migration"] = migration
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (
        np.load(source, allow_pickle=False) as before,
        np.load(output, allow_pickle=False) as after,
    ):
        migration["array_parity"] = {
            key: bool(np.array_equal(before[key], after[key]))
            for key in before.files
            if key not in ("metadata_json", "future_goals", "reference_sha256")
        }
    if not all(migration["array_parity"].values()):
        raise RuntimeError("Physical teacher arrays changed during migration")
    migration["output_teacher"] = str(output.resolve())
    migration["output_teacher_sha256"] = sha256(output)
    return migration


def reconstruction_diagnosis(reference, before, after, args):
    """Original camera-space human versus prescribed robot in the same frame."""
    with np.load(args.body, allow_pickle=False) as source:
        body = {key: source[key] for key in source.files}
    common = json.loads(args.common_world.read_text())
    rotation = np.asarray(common["rotation_camera_to_world"])
    translation = np.asarray(common["translation_camera_to_world"])
    names = body["joint_names"].tolist()
    feet = [names.index(name) for name in ("left_foot", "right_foot")]
    ankles = [names.index(name) for name in ("left_ankle", "right_ankle")]
    rows = []
    for time in (2.76, 2.96, 3.10, 3.20, 3.30, 3.40, 3.60):
        frame = int(np.argmin(abs(reference.time_physics - time)))
        dense = frame * 8
        human_frame = int(np.argmin(abs(body["time_seconds"] - reference.time_video[frame])))
        camera = body["joints_camera"][human_frame]
        human_world = camera @ rotation.T + translation
        board = reference.board_state[frame]
        board_rotation = before["board_rotation"][dense]
        image = camera @ body["K_fullimg"][human_frame].T
        image = image[:, :2] / image[:, 2:]
        detected = body["keypoints2d_coco17"][human_frame, [15, 16]]
        rows.append(
            {
                "physical_time_s": time,
                "nearest_human_video_time_s": float(body["time_seconds"][human_frame]),
                "source_frame": int(body["nearest_source_frame"][human_frame]),
                "human_raw_foot_joint_board_local_m": (
                    (human_world[feet] - board[:3]) @ board_rotation
                ).tolist(),
                "robot_before_foot_site_board_local_m": before["foot_site_board_local_m"][
                    dense
                ].tolist(),
                "robot_after_foot_site_board_local_m": after["foot_site_board_local_m"][
                    dense
                ].tolist(),
                "human_raw_foot_joint_pixels": image[feet].tolist(),
                "human_ankle_vs_detector_ankle_residual_px": np.linalg.norm(
                    image[ankles] - detected[:, :2], axis=1
                ).tolist(),
                "detector_ankle_confidence": detected[:, 2].tolist(),
            }
        )
    return {
        "body_sha256": sha256(args.body),
        "common_world_sha256": sha256(args.common_world),
        "rows": rows,
        "caveat": "SMPL-X foot joints and G1 sole sites are different landmarks. Camera projections check alignment only; human raw 3D, fixed camera and metric scale are estimates, not ground truth.",
    }


def plot_diagnostics(before, after, shift, board_shift, times, output, window):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    colours = ("#1570b0", "#d66b13")
    dense_time = before["time_physics"]
    for side, colour in enumerate(colours):
        for arrays, style, name in ((before, "--", "before"), (after, "-", "after")):
            axes[0, 0].plot(
                dense_time,
                arrays["foot_site_board_local_m"][:, side, 1],
                style,
                color=colour,
                label=f"{'left' if side == 0 else 'right'} {name}",
            )
            axes[0, 1].plot(dense_time, arrays["sole_gap_m"][:, side], style, color=colour)
    axes[0, 0].set(title="Foot-site lateral coordinate in board frame", ylabel="m")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Entire seven-capsule sole versus centre plane", ylabel="m")
    for axis, colour in enumerate(("#1570b0", "#cf522f", "#189865")):
        axes[1, 0].plot(times, shift[:, axis], color=colour, label="XYZ"[axis])
        axes[1, 1].plot(times, board_shift[:, axis], color=colour, label="XYZ"[axis])
    axes[1, 0].set(title="Robot root XYZ change", ylabel="m")
    axes[1, 1].set(title="Board XYZ change", ylabel="m")
    for a in axes[1]:
        a.legend()
    for arrays, style, name in ((before, "--", "before"), (after, "-", "after")):
        axes[2, 0].plot(dense_time, arrays["system_com_world_m"][:, 1], style, label=name)
        axes[2, 1].plot(dense_time, arrays["system_com_world_m"][:, 2], style, label=name)
    axes[2, 0].set(title="Full-system COM horizontal Y", ylabel="m")
    axes[2, 1].set(title="Full-system COM height", ylabel="m")
    for a in axes.flat:
        a.set_xlim(2.70, 4.05)
        a.set_xlabel("physical seconds (video / 1.5 assumption)")
        a.axvspan(*window, color="#dddddd", alpha=0.35)
        a.grid(alpha=0.2)
    fig.suptitle("Prescribed kinematic repair and gravity prior; no physics rollout")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def board_timing_prior(reference, samples):
    """Video-timed support geometry; contacts stay unknown, no forces inferred."""
    time = reference.time_physics
    floor = samples["board_floor_m"][::8]
    lift_end = 3.12  # video4.68: nearest node after tail-near-floor4.671
    observed_start, observed_end = 3.16, 3.40
    landing = 3.56  # video5.34: clearly grounded source frame
    before_weight = 1 - smoothstep((time - lift_end) / (observed_start - lift_end))
    after_weight = smoothstep((time - observed_end) / (landing - observed_end))
    weight = np.maximum(before_weight, after_weight)
    weight[time <= 2.78 + 1e-8] = 0
    shift = np.zeros((len(time), 3))
    shift[:, 2] = -floor * weight
    result = copy.deepcopy(reference)
    result.board_state[:, :3] += shift
    result.board_state[:, 11:14] += np.gradient(shift, 0.02, axis=0)
    return (
        result,
        shift,
        {
            "method": "Video-timed full-board lower-envelope Z correction; rotations and XY unchanged",
            "source_visual_tail_near_ground_through_video_s": 4.671,
            "last_floor_pinned_physical_node_s": lift_end,
            "original_pnp_interpolation_nodes_preserved_physical_s": [observed_start, observed_end],
            "source_visual_landing_clearly_grounded_video_s": 5.338,
            "floor_pinned_again_physical_s": landing,
            "board_z_max_change_m": float(np.abs(shift[:, 2]).max()),
            "contact_labels_changed": False,
            "meaning": "Engineering geometric support timing from visual review; no measured contact force; short transition/high acceleration must be reported",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/ollie/self_collision_corrected/joint_reference_50hz.npz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/ollie/air_relative_video_timed/joint_reference_50hz.npz"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ollie/air_repair/video_timed"))
    parser.add_argument("--preserve-through", type=float, default=2.76)
    parser.add_argument("--fixed-suffix-from", type=float, default=4.0)
    parser.add_argument("--foot-lateral-limit", type=float, default=0.16)
    parser.add_argument("--centre-lateral-limit", type=float, default=0.08)
    parser.add_argument("--com-window", type=float, nargs=2, default=[3.16, 3.48])
    parser.add_argument("--com-sigma", type=float, default=0.003)
    parser.add_argument(
        "--board-timing-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: source-video support/landing timing correction; disable only for explicit diagnostic comparisons",
    )
    parser.add_argument("--acceleration-sigma", type=float, default=20.0)
    parser.add_argument("--solver-iterations", type=int, default=300)
    parser.add_argument(
        "--move-board",
        action="store_true",
        help="Optimize board XYZ as uncertain reconstruction; rotations/wheels remain exact",
    )
    parser.add_argument("--board-prior-sigma", type=float, default=0.2)
    parser.add_argument("--reprojection-sigma", type=float, default=5.0)
    parser.add_argument(
        "--body", type=Path, default=Path("logs/ollie_gvhmr/ollie/body_world_raw.npz")
    )
    parser.add_argument("--common-world", type=Path, default=Path("data/ollie/common_world.json"))
    parser.add_argument(
        "--board-observations",
        type=Path,
        default=Path("logs/ollie/board_tracking/board_image_observations.json"),
    )
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.out.resolve() or args.out.exists():
        raise ValueError("Only a fresh independent output branch is allowed")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    reference = JointReference.load(args.source)
    if args.board_timing_prior and (
        not np.isclose(reference.metadata["time_scale"], 1.5)
        or not np.isclose(reference.metadata["physics_fps"], 50)
    ):
        raise ValueError(
            "Reviewed Ollie timing priors currently apply only to the explicit scale=1.5, 50Hz branch"
        )
    source_sha = sha256(args.source)
    geometry = Geometry(reference)
    samples, before = geometry.sample(geometry.poses())
    fit_reference, timing_shift, timing_report = (
        reference,
        np.zeros((len(reference.time_physics), 3)),
        None,
    )
    fit_samples = samples
    if args.board_timing_prior:
        fit_reference, timing_shift, timing_report = board_timing_prior(reference, samples)
        fit_samples, _ = geometry.sample(
            geometry.poses(fit_reference.robot_state, fit_reference.board_state),
            collect_contacts=False,
        )
    shift, board_shift, solver = fit_root(fit_reference, geometry, fit_samples, args)
    result = copy.deepcopy(fit_reference)
    result.robot_state[:, :3] += shift
    result.robot_state[:, 36:39] += np.gradient(
        shift, 1 / reference.metadata["physics_fps"], axis=0
    )
    result.board_state[:, :3] += board_shift
    result.board_state[:, 11:14] += np.gradient(
        board_shift, 1 / reference.metadata["physics_fps"], axis=0
    )
    board_shift = board_shift + timing_shift
    protected = reference.time_physics <= args.preserve_through + 1e-8
    prefix_equal = np.array_equal(result.state[protected], reference.state[protected])
    if not prefix_equal:
        raise RuntimeError("Protected complete state prefix changed")
    after_samples, after = geometry.sample(geometry.poses(result.robot_state, result.board_state))
    speed = np.linalg.norm(result.robot_state[:, 36:39], axis=1)
    acceleration = np.gradient(
        result.robot_state[:, 36:39], 1 / reference.metadata["physics_fps"], axis=0
    )
    source_acceleration = np.gradient(
        reference.robot_state[:, 36:39], 1 / reference.metadata["physics_fps"], axis=0
    )
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": source_sha,
        "method": "Robot root and optional board XYZ relative geometry repair; prescribed kinematics and ballistic prior, no dynamics rollout",
        "solver": solver,
        "board_timing_prior": timing_report,
        "protected_prefix_through_physical_s": args.preserve_through,
        "protected_complete_state_exact": prefix_equal,
        "future_through_2_96_exact": bool(
            np.array_equal(
                result.state[reference.time_physics <= 2.96 + 1e-8],
                reference.state[reference.time_physics <= 2.96 + 1e-8],
            )
        ),
        "root_translation_max_change_m": float(np.linalg.norm(shift, axis=1).max()),
        "root_translation_rms_change_m": float(np.sqrt(np.mean(np.sum(shift**2, axis=1)))),
        "root_rotation_exact": bool(
            np.array_equal(result.robot_state[:, 3:7], reference.robot_state[:, 3:7])
        ),
        "all_29_joint_positions_and_velocities_exact": bool(
            np.array_equal(result.robot_state[:, 7:36], reference.robot_state[:, 7:36])
            and np.array_equal(result.robot_state[:, 42:], reference.robot_state[:, 42:])
        ),
        "board_full_state_exact": bool(np.array_equal(result.board_state, reference.board_state)),
        "board_rotation_wheel_positions_velocities_exact": bool(
            np.array_equal(result.board_state[:, 3:11], reference.board_state[:, 3:11])
            and np.array_equal(result.board_state[:, 14:], reference.board_state[:, 14:])
        ),
        "board_translation_max_change_m": float(np.linalg.norm(board_shift, axis=1).max()),
        "contact_phase_confidence_labels_exact": bool(
            all(
                np.array_equal(getattr(result, name), getattr(reference, name))
                for name in ("contacts", "phase", "confidence")
            )
        ),
        "root_speed_max_before_after_m_s": [
            float(np.linalg.norm(reference.robot_state[:, 36:39], axis=1).max()),
            float(speed.max()),
        ],
        "root_acceleration_max_before_after_m_s2": [
            float(np.linalg.norm(source_acceleration, axis=1).max()),
            float(np.linalg.norm(acceleration, axis=1).max()),
        ],
        "new_root_velocity_max_change_m_s": float(
            np.linalg.norm(
                result.robot_state[:, 36:39] - reference.robot_state[:, 36:39], axis=1
            ).max()
        ),
        "before_dense_geometry": before,
        "after_dense_geometry": after,
        "before_com_ballistic_fit": ballistic_fit(
            samples["time_physics"], samples["system_com_world_m"], args.com_window
        ),
        "after_com_ballistic_fit": ballistic_fit(
            after_samples["time_physics"], after_samples["system_com_world_m"], args.com_window
        ),
        "com_window_sensitivity": [
            {
                "window_s": window,
                "before": ballistic_fit(
                    samples["time_physics"], samples["system_com_world_m"], window
                ),
                "after": ballistic_fit(
                    after_samples["time_physics"], after_samples["system_com_world_m"], window
                ),
            }
            for window in ([3.0, 3.56], [3.06, 3.50], [3.14, 3.42])
        ],
        "source_reconstruction_diagnosis": reconstruction_diagnosis(
            reference, samples, after_samples, args
        ),
        "unresolved": [
            "Monocular human and board have incompatible metric depth/shape; no new observed 3D contact is inferred",
            "Upper board halfspace and lateral envelopes are explicit engineering assumptions for this Ollie",
            "No force/torque feasibility, impact feasibility, robustness or policy success is certified",
            "Board pose, metric scale and slowdown=1.5 remain uncertain input hypotheses",
        ],
    }
    result.metadata["air_relative_repair"] = report
    result.metadata["provenance"]["air_relative_source"] = str(args.source.resolve())
    result.metadata["provenance"]["air_relative_source_sha256"] = source_sha
    result.save(args.out)
    report["output"] = str(args.out.resolve())
    report["output_sha256"] = sha256(args.out)
    if args.teacher:
        report["teacher_migration"] = migrate_teacher(
            args.teacher,
            args.log_dir / "teacher_migrated.npz",
            reference,
            result,
            args.out,
            source_sha,
        )
    np.savez_compressed(args.log_dir / "geometry_before.npz", **samples)
    np.savez_compressed(args.log_dir / "geometry_after.npz", **after_samples)
    plot_diagnostics(
        samples,
        after_samples,
        shift,
        board_shift,
        reference.time_physics,
        args.log_dir / "kinematic_repair_diagnostics.png",
        args.com_window,
    )
    np.savez_compressed(
        args.log_dir / "root_translation_repair.npz",
        time_physics=reference.time_physics,
        root_translation_change_m=shift,
        board_translation_change_m=board_shift,
        root_acceleration_before=source_acceleration,
        root_acceleration_after=acceleration,
    )
    (args.log_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "output",
                    "output_sha256",
                    "protected_complete_state_exact",
                    "future_through_2_96_exact",
                    "root_translation_max_change_m",
                    "root_speed_max_before_after_m_s",
                    "root_acceleration_max_before_after_m_s2",
                    "before_com_ballistic_fit",
                    "after_com_ballistic_fit",
                    "after_dense_geometry",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
