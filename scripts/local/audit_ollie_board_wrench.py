"""Conditional board force feasibility from real inertia and finite geometry.

No forces, accelerations or poses are applied in a rollout. Wheel accelerations
are eliminated as unknown passive DOFs. Contact candidates are not measured forces;
inner/outer friction approximations and incomplete spatial witnesses stay explicit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import linprog

from supermimic.ollie import physics
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tangent_frame(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    seed = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(seed, normal)
    tangent /= np.linalg.norm(tangent)
    return np.column_stack([normal, tangent, np.cross(normal, tangent)])


@dataclass
class Witness:
    board_geom: int
    other_geom: int
    point: np.ndarray
    frame: np.ndarray
    gap_m: float
    origin: str
    condim: int
    friction: np.ndarray


class BoardWrenchAudit:
    def __init__(self, gap_band=0.002, moment_length=0.4):
        self.model, self.data, self.index = compile_scene(timestep=0.0025)
        model, index = self.model, self.index
        self.gap_band, self.moment_length = gap_band, moment_length
        self.indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
        self.rows = np.r_[np.arange(index.board_qvel, index.board_qvel + 6), index.wheel_qvel]
        self.board_mask = model.body_rootid == model.body_rootid[index.board_body]
        self.board_geoms = [
            g
            for g in range(model.ngeom)
            if self.board_mask[model.geom_bodyid[g]] and model.geom_contype[g]
        ]
        self.feet = [
            g
            for g in range(model.ngeom)
            if re.fullmatch(r"(left|right)_foot[1-7]_collision", model.geom(g).name or "")
        ]
        self.ground = model.geom("ground").id
        if model.neq or model.nu != 29 or mujoco.get_mjcb_control() is not None:
            raise ValueError(
                "Only the original passive-board scene without hidden constraints is supported"
            )
        if np.any(model.dof_frictionloss[index.wheel_qvel]):
            raise ValueError(
                "Nonzero passive wheel dry-friction requires an additional bounded torque variable"
            )
        self.jac, self.rot_jac = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def parameters(self, first, second):
        model = self.model
        for pair in range(model.npair):
            if {first, second} == {int(model.pair_geom1[pair]), int(model.pair_geom2[pair])}:
                return int(model.pair_dim[pair]), model.pair_friction[pair].copy()
        priorities = model.geom_priority[[first, second]]
        chosen = (
            first
            if priorities[0] > priorities[1]
            else second
            if priorities[1] > priorities[0]
            else None
        )
        friction = (
            model.geom_friction[chosen]
            if chosen is not None
            else np.maximum(model.geom_friction[first], model.geom_friction[second])
        )
        condim = int(
            model.geom_condim[chosen]
            if chosen is not None
            else max(model.geom_condim[first], model.geom_condim[second])
        )
        return condim, np.maximum(np.asarray(friction)[[0, 0, 1, 2, 2]], 1e-5)

    def enabled(self, first, second):
        model = self.model
        return bool(
            (model.geom_contype[first] & model.geom_conaffinity[second])
            or (model.geom_contype[second] & model.geom_conaffinity[first])
        )

    def witnesses(self):
        model, data = self.model, self.data
        result, seen = [], set()

        def append(
            board, other, point, normal, gap, origin, frame=None, condim=None, friction=None
        ):
            if (
                gap > self.gap_band
                or np.linalg.norm(normal) < 1e-10
                or not self.enabled(board, other)
            ):
                return
            normal = np.asarray(normal) / np.linalg.norm(normal)
            key = (
                board,
                other,
                *np.rint(np.asarray(point) / 1e-5).astype(int),
                *np.rint(normal * 1e4).astype(int),
            )
            if key in seen:
                return
            seen.add(key)
            if condim is None:
                condim, friction = self.parameters(board, other)
            result.append(
                Witness(
                    board,
                    other,
                    np.array(point),
                    tangent_frame(normal) if frame is None else frame.copy(),
                    float(gap),
                    origin,
                    int(condim),
                    np.array(friction),
                )
            )

        for contact in data.contact[: data.ncon]:
            first, second = map(int, contact.geom)
            if first in self.board_geoms and (second in self.feet or second == self.ground):
                append(
                    first,
                    second,
                    contact.pos,
                    -contact.frame[:3],
                    contact.dist,
                    "native_manifold",
                    -contact.frame.reshape(3, 3).T,
                    contact.dim,
                    contact.friction,
                )
            elif second in self.board_geoms and (first in self.feet or first == self.ground):
                append(
                    second,
                    first,
                    contact.pos,
                    contact.frame[:3],
                    contact.dist,
                    "native_manifold",
                    contact.frame.reshape(3, 3).T,
                    contact.dim,
                    contact.friction,
                )

        for foot in self.feet:
            for board in self.board_geoms:
                if not self.enabled(foot, board):
                    continue
                if (
                    np.linalg.norm(data.geom_xpos[foot] - data.geom_xpos[board])
                    > model.geom_rbound[foot] + model.geom_rbound[board] + self.gap_band
                ):
                    continue
                segment = np.zeros(6)
                distance = float(
                    mujoco.mj_geomDistance(model, data, foot, board, self.gap_band + 0.001, segment)
                )
                delta = segment[3:] - segment[:3]
                if abs(distance) > 1e-10 and np.linalg.norm(delta) > 1e-10:
                    append(board, foot, segment[3:], delta / distance, distance, "geom_distance")
                # Parallel capsule/box contacts can support a distributed line.
                # Add cap-end witnesses only if the proposed point is on the
                # actual capsule boundary, rather than inside the union of spheres.
                if (
                    model.geom_type[foot] != mujoco.mjtGeom.mjGEOM_CAPSULE
                    or model.geom_type[board] != mujoco.mjtGeom.mjGEOM_BOX
                ):
                    continue
                center, axis = data.geom_xpos[foot], data.geom_xmat[foot].reshape(3, 3)[:, 2]
                radius, half = model.geom_size[foot, :2]
                rotation, origin, size = (
                    data.geom_xmat[board].reshape(3, 3),
                    data.geom_xpos[board],
                    model.geom_size[board],
                )
                for endpoint in (center - half * axis, center + half * axis):
                    local = rotation.T @ (endpoint - origin)
                    closest = origin + rotation @ np.clip(local, -size, size)
                    vector, length = endpoint - closest, float(np.linalg.norm(endpoint - closest))
                    if length < 1e-10:
                        continue
                    outward = vector / length
                    foot_point = endpoint - radius * outward
                    nearest_axis = (
                        center + np.clip((foot_point - center) @ axis, -half, half) * axis
                    )
                    surface_error = abs(np.linalg.norm(foot_point - nearest_axis) - radius)
                    if surface_error <= 1e-5:
                        append(
                            board, foot, closest, -outward, length - radius, "capsule_cap_endpoint"
                        )

        for board in self.board_geoms:
            rotation, center, size = (
                data.geom_xmat[board].reshape(3, 3),
                data.geom_xpos[board],
                model.geom_size[board],
            )
            kind = model.geom_type[board]
            points = []
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                points = [
                    center + rotation @ (size * signs)
                    for signs in map(np.array, itertools.product((-1, 1), repeat=3))
                ]
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                axis = rotation[:, 2]
                radial = np.array([0.0, 0.0, -1.0]) + axis * axis[2]
                norm = np.linalg.norm(radial)
                if norm > 1e-10:
                    points = [
                        center + offset * size[1] * axis + size[0] * radial / norm
                        for offset in (-1.0, 0.0, 1.0)
                    ]
            else:
                raise ValueError("Board-ground witnesses require authored boxes/cylinders")
            for point in points:
                append(
                    board,
                    self.ground,
                    point,
                    np.array([0.0, 0.0, 1.0]),
                    point[2],
                    "ground_exact_primitive_support",
                )
        return result

    def board_pose_summary(self, qpos):
        model, data = self.model, self.data
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        mass = model.body_mass[self.board_mask].sum()
        com = (model.body_mass[self.board_mask, None] * data.xipos[self.board_mask]).sum(0) / mass
        minima = []
        for geom in self.board_geoms:
            rotation, size = data.geom_xmat[geom].reshape(3, 3), model.geom_size[geom]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(rotation[2]) @ size
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = size[0] * np.sqrt(max(0.0, 1 - rotation[2, 2] ** 2)) + size[1] * abs(
                    rotation[2, 2]
                )
            else:
                raise ValueError("Unsupported board primitive")
            minima.append(data.geom_xpos[geom, 2] - extent)
        return com, float(min(minima))

    def inertia(self, qpos, qvel, acceleration, wheel_speed):
        model, data, index = self.model, self.data, self.index
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.qvel[index.wheel_qvel] = wheel_speed
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        if np.abs(data.qfrc_actuator[self.rows]).max() > 1e-10:
            raise ValueError("A board/wheel actuator unexpectedly contributes generalized force")
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        known = acceleration.copy()
        known[index.wheel_qvel] = 0  # Unknown acceleration is eliminated below, never assumed zero.
        rhs = (mass @ known + data.qfrc_bias - data.qfrc_passive)[self.rows]
        coupling = mass[np.ix_(self.rows[:6], index.wheel_qvel)]
        wheel_mass = mass[np.ix_(index.wheel_qvel, index.wheel_qvel)]
        reduced = rhs[:6] - coupling @ np.linalg.solve(wheel_mass, rhs[6:])
        return rhs, reduced, coupling, wheel_mass

    def contact_matrix(self, witnesses):
        model, data = self.model, self.data
        matrices = []
        for witness in witnesses:
            mujoco.mj_jac(
                model,
                data,
                self.jac,
                self.rot_jac,
                witness.point,
                int(model.geom_bodyid[witness.board_geom]),
            )
            force, torque = self.jac.T @ witness.frame, self.rot_jac.T @ witness.frame
            block = (
                force
                if witness.condim <= 3
                else np.concatenate(
                    [force, torque[:, :1] if witness.condim == 4 else torque], axis=1
                )
            )
            matrices.append(block[self.rows, : witness.condim])
        return matrices


def friction_generators(witness, outer):
    dimension = witness.condim
    if dimension == 1:
        return np.ones((1, 1))
    if dimension not in (3, 4, 6):
        raise ValueError(f"Unsupported contact dimension {dimension}")
    mu = witness.friction[: dimension - 1]
    if outer:
        return np.column_stack(
            [
                np.r_[1.0, np.array(signs) * mu]
                for signs in itertools.product((-1.0, 1.0), repeat=dimension - 1)
            ]
        )
    return np.column_stack(
        [
            np.r_[1.0, sign * mu * np.eye(dimension - 1)[axis]]
            for axis in range(dimension - 1)
            for sign in (-1.0, 1.0)
        ]
    )


def solve_wrench(
    witnesses, matrices, rhs, coupling, wheel_mass, spin_rhs, spin_policy, outer, moment_length
):
    reduced_matrices, generators, blocks = [], [], []
    for witness, matrix in zip(witnesses, matrices, strict=True):
        generator = friction_generators(witness, outer)
        reduced = matrix[:6] - coupling @ np.linalg.solve(wheel_mass, matrix[6:])
        reduced_matrices.append(reduced @ generator)
        generators.append(generator)
        blocks.append(matrix)
    count = sum(matrix.shape[1] for matrix in reduced_matrices)
    contact = np.concatenate(reduced_matrices, axis=1) if count else np.empty((6, 0))
    target = rhs[:6] - coupling @ np.linalg.solve(wheel_mass, rhs[6:])
    free_spin = spin_policy == "free_affine_relaxation"
    columns = np.concatenate([contact, -spin_rhs], axis=1) if free_spin else contact
    scales = np.r_[np.ones(3), np.full(3, 1 / moment_length)]
    bounds = [(0, None)] * count + ([(None, None)] * 4 if free_spin else [])
    objective = np.r_[np.ones(count), np.zeros(4 if free_spin else 0)]
    if columns.shape[1]:
        exact = linprog(
            objective,
            A_eq=columns * scales[:, None],
            b_eq=target * scales,
            bounds=bounds,
            method="highs",
        )
    else:
        exact = None
    if exact is not None and exact.success:
        variables, status = exact.x, "EXACT_FEASIBLE_FOR_WITNESS_CONE"
    else:
        augmented = np.concatenate([columns, np.eye(6), -np.eye(6)], axis=1)
        relaxed = linprog(
            np.r_[objective * 1e-9, scales, scales],
            A_eq=augmented * scales[:, None],
            b_eq=target * scales,
            bounds=bounds + [(0, None)] * 12,
            method="highs",
        )
        if not relaxed.success:
            raise RuntimeError("Unexplained-wrench LP failed: " + relaxed.message)
        variables, status = (
            relaxed.x[: columns.shape[1]],
            "NONZERO_UNEXPLAINED_WRENCH_FOR_WITNESS_CONE",
        )
    spin = variables[count : count + 4] if free_spin else np.zeros(4)
    local, generalized, offset = [], np.zeros(10), 0
    for block, generator in zip(blocks, generators, strict=True):
        force = generator @ variables[offset : offset + generator.shape[1]]
        offset += generator.shape[1]
        local.append(force)
        generalized += block @ force
    # The caller supplies full affine wheel-speed rhs separately; only reduced
    # residual is needed here. Recover qdd after reconstructing the actual h.
    residual = (
        target
        + spin_rhs @ spin
        - generalized[:6]
        + coupling @ np.linalg.solve(wheel_mass, generalized[6:])
    )
    if np.max(np.abs(residual[:3])) <= 1e-3 and np.max(np.abs(residual[3:])) <= 1e-4:
        status = "FEASIBLE_FOR_WITNESS_CONE_WITHIN_RECORDED_TOLERANCE"
    elliptic_violation = max(
        (
            float(np.linalg.norm(force[1:] / w.friction[: w.condim - 1]) - force[0])
            for force, w in zip(local, witnesses, strict=True)
        ),
        default=0.0,
    )
    return (
        {
            "status": status,
            "force_residual_world_n": residual[:3].tolist(),
            "torque_residual_board_nm": residual[3:].tolist(),
            "force_residual_norm_n": float(np.linalg.norm(residual[:3])),
            "torque_residual_norm_nm": float(np.linalg.norm(residual[3:])),
            "minimum_unexplained_force_equivalent_l1_n": float(np.abs(residual) @ scales),
            "normal_force_sum_n": float(sum(force[0] for force in local)),
            "elliptic_friction_cone_max_violation_force_equivalent_n": elliptic_violation,
            "local_contact_wrenches": [force.tolist() for force in local],
            "generalized_contact_force_10": generalized.tolist(),
            "wheel_spin_adjustment_rad_s": spin.tolist(),
            "lp_exact_status": int(exact.status) if exact is not None else None,
        },
        generalized,
        spin,
    )


def audit_frame(audit, qpos, qvel, acceleration, speeds):
    rhs0, reduced0, coupling, wheel_mass = audit.inertia(qpos, qvel, acceleration, np.zeros(4))
    witnesses = audit.witnesses()
    matrices = audit.contact_matrix(witnesses)
    geometry_depth = max(
        (max(0.0, -float(c.dist)) for c in audit.data.contact[: audit.data.ncon]), default=0.0
    )
    full_spin = np.column_stack(
        [audit.inertia(qpos, qvel, acceleration, np.eye(4)[i])[0] - rhs0 for i in range(4)]
    )
    reduced_spin = full_spin[:6] - coupling @ np.linalg.solve(wheel_mass, full_spin[6:])
    errors = []
    for probe in (
        np.array([20.0, -20.0, 10.0, -10.0]),
        np.full(4, 20.0),
        np.full(4, -20.0),
        np.array([50.0, 10.0, -30.0, -40.0]),
    ):
        true = audit.inertia(qpos, qvel, acceleration, probe)[0]
        errors.append(float(np.abs(true - rhs0 - full_spin @ probe).max()))
    affine = max(errors) < 1e-7
    profiles = [("all_wheels_" + str(speed), np.full(4, speed), False) for speed in speeds]
    if affine:
        profiles.append(("free_affine_relaxation", np.zeros(4), True))
    results = []
    for name, wheel_spin, free in profiles:
        rhs, _reduced, _coupling, _wheel_mass = audit.inertia(qpos, qvel, acceleration, wheel_spin)
        cones = {}
        for outer in (False, True):
            record, generalized, adjustment = solve_wrench(
                witnesses,
                matrices,
                rhs,
                coupling,
                wheel_mass,
                reduced_spin,
                "free_affine_relaxation" if free else "fixed",
                outer,
                audit.moment_length,
            )
            true_rhs, _red, _c, _wm = audit.inertia(
                qpos, qvel, acceleration, wheel_spin + adjustment
            )
            wheel_acceleration = np.linalg.solve(wheel_mass, generalized[6:] - true_rhs[6:])
            residual10 = (
                generalized
                - true_rhs
                - np.r_[coupling @ wheel_acceleration, wheel_mass @ wheel_acceleration]
            )
            record.update(
                actual_wheel_spin_rad_s=(wheel_spin + adjustment).tolist(),
                unknown_passive_wheel_qdd_rad_s2=wheel_acceleration.tolist(),
                actual_full_10_equation_residual=residual10.tolist(),
                required_rhs_10=true_rhs.tolist(),
                effective_required_wrench_6=_red.tolist(),
                recovered_wheel_equation_residual_max=float(np.abs(residual10[6:]).max()),
                true_model_balance_verified=bool(
                    np.max(np.abs(residual10[:3])) <= 1e-3
                    and np.max(np.abs(residual10[3:6])) <= 1e-4
                    and np.max(np.abs(residual10[6:])) <= 1e-6
                ),
            )
            if (
                record["status"].startswith("FEASIBLE")
                and not record["true_model_balance_verified"]
            ):
                record["status"] = "AFFINE_RELAXATION_FAILED_TRUE_MODEL_RECHECK"
            cones["outer_box_relaxation" if outer else "inner_inscribed_pyramid"] = record
        interpretation = (
            "INNER_WITNESS_FORCE_BALANCE_FEASIBLE_NOT_A_PHYSICS_CERTIFICATE"
            if cones["inner_inscribed_pyramid"]["true_model_balance_verified"]
            else "OUTER_INFEASIBLE_FOR_THIS_FINITE_CONTACT_FAMILY"
            if not cones["outer_box_relaxation"]["true_model_balance_verified"]
            else "ELLIPTIC_WITNESS_FOUND_BY_OUTER_LP"
            if cones["outer_box_relaxation"][
                "elliptic_friction_cone_max_violation_force_equivalent_n"
            ]
            <= 1e-6
            else "UNDECIDED_BETWEEN_INNER_AND_OUTER_FRICTION_APPROXIMATIONS"
        )
        results.append(
            {
                "wheel_spin_profile": name,
                "wheel_spin_observed": False,
                "cones": cones,
                "interpretation": interpretation,
            }
        )
    audit.inertia(qpos, qvel, acceleration, np.zeros(4))
    rotation = audit.data.xmat[audit.index.board_body].reshape(3, 3)
    mass = np.zeros((audit.model.nv, audit.model.nv))
    mujoco.mj_fullM(audit.model, audit.data, mass)

    def witness_report(witness):
        values = asdict(witness)
        values.update(
            board_geom=audit.model.geom(witness.board_geom).name,
            other_geom=audit.model.geom(witness.other_geom).name,
            point=witness.point.tolist(),
            frame=witness.frame.tolist(),
            friction=witness.friction.tolist(),
        )
        normal_local = (
            audit.data.geom_xmat[witness.board_geom].reshape(3, 3).T @ witness.frame[:, 0]
        )
        values["normal_on_board_world"] = witness.frame[:, 0].tolist()
        values["normal_on_board_geom_local"] = normal_local.tolist()
        values["mode"] = (
            "board_ground"
            if witness.other_geom == audit.ground
            else "foot_on_upper_face_or_edge"
            if normal_local[2] < -0.5
            else "foot_under_lower_face_or_edge"
            if normal_local[2] > 0.5
            else "foot_on_side_or_wheel"
        )
        return values

    return {
        "candidate_contacts": [witness_report(w) for w in witnesses],
        "snapshot_maximum_geometry_penetration_m": geometry_depth,
        "snapshot_geometry_within_2mm": geometry_depth <= 0.002 + 1e-8,
        "contact_mode_counts": {
            mode: sum(witness_report(w)["mode"] == mode for w in witnesses)
            for mode in sorted({witness_report(w)["mode"] for w in witnesses})
        },
        "required_inertia_rhs_10_at_zero_wheel_spin": rhs0.tolist(),
        "effective_required_wrench_6_at_zero_wheel_spin": reduced0.tolist(),
        "effective_world_force_n": reduced0[:3].tolist(),
        "effective_world_torque_nm": (rotation @ reduced0[3:]).tolist(),
        "wheel_mass_matrix": wheel_mass.tolist(),
        "board_wheel_mass_coupling": coupling.tolist(),
        "native_mass_block_10x10": mass[np.ix_(audit.rows, audit.rows)].tolist(),
        "native_bias_minus_passive_10_at_zero_spin": (
            audit.data.qfrc_bias - audit.data.qfrc_passive
        )[audit.rows].tolist(),
        "full_10_rhs_wheel_speed_affine_coefficients": full_spin.tolist(),
        "wheel_speed_affine_verification_max_residual": max(errors),
        "wheel_speed_affine_relaxation_used": affine,
        "profiles": results,
    }


def validate_core(audit):
    """Small native checks which catch frame signs and passive-wheel mistakes."""
    model, data, index = audit.model, audit.data, audit.index
    alignments = []
    for contact in data.contact[: data.ncon]:
        first, second = map(int, contact.geom)
        if first in audit.feet and second in audit.board_geoms:
            foot, board, expected = first, second, contact.frame[:3]
        elif second in audit.feet and first in audit.board_geoms:
            foot, board, expected = second, first, -contact.frame[:3]
        else:
            continue
        segment = np.zeros(6)
        distance = mujoco.mj_geomDistance(model, data, foot, board, 0.003, segment)
        if distance < -1e-8:
            normal = (segment[3:] - segment[:3]) / distance
            alignments.append(float(normal @ expected))
    if not alignments or min(alignments) < 1 - 1e-8:
        raise AssertionError("Signed penetration witness normal disagrees with native contact")
    point, force, torque = (
        np.array([0.12, -0.03, 0.08]),
        np.array([2.0, 3.0, -4.0]),
        np.array([0.1, 0.2, -0.3]),
    )
    jp, jr, native = np.zeros((3, model.nv)), np.zeros((3, model.nv)), np.zeros(model.nv)
    mujoco.mj_jac(model, data, jp, jr, point, index.board_body)
    mujoco.mj_applyFT(model, data, force, torque, point, index.board_body, native)
    jac_error = float(np.max(np.abs(native - jp.T @ force - jr.T @ torque)))
    if jac_error > 1e-12:
        raise AssertionError("Contact wrench Jacobian disagrees with mj_applyFT")

    physics.reset_scene(model, data, index)
    data.qpos[index.robot_qpos : index.robot_qpos + 3] = [5, 0, 3]
    data.qpos[index.board_qpos : index.board_qpos + 3] = [0, 0, 2]
    data.qvel[index.board_qvel + 3 : index.board_qvel + 6] = [2, 1, 3]
    data.qvel[index.wheel_qvel] = 20
    mujoco.mj_forward(model, data)
    qpos, qvel, native_acceleration = data.qpos.copy(), data.qvel.copy(), data.qacc.copy()
    rhs, reduced, coupling, wheel_mass = audit.inertia(
        qpos, qvel, native_acceleration, qvel[index.wheel_qvel]
    )
    if audit.witnesses():
        raise AssertionError("Freefall check unexpectedly has a board contact")
    solved, _, _ = solve_wrench(
        [], [], rhs, coupling, wheel_mass, np.zeros((6, 4)), "fixed", False, audit.moment_length
    )
    wrong_zero_qdd_rhs_max = float(np.max(np.abs(rhs)))
    recovered = np.linalg.solve(wheel_mass, -rhs[6:])
    if (
        not solved["status"].startswith("FEASIBLE")
        or np.max(np.abs(reduced)) > 1e-10
        or np.max(np.abs(recovered - native_acceleration[index.wheel_qvel])) > 1e-8
        or np.max(np.abs(recovered)) < 1e-3
    ):
        raise AssertionError("Unknown passive wheel acceleration does not recover native dynamics")

    witness = Witness(0, 0, np.zeros(3), np.eye(3), 0, "synthetic", 3, np.full(5, 0.9))
    block, rhs = np.zeros((10, 3)), np.zeros(10)
    block[:3] = np.eye(3)
    rhs[:3] = [1, 0.54, 0.54]  # Inside the circle; outside the inscribed diamond.
    cones = {}
    for outer in (False, True):
        record, _, _ = solve_wrench(
            [witness],
            [block],
            rhs,
            np.zeros((6, 4)),
            np.eye(4),
            np.zeros((6, 4)),
            "fixed",
            outer,
            audit.moment_length,
        )
        cones["outer" if outer else "inner"] = record["status"]
    if cones["inner"].startswith("FEASIBLE") or not cones["outer"].startswith("FEASIBLE"):
        raise AssertionError("Inner and outer friction approximations were conflated")
    physics.reset_scene(model, data, index)
    return {
        "negative_gap_normal_alignment_min": min(alignments),
        "freefall_without_contacts_feasible": True,
        "unknown_passive_wheel_qdd_matches_native": True,
        "native_wheel_qdd_rad_s2": recovered.tolist(),
        "wrong_zero_qdd_original_rhs_max": wrong_zero_qdd_rhs_max,
        "schur_eliminated_rhs_max": float(np.max(np.abs(reduced))),
        "inner_vs_outer_distinction": cones,
        "applyFT_vs_jacobian_transpose_max_error": jac_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references",
        type=Path,
        nargs="+",
        default=[Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz")],
    )
    parser.add_argument(
        "--times", type=float, nargs="+", default=[3.10, 3.14, 3.18, 3.26, 3.34, 3.50, 3.58]
    )
    parser.add_argument("--wheel-speeds", type=float, nargs="+", default=[0.0, -20.0, 20.0])
    parser.add_argument("--gap-band-m", type=float, default=0.002)
    parser.add_argument("--moment-length-m", type=float, default=0.4)
    parser.add_argument(
        "--self-test", action="store_true", help="Run small CPU native-dynamics checks only"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.out_dir.exists()
        or not np.isfinite(
            [*args.times, *args.wheel_speeds, args.gap_band_m, args.moment_length_m]
        ).all()
        or not 0 < args.gap_band_m <= 0.002
        or args.moment_length_m <= 0
    ):
        parser.error("Need fresh output, finite times/spins and a (0,2mm] candidate band")
    torch.set_num_threads(2)
    audit = BoardWrenchAudit(args.gap_band_m, args.moment_length_m)
    if args.self_test:
        validation = validate_core(audit)
        args.out_dir.mkdir(parents=True)
        (args.out_dir / "validation.json").write_text(
            json.dumps(validation, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(validation, indent=2))
        return
    outputs = []
    for path in args.references:
        reference = JointReference.load(path)
        if not np.isclose(reference.metadata["physics_fps"], 50):
            raise ValueError("Requires 50Hz reference")
        qpos, qvel = audit.indices.unpack(torch.from_numpy(reference.state))
        qpos, qvel = qpos.numpy(), qvel.numpy()
        acceleration = np.gradient(qvel, 0.02, axis=0)
        board_summary = [audit.board_pose_summary(pose) for pose in qpos]
        board_com = np.stack([value[0] for value in board_summary])
        board_floor = np.asarray([value[1] for value in board_summary])
        board_mass = float(audit.model.body_mass[audit.board_mask].sum())
        frames = []
        for time in args.times:
            frame = round(time / 0.02)
            if (
                frame < 2
                or frame >= len(qpos) - 2
                or abs(reference.time_physics[frame] - time) > 1e-9
            ):
                raise ValueError("Time must be an interior 50Hz boundary")
            row = audit_frame(
                audit, qpos[frame], qvel[frame], acceleration[frame], args.wheel_speeds
            )
            row.update(
                time_physics_s=float(reference.time_physics[frame]),
                time_video_s=float(reference.time_video[frame]),
                frame=frame,
                reference_board_qvel_native=qvel[frame, audit.rows[:6]].tolist(),
                reference_board_qacc_native=acceleration[frame, audit.rows[:6]].tolist(),
                derivative_stencil_note="qacc is d(native qvel)/dt; free angular qvel/qacc are body components, linear components world. Stored pose-derived velocities may already span another +/-20ms; takeoff/impact boundary results are conditional.",
                conservative_pose_derivative_stencil_s=reference.time_physics[
                    [frame - 2, frame + 2]
                ].tolist(),
                board_ground_within_derivative_stencil=bool(
                    np.any(board_floor[frame - 2 : frame + 3] <= args.gap_band_m)
                ),
                board_com_force_from_direct_pose_second_difference_n=(
                    board_mass
                    * (
                        (board_com[frame + 1] - 2 * board_com[frame] + board_com[frame - 1])
                        / 0.02**2
                        - audit.model.opt.gravity
                    )
                ).tolist(),
            )
            frames.append(row)
        outputs.append(
            {"reference": str(path.resolve()), "reference_sha256": sha(path), "frames": frames}
        )
    report = {
        "schema": "supermimic.ollie.board_wrench_audit.v1",
        "references": outputs,
        "mujoco_version": mujoco.__version__,
        "audit_script_sha256": sha(__file__),
        "physics_module_sha256": sha(physics.__file__),
        "board_xml_sha256": sha(physics.BOARD_XML),
        "model_friction_cone_enum": int(audit.model.opt.cone),
        "gap_band_m": args.gap_band_m,
        "moment_length_m": args.moment_length_m,
        "equations": "(M*a + qfrc_bias - qfrc_passive)[board_free6,wheel4] = J_contact^T*f; wheel qdd is an unknown eliminated by Schur complement, never forced to the zero placeholder",
        "wheel_velocity_note": "Unobserved. Fixed 0/+/-20 scenarios and, only after actual-model affine verification, an unbounded-speed relaxation. Free-spin feasibility may require implausible wheel speeds and is not a trajectory certificate.",
        "friction_note": "Inner nonnegative single-axis rays are inside the elliptic cone; outer independent +/-mu*FN box contains it. Native manifold tangent axes retained; positive-gap witness frames are diagnostic choices. Inner failure does not prove real-cone failure; outer failure is conditional on this finite spatial witness family.",
        "spatial_note": "Native manifold plus closest finite-geom witness and capsule cap endpoints lying on the real capsule boundary; ground box vertices and cylinder support line endpoints. This improves distributed contact coverage but is not a proof of complete continuum patch extremals.",
        "scope": "Pointwise necessary force balance only; no measured forces, no force application, no robot torque/COM/complementarity/slip law/whole-trajectory certification. Large acceleration alone is not an infeasibility proof.",
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = []
    for source in outputs:
        for frame in source["frames"]:
            profile = next(
                (
                    p
                    for p in frame["profiles"]
                    if p["wheel_spin_profile"] == "free_affine_relaxation"
                ),
                frame["profiles"][0],
            )
            summary.append(
                {
                    "reference": Path(source["reference"]).parent.name,
                    "time": frame["time_physics_s"],
                    "contacts": len(frame["candidate_contacts"]),
                    "contact_mode_counts": frame["contact_mode_counts"],
                    "board_ground_within_derivative_stencil": frame[
                        "board_ground_within_derivative_stencil"
                    ],
                    "interpretation": profile["interpretation"],
                    "effective_world_force_at_zero_wheel_spin_n": frame["effective_world_force_n"],
                    "residual_profile": profile["wheel_spin_profile"],
                    "cones": {
                        k: {
                            "status": v["status"],
                            "force_error_n": v["force_residual_norm_n"],
                            "torque_error_nm": v["torque_residual_norm_nm"],
                        }
                        for k, v in profile["cones"].items()
                    },
                }
            )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
