"""Native MuJoCo PD trajectory search: a physical feasibility baseline, not SuperMimic.

Initial joint/root placement is solved once. During every rollout only the
robot's 29 position targets are supplied; the board is always passive.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import rollout
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from supermimic.ollie.physics import (
    BOARD_BODY_NAME,
    BOARD_GEOM_NAMES,
    BOARD_JOINT_NAMES,
    SceneIndex,
    contact_summary,
    make_scene_spec,
    reset_scene,
)
from supermimic.ollie.quasistatic import ContactStatics

PARAMETER_NAMES = (
    "crouch_delta", "rear_extension", "front_tuck", "pop_seconds",
    "front_delay_seconds", "air_tuck", "air_seconds", "rear_ankle_roll",
    "rear_ankle_pitch", "rear_hip_roll",
    "front_extension", "drive_seconds", "rear_pop_delay_seconds",
    "rear_tuck_delta", "rear_tuck_seconds", "front_press_seconds",
    "ankle_pop_lead_seconds",
    "front_pop_lead_seconds",
    "rear_leg_pop_lead_seconds",
)
LOWER = np.array([0.20, 0.10, 0.20, 0.06, -0.04, 0.20, 0.18, -0.15, -0.65, -0.25, 0.10, 0.03, 0.0, 0.40, 0.02, 0.03, 0.0, 0.0, 0.0])
UPPER = np.array([1.10, 0.70, 1.30, 0.20, 0.14, 1.20, 0.55, 0.15, 0.65, 0.25, 0.70, 0.10, 0.10, 1.50, 0.14, 0.18, 0.22, 0.14, 0.16])
DEFAULT = np.array([0.55, 0.30, 0.70, 0.12, 0.06, 0.65, 0.32, 0.0, 0.0, 0.0, 0.35, 0.05, 0.03, 1.0, 0.06, 0.10, 0.0, 0.0, 0.0])
CONTROL_DT = 0.02


@dataclass(frozen=True)
class BalanceGains:
    """Four support-feedback gains, selected by a bounded native CPU grid."""

    tilt_scale: float = 1.5
    rate_scale: float = 2.0
    com_position_gain: float = 2.4
    com_velocity_gain: float = 0.2

    def __post_init__(self):
        values = np.array([self.tilt_scale, self.rate_scale,
                           self.com_position_gain, self.com_velocity_gain])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("balance gains must be finite and nonnegative")


class FootBoardSupport:
    """Read resolved normal support force; never apply the measured force."""

    def __init__(self, model):
        self.model = model
        self.deck = {model.geom(name).id for name in ("board_center", "board_tail", "board_nose")}
        self.foot_side = {g: side for side, name in enumerate(("left", "right"))
                          for g in range(model.ngeom) if model.geom(g).name.startswith(name + "_foot")}

    def vertical_loads(self, data):
        loads = np.zeros(2)
        for contact_id, contact in enumerate(data.contact[:data.ncon]):
            if contact.efc_address < 0:
                continue
            a, b = map(int, contact.geom)
            side = self.foot_side.get(a) if b in self.deck else self.foot_side.get(b) if a in self.deck else None
            if side is None:
                continue
            surface = b if a in self.foot_side else a
            rotation = data.geom_xmat[surface].reshape(3, 3)
            normal_on_foot = np.asarray(contact.frame[:3]) * (1 if b in self.foot_side else -1)
            local_contact = rotation.T @ (contact.pos - data.geom_xpos[surface])
            if local_contact[2] < 0 or normal_on_foot @ rotation[:, 2] <= 0.5:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, contact_id, force)
            # Contact normal points geom0 -> geom1; its force acts positively
            # on geom1. Downward/underside contacts are not supporting the foot.
            upward = float(normal_on_foot[2])
            loads[side] += max(float(force[0]), 0) * max(upward, 0)
        return loads


@dataclass(frozen=True)
class BoardCollisionGeometry:
    """Exact support heights of this rigid deck and four symmetric cylinders.

    Wheel hinge rotation changes neither the cylinder centre nor its symmetry
    axis. This reduction is checked against the authored joint/geom axes.
    It is geometry only: positive clearance does not certify a contact event.
    """

    local_centres: np.ndarray
    local_rotations: np.ndarray
    sizes: np.ndarray
    wheel_count: int

    @classmethod
    def from_model(cls, model, index, qpos):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        names = [name.removesuffix("_joint") + "_collision" for name in BOARD_JOINT_NAMES]
        geoms = [model.geom(name).id for name in names + list(BOARD_GEOM_NAMES)]
        board_rotation = data.xmat[index.board_body].reshape(3, 3)
        centres = (data.geom_xpos[geoms] - data.xpos[index.board_body]) @ board_rotation
        rotations = np.einsum("ij,njk->nik", board_rotation.T,
                              data.geom_xmat[geoms].reshape(-1, 3, 3))
        for slot, name in enumerate(BOARD_JOINT_NAMES):
            joint = model.joint(name).id
            if model.geom_type[geoms[slot]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise ValueError("wheel support expects cylinder collision geometry")
            axis = board_rotation.T @ data.xaxis[joint]
            if abs(axis @ rotations[slot, :, 2]) < 1 - 1e-8:
                raise ValueError("wheel hinge must be parallel to its cylinder symmetry axis")
            if not np.allclose(data.xanchor[joint], data.geom_xpos[geoms[slot]], atol=1e-10):
                raise ValueError("wheel hinge must pass through its cylinder centre")
        if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in geoms[len(names):]):
            raise ValueError("board support expects deck and axle boxes")
        return cls(centres.copy(), rotations.copy(), model.geom_size[geoms].copy(), len(names))

    def clearances(self, board_qpos):
        values = np.atleast_2d(board_qpos)
        w, x, y, z = values[:, 3:7].T
        # Third row of the board's world rotation, for scalar-first quaternions.
        up = np.stack((2 * (x*z - w*y), 2 * (y*z + w*x),
                       1 - 2 * (x*x + y*y)), axis=1)
        centre_z = values[:, 2, None] + up @ self.local_centres.T
        directions = np.einsum("ni,gij->ngj", up, self.local_rotations)
        extent = (np.abs(directions) * self.sizes[None]).sum(axis=2)
        axis_z = np.clip(directions[:, :self.wheel_count, 2], -1, 1)
        extent[:, :self.wheel_count] = (
            self.sizes[:self.wheel_count, 0] * np.sqrt(np.maximum(0, 1 - axis_z**2))
            + self.sizes[:self.wheel_count, 1] * np.abs(axis_z)
        )
        return centre_z - extent


@dataclass
class ControlProblem:
    model: mujoco.MjModel
    index: SceneIndex
    qpos: np.ndarray
    qvel: np.ndarray
    nominal: np.ndarray
    joint_limits: np.ndarray
    sensor_slices: dict[str, slice]
    foot_targets: np.ndarray
    ik_report: dict
    leg_ik_tables: tuple | None = None
    catch_priority: bool = False
    board_geometry: BoardCollisionGeometry | None = None
    control_bias: np.ndarray | None = None
    return_seed_stance: bool = False
    free_air_tuck: bool = False


def make_control_problem(
    *, rear_x: float = -0.23, front_x: float = 0.22, timestep: float = 0.0025,
    rear_yaw_degrees: float = 135,
    front_yaw_degrees: float = 90,
    root_y_offset: float = 0.0,
    coordinated_ik: bool = True,
    balance_support_com: bool = False,
    catch_priority: bool = False,
    gravity_seed: bool = False,
    return_seed_stance: bool = False,
    free_air_tuck: bool = False,
) -> ControlProblem:
    """Build named sensors and an initially stationary, contact-aligned wide stance."""
    spec = make_scene_spec(timestep=timestep)
    for foot in ("left", "right"):
        spec.add_sensor(
            name=f"{foot}_foot_board", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{foot}_foot",
            reftype=mujoco.mjtObj.mjOBJ_BODY, refname=BOARD_BODY_NAME,
        )
    for wheel in BOARD_JOINT_NAMES:
        body = wheel.removesuffix("_joint")
        spec.add_sensor(
            name=f"{body}_world", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_BODY, objname=body,
        )
    spec.add_sensor(name="tail_tip_world", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="board_tail_tip")
    spec.add_sensor(name="robot_com_velocity", type=mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="pelvis")
    spec.add_sensor(name="tail_tip_velocity", type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="board_tail_tip")
    spec.add_sensor(name="tail_ground_force", type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM, objname="ground",
                    reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="board_tail",
                    intprm=[1 << int(mujoco.mjtConDataField.mjCONDATA_FORCE), 3, 1])
    spec.add_sensor(name="board_com_velocity", type=mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname=BOARD_BODY_NAME)
    for side in ("left", "right"):
        spec.add_sensor(name=f"{side}_force_on_board", type=mujoco.mjtSensor.mjSENS_CONTACT,
                        objtype=mujoco.mjtObj.mjOBJ_BODY, objname=f"{side}_ankle_roll_link",
                        reftype=mujoco.mjtObj.mjOBJ_BODY, refname=BOARD_BODY_NAME,
                        intprm=[1 << int(mujoco.mjtConDataField.mjCONDATA_FORCE), 3, 1])
    model = spec.compile()
    data = mujoco.MjData(model)
    index = SceneIndex.from_model(model)
    reset_scene(model, data, index)
    original = data.qpos.copy()
    leg_addresses = index.robot_joint_qpos[:12]
    leg_joints = np.array([model.joint(name).id for name in (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    )])
    feet = [model.site(f"{foot}_foot").id for foot in ("left", "right")]
    bodies = [model.body(f"{foot}_ankle_roll_link").id for foot in ("left", "right")]
    target_rotations = [data.xmat[b].reshape(3, 3).copy() for b in bodies]
    target_rotations[0] = Rotation.from_euler("z", rear_yaw_degrees - 90, degrees=True).as_matrix() @ target_rotations[0]
    target_rotations[1] = Rotation.from_euler("z", front_yaw_degrees - 90, degrees=True).as_matrix() @ target_rotations[1]
    target_positions = np.array([data.site_xpos[sid].copy() for sid in feet])
    target_positions[:, 0] = [rear_x, front_x]
    target_positions[:, 1] = -0.0012
    data.qpos[index.robot_qpos] = (rear_x + front_x) / 2
    data.qpos[index.robot_qpos + 1] += root_y_offset
    base_qpos = data.qpos.copy()
    start = np.r_[data.qpos[leg_addresses], data.qpos[index.robot_qpos + 2]]
    lower = np.r_[model.jnt_range[leg_joints, 0] + 0.005, 0.45]
    upper = np.r_[model.jnt_range[leg_joints, 1] - 0.005, 0.98]

    def residual(parameters):
        data.qpos[:] = base_qpos
        data.qpos[leg_addresses] = parameters[:12]
        data.qpos[index.robot_qpos + 2] = parameters[-1]
        mujoco.mj_forward(model, data)
        positions = np.concatenate([data.site_xpos[sid] for sid in feet])
        rotations = np.concatenate([
            Rotation.from_matrix(target.T @ data.xmat[bid].reshape(3, 3)).as_rotvec()
            for target, bid in zip(target_rotations, bodies, strict=True)
        ])
        return np.r_[100 * (positions - target_positions.ravel()),
                     1.5 * rotations, 0.025 * (parameters[:12] - start[:12])]

    solution = least_squares(residual, start, bounds=(lower, upper), max_nfev=160)
    residual(solution.x)
    # A rear flat foot can otherwise overlap the raised kicktail by millimetres.
    # Remove that initialization preload, rather than letting it launch the board.
    for _ in range(5):
        lift = np.zeros(2)
        for contact in data.contact[:data.ncon]:
            names = [model.geom(int(gid)).name for gid in contact.geom]
            if any(name.startswith("board_") for name in names):
                for side, foot in enumerate(("left", "right")):
                    if any(name.startswith(f"{foot}_foot") for name in names):
                        lift[side] = max(lift[side], -float(contact.dist) - 0.0002)
        if lift.max() < 1e-5:
            break
        target_positions[:, 2] += lift
        solution = least_squares(residual, solution.x, bounds=(lower, upper), max_nfev=160)
        residual(solution.x)
    position_error = np.linalg.norm(
        np.array([data.site_xpos[sid] for sid in feet]) - target_positions, axis=1
    )
    if position_error.max() > 0.006:
        raise RuntimeError(f"wide stance IK position error too large: {position_error}")
    nominal = data.qpos[index.robot_joint_qpos].copy()
    data.ctrl[index.robot_actuators] = nominal
    mujoco.mj_forward(model, data)
    joint_ids = np.array([model.joint(i).id for i in range(model.njnt)
                          if model.jnt_qposadr[i] in set(index.robot_joint_qpos.tolist())])
    # The robot joints retain their original G1 order in the compiled model.
    limits = model.jnt_range[joint_ids].copy()
    slices = {}
    for name in ["left_foot_board", "right_foot_board", "tail_tip_world",
                 "robot_com_velocity", "tail_tip_velocity", "tail_ground_force",
                 "board_com_velocity", "left_force_on_board", "right_force_on_board"] + [
        f"{wheel.removesuffix('_joint')}_world" for wheel in BOARD_JOINT_NAMES
    ]:
        sensor = model.sensor(name)
        address = int(model.sensor_adr[sensor.id])
        slices[name] = slice(address, address + int(model.sensor_dim[sensor.id]))
    report = {
        "solver_success": bool(solution.success), "position_error_m": position_error.tolist(),
        "target_feet_world": target_positions.tolist(),
        "actual_feet_world": [data.site_xpos[sid].tolist() for sid in feet],
        "initial_contacts": contact_summary(model, data),
        "robot_root_height_m": float(data.qpos[index.robot_qpos + 2]),
        "original_root_height_m": float(original[index.robot_qpos + 2]),
        "initialization_only": True,
        "rear_foot_yaw_degrees": rear_yaw_degrees,
        "front_foot_yaw_degrees": front_yaw_degrees,
        "root_y_offset_m": root_y_offset,
        "rear_foot_x_m": rear_x,
    }
    foot_targets = np.stack([data.sensordata[slices[f"{foot}_foot_board"]].copy()
                             for foot in ("left", "right")])
    tables = None
    if coordinated_ik:
        # Precompute physically bounded joint targets for vertical support motion.
        # No root edits occur in a rollout; this is only a target-pose generator.
        base = data.qpos.copy()
        fixed_positions = np.array([data.site_xpos[sid].copy() for sid in feet])
        fixed_rotations = [data.xmat[b].reshape(3, 3).copy() for b in bodies]
        robot_body = model.body("pelvis").id
        fixed_com = data.subtree_com[robot_body, :2].copy()
        support_lower = np.r_[lower[:12], base[index.robot_qpos:index.robot_qpos + 2] - 0.12] if balance_support_com else lower[:12]
        support_upper = np.r_[upper[:12], base[index.robot_qpos:index.robot_qpos + 2] + 0.12] if balance_support_com else upper[:12]
        samples = [nominal[:12].copy()]
        errors = []
        for direction, count in ((-1, 18), (1, 7)):
            guess = np.r_[nominal[:12], base[index.robot_qpos:index.robot_qpos + 2]] if balance_support_com else nominal[:12].copy()
            for step in range(1, count + 1):
                height = base[index.robot_qpos + 2] + direction * step * 0.01

                def support_residual(joints, support_height=height):
                    data.qpos[:] = base
                    data.qpos[index.robot_qpos + 2] = support_height
                    data.qpos[leg_addresses] = joints[:12]
                    if balance_support_com:
                        data.qpos[index.robot_qpos:index.robot_qpos + 2] = joints[-2:]
                    mujoco.mj_forward(model, data)
                    pos = np.concatenate([data.site_xpos[sid] for sid in feet])
                    rot = np.concatenate([
                        Rotation.from_matrix(target.T @ data.xmat[bid].reshape(3, 3)).as_rotvec()
                        for target, bid in zip(fixed_rotations, bodies, strict=True)
                    ])
                    com = 50 * (data.subtree_com[robot_body, :2] - fixed_com) if balance_support_com else []
                    return np.r_[100 * (pos - fixed_positions.ravel()), 1.5 * rot, com,
                                 0.005 * (joints[:12] - nominal[:12])]

                fitted = least_squares(support_residual, guess,
                                       bounds=(support_lower, support_upper), max_nfev=90)
                residuals = support_residual(fitted.x)
                pos_error = float(np.linalg.norm(residuals[:6].reshape(2, 3), axis=1).max() / 100)
                rot_error = float(np.linalg.norm(residuals[6:12].reshape(2, 3), axis=1).max() / 1.5)
                if pos_error > 0.001 or rot_error > 0.03:
                    break
                samples.append(fitted.x[:12].copy())
                errors.append((pos_error, rot_error))
                guess = fitted.x
        data.qpos[:] = base
        data.ctrl[index.robot_actuators] = nominal
        mujoco.mj_forward(model, data)
        samples = np.asarray(samples)
        tables = []
        for start in (0, 6):
            keys = samples[:, start + 3] - nominal[start + 3]
            order = np.argsort(keys)
            keep = np.r_[True, np.diff(keys[order]) > 1e-6]
            tables.append((keys[order][keep], samples[order][keep, start:start + 6]))
        tables = tuple(tables)
        report["coordinated_leg_ik"] = {
            "samples": len(samples), "maximum_foot_position_error_m": max((e[0] for e in errors), default=0),
            "maximum_foot_rotation_error_rad": max((e[1] for e in errors), default=0),
            "knee_delta_ranges": [[float(keys[0]), float(keys[-1])] for keys, _ in tables],
            "target_generation_only": True,
            "preserve_support_com_xy": balance_support_com,
        }
    geometry = BoardCollisionGeometry.from_model(model, index, data.qpos)
    report["free_air_tuck"] = {
        "enabled": free_air_tuck, "hip_delta_per_knee_delta": -1.0,
        "outward_hip_roll_per_knee_delta": 0.3,
        "ankle_targets": "initial stance joint angles; no stance orientation constraint",
        "arm_clearance": {"shoulder_pitch_rad": -0.4,
                          "shoulder_roll_rad": [0.65, -0.65], "elbow_rad": 1.4},
        "stance_crouch_and_drive_leg_targets_unchanged": True,
    }
    control_bias = None
    if return_seed_stance and not gravity_seed:
        raise ValueError("return_seed_stance requires gravity_seed")
    if gravity_seed:
        solver = ContactStatics(model, index)
        solution = solver.solve(data)
        if not solution.success:
            raise RuntimeError(f"initial gravity seed failed: {solution.diagnostics()}")
        seeded, raw_target = solver.pd_target(nominal, np.zeros(29), solution)
        control_bias = seeded - nominal
        report["quasistatic_gravity_seed"] = {
            **solution.diagnostics(), "bias_rad": control_bias.tolist(),
            "clipped_targets": int(np.count_nonzero(raw_target != seeded)),
            "assumed_contacts_not_applied_forces": True,
        }
        report["return_seed_stance"] = return_seed_stance
    return ControlProblem(model, index, data.qpos.copy(), data.qvel.copy(),
                          nominal, limits, slices, foot_targets, report, tables, catch_priority,
                          geometry, control_bias, return_seed_stance, free_air_tuck)


def control_trajectory(problem: ControlProblem, parameters, seconds: float = 2.2) -> np.ndarray:
    """Smooth joint targets for settle, crouch, rear-foot pop, tuck and landing."""
    raw = np.asarray(parameters, dtype=float)
    if raw.shape in ((13,), (16,), (17,), (18,)):
        raw = np.r_[raw, DEFAULT[len(raw):]]
    if raw.shape != DEFAULT.shape:
        raise ValueError(f"expected 13, 16, 17, 18 or {len(DEFAULT)} finite control parameters")
    if not np.isfinite(raw).all():
        raise ValueError("control parameters must be finite")
    p = np.clip(raw, LOWER, UPPER)
    (crouch, extension, front_tuck, pop_time, delay, air_tuck, air_time,
     ankle_roll, ankle_pitch, hip_roll, front_extension, drive_time, rear_delay,
     rear_tuck, rear_tuck_time, front_press_time, ankle_pop_lead, front_pop_lead,
     rear_leg_pop_lead) = p
    dt = problem.model.opt.timestep
    substeps = round(CONTROL_DT / dt)
    if not np.isclose(substeps * dt, CONTROL_DT):
        raise ValueError("physics timestep must divide the 50 Hz control interval")
    time = (np.arange(round(seconds / dt)) // substeps) * CONTROL_DT
    target = np.repeat(problem.nominal[None], len(time), axis=0)

    def leg_pose(knee_delta, side, roll_delta=0.0):
        pose = problem.nominal.copy()
        start = 0 if side == "left" else 6
        if problem.leg_ik_tables is None:
            pose[start] -= 0.50 * knee_delta
            pose[start + 3] += knee_delta
            pose[start + 4] -= 0.50 * knee_delta
        else:
            keys, values = problem.leg_ik_tables[0 if side == "left" else 1]
            pose[start:start + 6] = [np.interp(knee_delta, keys, values[:, j]) for j in range(6)]
        pose[start + 5] += roll_delta
        return pose

    def air_pose(knee_delta, side):
        if not problem.free_air_tuck:
            return leg_pose(knee_delta, side)
        pose = problem.nominal.copy()
        start = 0 if side == "left" else 6
        pose[start] -= knee_delta
        pose[start + 1] += (1 if side == "left" else -1) * 0.3 * knee_delta
        pose[start + 3] += knee_delta
        return np.clip(pose, problem.joint_limits[:, 0] + 0.005,
                       problem.joint_limits[:, 1] - 0.005)

    def curve(times, values):
        # Cubic smoothstep interpolation has zero target velocity at each knot.
        segment = np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2)
        alpha = np.clip((time - np.array(times)[segment]) /
                        (np.array(times)[segment + 1] - np.array(times)[segment]), 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        return np.array(values)[segment] * (1 - alpha[:, None]) + np.array(values)[segment + 1] * alpha[:, None]

    crouch_end = 0.42
    drive_end = crouch_end + drive_time
    pop_end = drive_end + pop_time
    air_end = pop_end + max(delay, rear_delay) + air_time
    for side, offset in (("left", rear_delay), ("right", delay)):
        pop_peak = pop_end + offset
        if side == "right":
            pop_peak = max(drive_end + 0.02, pop_peak - front_pop_lead)
        else:
            pop_peak = max(drive_end + 0.02, pop_peak - rear_leg_pop_lead)
        crouch_pose = leg_pose(crouch, side)
        drive_pose = leg_pose(-extension if side == "left" else -front_extension, side)
        pop_pose = leg_pose(-extension if side == "left" else front_tuck, side,
                            ankle_roll if side == "left" else 0.0)
        if side == "right" and problem.free_air_tuck:
            pop_pose = air_pose(front_tuck, side)
        if side == "left":
            pop_pose[4] += ankle_pitch
            pop_pose[1] += hip_roll
        tucked_pose = air_pose(rear_tuck if side == "left" else air_tuck, side)
        land_pose = problem.nominal if problem.return_seed_stance else leg_pose(0.12, side)
        tuck_time = pop_peak + (rear_tuck_time if side == "left" else front_press_time)
        times = [0.0, 0.10, crouch_end, drive_end, pop_peak, tuck_time]
        poses = [problem.nominal, problem.nominal, crouch_pose, drive_pose, pop_pose, tucked_pose]
        if tuck_time < air_end - 1e-9:
            times.append(air_end)
            poses.append(tucked_pose)
        path = curve(times + [air_end + 0.30, seconds], poses + [land_pose, land_pose])
        start = 0 if side == "left" else 6
        target[:, start:start + 6] = path[:, start:start + 6]
        if side == "left" and ankle_pop_lead > 0:
            # Move only the rear ankle pulse earlier, overlapping the bilateral
            # drive. The leg extension/tuck trajectory is not reset or moved.
            pulse = [np.zeros(2), np.zeros(2), np.array([ankle_pitch, ankle_roll]),
                     np.zeros(2), np.zeros(2)]
            old_times = [0.0, drive_end, pop_peak, tuck_time, seconds]
            early_times = [0.0, drive_end - ankle_pop_lead,
                           pop_peak - ankle_pop_lead,
                           tuck_time - ankle_pop_lead, seconds]
            target[:, [4, 5]] += curve(early_times, pulse) - curve(old_times, pulse)
    if problem.free_air_tuck:
        # Prepare arm clearance before free tuck: static FK at d=.8..1.5 showed
        # wrist/thigh intersections with the original hanging-arm posture.
        # This is still only bounded joint targets through the original motors.
        air_arms = problem.nominal.copy()
        air_arms[[15, 22]] = -0.4
        air_arms[[16, 23]] = [0.65, -0.65]
        air_arms[[18, 25]] = 1.4
        arm_path = curve([0.0, 0.10, crouch_end, air_end, air_end + 0.30, seconds],
                         [problem.nominal, problem.nominal, air_arms, air_arms,
                          problem.nominal, problem.nominal])
        target[:, 15:] = arm_path[:, 15:]
    if problem.control_bias is not None:
        target += problem.control_bias
    target = np.clip(target, problem.joint_limits[:, 0] + 0.005,
                     problem.joint_limits[:, 1] - 0.005)
    control = np.zeros((len(time), problem.model.nu))
    control[:, problem.index.robot_actuators] = target
    return control


def initial_state(problem):
    data = mujoco.MjData(problem.model)
    data.qpos[:] = problem.qpos
    data.qvel[:] = problem.qvel
    mujoco.mj_forward(problem.model, data)
    state = np.empty(mujoco.mj_stateSize(problem.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(problem.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return state


def balance_callback(problem: ControlProblem, gains: BalanceGains | None = None):
    """50 Hz held ankle feedback; positive projected tilt corrects supported root tilt.

    The sign is backed by the independent +/- perturbation probe recorded in
    docs/14: positive targets improve 0.6 s deviations, with a nonmonotone 0.3 s
    transient. This is still an untuned engineering controller, not a stable actor.
    """
    index = problem.index
    gains = gains or BalanceGains()
    ankles = index.robot_actuators[[4, 5, 10, 11]]
    ankle_joints = problem.model.actuator_trnid[ankles, 0]
    root_body = problem.model.body("pelvis").id
    support_contact = FootBoardSupport(problem.model)
    support_offset = problem.foot_targets.mean(axis=0)
    initial_quat = problem.qpos[index.robot_qpos + 3:index.robot_qpos + 7]
    yaw = 2 * np.arctan2(initial_quat[3], initial_quat[0])
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
    cache = {}

    def callback(model, data):
        key = data.qpos.ctypes.data
        tick = int(np.floor((data.time + 1e-9) / CONTROL_DT))
        old = cache.get(key)
        if old is not None and old[0] == tick and data.time >= old[1] - 1e-9:
            data.ctrl[ankles] += old[2]
            data.ctrl[index.robot_actuators] = np.clip(
                data.ctrl[index.robot_actuators], problem.joint_limits[:, 0] + 0.005,
                problem.joint_limits[:, 1] - 0.005,
            )
            return
        qw, qx, qy, qz = data.qpos[index.robot_qpos + 3:index.robot_qpos + 7]
        w, x, y, z = c * qw + s * qz, c * qx + s * qy, c * qy - s * qx, c * qz - s * qw
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        angular_world = data.xmat[root_body].reshape(3, 3) @ data.qvel[index.robot_qvel + 3:index.robot_qvel + 6]
        angular = np.array([angular_world[:2] @ forward, angular_world[:2] @ lateral])
        loads = support_contact.vertical_loads(data)
        if not (loads > 1.0).any():
            cache[key] = (tick, float(data.time), np.zeros(4))
            return
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jacobian, root_body)
        velocity = (jacobian @ data.qvel)[:2]
        support = (data.xpos[index.board_body]
                   + data.xmat[index.board_body].reshape(3, 3) @ support_offset)[:2]
        com_error = data.subtree_com[root_body, :2] - support
        pitch_correction = np.clip(
            1.4 * gains.tilt_scale * pitch + 0.14 * gains.rate_scale * angular[1]
            + gains.com_position_gain * (com_error @ forward)
            + gains.com_velocity_gain * (velocity @ forward), -0.35, 0.35
        )
        roll_correction = np.clip(
            0.8 * gains.tilt_scale * roll + 0.10 * gains.rate_scale * angular[0]
            - gains.com_position_gain * (com_error @ lateral)
            - gains.com_velocity_gain * (velocity @ lateral), -0.15, 0.15
        )
        # The rear foot is open relative to the torso. Project torso tilt onto
        # each actual world-space ankle axis instead of applying identical
        # pitch/roll commands to differently oriented feet.
        tilt_world = np.r_[forward * roll_correction + lateral * pitch_correction, 0.0]
        corrections = np.clip(data.xaxis[ankle_joints] @ tilt_world,
                              [-0.35, -0.15, -0.35, -0.15], [0.35, 0.15, 0.35, 0.15])
        corrections *= np.repeat(loads > 1.0, 2)
        cache[key] = (tick, float(data.time), corrections)
        data.ctrl[ankles] += corrections
        data.ctrl[index.robot_actuators] = np.clip(
            data.ctrl[index.robot_actuators], problem.joint_limits[:, 0] + 0.005,
            problem.joint_limits[:, 1] - 0.005,
        )

    return callback


def placement_callback(problem: ControlProblem):
    """Bounded horizontal foot-placement feedback via real joint Jacobians.

    Vertical pop/tuck targets are retained. This is an optional full-state
    engineering controller, not a learned actor or a force applied to the board.
    """
    model, index = problem.model, problem.index
    board = model.body(BOARD_BODY_NAME).id
    feet = [model.site(f"{side}_foot").id for side in ("left", "right")]
    board_geoms = {g for g in range(model.ngeom) if model.geom(g).name.startswith(("board_", "wheel_"))}
    foot_geoms = [{g for g in range(model.ngeom) if model.geom(g).name.startswith(f"{side}_foot")}
                  for side in ("left", "right")]
    cache = {}

    def callback(model, data):
        key = data.qpos.ctypes.data
        tick = int(np.floor((data.time + 1e-9) / CONTROL_DT))
        old = cache.get(key)
        if old is not None and old[0] == tick and data.time >= old[1] - 1e-9:
            correction = old[2]
        else:
            fresh = old is None or data.time < old[1] - 1e-9
            correction = np.zeros(29) if fresh else old[2].copy()
            seen_air = np.zeros(2, dtype=bool) if fresh else old[3].copy()
            board_rotation = data.xmat[board].reshape(3, 3)
            touching = [False, False]
            for contact in data.contact[:data.ncon]:
                a, b = map(int, contact.geom)
                for side in (0, 1):
                    if (a in board_geoms and b in foot_geoms[side]) or (b in board_geoms and a in foot_geoms[side]):
                        touching[side] = True
            for side, foot in enumerate(feet):
                relative = board_rotation.T @ (data.site_xpos[foot] - data.xpos[board])
                if touching[side] or relative[2] - problem.foot_targets[side, 2] < 0.015:
                    # After the airborne correction, hold its last value on
                    # touchdown. Updating a fixed-board Jacobian while standing
                    # on a free board would otherwise push the support away.
                    continue
                if data.qpos[index.robot_qpos + 2] < 0.55:
                    continue
                seen_air[side] = True
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacobian, None, foot)
                start = side * 6
                columns = index.robot_joint_qvel[start:start + 6]
                horizontal = jacobian[:2, columns]
                desired = data.xpos[board] + board_rotation @ problem.foot_targets[side]
                board_velocity = data.qvel[index.board_qvel:index.board_qvel + 3]
                angular = data.qvel[index.board_qvel + 3:index.board_qvel + 6]
                desired_velocity = board_velocity + board_rotation @ np.cross(angular, problem.foot_targets[side])
                velocity = jacobian @ data.qvel
                error = (desired - data.site_xpos[foot])[:2] + 0.06 * (desired_velocity - velocity)[:2]
                delta = horizontal.T @ np.linalg.solve(horizontal @ horizontal.T + 0.001 * np.eye(2), error)
                correction[start:start + 6] = np.clip(0.75 * delta, -0.30, 0.30)
            cache[key] = (tick, float(data.time), correction, seen_air)
        data.ctrl[index.robot_actuators] = np.clip(
            data.ctrl[index.robot_actuators] + correction,
            problem.joint_limits[:, 0] + 0.005, problem.joint_limits[:, 1] - 0.005,
        )

    return callback


def feedback_callback(problem: ControlProblem, *, balance=False, placement=False, balance_gains=None):
    callbacks = ([balance_callback(problem, balance_gains)] if balance else []) + ([placement_callback(problem)] if placement else [])
    if not callbacks:
        return None

    def callback(model, data):
        for function in callbacks:
            function(model, data)

    return callback


def score_rollout(problem: ControlProblem, states, sensors, *, sensor_stage_offset_s=0.0) -> tuple[float, dict]:
    """Reward a bounded board jump and catch, rather than simply throwing the board."""
    model, index = problem.model, problem.index
    if (not np.isfinite(states).all() or not np.isfinite(sensors).all()
            or not np.all(np.diff(states[:, 0]) > 0)):
        return -1000.0, {"score": -1000.0, "catch_search_eligible": False,
                         "numerical_failure": "nonfinite state/sensor or nonmonotonic time"}
    sample_dt = float(np.median(np.diff(states[:, 0]))) if len(states) > 1 else CONTROL_DT
    q = states[:, 1:1 + model.nq]
    board_pos = q[:, index.board_qpos:index.board_qpos + 3]
    robot_pos = q[:, index.robot_qpos:index.robot_qpos + 3]
    robot_quat = q[:, index.robot_qpos + 3:index.robot_qpos + 7]
    upright = 1 - 2 * (robot_quat[:, 1]**2 + robot_quat[:, 2]**2)
    board_quat = q[:, index.board_qpos + 3:index.board_qpos + 7]
    board_upright = 1 - 2 * (board_quat[:, 1]**2 + board_quat[:, 2]**2)
    min_board_upright = float(board_upright.min())
    actual_joints = q[:, index.robot_joint_qpos]
    excess = np.maximum(problem.joint_limits[:, 0] - actual_joints,
                        actual_joints - problem.joint_limits[:, 1]).clip(0)
    maximum_actual_joint_violation = float(excess.max())
    geometry = problem.board_geometry or BoardCollisionGeometry.from_model(model, index, problem.qpos)
    collision_heights = geometry.clearances(q[:, index.board_qpos:index.board_qpos + 7])
    wheel_height = collision_heights[:, :geometry.wheel_count]
    minimum_wheel_height = wheel_height.min(axis=1)
    minimum_board_height = collision_heights.min(axis=1)
    left = sensors[:, problem.sensor_slices["left_foot_board"]]
    right = sensors[:, problem.sensor_slices["right_foot_board"]]
    feet = np.stack([left, right], axis=1)
    final = slice(-max(round(0.25 / sample_dt), 1), None)
    foot_error = np.linalg.norm(feet[:, :, :2] - problem.foot_targets[:, :2], axis=2)
    height_error = np.abs(feet[:, :, 2] - problem.foot_targets[:, 2])
    landing_error = float(np.mean(foot_error[final]) + np.mean(height_error[final]))
    board_escape = float(np.max(np.linalg.norm(board_pos[:, :2] - robot_pos[:, :2], axis=1)))
    clearance = float(max(minimum_wheel_height.max(), 0))
    edges = np.diff(np.r_[False, minimum_board_height > 0.005, False].astype(np.int8))
    windows = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    geometric_air_seconds = float(windows.max(initial=0) * sample_dt)
    root_peak = float(robot_pos[:, 2].max() - problem.qpos[index.robot_qpos + 2])
    crouch_end = max(1, round(0.42 / sample_dt))
    takeoff_end = max(crouch_end + 1, round(1.0 / sample_dt))
    crouch_height = float(robot_pos[:crouch_end, 2].min())
    extension_rise = float(max(robot_pos[crouch_end:takeoff_end, 2].max() - crouch_height, 0))
    # A nose-up board is a dense stepping stone; it does not count as a completed pop.
    front_height = wheel_height[:, :2].mean(axis=1)
    rear_height = wheel_height[:, 2:].mean(axis=1)
    nose_rise = float(max((front_height - rear_height).max(), 0))
    tail_tip_height = float(sensors[crouch_end:takeoff_end, problem.sensor_slices["tail_tip_world"]][:, 2].min())
    tail_approach = float(np.clip((0.060 - tail_tip_height) / 0.055, 0, 1))
    # Include all deck/axle box corners and oriented wheel cylinder extents.
    # Actual contact-free flight is certified separately at every physics step.
    flight_end = min(len(states), round(1.35 / sample_dt))
    board_clearance = float(max(minimum_board_height[crouch_end:flight_end].max(), 0))
    min_upright = float(upright.min())
    upward_speed = float(max(states[crouch_end:takeoff_end, 1 + model.nq + index.robot_qvel + 2].max(), 0))
    final_root_height = float(robot_pos[final, 2].mean())
    survival = float(min(np.clip((min_upright - 0.4) / 0.4, 0, 1),
                         np.clip((final_root_height - 0.3) / 0.35, 0, 1),
                         np.clip((min_board_upright - 0.4) / 0.35, 0, 1)))
    pop_gate = float(np.clip((0.030 - tail_tip_height) / 0.020, 0, 1))
    impact = {}
    dynamics_reward = 0.6 * min(upward_speed, 2.0)
    com_slice = problem.sensor_slices["robot_com_velocity"]
    tail_force_slice = problem.sensor_slices["tail_ground_force"]
    if sensors.shape[1] >= tail_force_slice.stop:
        com_velocity = sensors[:, com_slice]
        tail_velocity = sensors[:, problem.sensor_slices["tail_tip_velocity"]]
        tail_force = sensors[:, tail_force_slice][:, 2]
        dynamics_time = states[:, 0] - sensor_stage_offset_s
        pop_window = (dynamics_time >= 0.35) & (dynamics_time <= 1.05)
        valid = np.flatnonzero(pop_window)
        peak = int(valid[np.argmax(com_velocity[valid, 2])])
        impact = {"maximum_pop_com_upward_speed_m_s": float(com_velocity[peak, 2]),
                  "maximum_pop_com_upward_speed_time_s": float(dynamics_time[peak]),
                  "first_tail_force_time_s": None,
                  "com_upward_speed_at_tail_m_s": None,
                  "com_horizontal_speed_at_tail_m_s": None}
        dynamics_reward = 0.6 * min(max(float(com_velocity[peak, 2]), 0), 2.0)
        tail_events = np.flatnonzero(pop_window & (tail_force > 0.1))
        if len(tail_events):
            first = int(tail_events[0])
            t0 = dynamics_time[first]
            preceding = (dynamics_time >= t0 - 0.020 - 1e-9) & (dynamics_time < t0 + 1e-9)
            downward = max(float(-tail_velocity[preceding, 2].min()), 0)
            upward = float(com_velocity[first, 2])
            horizontal = float(np.linalg.norm(com_velocity[first, :2]))
            impact.update(first_tail_force_time_s=float(t0),
                          com_upward_speed_at_tail_m_s=upward,
                          com_horizontal_speed_at_tail_m_s=horizontal,
                          tail_pre20ms_downward_speed_m_s=downward,
                          tail_delay_after_com_peak_s=float(t0 - dynamics_time[peak]))
            # A root-speed peak alone can precede tail impact by 170ms. Reward
            # the actual COM momentum present at a real force-producing impact.
            dynamics_reward += (2.5 * min(max(upward, 0), 1.2)
                                - 1.5 * min(horizontal, 1.0) + 0.4 * min(downward, 1.5))
            last_force = problem.sensor_slices["right_force_on_board"]
            if sensors.shape[1] >= last_force.stop:
                post = (dynamics_time >= t0 - 1e-9) & (dynamics_time < t0 + 0.04 - 1e-9)
                rear_force = sensors[:, problem.sensor_slices["left_force_on_board"]][:, 2]
                front_force = sensors[:, last_force][:, 2]
                rear_impulse = float(np.maximum(-rear_force[post], 0).sum() * sample_dt)
                front_impulse = float(np.maximum(-front_force[post], 0).sum() * sample_dt)
                after = int(np.argmin(np.abs(dynamics_time - (t0 + 0.04))))
                board_up = float(sensors[after, problem.sensor_slices["board_com_velocity"]][2])
                impact.update(rear_downward_impulse_on_board_after_tail_40ms_n_s=rear_impulse,
                              front_downward_impulse_on_board_after_tail_40ms_n_s=front_impulse,
                              board_com_upward_speed_after_tail_40ms_m_s=board_up)
                dynamics_reward += (1.5 * min(max(board_up, 0), 1.0)
                                    - 0.4 * rear_impulse - 0.2 * front_impulse)
    score = (
        survival * (20 * min(clearance, 0.25) + 2 * min(geometric_air_seconds, 0.45)
                    + 40 * min(board_clearance, 0.15) * pop_gate)
        + 2 * min(max(root_peak, 0), 0.35) - 5 * landing_error
        + 3 * min(extension_rise, 0.30) + 2 * min(nose_rise, 0.20)
        + dynamics_reward
        + 1.5 * tail_approach
        - 2 * max(board_escape - 0.35, 0) - 3 * max(0.6 - min_upright, 0)
        - 3 * max(0.60 - final_root_height, 0)
        - 4 * max(0.70 - min_board_upright, 0)
    )
    catch_eligible = board_clearance >= 0.008 and pop_gate >= 0.8 and survival >= 0.85
    if problem.catch_priority:
        # This is only a geometric search gate; actual contacts and continuous
        # flight are audited in the final, full-resolution native rollout.
        vector = 2 * np.cross(board_quat[:, None, 1:], feet)
        feet_world = (board_pos[:, None] + feet + board_quat[:, None, :1] * vector
                      + np.cross(board_quat[:, None, 1:], vector))
        floor_proxy = float(np.clip((0.040 - feet_world[final, :, 2].mean()) / 0.04, 0, 1))
        score = ((100.0 if catch_eligible else -100.0) + 2 * min(extension_rise, 0.25)
                 - 20 * landing_error - 8 * floor_proxy
                 - 4 * max(board_escape - 0.25, 0) - 4 * max(0.8 - min_upright, 0))
    score -= 30 * max(maximum_actual_joint_violation - 0.02, 0)
    return score, {
        "score": float(score), "maximum_all_wheel_clearance_m": clearance,
        "whole_board_geometric_clearance_above_5mm_longest_s": geometric_air_seconds,
        "robot_root_rise_m": root_peak, "minimum_upright_cosine": min_upright,
        "minimum_board_upright_cosine": min_board_upright,
        "landing_foot_error_m": landing_error, "maximum_board_robot_xy_distance_m": board_escape,
        "final_robot_root_height_m": final_root_height,
        "root_extension_from_crouch_m": extension_rise,
        "maximum_nose_above_rear_wheels_m": nose_rise,
        "maximum_pop_root_upward_speed_mps": upward_speed,
        "minimum_pop_tail_tip_height_m": tail_tip_height,
        "maximum_whole_board_clearance_m": float(minimum_board_height.max()),
        "maximum_early_whole_board_clearance_m": board_clearance,
        "search_metrics_certify_contact_free_flight": False,
        "survival_reward_gate": survival,
        "catch_search_eligible": bool(catch_eligible),
        "search_observation_dt": sample_dt,
        "maximum_actual_joint_limit_violation_rad": maximum_actual_joint_violation,
        "dynamics_sensor_stage_offset_s": sensor_stage_offset_s,
        **impact,
    }


def feedback_rollout(problem, parameters, *, seconds=2.2, balance=False, placement=False, balance_gains=None):
    """Continuous simulation with feedback sampled after resolved constraints.

    This deliberately does not use mjcb_control: that callback precedes the
    current constraint solve, so current contacts and previous forces can differ.
    The caller must leave the global control callback unset.
    """
    model, index = problem.model, problem.index
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = problem.qpos, problem.qvel
    mujoco.mj_forward(model, data)
    callback = feedback_callback(problem, balance=balance, placement=placement, balance_gains=balance_gains)
    substeps = round(CONTROL_DT / model.opt.timestep)
    nominal = control_trajectory(problem, parameters, seconds)[::substeps]
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states, sensors, actual_controls = [], [], []
    for target in nominal:
        data.ctrl[:] = target
        if callback:
            callback(model, data)
        actual_controls.append(data.ctrl[index.robot_actuators].copy())
        mujoco.mj_step(model, data, nstep=substeps)
        mujoco.mj_forward(model, data)
        state = np.empty(state_size)
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        states.append(state)
        sensors.append(data.sensordata.copy())
    return np.asarray(states), np.asarray(sensors), np.asarray(actual_controls)


def evaluate_batch(problem, parameters, *, seconds=2.2, threads=4, balance=False, placement=False, balance_gains=None):
    previous = mujoco.get_mjcb_control()
    try:
        mujoco.set_mjcb_control(None)
        if balance or placement:
            def evaluate_one(parameters):
                state, sensor, _ = feedback_rollout(
                    problem, parameters, seconds=seconds, balance=balance, placement=placement,
                    balance_gains=balance_gains,
                )
                return score_rollout(problem, state, sensor)

            with ThreadPoolExecutor(max_workers=threads) as executor:
                return list(executor.map(evaluate_one, parameters))
        controls = np.stack([control_trajectory(problem, p, seconds) for p in parameters])
        start = initial_state(problem)[None]
        states, sensors = rollout.rollout(
            problem.model, [mujoco.MjData(problem.model) for _ in range(threads)],
            start, controls, persistent_pool=True,
        )
        # mj_step integrates qpos/qvel after evaluating position sensors. Thus
        # native rollout's next sensor sample belongs to the preceding state.
        # Shift only FramePos samples. Velocity and contact-force sensors remain
        # paired at the pre-integration state where their actual force was used;
        # their timestamp is reported with a one-physics-step offset below.
        aligned = sensors.copy()
        for episode, state in enumerate(states):
            if not np.isfinite(state).all() or not np.all(np.diff(state[:, 0]) > 0):
                continue
            final_data = mujoco.MjData(problem.model)
            final_data.qpos[:] = state[-1, 1:1 + problem.model.nq]
            final_data.qvel[:] = state[-1, 1 + problem.model.nq:1 + problem.model.nq + problem.model.nv]
            mujoco.mj_forward(problem.model, final_data)
            for name, sensor_slice in problem.sensor_slices.items():
                if problem.model.sensor_type[problem.model.sensor(name).id] != mujoco.mjtSensor.mjSENS_FRAMEPOS:
                    continue
                aligned[episode, :-1, sensor_slice] = sensors[episode, 1:, sensor_slice]
                aligned[episode, -1, sensor_slice] = final_data.sensordata[sensor_slice]
        sensors = aligned
    finally:
        mujoco.set_mjcb_control(previous)
    return [score_rollout(problem, state, sensor, sensor_stage_offset_s=problem.model.opt.timestep)
            for state, sensor in zip(states, sensors, strict=True)]
