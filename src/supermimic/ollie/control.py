"""Native MuJoCo PD trajectory search: a physical feasibility baseline, not SuperMimic.

Initial joint/root placement is solved once. During every rollout only the
robot's 29 position targets are supplied; the board is always passive.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import rollout
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from supermimic.ollie.physics import (
    BOARD_BODY_NAME,
    BOARD_JOINT_NAMES,
    WHEEL_RADIUS,
    SceneIndex,
    contact_summary,
    make_scene_spec,
    reset_scene,
)

PARAMETER_NAMES = (
    "crouch_delta", "rear_extension", "front_tuck", "pop_seconds",
    "front_delay_seconds", "air_tuck", "air_seconds", "rear_ankle_roll",
    "rear_ankle_pitch", "rear_hip_roll",
    "front_extension", "drive_seconds", "rear_pop_delay_seconds",
    "rear_tuck_delta", "rear_tuck_seconds", "front_press_seconds",
)
LOWER = np.array([0.20, 0.10, 0.20, 0.06, -0.04, 0.20, 0.18, -0.15, -0.65, -0.25, 0.10, 0.03, 0.0, 0.40, 0.02, 0.03])
UPPER = np.array([1.10, 0.70, 1.30, 0.20, 0.14, 1.20, 0.55, 0.15, 0.65, 0.25, 0.70, 0.10, 0.10, 1.50, 0.14, 0.18])
DEFAULT = np.array([0.55, 0.30, 0.70, 0.12, 0.06, 0.65, 0.32, 0.0, 0.0, 0.0, 0.35, 0.05, 0.03, 1.0, 0.06, 0.10])
CONTROL_DT = 0.02


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


def make_control_problem(
    *, rear_x: float = -0.23, front_x: float = 0.22, timestep: float = 0.0025,
    rear_yaw_degrees: float = 135,
    front_yaw_degrees: float = 90,
    root_y_offset: float = 0.0,
    coordinated_ik: bool = True,
    balance_support_com: bool = False,
    catch_priority: bool = False,
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
    for name in ["left_foot_board", "right_foot_board", "tail_tip_world"] + [
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
    return ControlProblem(model, index, data.qpos.copy(), data.qvel.copy(),
                          nominal, limits, slices, foot_targets, report, tables, catch_priority)


def control_trajectory(problem: ControlProblem, parameters, seconds: float = 2.2) -> np.ndarray:
    """Smooth joint targets for settle, crouch, rear-foot pop, tuck and landing."""
    raw = np.asarray(parameters, dtype=float)
    if raw.shape == (13,):
        raw = np.r_[raw, DEFAULT[13:]]
    p = np.clip(raw, LOWER, UPPER)
    (crouch, extension, front_tuck, pop_time, delay, air_tuck, air_time,
     ankle_roll, ankle_pitch, hip_roll, front_extension, drive_time, rear_delay,
     rear_tuck, rear_tuck_time, front_press_time) = p
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
        crouch_pose = leg_pose(crouch, side)
        drive_pose = leg_pose(-extension if side == "left" else -front_extension, side)
        pop_pose = leg_pose(-extension if side == "left" else front_tuck, side,
                            ankle_roll if side == "left" else 0.0)
        if side == "left":
            pop_pose[4] += ankle_pitch
            pop_pose[1] += hip_roll
        tucked_pose = leg_pose(rear_tuck if side == "left" else air_tuck, side)
        land_pose = leg_pose(0.12, side)
        tuck_time = pop_end + offset + (rear_tuck_time if side == "left" else front_press_time)
        times = [0.0, 0.10, crouch_end, drive_end, pop_end + offset, tuck_time]
        poses = [problem.nominal, problem.nominal, crouch_pose, drive_pose, pop_pose, tucked_pose]
        if tuck_time < air_end - 1e-9:
            times.append(air_end)
            poses.append(tucked_pose)
        path = curve(times + [air_end + 0.30, seconds], poses + [land_pose, land_pose])
        start = 0 if side == "left" else 6
        target[:, start:start + 6] = path[:, start:start + 6]
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


def balance_callback(problem: ControlProblem):
    """50 Hz held ankle feedback; only ctrl changes, no root forces or state edits."""
    index = problem.index
    ankles = index.robot_actuators[[4, 5, 10, 11]]
    ankle_joints = problem.model.actuator_trnid[ankles, 0]
    root_body = problem.model.body("pelvis").id
    foot_sites = [problem.model.site(f"{side}_foot").id for side in ("left", "right")]
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
            data.ctrl[ankles] -= old[2]
            data.ctrl[index.robot_actuators] = np.clip(
                data.ctrl[index.robot_actuators], problem.joint_limits[:, 0] + 0.005,
                problem.joint_limits[:, 1] - 0.005,
            )
            return
        qw, qx, qy, qz = data.qpos[index.robot_qpos + 3:index.robot_qpos + 7]
        w, x, y, z = c * qw + s * qz, c * qx + s * qy, c * qy - s * qx, c * qz - s * qw
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        angular = data.qvel[index.robot_qvel + 3:index.robot_qvel + 6]
        velocity = data.qvel[index.robot_qvel:index.robot_qvel + 2]
        support = data.site_xpos[foot_sites, :2].mean(axis=0)
        com_error = data.subtree_com[root_body, :2] - support
        pitch_correction = np.clip(
            1.4 * pitch + 0.14 * angular[1] + 1.2 * (com_error @ forward)
            + 0.16 * (velocity @ forward), -0.35, 0.35
        )
        roll_correction = np.clip(
            0.8 * roll + 0.10 * angular[0] - 0.7 * (com_error @ lateral)
            - 0.10 * (velocity @ lateral), -0.15, 0.15
        )
        # The rear foot is open relative to the torso. Project torso tilt onto
        # each actual world-space ankle axis instead of applying identical
        # pitch/roll commands to differently oriented feet.
        tilt_world = np.r_[forward * roll_correction + lateral * pitch_correction, 0.0]
        corrections = np.clip(data.xaxis[ankle_joints] @ tilt_world,
                              [-0.35, -0.15, -0.35, -0.15], [0.35, 0.15, 0.35, 0.15])
        cache[key] = (tick, float(data.time), corrections)
        data.ctrl[ankles] -= corrections
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


def feedback_callback(problem: ControlProblem, *, balance=False, placement=False):
    callbacks = ([balance_callback(problem)] if balance else []) + ([placement_callback(problem)] if placement else [])
    if not callbacks:
        return None

    def callback(model, data):
        for function in callbacks:
            function(model, data)

    return callback


def score_rollout(problem: ControlProblem, states, sensors) -> tuple[float, dict]:
    """Reward a bounded board jump and catch, rather than simply throwing the board."""
    model, index = problem.model, problem.index
    q = states[:, 1:1 + model.nq]
    board_pos = q[:, index.board_qpos:index.board_qpos + 3]
    robot_pos = q[:, index.robot_qpos:index.robot_qpos + 3]
    robot_quat = q[:, index.robot_qpos + 3:index.robot_qpos + 7]
    upright = 1 - 2 * (robot_quat[:, 1]**2 + robot_quat[:, 2]**2)
    board_quat = q[:, index.board_qpos + 3:index.board_qpos + 7]
    board_upright = 1 - 2 * (board_quat[:, 1]**2 + board_quat[:, 2]**2)
    min_board_upright = float(board_upright.min())
    wheel_height = np.stack([
        sensors[:, problem.sensor_slices[f"{name.removesuffix('_joint')}_world"]][:, 2]
        for name in BOARD_JOINT_NAMES
    ], axis=1) - WHEEL_RADIUS
    minimum_wheel_height = wheel_height.min(axis=1)
    left = sensors[:, problem.sensor_slices["left_foot_board"]]
    right = sensors[:, problem.sensor_slices["right_foot_board"]]
    feet = np.stack([left, right], axis=1)
    final = slice(-max(round(0.25 / model.opt.timestep), 1), None)
    foot_error = np.linalg.norm(feet[:, :, :2] - problem.foot_targets[:, :2], axis=2)
    height_error = np.abs(feet[:, :, 2] - problem.foot_targets[:, 2])
    landing_error = float(np.mean(foot_error[final]) + np.mean(height_error[final]))
    board_escape = float(np.max(np.linalg.norm(board_pos[:, :2] - robot_pos[:, :2], axis=1)))
    clearance = float(max(minimum_wheel_height.max(), 0))
    airborne_seconds = float(np.count_nonzero(minimum_wheel_height > 0.005) * model.opt.timestep)
    root_peak = float(robot_pos[:, 2].max() - problem.qpos[index.robot_qpos + 2])
    crouch_end = max(1, round(0.42 / model.opt.timestep))
    takeoff_end = max(crouch_end + 1, round(1.0 / model.opt.timestep))
    crouch_height = float(robot_pos[:crouch_end, 2].min())
    extension_rise = float(max(robot_pos[crouch_end:takeoff_end, 2].max() - crouch_height, 0))
    # A nose-up board is a dense stepping stone; it does not count as a completed pop.
    front_height = wheel_height[:, :2].mean(axis=1)
    rear_height = wheel_height[:, 2:].mean(axis=1)
    nose_rise = float(max((front_height - rear_height).max(), 0))
    tail_tip_height = float(sensors[crouch_end:takeoff_end, problem.sensor_slices["tail_tip_world"]][:, 2].min())
    tail_approach = float(np.clip((0.060 - tail_tip_height) / 0.055, 0, 1))
    # Wheel lift while balancing on the tail is not board flight. The tip-site
    # lower bound is only a dense search proxy; final success uses actual contacts.
    tail_tip = sensors[:, problem.sensor_slices["tail_tip_world"]][:, 2]
    flight_end = min(len(states), round(1.35 / model.opt.timestep))
    board_clearance_proxy = float(max(np.minimum(minimum_wheel_height, tail_tip - 0.009)[crouch_end:flight_end].max(), 0))
    min_upright = float(upright.min())
    upward_speed = float(max(states[crouch_end:takeoff_end, 1 + model.nq + index.robot_qvel + 2].max(), 0))
    final_root_height = float(robot_pos[final, 2].mean())
    survival = float(min(np.clip((min_upright - 0.4) / 0.4, 0, 1),
                         np.clip((final_root_height - 0.3) / 0.35, 0, 1),
                         np.clip((min_board_upright - 0.4) / 0.35, 0, 1)))
    pop_gate = float(np.clip((0.030 - tail_tip_height) / 0.020, 0, 1))
    score = (
        survival * (20 * min(clearance, 0.25) + 2 * min(airborne_seconds, 0.45)
                    + 40 * min(board_clearance_proxy, 0.15) * pop_gate)
        + 2 * min(max(root_peak, 0), 0.35) - 5 * landing_error
        + 3 * min(extension_rise, 0.30) + 2 * min(nose_rise, 0.20)
        + 0.6 * min(upward_speed, 2.0)
        + 1.5 * tail_approach
        - 2 * max(board_escape - 0.35, 0) - 3 * max(0.6 - min_upright, 0)
        - 3 * max(0.60 - final_root_height, 0)
        - 4 * max(0.70 - min_board_upright, 0)
    )
    catch_eligible = board_clearance_proxy >= 0.008 and pop_gate >= 0.8 and survival >= 0.85
    if problem.catch_priority:
        # Second-stage search retains genuine early board-release candidates,
        # then minimizes missed catches instead of returning to a grounded squat.
        vector = 2 * np.cross(board_quat[:, None, 1:], feet)
        feet_world = (board_pos[:, None] + feet + board_quat[:, None, :1] * vector
                      + np.cross(board_quat[:, None, 1:], vector))
        floor_proxy = float(np.clip((0.040 - feet_world[final, :, 2].mean()) / 0.04, 0, 1))
        score = ((100.0 if catch_eligible else -100.0) + 2 * min(extension_rise, 0.25)
                 - 20 * landing_error - 8 * floor_proxy
                 - 4 * max(board_escape - 0.25, 0) - 4 * max(0.8 - min_upright, 0))
    if not np.isfinite(states).all() or not np.all(np.diff(states[:, 0]) > 0):
        score = -1000.0
    return score, {
        "score": float(score), "maximum_all_wheel_clearance_m": clearance,
        "all_wheels_airborne_seconds": airborne_seconds,
        "robot_root_rise_m": root_peak, "minimum_upright_cosine": min_upright,
        "minimum_board_upright_cosine": min_board_upright,
        "landing_foot_error_m": landing_error, "maximum_board_robot_xy_distance_m": board_escape,
        "final_robot_root_height_m": final_root_height,
        "root_extension_from_crouch_m": extension_rise,
        "maximum_nose_above_rear_wheels_m": nose_rise,
        "maximum_pop_root_upward_speed_mps": upward_speed,
        "minimum_pop_tail_tip_height_m": tail_tip_height,
        "maximum_board_clearance_proxy_m": board_clearance_proxy,
        "survival_reward_gate": survival,
        "catch_search_eligible": bool(catch_eligible),
    }


def evaluate_batch(problem, parameters, *, seconds=2.2, threads=4, balance=False, placement=False):
    controls = np.stack([control_trajectory(problem, p, seconds) for p in parameters])
    start = initial_state(problem)[None]
    previous = mujoco.get_mjcb_control()
    try:
        if balance or placement:
            mujoco.set_mjcb_control(feedback_callback(problem, balance=balance, placement=placement))
        states, sensors = rollout.rollout(
            problem.model, [mujoco.MjData(problem.model) for _ in range(threads)],
            start, controls, persistent_pool=True,
        )
    finally:
        mujoco.set_mjcb_control(previous)
    return [score_rollout(problem, state, sensor)
            for state, sensor in zip(states, sensors, strict=True)]
