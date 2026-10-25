"""One bounded native-physics pop-event search; no learned model or force control.

Replay the actual 53 prefix to 2.60 s, search 29 absolute PD targets on scratch
branches, then execute exactly one already evaluated sequence to 3.40 s.
The failed 54 motion is used only by the historical diagnosis, never as a goal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.optimize import brentq

from scripts.local.continue_ollie_sampling_mpc import compare_models, fingerprint
from scripts.local.evaluate_ollie_policy import continuous_windows, outcome
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, make_scene_spec

PREFIX_HASH = "559fdfb376120a1529bbe5e3de8f0537103c7f4001a4ed3934e5be7152b9b96e"
REFERENCE_HASH = "38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7"
FEEDFORWARD_HASH = "08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115"
PREVIOUS_HASH = "44467039d304d547af23557136ee96cbaeca347ba7ab0ab701564ea6658a0fcc"


@dataclass(frozen=True)
class Config:
    prefix_controls: int = 130
    horizon: int = 40
    candidates: int = 256
    rounds: int = 24
    elites: int = 16
    knots: int = 6
    dimensions: int = 15
    sigma: float = 0.25
    sigma_floor: float = 0.025
    seed: int = 60
    threads: int = 16
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    wall_budget_s: float = 300.0
    front_unload_n: float = 40.0
    target_com_vz: float = 1.0
    target_nose_up_impulse_nms: float = 0.5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rotate(q, v):
    xyz = q[..., 1:]
    cross = 2 * np.cross(xyz, v)
    return v + q[..., :1] * cross + np.cross(xyz, cross)


def inverse_rotate(q, v):
    return rotate(q * [1, -1, -1, -1], v)


def rolling_mean(values, length):
    summed = np.concatenate([np.zeros((*values.shape[:-1], 1)), values.cumsum(axis=-1)], -1)
    return (summed[..., length:] - summed[..., :-length]) / length


def replay_prefix(controller, archive, config):
    """One original initialization and actual controls, without copying an endpoint."""
    data, model = controller.data, controller.model
    records = [controller.observer.observe(data)]
    max_error = {name: 0.0 for name in ("qpos", "qvel", "state")}
    for frame in range(config.prefix_controls + 1):
        values = {
            "qpos": data.qpos,
            "qvel": data.qvel,
            "state": controller.indices.pack(data.qpos, data.qvel).numpy(),
        }
        for name, value in values.items():
            max_error[name] = max(max_error[name], float(np.max(abs(value - archive[name][frame]))))
            if not np.array_equal(value, archive[name][frame]):
                raise RuntimeError(f"Prefix {name} differs at boundary {frame}")
        if frame == config.prefix_controls:
            break
        data.ctrl[controller.index.robot_actuators] = archive["action"][frame]
        for _ in range(8):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            row = controller.observer.observe(data)
            if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                raise RuntimeError("Prefix has external forces")
            records.append(row)
    source = json.loads((controller.prefix_path.parent / "contacts.json").read_text())
    # The original preparation recorder adds tracking diagnostics to the
    # ContactObserver row. Compare every observer field, not extra columns.
    if len(source) < len(records) or any(
        any(row[name] != original[name] for name in row)
        for row, original in zip(records, source, strict=False)
    ):
        raise RuntimeError("Prefix contact records differ")
    return records, {
        "controls": config.prefix_controls,
        "endpoint_s": float(data.time),
        "qpos_qvel_state_max_error": max_error,
        "all_observer_fields_exact": True,
        "hot_fingerprint": fingerprint(data, model),
        "initializations": 1,
        "phase_resets": 0,
    }


def add_sensors(controller):
    spec = make_scene_spec(timestep=0.0025)
    descriptions = []
    for name, kind, objtype, objname in (
        ("rootpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_XBODY, "pelvis"),
        ("rootquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_XBODY, "pelvis"),
        ("boardpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_XBODY, "board"),
        ("boardquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_XBODY, "board"),
        ("robot_com", mujoco.mjtSensor.mjSENS_SUBTREECOM, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        ("board_com", mujoco.mjtSensor.mjSENS_SUBTREECOM, mujoco.mjtObj.mjOBJ_BODY, "board"),
        ("robot_com_v", mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        ("board_com_v", mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtObj.mjOBJ_BODY, "board"),
        ("tailpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_GEOM, "board_tail"),
        ("tailquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_GEOM, "board_tail"),
    ):
        spec.add_sensor(name="pop_" + name, type=kind, objtype=objtype, objname=objname)
        descriptions.append(name)
    found = 1 << int(mujoco.mjtConDataField.mjCONDATA_FOUND)
    distance = 1 << int(mujoco.mjtConDataField.mjCONDATA_DIST)
    wrench_fields = found
    for field in ("FORCE", "TORQUE", "POS"):
        wrench_fields |= 1 << int(getattr(mujoco.mjtConDataField, "mjCONDATA_" + field))
    spec.add_sensor(
        name="pop_depth", type=mujoco.mjtSensor.mjSENS_CONTACT, intprm=[found | distance, 1, 1]
    )
    descriptions.append("depth")
    spec.add_sensor(
        name="pop_robot_ground",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="pelvis",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="ground",
        intprm=[found, 0, 1],
    )
    descriptions.append("robot_ground")
    controller.pairs = {}
    for side in ("left", "right"):
        ids = [controller.model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)]
        bodies = controller.model.geom_bodyid[ids]
        if np.any(bodies != bodies[0]):
            raise ValueError("Expected seven authored capsules on one foot body")
        foot_body = controller.model.body(int(bodies[0])).name
        for box in ("center", "tail", "nose"):
            name = f"{side}_{box}"
            spec.add_sensor(
                name="pop_" + name,
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname="board_" + box,
                reftype=mujoco.mjtObj.mjOBJ_XBODY,
                refname=foot_body,
                intprm=[wrench_fields, 3, 1],
            )
            controller.pairs[name] = (controller.model.geom("board_" + box).id, set(ids))
            descriptions.append(name)
    spec.add_sensor(
        name="pop_tail_ground",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_GEOM,
        objname="ground",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="board_tail",
        intprm=[wrench_fields, 3, 1],
    )
    controller.pairs["tail_ground"] = (
        controller.model.geom("ground").id,
        {controller.model.geom("board_tail").id},
    )
    descriptions.append("tail_ground")
    controller.branch_model = spec.compile()
    controller.pool_data = [mujoco.MjData(controller.branch_model) for _ in range(16)]
    controller.slices = {}
    for name in descriptions:
        i = controller.branch_model.sensor("pop_" + name).id
        start = int(controller.branch_model.sensor_adr[i])
        controller.slices[name] = slice(start, start + int(controller.branch_model.sensor_dim[i]))
    for name in controller.pairs:
        i = controller.branch_model.sensor("pop_" + name).id
        if controller.branch_model.sensor_needstage[i] != mujoco.mjtStage.mjSTAGE_ACC:
            raise RuntimeError("Wrench sensor must run at the acceleration stage")
    return compare_models(controller.model, controller.branch_model)


def net_wrench(model, data, first_geom, second_geoms):
    """Independent force and moment about world origin, acting on the second set."""
    force, moment, count = np.zeros(3), np.zeros(3), 0
    local = np.empty(6)
    for i, contact in enumerate(data.contact[: data.ncon]):
        first, second = map(int, contact.geom)
        sign = (
            1
            if first == first_geom and second in second_geoms
            else (-1 if second == first_geom and first in second_geoms else 0)
        )
        if not sign:
            continue
        count += 1
        mujoco.mj_contactForce(model, data, i, local)
        rotation = contact.frame.reshape(3, 3).T
        f = sign * (rotation @ local[:3])
        force += f
        moment += sign * (rotation @ local[3:]) + np.cross(contact.pos, f)
    return count, force, moment


def native_geometry(controller, data, qvel_before):
    model, ix = controller.model, controller.index
    root = model.body("pelvis").id
    tail = model.geom("board_tail").id
    jac = np.empty((3, model.nv))
    result = {}
    for name, body in (("root", root), ("board", ix.board_body)):
        result[name + "pos"] = data.xpos[body].copy()
        result[name + "quat"] = data.xquat[body].copy()
    for name, body in (("robot", root), ("board", ix.board_body)):
        result[name + "_com"] = data.subtree_com[body].copy()
        mujoco.mj_jacSubtreeCom(model, data, jac, body)
        result[name + "_com_v"] = jac @ qvel_before
    result["tailpos"] = data.geom_xpos[tail].copy()
    # geom frame orientation includes the authored kicktail rotation.
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.geom_xmat[tail])
    result["tailquat"] = quat
    return result


def branch_proof(controller, target, predicted, sensors):
    """Prove one complete already simulated branch without advancing the main data."""
    unchanged = fingerprint(controller.data, controller.model)
    scratch = copy.copy(controller.data)
    state_error, pose_error, force_error, moment_error, count_error = 0.0, 0.0, 0.0, 0.0, 0.0
    model, sl = controller.model, controller.slices
    cursor = 0
    for control in target:
        scratch.ctrl[controller.index.robot_actuators] = control
        for _ in range(8):
            before_qvel = scratch.qvel.copy()
            mujoco.mj_step(model, scratch)
            expected = native_geometry(controller, scratch, before_qvel)
            for name, value in expected.items():
                observed = sensors[cursor, sl[name]]
                if name.endswith("quat"):
                    pose_error = max(
                        pose_error,
                        float(min(abs(observed - value).max(), abs(observed + value).max())),
                    )
                else:
                    pose_error = max(pose_error, float(abs(observed - value).max()))
            for name, pair in controller.pairs.items():
                count, force, moment = net_wrench(model, scratch, *pair)
                value = sensors[cursor, sl[name]]
                count_error = max(count_error, abs(float(value[0]) - count))
                force_error = max(force_error, float(abs(value[1:4] - force).max()))
                moment_error = max(
                    moment_error,
                    float(abs(value[4:7] + np.cross(value[7:10], value[1:4]) - moment).max()),
                )
            state_error = max(
                state_error, float(abs(controller.state(scratch) - predicted[cursor]).max())
            )
            mujoco.mj_forward(model, scratch)
            cursor += 1
    if (
        state_error
        or pose_error > 1e-10
        or force_error > 1e-7
        or moment_error > 1e-7
        or count_error
    ):
        raise RuntimeError(
            f"Native branch/sensor proof failed: {state_error, pose_error, force_error, moment_error, count_error}"
        )
    if fingerprint(controller.data, model) != unchanged:
        raise RuntimeError("Scratch proof mutated the actual scene")
    return {
        "fullphysics_max_error": state_error,
        "pose_and_com_velocity_max_error": pose_error,
        "force_max_error_n": force_error,
        "world_origin_moment_max_error_nm": moment_error,
        "contact_found_count_max_error": count_error,
        "actual_scene_unchanged": True,
        "physics_samples": cursor,
        "sensor_timing": "All observations before integration; FORCE/TORQUE after that substep's forward dynamics, including its new control. FULLPHYSICS after integration.",
    }


def event_cost(controller, states, sensors, targets, config):
    """Engineering event guidance only; actual phase records decide what occurred."""
    sl, model, ix = controller.slices, controller.model, controller.index
    start = np.broadcast_to(controller.state(), (len(states), 1, states.shape[-1]))
    pre = np.concatenate([start, states[:, :-1]], axis=1)
    t = np.arange(config.horizon * 8) * config.physics_dt
    bq, bp = sensors[..., sl["boardquat"]], sensors[..., sl["boardpos"]]
    rootq = sensors[..., sl["rootquat"]]
    board_up = 1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)
    root_up = 1 - 2 * (rootq[..., 1] ** 2 + rootq[..., 2] ** 2)
    board_x = rotate(bq, np.array([1.0, 0.0, 0.0]))
    nose_up = np.arctan2(board_x[..., 2], board_up)
    tq = sensors[..., sl["tailquat"]]
    tail_size = model.geom_size[model.geom("board_tail").id]
    zrow = np.stack(
        [
            2 * (tq[..., 1] * tq[..., 3] - tq[..., 0] * tq[..., 2]),
            2 * (tq[..., 2] * tq[..., 3] + tq[..., 0] * tq[..., 1]),
            1 - 2 * (tq[..., 1] ** 2 + tq[..., 2] ** 2),
        ],
        -1,
    )
    tail_bottom = sensors[..., sl["tailpos"]][..., 2] - np.sum(abs(zrow) * tail_size, -1)
    axle = bp + rotate(bq, np.array([-0.21, 0.0, -0.025]))
    foot_force, moment_world = {}, np.zeros_like(bp)
    for side in ("left", "right"):
        total = np.zeros_like(bp)
        for box in ("center", "tail", "nose"):
            value = sensors[..., sl[f"{side}_{box}"]]
            f = value[..., 1:4]
            total += f
            moment_world -= value[..., 4:7] + np.cross(value[..., 7:10] - axle, f)
        foot_force[side] = np.maximum(total[..., 2], 0)
    moment_y = inverse_rotate(bq, moment_world)[..., 1]
    root = model.body("pelvis").id
    masses = np.array([model.body_subtreemass[root], model.body_subtreemass[ix.board_body]])
    com_v = (
        masses[0] * sensors[..., sl["robot_com_v"]] + masses[1] * sensors[..., sl["board_com_v"]]
    ) / masses.sum()
    depth = np.maximum(-sensors[..., sl["depth"]][..., 1], 0)
    ground = sensors[..., sl["robot_ground"]][..., 0] > 0
    q = pre[..., 1 + ix.robot_joint_qpos]
    limit = np.maximum(np.maximum(controller.limits[:, 0] - q, q - controller.limits[:, 1]), 0).max(
        -1
    )
    violations = (
        (depth.max(1) > 0.01).astype(int)
        + ground.any(1).astype(int)
        + (board_up.min(1) <= 0).astype(int)
        + (limit.max(1) > 0.05).astype(int)
    )
    safety = 1e6 * violations + 1e5 * (np.maximum(depth.max(1) - 0.01, 0) / 0.01) ** 2
    safety += 1e5 * (np.maximum(limit.max(1) - 0.05, 0) / 0.05) ** 2
    window = (t >= 0.12) & (t <= 0.50)
    front_min = rolling_mean(foot_force["left"][:, window], 8).min(1)
    unload = 50 * np.minimum(front_min / config.front_unload_n, 10) ** 2
    progress = (t >= 0.16) & (t <= 0.70)
    bottom_min = rolling_mean(tail_bottom[:, progress], 2).min(1)
    angle_max = nose_up[:, progress].max(1)
    tail_force = np.maximum(sensors[..., sl["tail_ground"]][..., 3], 0)
    tail_peak = rolling_mean(tail_force[:, progress], 2).max(1)
    geometry = 1000 * (np.maximum(bottom_min, 0) / 0.068) ** 2
    geometry += (
        100
        * (np.maximum(controller.required_nose_up - angle_max, 0) / controller.required_nose_up)
        ** 2
    )
    tail_event = 100 * np.maximum(1 - tail_peak / 5, 0) ** 2
    torque_window = (t >= 0.12) & (t <= 0.56)
    pitch_impulse = np.maximum(-moment_y[:, torque_window], 0).sum(1) * config.physics_dt
    drive = (
        100
        * (
            np.maximum(config.target_nose_up_impulse_nms - pitch_impulse, 0)
            / config.target_nose_up_impulse_nms
        )
        ** 2
    )
    vz_peak = com_v[:, (t >= 0.14) & (t <= 0.72), 2].max(1)
    impulse = 200 * (np.maximum(config.target_com_vz - vz_peak, 0) / config.target_com_vz) ** 2
    eligible_unload = rolling_mean(foot_force["left"], 8) < config.front_unload_n
    first_unload = np.where(
        eligible_unload, np.arange(eligible_unload.shape[1])[None] + 7, 1e6
    ).min(1)
    first_tail = np.where(tail_force > 0.1, np.arange(len(t))[None], 1e6).min(1)
    order = 1000 * ((first_tail < 1e6) & (first_tail < first_unload)).astype(float)
    drift = 5 * np.mean(
        np.sum(((bp[..., :2] - controller.data.xpos[ix.board_body, :2]) / 0.15) ** 2, -1), 1
    )
    upright = 20 * np.mean((np.maximum(0.8 - root_up, 0) / 0.2) ** 2, 1)
    previous = controller.data.ctrl[ix.robot_actuators]
    delta = np.diff(
        np.concatenate([np.broadcast_to(previous, (len(targets), 1, 29)), targets], 1), axis=1
    )
    smooth = 0.1 * np.mean((delta / 0.25) ** 2, (1, 2))
    terms = {
        "safety": safety,
        "front_unload": unload,
        "tail_geometry": geometry,
        "tail_force_event": tail_event,
        "nose_up_foot_impulse": drive,
        "system_com_upward_velocity": impulse,
        "event_order": order,
        "board_xy_drift": drift,
        "pelvis_upright_soft": upright,
        "target_change": smooth,
    }
    cost = sum(terms.values())
    valid = np.isfinite(states).all((1, 2)) & np.isfinite(sensors).all((1, 2)) & np.isfinite(cost)
    valid &= np.isclose(
        states[:, -1, 0], controller.data.time + config.horizon * 0.02, rtol=0, atol=1e-9
    )
    metrics = {
        "maximum_penetration_m": depth.max(1),
        "robot_ground_any": ground.any(1),
        "board_min_upright": board_up.min(1),
        "front_min_20ms_upward_force_n": front_min,
        "tail_min_5ms_mean_bottom_m": bottom_min,
        "nose_up_max_rad": angle_max,
        "tail_peak_5ms_mean_force_n": tail_peak,
        "nose_up_foot_angular_impulse_nms": pitch_impulse,
        "system_com_vz_peak_m_s": vz_peak,
        "mechanical_limit_peak_rad": limit.max(1),
    }
    return np.where(valid, cost, np.inf), terms, metrics


def make_targets(knots, nominal, config):
    x = np.linspace(0, config.horizon - 1, config.knots)
    residual = np.stack(
        [
            np.stack(
                [
                    np.interp(np.arange(config.horizon), x, row[:, i])
                    for i in range(config.dimensions)
                ],
                -1,
            )
            for row in knots
        ]
    )
    target = np.broadcast_to(nominal, (len(knots), config.horizon, 29)).copy()
    target[:, :, :15] += residual
    return target


def geometry_diagnosis(controller):
    model, ix = controller.model, controller.index
    scratch = mujoco.MjData(model)
    tail = model.geom("board_tail").id
    rear = [model.geom(f"wheel_rear_{side}_collision").id for side in ("left", "right")]

    def height(angle):
        scratch.qpos[ix.board_qpos : ix.board_qpos + 7] = [
            0,
            0,
            0,
            np.cos(angle / 2),
            0,
            np.sin(angle / 2),
            0,
        ]
        mujoco.mj_forward(model, scratch)
        low = []
        for geom in rear:
            az = scratch.geom_xmat[geom].reshape(3, 3)[2, 2]
            radius, width = model.geom_size[geom, :2]
            low.append(
                scratch.geom_xpos[geom, 2]
                - (radius * np.sqrt(max(0, 1 - az * az)) + width * abs(az))
            )
        scratch.qpos[ix.board_qpos + 2] -= min(low)
        mujoco.mj_forward(model, scratch)
        bottom = (
            scratch.geom_xpos[tail, 2]
            - abs(scratch.geom_xmat[tail].reshape(3, 3)[2]) @ model.geom_size[tail]
        )
        return float(bottom), float(scratch.qpos[ix.board_qpos + 2])

    angle = brentq(lambda x: height(x)[0], -0.9, 0)
    controller.required_nose_up = -angle
    return {
        "assumption": "Pure pitch about unchanged rear-wheel ground support, no roll; exact box/cylinder envelope, not a universal roll/pitch contact angle.",
        "required_pitch_y_rad": angle,
        "required_nose_up_deg": -angle * 180 / np.pi,
        "board_root_z_at_touch_m": height(angle)[1],
        "method": "brentq tail full-box minimum Z=0 with rear wheel full cylinder tangency",
        "is_physical_simulation": False,
    }


def prior_diagnosis(controller, out_dir):
    """Recompute historical evidence without using 54 anywhere in this search."""
    from scripts.local.continue_ollie_sampling_mpc import (
        ProposedConfig,
        add_readonly_contact_sensors,
        continuation_cost,
        prepare_planner,
    )
    from scripts.local.continue_ollie_sampling_mpc import (
        replay_prefix as replay56,
    )
    from supermimic.ollie.reference import JointReference

    path = Path("logs/ollie_sampling_mpc_continuation_01/actual_rollout.npz")
    if digest(path) != PREVIOUS_HASH:
        raise ValueError("Historical 56 rollout changed")
    with np.load(path, allow_pickle=False) as data:
        actual = {name: data[name].copy() for name in data.files}
    model, ix = controller.model, controller.index
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = actual["qpos"][0]
    scratch.qvel[:] = actual["qvel"][0]
    scratch.ctrl[ix.robot_actuators] = actual["qpos"][0, ix.robot_joint_qpos]
    mujoco.mj_forward(model, scratch)
    feet = {
        side: {model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)}
        for side in ("left", "right")
    }
    boxes = [model.geom("board_" + n).id for n in ("center", "tail", "nose")]
    root = model.body("pelvis").id
    masses = [model.body_subtreemass[root], model.body_subtreemass[ix.board_body]]
    jac = np.empty((3, model.nv))
    rows = []
    for step, target in enumerate(actual["action"]):
        scratch.ctrl[ix.robot_actuators] = target
        for _ in range(8):
            mujoco.mj_step(model, scratch)
            mujoco.mj_forward(model, scratch)
            if not 2.3 <= scratch.time <= 3.6:
                continue
            R = scratch.xmat[ix.board_body].reshape(3, 3)
            axle = scratch.xpos[ix.board_body] + R @ np.array([-0.21, 0, -0.025])
            row = {
                "time_s": float(scratch.time),
                "root_vz": float(scratch.qvel[ix.robot_qvel + 2]),
                "pitch_y_rad": float(np.arctan2(-R[2, 0], R[2, 2])),
            }
            tail = model.geom("board_tail").id
            row["tail_bottom_m"] = float(
                scratch.geom_xpos[tail, 2]
                - abs(scratch.geom_xmat[tail].reshape(3, 3)[2]) @ model.geom_size[tail]
            )
            for side, foot_geoms in feet.items():
                f = np.zeros(3)
                mo = np.zeros(3)
                cop_num = 0.0
                tail_load = 0.0
                for box in boxes:
                    _, f_on_foot, m_on_foot = net_wrench(model, scratch, box, foot_geoms)
                    f += f_on_foot
                    mo += m_on_foot
                    if box == tail:
                        tail_load = max(float(f_on_foot[2]), 0)
                for i, contact in enumerate(scratch.contact[: scratch.ncon]):
                    g0, g1 = map(int, contact.geom)
                    box = g1 if g0 in foot_geoms else g0 if g1 in foot_geoms else None
                    if box not in boxes:
                        continue
                    value = np.empty(6)
                    mujoco.mj_contactForce(model, scratch, i, value)
                    if value[0] < 0.1:
                        continue
                    sign = -1 if g0 in foot_geoms else 1
                    board_force = sign * (contact.frame.reshape(3, 3).T @ value[:3])
                    cop_num += (R.T @ (contact.pos - scratch.xpos[ix.board_body]))[0] * max(
                        -board_force[2], 0
                    )
                load = max(float(f[2]), 0)
                row[side + "_upward_n"] = load
                row[side + "_cop_x_m"] = float(cop_num / load) if load else None
                row[side + "_board_pitch_torque_nm"] = float((R.T @ (-mo + np.cross(axle, f)))[1])
                row[side + "_tail_upward_n"] = tail_load
            cv = []
            for body in (root, ix.board_body):
                mujoco.mj_jacSubtreeCom(model, scratch, jac, body)
                cv.append(jac @ scratch.qvel)
            row["system_com_vz"] = float(
                (masses[0] * cv[0][2] + masses[1] * cv[1][2]) / sum(masses)
            )
            rows.append(row)
    errors = {
        "qpos": float(abs(scratch.qpos - actual["qpos"][-1]).max()),
        "qvel": float(abs(scratch.qvel - actual["qvel"][-1]).max()),
    }
    if any(errors.values()):
        raise RuntimeError("Historical 56 actual replay differs")
    c = NativeSamplingMPC(
        Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
        Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"),
        seed=56,
    )
    try:
        c.prefix_path = Path("logs/ollie_sampling_mpc_prep_01/actual_rollout.npz")
        with np.load(c.prefix_path, allow_pickle=False) as a:
            prefix = {k: a[k].copy() for k in a.files}
        c.horizon = 48
        c.knots = np.linspace(0, 47, 6)
        replay56(c, prefix)
        add_readonly_contact_sensors(c)
        old_ref = JointReference.load(
            "data/ollie/landing_ballistic_v1/failed_joint_reference_50hz.npz"
        )
        prepare_planner(c, old_ref, ProposedConfig(), c.prefix_path)
        with np.load(
            "logs/ollie_sampling_mpc_continuation_01/selected_plans.npz", allow_pickle=False
        ) as a:
            states = a["fullphysics"][:1].copy()
            targets = a["action"][:1].copy()
        pre = np.concatenate([c.state()[None, None, :], states[:, :-1]], 1)
        d = mujoco.MjData(c.branch_model)
        s = []
        for state in pre[0]:
            mujoco.mj_setState(c.branch_model, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(c.branch_model, d)
            s.append(d.sensordata.copy())
        captured = {}

        def profile(frame, event, arg):
            if event == "return" and frame.f_code is continuation_cost.__code__:
                captured.update(frame.f_locals)

        try:
            sys.setprofile(profile)
            cost, _ = continuation_cost(c, states, np.asarray(s)[None], targets, 138)
        finally:
            sys.setprofile(None)
        names = {
            "feet": "foot",
            "relative_com": "com_loss",
            "relative_root_orientation": "pose",
            "joint_position": "joint",
            "board_z": "board_z",
            "board_quat": "board_quat",
            "contact_witness": "foot_contact",
            "unsafe": "unsafe",
            "penetration": "penetration",
        }
        terms = {
            key: float(captured[name].mean() + 2 * captured[name][0, -1])
            for key, name in names.items()
        }
        root_penalty = 1e4 * (np.maximum(captured["root_low"], 0) / 0.05) ** 2
        old_cost = {
            "reconstructed_first_winner_cost": float(cost[0]),
            "root_reference_low_penalty": float(root_penalty.mean() + 2 * root_penalty[0, -1]),
            "root_reference_fraction": float(
                (root_penalty.mean() + 2 * root_penalty[0, -1]) / cost[0]
            ),
            "other_terms_selected": terms,
            "old_reference_is_failed_and_used_only_for_this_diagnosis": True,
        }
    finally:
        c.pool.close()
    near = lambda t: min(rows, key=lambda r: abs(r["time_s"] - t))
    peak = min((r for r in rows if 2.76 <= r["time_s"] <= 3.14), key=lambda r: r["pitch_y_rad"])
    result = {
        "previous_actual_path": str(path.resolve()),
        "previous_actual_sha256": PREVIOUS_HASH,
        "actual_replay_endpoint_error": errors,
        "samples": [
            near(t) for t in (2.4, 2.5, 2.6, 2.7, 2.76, 2.8, 2.9, 2.94, 3.0, 3.04, 3.08, 3.1, 3.14)
        ],
        "peak_nose_up_before_3_14": peak,
        "first_plan_cost_decomposition": old_cost,
        "cause_scope": "Observed control/timing and objective failure; not a proof that this passive board or every control is impossible.",
    }
    write_json(out_dir / "prior_56_diagnosis.json", result)
    write_json(out_dir / "prior_56_force_samples.json", rows)
    return result


def execute_winner(controller, prefix, prefix_records, targets, predicted, config):
    model, data, ix = controller.model, controller.data, controller.index
    records = list(prefix_records)
    poses = [v.copy() for v in prefix["qpos"][: config.prefix_controls + 1]]
    velocities = [v.copy() for v in prefix["qvel"][: config.prefix_controls + 1]]
    states = [v.copy() for v in prefix["state"][: config.prefix_controls + 1]]
    actions = [v.copy() for v in prefix["action"][: config.prefix_controls]]
    maximum = 0.0
    for frame, target in enumerate(targets):
        data.ctrl[ix.robot_actuators] = target
        for substep in range(8):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            row = controller.observer.observe(data)
            records.append(row)
            if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                raise RuntimeError("Actual winner has external forces")
            error = float(abs(controller.state() - predicted[frame * 8 + substep]).max())
            maximum = max(maximum, error)
            if (
                not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
                or error > 1e-10
            ):
                raise RuntimeError("Actual winner differs from its evaluated branch")
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        states.append(controller.indices.pack(data.qpos, data.qvel).numpy())
        actions.append(target.copy())
    result = outcome(records, full_reference=False, numerical_failure=None)
    result.update(
        schema="supermimic.ollie.pop_events_actual.v1",
        scope="bounded pop-event diagnostic, not complete Ollie",
        actual_endpoint_s=float(data.time),
        initializations=1,
        phase_resets=0,
        external_forces_applied=False,
        board_actuators=0,
        verified_teacher=False,
        supermimic_training_performed=False,
        actual_prediction_max_error=maximum,
    )
    return (
        result,
        {
            "qpos": np.asarray(poses),
            "qvel": np.asarray(velocities),
            "state": np.asarray(states),
            "action": np.asarray(actions),
            "time_physics": np.arange(len(states)) * 0.02,
        },
        records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory must be new")
    started = time.monotonic()
    config = Config()
    paths = {
        "prefix": Path("logs/ollie_sampling_mpc_prep_01/actual_rollout.npz"),
        "reference": Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
        "feedforward": Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"),
    }
    for name, expected in (
        ("prefix", PREFIX_HASH),
        ("reference", REFERENCE_HASH),
        ("feedforward", FEEDFORWARD_HASH),
    ):
        if digest(paths[name]) != expected:
            raise ValueError(f"Frozen {name} changed")
    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema": "supermimic.ollie.pop_events_manifest.v1",
        "config": asdict(config),
        "script_sha256": digest(__file__),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in paths.items()
        },
        "known_failed_54_is_search_goal": False,
        "future_source_root_feet_board_tracking": False,
        "cost_phases_elapsed_s": {
            "unload": [0.12, 0.50],
            "tail_geometry": [0.16, 0.70],
            "nose_up_impulse": [0.12, 0.56],
            "upward_com_v": [0.14, 0.72],
        },
        "objective_description": "High-priority native body-ground/inversion/depth>10mm/mechanical violation>0.05rad; finite tail geometry, actual foot-board netforce, signed rear-axle foot moment, COM velocity; no prescribed flight pose.",
        "source_timing_scale_used_for_control": False,
    }
    write_json(args.out_dir / "frozen_manifest.json", manifest)
    with np.load(paths["prefix"], allow_pickle=False) as a:
        prefix = {k: a[k].copy() for k in a.files}
    c = NativeSamplingMPC(paths["reference"], paths["feedforward"], seed=config.seed)
    try:
        c.prefix_path = paths["prefix"]
        c.horizon = config.horizon
        c.candidates = config.candidates
        c.knots = np.linspace(0, config.horizon - 1, config.knots)
        records, prefix_proof = replay_prefix(c, prefix, config)
        write_json(args.out_dir / "prefix_proof.json", prefix_proof)
        geometry = geometry_diagnosis(c)
        write_json(args.out_dir / "tail_touch_geometry.json", geometry)
        diagnosis = prior_diagnosis(c, args.out_dir)
        manifest["historical_root_reference_cost_fraction"] = diagnosis[
            "first_plan_cost_decomposition"
        ]["root_reference_fraction"]
        manifest["sensor_model_physics_proof"] = add_sensors(c)
        future = np.minimum(
            config.prefix_controls + np.arange(config.horizon), len(prefix["action"]) - 1
        )
        nominal = prefix["action"][future].copy()
        np.savez_compressed(
            args.out_dir / "nominal_targets.npz", target=nominal, source_action_index=future
        )
        unchanged = fingerprint(c.data, c.model)
        rng = np.random.default_rng(config.seed + 1000)
        knots = rng.normal(0, config.sigma, (config.candidates, config.knots, config.dimensions))
        knots[0] = 0
        targets = np.clip(make_targets(knots, nominal, config), *c.target_limits.T)
        t0 = time.monotonic()
        states, sensors = c.rollout(targets)
        costs, terms, metrics = event_cost(c, states, sensors, targets, config)
        benchmark_s = time.monotonic() - t0
        proof = branch_proof(c, targets[0], states[0], sensors[0])
        proof.update(
            benchmark_batch_wall_s=benchmark_s, projected_24_rounds_s=benchmark_s * config.rounds
        )
        write_json(args.out_dir / "first_branch_proof.json", proof)
        if fingerprint(c.data, c.model) != unchanged:
            raise RuntimeError("Benchmark mutated actual scene")
        remaining = config.wall_budget_s - (time.monotonic() - started)
        if benchmark_s * config.rounds + 20 > remaining:
            write_json(
                args.out_dir / "result.json",
                {
                    "status": "BUDGET_REFUSED",
                    "actual_endpoint_s": float(c.data.time),
                    "benchmark": proof,
                    "search_rounds_executed": 0,
                },
            )
            return
        mean = np.zeros((config.knots, config.dimensions))
        std = np.full_like(mean, config.sigma)
        rng = np.random.default_rng(config.seed)
        best = None
        reports = []
        for iteration in range(config.rounds):
            if time.monotonic() - started > config.wall_budget_s - 15:
                write_json(args.out_dir / "search_rounds.json", reports)
                write_json(
                    args.out_dir / "result.json",
                    {
                        "status": "BUDGET_STOPPED_NO_ACTUAL_EXECUTION",
                        "completed_rounds": iteration,
                        "actual_endpoint_s": float(c.data.time),
                    },
                )
                return
            knots = rng.normal(mean, std, (config.candidates, config.knots, config.dimensions))
            knots[0] = 0
            knots[1] = mean
            if best is not None:
                knots[2] = best["knots"]
            targets = np.clip(make_targets(knots, nominal, config), *c.target_limits.T)
            states, sensors = c.rollout(targets)
            costs, terms, metrics = event_cost(c, states, sensors, targets, config)
            order = np.argsort(costs)
            valid = order[np.isfinite(costs[order])]
            if not len(valid):
                raise RuntimeError("No complete finite candidate")
            winner = int(valid[0])
            elite = valid[: config.elites]
            if best is None or costs[winner] < best["cost"]:
                best = {
                    "cost": float(costs[winner]),
                    "knots": knots[winner].copy(),
                    "targets": targets[winner].copy(),
                    "states": states[winner].copy(),
                    "sensors": sensors[winner].copy(),
                    "round": iteration,
                    "candidate": winner,
                    "terms": {name: float(v[winner]) for name, v in terms.items()},
                    "metrics": {name: float(v[winner]) for name, v in metrics.items()},
                }
            mean = knots[elite].mean(0)
            std = np.clip(knots[elite].std(0), config.sigma_floor, config.sigma)
            row = {
                "round": iteration,
                "costs": [float(v) if np.isfinite(v) else None for v in costs],
                "elite_ids": elite.tolist(),
                "round_winner_id": winner,
                "round_winner_cost": float(costs[winner]),
                "best_so_far_cost": best["cost"],
                "round_winner_terms": {name: float(v[winner]) for name, v in terms.items()},
                "round_winner_metrics": {name: float(v[winner]) for name, v in metrics.items()},
                "wall_s": time.monotonic() - started,
            }
            reports.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("round", "round_winner_cost", "best_so_far_cost", "wall_s")
                    }
                ),
                flush=True,
            )
        write_json(args.out_dir / "search_rounds.json", reports)
        if fingerprint(c.data, c.model) != unchanged:
            raise RuntimeError("Search mutated actual scene")
        winner_proof = branch_proof(c, best["targets"], best["states"], best["sensors"])
        write_json(args.out_dir / "winner_branch_proof.json", winner_proof)
        np.savez_compressed(
            args.out_dir / "winner_evaluated_branch.npz",
            action=best["targets"],
            fullphysics=best["states"],
            sensors=best["sensors"],
            knots=best["knots"],
        )
        result, arrays, actual_records = execute_winner(
            c, prefix, records, best["targets"], best["states"], config
        )
        np.savez_compressed(
            args.out_dir / "actual_rollout.npz",
            **arrays,
            control_dt=np.array(0.02),
            robot_joint_names=np.asarray(G1_JOINT_NAMES),
            board_joint_names=np.asarray(BOARD_JOINT_NAMES),
            verified_teacher=np.array(False),
        )
        write_json(args.out_dir / "contacts.json", actual_records)
        times = np.array([row["time_physics"] for row in actual_records])
        tails = np.array([row["tail_ground"] for row in actual_records])
        actual_summary = {
            "tail_windows": continuous_windows(tails, times),
            "first_tail_ground_s": next(
                (row["time_physics"] for row in actual_records if row["tail_ground"]), None
            ),
            "whole_board_max_clearance_m": max(
                row["minimum_board_clearance_m"] for row in actual_records
            ),
            "robot_ground_first_s": result["robot_ground_first_time_s"],
            "maximum_penetration_m": result["maximum_penetration_m"],
            "any_true_tail_contact": bool(tails.any()),
            "does_not_certify_full_ollie": True,
        }
        result.update(
            status="COMPLETED_ONE_PHASE_ACTUAL",
            config=asdict(config),
            wall_seconds=time.monotonic() - started,
            winner={
                name: best[name] for name in ("cost", "round", "candidate", "terms", "metrics")
            },
            phase_events=actual_summary,
            full_reference_episode_completed=False,
        )
        write_json(args.out_dir / "result.json", result)
        write_json(
            args.out_dir / "output_hashes.json",
            {
                p.name: digest(p)
                for p in args.out_dir.iterdir()
                if p.is_file() and p.name != "output_hashes.json"
            },
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        c.pool.close()


if __name__ == "__main__":
    main()
