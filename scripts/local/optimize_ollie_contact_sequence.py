"""One bounded, full native-physics contact-sequence search on an explicit model.

The actual scene is initialized once and replays its verified preparation.
Only one already evaluated complete sequence of the original 29 PD targets is
then executed. No failed airborne reference, external force, phase reset,
learned model, or learned policy is used.
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

from scripts.local.continue_ollie_sampling_mpc import (
    compare_models,
    fingerprint,
    finite_capsule_witness_gap,
)
from scripts.local.evaluate_ollie_policy import outcome
from scripts.local.optimize_ollie_pop_events import net_wrench, replay_prefix, rolling_mean
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC, inverse_rotate
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_JOINT_NAMES, make_scene_spec

REFERENCE_HASH = "38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7"
FEEDFORWARD_HASH = "08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115"
OLD_PRIOR_HASH = "43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d"


@dataclass(frozen=True)
class Config:
    prefix_controls: int = 130
    horizon: int = 127
    candidates: int = 256
    rounds: int = 64
    elites: int = 16
    dimensions: int = 15
    knot_times_s: tuple = (
        0.0,
        0.04,
        0.08,
        0.12,
        0.16,
        0.20,
        0.26,
        0.34,
        0.44,
        0.56,
        0.72,
        0.90,
        1.16,
        1.50,
        1.94,
        2.52,
    )
    sigma_rad: float = 0.25
    sigma_floor_rad: float = 0.025
    seed: int = 65
    threads: int = 16
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    wall_budget_s: float = 300.0
    winner_proof_and_execution_reserve_s: float = 10.0
    actual_execution_reserve_s: float = 5.0
    foot_deck_friction: float = 0.9
    # Full score and the final winner always use identical weights. Only the
    # first 16 rounds' distribution updates de-emphasize the recovery terms.
    early_sync_rounds: int = 16
    early_recovery_weight: float = 0.15
    selection_schedule: str = "16_sync_plus_physical_rounds_then_48_full_rounds"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rotate(q, v):
    return inverse_rotate(q * [1, -1, -1, -1], v)


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def envelope_z(model, geom, position, quaternion):
    """Exact lower support of an authored box or oriented wheel cylinder."""
    q = quaternion
    zrow = np.stack(
        [
            2 * (q[..., 1] * q[..., 3] - q[..., 0] * q[..., 2]),
            2 * (q[..., 2] * q[..., 3] + q[..., 0] * q[..., 1]),
            1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2),
        ],
        -1,
    )
    size = model.geom_size[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        extent = (np.abs(zrow) * size).sum(-1)
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.sqrt(np.maximum(0, 1 - zrow[..., 2] ** 2))
        extent += size[1] * np.abs(zrow[..., 2])
    else:
        raise ValueError("Whole-board envelope accepts the five boxes and four cylinders only")
    return position[..., 2] - extent


def add_sensors(controller, config):
    """Compile the identical explicit-pair model with read-only observations."""
    spec = make_scene_spec(timestep=config.physics_dt, foot_deck_friction=config.foot_deck_friction)
    controller.sensor_descriptions = {}

    def add_pose(name, objtype, objname):
        for suffix, kind in (
            ("pos", mujoco.mjtSensor.mjSENS_FRAMEPOS),
            ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ):
            spec.add_sensor(
                name="seq_" + name + suffix, type=kind, objtype=objtype, objname=objname
            )
            controller.sensor_descriptions[name + suffix] = (kind, objtype, objname)

    for name, body in (("root", "pelvis"), ("board", "board")):
        add_pose(name, mujoco.mjtObj.mjOBJ_XBODY, body)
        for suffix, kind in (
            ("com", mujoco.mjtSensor.mjSENS_SUBTREECOM),
            ("com_v", mujoco.mjtSensor.mjSENS_SUBTREELINVEL),
        ):
            key = name + "_" + suffix
            spec.add_sensor(
                name="seq_" + key, type=kind, objtype=mujoco.mjtObj.mjOBJ_BODY, objname=body
            )
            controller.sensor_descriptions[key] = (kind, mujoco.mjtObj.mjOBJ_BODY, body)
    controller.board_geoms = list(BOARD_GEOM_NAMES) + [
        name.removesuffix("_joint") + "_collision" for name in BOARD_JOINT_NAMES
    ]
    if len(controller.board_geoms) != 9:
        raise ValueError("Expected all five board/axle boxes and four wheel cylinders")
    for geom in controller.board_geoms:
        add_pose(geom + "_", mujoco.mjtObj.mjOBJ_GEOM, geom)
    found = 1 << int(mujoco.mjtConDataField.mjCONDATA_FOUND)
    fields = found
    for name in ("FORCE", "TORQUE", "POS"):
        fields |= 1 << int(getattr(mujoco.mjtConDataField, "mjCONDATA_" + name))
    controller.pairs, controller.foot_geometry = {}, {}

    def contact(name, first, second, second_type=mujoco.mjtObj.mjOBJ_GEOM):
        spec.add_sensor(
            name="seq_" + name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=first,
            reftype=second_type,
            refname=second,
            intprm=[fields, 3, 1],
        )

    for side in ("left", "right"):
        ids = [controller.model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)]
        bodies = controller.model.geom_bodyid[ids]
        if np.any(bodies != bodies[0]) or np.any(
            controller.model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_CAPSULE
        ):
            raise ValueError("Expected the authored seven fixed capsules per foot")
        body = controller.model.body(int(bodies[0])).name
        add_pose(side + "_body_", mujoco.mjtObj.mjOBJ_XBODY, body)
        # The shared finite-witness function uses these exact sensor keys.
        controller.foot_geometry[side] = {
            "body": int(bodies[0]),
            "geoms": ids,
            "center": controller.model.geom_pos[ids].copy(),
            "axis": rotate(controller.model.geom_quat[ids], np.array([0.0, 0.0, 1.0])),
            "size": controller.model.geom_size[ids, :2].copy(),
        }
        for box in ("center", "tail", "nose"):
            name = side + "_" + box
            contact(name, "board_" + box, body, mujoco.mjtObj.mjOBJ_XBODY)
            controller.pairs[name] = (controller.model.geom("board_" + box).id, set(ids))
    for geom in controller.board_geoms:
        name = "ground_" + geom
        contact(name, "ground", geom)
        controller.pairs[name] = (
            controller.model.geom("ground").id,
            {controller.model.geom(geom).id},
        )
    distance = 1 << int(mujoco.mjtConDataField.mjCONDATA_DIST)
    spec.add_sensor(
        name="seq_depth", type=mujoco.mjtSensor.mjSENS_CONTACT, intprm=[found | distance, 1, 1]
    )
    spec.add_sensor(
        name="seq_robot_ground",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="pelvis",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="ground",
        intprm=[found, 0, 1],
    )
    controller.branch_model = spec.compile()
    controller.pool_data = [mujoco.MjData(controller.branch_model) for _ in range(config.threads)]
    controller.slices = {}
    for name in (*controller.sensor_descriptions, *controller.pairs, "depth", "robot_ground"):
        sensor = controller.branch_model.sensor("seq_" + name).id
        begin = int(controller.branch_model.sensor_adr[sensor])
        controller.slices[name] = slice(
            begin, begin + int(controller.branch_model.sensor_dim[sensor])
        )
    for name in controller.pairs:
        sensor = controller.branch_model.sensor("seq_" + name).id
        if controller.branch_model.sensor_needstage[sensor] != mujoco.mjtStage.mjSTAGE_ACC:
            raise RuntimeError("Native contact wrench sensor must be acceleration-stage")
    return compare_models(controller.model, controller.branch_model)


def branch_proof(controller, targets, states, sensors):
    """Independently verify every observation and substep without advancing main."""
    model, sl = controller.model, controller.slices
    unchanged = fingerprint(controller.data, model)
    scratch = copy.copy(controller.data)
    maxima = {
        "fullphysics": 0.0,
        "pose_com_velocity": 0.0,
        "force_n": 0.0,
        "moment_nm": 0.0,
        "contact_count": 0.0,
        "depth_m": 0.0,
        "robot_ground_count": 0.0,
        "board_envelope_m": 0.0,
    }
    ground = model.geom("ground").id
    root = model.body("pelvis").id
    tree = model.body_rootid[root]
    jac = np.empty((3, model.nv))
    cursor = 0
    for target in targets:
        scratch.ctrl[controller.index.robot_actuators] = target
        for _ in range(8):
            before_velocity = scratch.qvel.copy()
            mujoco.mj_step(model, scratch)
            for name, (kind, objtype, objname) in controller.sensor_descriptions.items():
                if kind in (
                    mujoco.mjtSensor.mjSENS_SUBTREECOM,
                    mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
                ):
                    body = model.body(objname).id
                    if kind == mujoco.mjtSensor.mjSENS_SUBTREECOM:
                        expected = scratch.subtree_com[body]
                    else:
                        mujoco.mj_jacSubtreeCom(model, scratch, jac, body)
                        expected = jac @ before_velocity
                elif objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    geom = model.geom(objname).id
                    expected = scratch.geom_xpos[geom]
                    if kind == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                        expected = np.empty(4)
                        mujoco.mju_mat2Quat(expected, scratch.geom_xmat[geom])
                else:
                    body = model.body(objname).id
                    expected = (
                        scratch.xquat[body]
                        if kind == mujoco.mjtSensor.mjSENS_FRAMEQUAT
                        else scratch.xpos[body]
                    )
                value = sensors[cursor, sl[name]]
                error = abs(value - expected).max()
                if kind == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                    error = min(error, abs(value + expected).max())
                maxima["pose_com_velocity"] = max(maxima["pose_com_velocity"], float(error))
            for name, pair in controller.pairs.items():
                count, force, moment = net_wrench(model, scratch, *pair)
                value = sensors[cursor, sl[name]]
                maxima["contact_count"] = max(maxima["contact_count"], abs(value[0] - count))
                maxima["force_n"] = max(maxima["force_n"], float(abs(value[1:4] - force).max()))
                moment_sensed = value[4:7] + np.cross(value[7:10], value[1:4])
                maxima["moment_nm"] = max(
                    maxima["moment_nm"], float(abs(moment_sensed - moment).max())
                )
            native_min = min((float(c.dist) for c in scratch.contact[: scratch.ncon]), default=0.0)
            depth_value = sensors[cursor, sl["depth"]]
            maxima["contact_count"] = max(
                maxima["contact_count"], abs(depth_value[0] - scratch.ncon)
            )
            maxima["depth_m"] = max(maxima["depth_m"], abs(depth_value[1] - native_min))
            robot_contacts = sum(
                (
                    int(c.geom[0]) == ground
                    and model.body_rootid[model.geom_bodyid[int(c.geom[1])]] == tree
                )
                or (
                    int(c.geom[1]) == ground
                    and model.body_rootid[model.geom_bodyid[int(c.geom[0])]] == tree
                )
                for c in scratch.contact[: scratch.ncon]
            )
            maxima["robot_ground_count"] = max(
                maxima["robot_ground_count"],
                abs(sensors[cursor, sl["robot_ground"]][0] - robot_contacts),
            )
            for name in controller.board_geoms:
                geom = model.geom(name).id
                rotation = scratch.geom_xmat[geom].reshape(3, 3)
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                    extent = abs(rotation[2]) @ model.geom_size[geom]
                else:
                    axis_z = rotation[2, 2]
                    radius, half_width = model.geom_size[geom, :2]
                    extent = radius * np.sqrt(max(0, 1 - axis_z**2)) + half_width * abs(axis_z)
                native = scratch.geom_xpos[geom, 2] - extent
                sensed = envelope_z(
                    model,
                    geom,
                    sensors[cursor, sl[name + "_pos"]],
                    sensors[cursor, sl[name + "_quat"]],
                )
                maxima["board_envelope_m"] = max(
                    maxima["board_envelope_m"], abs(float(sensed) - native)
                )
            maxima["fullphysics"] = max(
                maxima["fullphysics"], float(abs(controller.state(scratch) - states[cursor]).max())
            )
            mujoco.mj_forward(model, scratch)
            cursor += 1
    if (
        maxima["fullphysics"]
        or maxima["contact_count"]
        or maxima["robot_ground_count"]
        or max(maxima[key] for key in ("pose_com_velocity", "depth_m", "board_envelope_m")) > 1e-10
        or max(maxima[key] for key in ("force_n", "moment_nm")) > 1e-7
        or fingerprint(controller.data, model) != unchanged
    ):
        raise RuntimeError(f"Independent branch proof failed: {maxima}")
    return {
        "max_errors": {key: float(value) for key, value in maxima.items()},
        "substeps": cursor,
        "main_scene_unchanged": True,
        "sensor_timing": "pre-integration geometry/COMv and new-control forward-dynamics contact wrenches; FULLPHYSICS after integration",
    }


def min_windows(values, length):
    return np.minimum.reduce(
        [values[:, k : values.shape[1] - length + k + 1] for k in range(length)]
    )


def max_windows(values, length):
    return np.maximum.reduce(
        [values[:, k : values.shape[1] - length + k + 1] for k in range(length)]
    )


def sequence_cost(controller, states, sensors, targets, config):
    """Coupled event timing and complete recovery; guidance is not certification."""
    sl, model, ix = controller.slices, controller.model, controller.index
    count, samples = states.shape[:2]
    dt = config.physics_dt
    times = np.arange(samples) * dt
    initial = np.broadcast_to(controller.state(), (count, 1, states.shape[-1]))
    pre = np.concatenate([initial, states[:, :-1]], axis=1)
    bq, bp = sensors[..., sl["boardquat"]], sensors[..., sl["boardpos"]]
    rq, rp = sensors[..., sl["rootquat"]], sensors[..., sl["rootpos"]]
    board_up = 1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)
    root_up = 1 - 2 * (rq[..., 1] ** 2 + rq[..., 2] ** 2)
    lower = np.stack(
        [
            envelope_z(
                model,
                model.geom(name).id,
                sensors[..., sl[name + "_pos"]],
                sensors[..., sl[name + "_quat"]],
            )
            for name in controller.board_geoms
        ],
        axis=-1,
    ).min(-1)
    floor_force = np.stack(
        [
            np.maximum(sensors[..., sl["ground_" + name]][..., 3], 0)
            for name in controller.board_geoms
        ],
        axis=-1,
    )
    any_floor = floor_force.max(-1)
    foot = {}
    for side in ("left", "right"):
        foot[side] = sum(
            sensors[..., sl[side + "_" + box]][..., 1:4] for box in ("center", "tail", "nose")
        )
    front_load = np.maximum(foot["left"][..., 2], 0)
    rear_load = np.maximum(foot["right"][..., 2], 0)
    tail_force = np.maximum(sensors[..., sl["ground_board_tail"]][..., 3], 0)
    masses = np.array(
        [model.body_subtreemass[model.body("pelvis").id], model.body_subtreemass[ix.board_body]]
    )
    com_vz = (
        masses[0] * sensors[..., sl["root_com_v"]][..., 2]
        + masses[1] * sensors[..., sl["board_com_v"]][..., 2]
    ) / masses.sum()
    # A front-foot force on the board is the opposite of its native contact
    # force on the foot. This must be positive in world Z to be a lifting task.
    front_lift = np.maximum(-sensors[..., sl["left_nose"]][..., 3], 0)
    foot_gap = finite_capsule_witness_gap(controller, sensors[:, :361:2], "left", "board_nose")
    # Interpolation is only for this soft finite-witness task. Native wrench,
    # envelope, support and actual outcome remain at the full 400 Hz.
    gap = np.stack(
        [np.interp(np.arange(samples), np.arange(0, min(samples, 361), 2), row) for row in foot_gap]
    )
    q = pre[..., 1 + ix.robot_joint_qpos]
    limits = np.maximum(
        np.maximum(controller.limits[:, 0] - q, q - controller.limits[:, 1]), 0
    ).max(-1)
    depth = np.maximum(-sensors[..., sl["depth"]][..., 1], 0)
    ground = sensors[..., sl["robot_ground"]][..., 0] > 0
    # Continuous severity keeps candidate ranking informative even when all
    # candidates miss the complete task. Actual acceptance still uses v3.
    safety = 4000 * ground.mean(1)
    safety += 1000 * (depth.max(1) / 0.01) ** 2 + 100 * np.mean((depth / 0.005) ** 2, 1)
    safety += 1000 * (limits.max(1) / 0.05) ** 2
    safety += 2000 * np.mean((np.maximum(-board_up, 0) / 0.5) ** 2, 1)
    safety += 300 * np.mean((np.maximum(0.2 - root_up, 0) / 0.2) ** 2, 1)
    # Joint sample at 5.14 is post-step and is also included in limit severity.
    finalq = states[:, -1, 1 + ix.robot_joint_qpos]
    final_limit = np.maximum(
        np.maximum(controller.limits[:, 0] - finalq, finalq - controller.limits[:, 1]), 0
    ).max(-1)
    safety += 1000 * (final_limit / 0.05) ** 2
    clear40 = min_windows(lower, 16)
    floor40 = max_windows(any_floor, 16)
    flight_cost = 2200 * (np.maximum(0.008 - clear40, 0) / 0.04) ** 2
    flight_cost += 800 * np.minimum(floor40 / 5, 1) ** 2
    front20 = rolling_mean(front_load, 8)
    rear20 = rolling_mean(rear_load, 8)
    tail5 = rolling_mean(tail_force, 2)
    best_sync = np.full(count, np.inf)
    best_anchor, best_flight = np.zeros(count, int), np.zeros(count, int)
    best_parts = {
        name: np.zeros(count)
        for name in (
            "tail_touch",
            "upward_at_tail",
            "front_unload",
            "rear_release",
            "front_drag",
            "flight",
        )
    }
    # Every component below is conditioned on the SAME impact anchor. Later
    # events use bounded windows relative to it, rather than unrelated peaks.
    for anchor in np.flatnonzero((times >= 0.10) & (times <= 0.70)):
        release_start, release_end = anchor + 4, anchor + 25
        drag_start, drag_end = anchor, anchor + 49
        lift_impulse = front_lift[:, drag_start:drag_end].sum(1) * dt
        drag_gap = gap[:, drag_start:drag_end].min(1)
        tail_bottom = envelope_z(
            model,
            model.geom("board_tail").id,
            sensors[:, anchor, sl["board_tail_pos"]],
            sensors[:, anchor, sl["board_tail_quat"]],
        )
        parts = {
            "tail_touch": 700 * np.maximum(1 - tail5[:, anchor] / 5, 0) ** 2
            + 300 * (np.maximum(tail_bottom, 0) / 0.068) ** 2,
            "upward_at_tail": 600 * (np.maximum(0.6 - com_vz[:, anchor], 0) / 0.6) ** 2,
            "front_unload": 100 * np.minimum(front20[:, anchor - 7] / 40, 5) ** 2,
            "rear_release": 200
            * np.minimum(rear20[:, release_start:release_end].min(1) / 40, 5) ** 2,
            "front_drag": 400 * np.minimum(drag_gap / 0.04, 5) ** 2
            + 400 * np.maximum(1 - lift_impulse / 0.05, 0) ** 2,
        }
        eligible = flight_cost[:, anchor + 2 : anchor + 161]
        flight_slot = np.argmin(eligible, axis=1)
        parts["flight"] = eligible[np.arange(count), flight_slot]
        # A continuous 40 ms window has a lower-clearance goal; a true peak
        # height task adds the original 15 mm scale in its associated interval.
        peak = lower[:, anchor + 2 : anchor + 177].max(1)
        parts["flight"] += 500 * (np.maximum(0.015 - peak, 0) / 0.015) ** 2
        value = sum(parts.values())
        improved = value < best_sync
        best_sync[improved] = value[improved]
        best_anchor[improved] = anchor
        best_flight[improved] = anchor + 2 + flight_slot[improved]
        for name, part in best_parts.items():
            part[improved] = parts[name][improved]
    final = times >= config.horizon * config.control_dt - 0.3
    qvel = pre[..., 1 + model.nq :]
    rv = np.linalg.norm(qvel[..., ix.robot_qvel : ix.robot_qvel + 3], axis=-1)
    rw = np.linalg.norm(qvel[..., ix.robot_qvel + 3 : ix.robot_qvel + 6], axis=-1)
    bv = np.linalg.norm(qvel[..., ix.board_qvel : ix.board_qvel + 3], axis=-1)
    bw = np.linalg.norm(qvel[..., ix.board_qvel + 3 : ix.board_qvel + 6], axis=-1)
    support = 600 * np.mean(
        (1 - np.clip(front_load[:, final] / 10, 0, 1)) ** 2
        + (1 - np.clip(rear_load[:, final] / 10, 0, 1)) ** 2,
        1,
    )
    wheel_indices = [
        i for i, name in enumerate(controller.board_geoms) if name.startswith("wheel_")
    ]
    support += 600 * np.mean(
        (1 - np.clip(floor_force[:, final][:, :, wheel_indices] / 1, 0, 1)) ** 2, (1, 2)
    )
    terminal = 300 * np.mean((np.maximum(0.8 - root_up[:, final], 0) / 0.2) ** 2, 1)
    terminal += 300 * np.mean((np.maximum(0.95 - board_up[:, final], 0) / 0.2) ** 2, 1)
    terminal += 300 * np.mean((np.maximum(0.55 - (rp - bp)[:, final, 2], 0) / 0.2) ** 2, 1)
    terminal += 100 * np.mean(
        (np.maximum(rv[:, final] - 0.3, 0) / 0.3) ** 2
        + (np.maximum(rw[:, final] - 0.8, 0) / 0.8) ** 2
        + (np.maximum(bv[:, final] - 0.35, 0) / 0.35) ** 2
        + (np.maximum(bw[:, final] - 0.8, 0) / 0.8) ** 2,
        1,
    )
    # Catch is sought after the selected airborne window. The mean over a
    # 100ms window prevents a single incidental force spike satisfying it.
    catch = np.empty(count)
    for candidate in range(count):
        begin = min(best_flight[candidate] + 16, samples - 80)
        end = min(begin + 240, samples)
        force_quality = (1 - np.clip(front_load[candidate, begin:end] / 10, 0, 1)) ** 2 + (
            1 - np.clip(rear_load[candidate, begin:end] / 10, 0, 1)
        ) ** 2
        catch[candidate] = 500 * rolling_mean(force_quality[None], 40).min()
    drift = 10 * np.mean(
        np.sum(((bp[..., :2] - controller.data.xpos[ix.board_body, :2]) / 0.3) ** 2, -1), 1
    )
    previous = controller.data.ctrl[ix.robot_actuators]
    delta = np.diff(np.concatenate([np.broadcast_to(previous, (count, 1, 29)), targets], 1), axis=1)
    smooth = 0.1 * np.mean((delta / 0.25) ** 2, (1, 2))
    terms = {
        **best_parts,
        "physical_severity": safety,
        "catch_support": catch,
        "final_support": support,
        "final_stability": terminal,
        "board_xy_drift": drift,
        "target_change": smooth,
    }
    cost = sum(terms.values())
    valid = np.isfinite(states).all((1, 2)) & np.isfinite(sensors).all((1, 2)) & np.isfinite(cost)
    valid &= np.isclose(
        states[:, -1, 0],
        controller.data.time + config.horizon * config.control_dt,
        rtol=0,
        atol=1e-9,
    )
    metrics = {
        "anchor_time_s": config.prefix_controls * config.control_dt + times[best_anchor],
        "flight_window_guidance_start_s": config.prefix_controls * config.control_dt
        + times[best_flight],
        "com_vz_at_anchor_m_s": com_vz[np.arange(count), best_anchor],
        "front_nose_gap_at_anchor_m": gap[np.arange(count), best_anchor],
        "maximum_whole_board_clearance_m": lower.max(1),
        "maximum_penetration_m": depth.max(1),
        "robot_ground_fraction": ground.mean(1),
        "maximum_actual_joint_limit_violation_rad": np.maximum(limits.max(1), final_limit),
        "minimum_board_upright": board_up.min(1),
        "minimum_robot_upright": root_up.min(1),
        "final_front_support_fraction_proxy": (front_load[:, final] > 1).mean(1),
        "final_rear_support_fraction_proxy": (rear_load[:, final] > 1).mean(1),
    }
    return np.where(valid, cost, np.inf), terms, metrics


def interpolate_knots(knots, nominal, config):
    grid = np.arange(config.horizon) * config.control_dt
    result = np.broadcast_to(nominal, (len(knots), *nominal.shape)).copy()
    for candidate, row in enumerate(knots):
        for joint in range(config.dimensions):
            result[candidate, :, joint] += np.interp(grid, config.knot_times_s, row[:, joint])
    return result


def fit_knots(target, nominal, config):
    grid = np.arange(config.horizon) * config.control_dt
    return np.stack(
        [
            np.interp(config.knot_times_s, grid, target[:, j] - nominal[:, j])
            for j in range(config.dimensions)
        ],
        -1,
    )


def prepare_priors(controller, args, config):
    grid = np.arange(config.horizon) * config.control_dt
    start = controller.data.ctrl[controller.index.robot_actuators].copy()
    kp = controller.model.actuator_gainprm[controller.index.robot_actuators, 0]
    standing = controller.reference.robot_state[0, 7:36] + controller.ff_tau[0] / kp
    standing = np.clip(standing, *controller.target_limits.T)
    blend = smoothstep((grid - 0.65) / 1.15)[:, None]
    nominal = start * (1 - blend) + standing * blend
    with np.load(args.old_prior, allow_pickle=False) as archive:
        old = archive["action"][130:170].copy()
    if old.shape != (40, 29):
        raise ValueError("Old mu=.6 prior must contain exactly its 40 evaluated pop controls")
    prior = np.repeat(old[-1:], config.horizon, axis=0)
    prior[:40] = old
    blend = smoothstep((grid - 0.80) / 0.90)[:, None]
    prior = prior * (1 - blend) + standing * blend
    priors = {"old_mu06_pop_controls_plus_standing_target": prior}
    if args.ik_prior:
        with np.load(args.ik_prior, allow_pickle=False) as archive:
            target = archive["action_targets"].copy()
            stamps = archive["time_since_prefix"].copy()
        if target.ndim != 2 or target.shape != (len(stamps), 29) or len(stamps) < 2:
            raise ValueError("IK prior requires action_targets(K,29), time_since_prefix(K)")
        if (
            not np.isfinite(target).all()
            or not np.isfinite(stamps).all()
            or np.any(np.diff(stamps) <= 0)
        ):
            raise ValueError("IK prior must be finite and strictly time-ordered")
        priors["kinematic_ik_target_not_a_physical_demonstration"] = np.stack(
            [np.interp(grid, stamps, target[:, j]) for j in range(29)], -1
        )
    for name, value in priors.items():
        priors[name] = np.clip(value, *controller.target_limits.T)
    return (
        nominal,
        priors,
        {
            "nominal": "last actual prefix target held to +.65s, quintic to valid initial standing q+tau0/Kp by +1.80s",
            "old_prior": "old mu=.6 actual controls, resimulated in mu=.9; no endpoint/state copy; not a teacher",
            "standing_target": standing.tolist(),
            "ik_prior": "kinematic target prior only; dynamically re-evaluated"
            if args.ik_prior
            else None,
        },
    )


def execute_winner(controller, prefix, prefix_records, targets, predicted, config):
    model, data, ix = controller.model, controller.data, controller.index
    records = list(prefix_records)
    arrays = {
        name: [v.copy() for v in prefix[name][: config.prefix_controls + 1]]
        for name in ("state", "qpos", "qvel")
    }
    actions = list(prefix["action"][: config.prefix_controls].copy())
    times = list(prefix["time_physics"][: config.prefix_controls + 1])
    failure, partial, maximum = None, None, 0.0
    for frame, target in enumerate(targets):
        advanced, committed = 0, False
        try:
            data.ctrl[ix.robot_actuators] = target
            for substep in range(8):
                mujoco.mj_step(model, data)
                advanced += 1
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Native numerical failure")
                row = controller.observer.observe(data)
                records.append(row)
                if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                    raise RuntimeError("Unexpected external force")
                error = float(abs(controller.state() - predicted[frame * 8 + substep]).max())
                maximum = max(maximum, error)
            # Save every completed control before reporting a prediction error.
            actions.append(target.copy())
            arrays["state"].append(controller.indices.pack(data.qpos, data.qvel).numpy())
            arrays["qpos"].append(data.qpos.copy())
            arrays["qvel"].append(data.qvel.copy())
            times.append(float(data.time))
            committed = True
            if maximum > 1e-10:
                raise RuntimeError("Actual winner differs from its evaluated full branch")
        except (RuntimeError, ValueError) as error:
            failure = str(error)
            if advanced and not committed:
                partial = {
                    "control_step": config.prefix_controls + frame,
                    "target": target.tolist(),
                    "native_substeps_advanced": advanced,
                    "actual_time_s": float(data.time),
                    "omitted_from_regular_20ms_boundary_arrays": True,
                }
            break
    complete = len(actions) == 257 and abs(data.time - 5.14) < 1e-9 and failure is None
    result = outcome(records, full_reference=complete, numerical_failure=failure)
    result.update(
        schema="supermimic.ollie.contact_sequence_actual.v1",
        initializations=1,
        phase_resets=0,
        root_board_writes_after_initialization=0,
        board_actuators=0,
        external_forces_applied=False,
        actual_control_dim=29,
        actual_endpoint_s=float(data.time),
        partial_control=partial,
        actual_prediction_max_error=maximum,
        verified_teacher=False,
        supermimic_training_performed=False,
        foot_deck_friction=config.foot_deck_friction,
        model_material_is_unvalidated_engineering_assumption=True,
    )
    packed = {name: np.asarray(value) for name, value in arrays.items()}
    packed.update(
        action=np.asarray(actions),
        time_physics=np.asarray(times),
        raw_terminal_qpos=data.qpos.copy(),
        raw_terminal_qvel=data.qvel.copy(),
        raw_terminal_time_s=np.asarray(data.time),
    )
    return result, packed, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "benchmark", "run"), default="inspect")
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--prefix-sha256", required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
    )
    parser.add_argument(
        "--feedforward",
        type=Path,
        default=Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"),
    )
    parser.add_argument(
        "--old-prior",
        type=Path,
        default=Path("logs/ollie_pop_events_preflight_02/actual_rollout.npz"),
    )
    parser.add_argument("--ik-prior", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory must be new")
    config, started = Config(), time.monotonic()
    sources = {}
    for name, expected in (
        ("prefix", args.prefix_sha256),
        ("reference", REFERENCE_HASH),
        ("feedforward", FEEDFORWARD_HASH),
        ("old_prior", OLD_PRIOR_HASH),
    ):
        path = getattr(args, name)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Frozen {name} SHA differs")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}
    if args.ik_prior:
        sources["ik_prior"] = {
            "path": str(args.ik_prior.resolve()),
            "sha256": digest(args.ik_prior),
        }
    with np.load(args.prefix, allow_pickle=False) as archive:
        prefix = {name: archive[name].copy() for name in archive.files}
    shapes = {"state": (131, 92), "qpos": (131, 47), "qvel": (131, 45), "action": (130, 29)}
    if any(
        prefix[name].shape != shape or not np.isfinite(prefix[name]).all()
        for name, shape in shapes.items()
    ):
        raise ValueError("Expected the exact new-model 2.60 s prefix")
    if (
        tuple(prefix["robot_joint_names"]) != G1_JOINT_NAMES
        or tuple(prefix["board_joint_names"]) != BOARD_JOINT_NAMES
        or not np.array_equal(prefix["time_physics"], np.arange(131) * 0.02)
    ):
        raise ValueError("Prefix names/time convention mismatch")
    manifest = {
        "schema": "supermimic.ollie.contact_sequence_manifest.v1",
        "mode": args.mode,
        "sources": sources,
        "config": asdict(config),
        "script_sha256": digest(__file__),
        "mujoco_version": mujoco.__version__,
        "failed_54_used": False,
        "verified_teacher": False,
        "physical_model": "explicit 42 foot/deck pairs mu=.9; uncalibrated engineering model",
        "selection": "one complete previously evaluated branch; never an untested CEM mean",
    }
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "frozen_manifest.json", manifest)
    if args.mode == "inspect":
        print(json.dumps(manifest, allow_nan=False))
        return
    controller = NativeSamplingMPC(
        args.reference,
        args.feedforward,
        seed=config.seed,
        foot_deck_friction=config.foot_deck_friction,
    )
    controller.prefix_path = args.prefix
    controller.horizon = config.horizon
    search_rounds, best = [], None
    try:
        prefix_records, proof = replay_prefix(controller, prefix, config)
        write_json(args.out_dir / "prefix_proof.json", proof)
        model_proof = add_sensors(controller, config)
        if (
            controller.model.npair != 42
            or controller.model.neq
            or controller.model.na
            or controller.model.nmocap
        ):
            raise RuntimeError("Unexpected physical model/actuation state")
        nominal, priors, prior_metadata = prepare_priors(controller, args, config)
        write_json(
            args.out_dir / "model_and_prior_proof.json",
            {"model": model_proof, "prior": prior_metadata},
        )
        np.savez_compressed(args.out_dir / "target_priors.npz", nominal=nominal, **priors)
        mean = fit_knots(next(iter(priors.values())), nominal, config)
        std = np.full_like(mean, config.sigma_rad)
        for iteration in range(config.rounds):
            if time.monotonic() - started >= config.wall_budget_s - 5:
                raise TimeoutError("Fixed 300s budget: incomplete search; no winner executed")
            params = controller.rng.normal(mean, std, (config.candidates, *mean.shape))
            params[0] = mean
            params[1] = 0
            targets = interpolate_knots(params, nominal, config)
            labels = {0: "current_mean_evaluated_now", 1: "standing_nominal"}
            slot = 2
            if best is not None:
                targets[slot], params[slot] = best[1], best[4]
                labels[slot] = "previous_evaluated_full_winner"
                slot += 1
            for name, prior in priors.items():
                targets[slot], params[slot] = prior, fit_knots(prior, nominal, config)
                labels[slot] = name
                slot += 1
            targets = np.clip(targets, *controller.target_limits.T)
            begin = time.monotonic()
            states, sensors = controller.rollout(targets)
            rollout_wall = time.monotonic() - begin
            begin = time.monotonic()
            costs, terms, metrics = sequence_cost(controller, states, sensors, targets, config)
            cost_wall = time.monotonic() - begin
            order = np.argsort(costs)
            finite = order[np.isfinite(costs[order])]
            if not len(finite):
                raise RuntimeError("All full native branches are incomplete or nonfinite")
            winner = int(finite[0])
            if iteration < config.early_sync_rounds:
                sync_names = {
                    "tail_touch",
                    "upward_at_tail",
                    "front_unload",
                    "rear_release",
                    "front_drag",
                    "flight",
                    "physical_severity",
                }
                selection = sum(
                    value * (1 if name in sync_names else config.early_recovery_weight)
                    for name, value in terms.items()
                )
                selection = np.where(np.isfinite(costs), selection, np.inf)
            else:
                selection = costs
            selection_order = np.argsort(selection)
            elites = selection_order[np.isfinite(selection[selection_order])][: config.elites]
            report = {
                "round": iteration,
                "rollout_wall_s": rollout_wall,
                "cost_wall_s": cost_wall,
                "cost": [float(v) if np.isfinite(v) else None for v in costs],
                "distribution_update_cost": [
                    float(v) if np.isfinite(v) else None for v in selection
                ],
                "distribution_update_phase": "early_sync"
                if iteration < config.early_sync_rounds
                else "full",
                "valid_mask": np.isfinite(costs).tolist(),
                "elite_ids": elites.tolist(),
                "special_candidate_labels": labels,
                "best_candidate": winner,
                "best_cost": float(costs[winner]),
                "best_terms": {name: float(value[winner]) for name, value in terms.items()},
                "best_metrics": {name: float(value[winner]) for name, value in metrics.items()},
                "special_candidate_terms": {
                    str(candidate): {
                        "cost": float(costs[candidate]) if np.isfinite(costs[candidate]) else None,
                        "terms": {
                            name: float(value[candidate]) if np.isfinite(value[candidate]) else None
                            for name, value in terms.items()
                        },
                    }
                    for candidate in labels
                },
            }
            if best is None or costs[winner] < best[0]:
                best = (
                    float(costs[winner]),
                    targets[winner].copy(),
                    states[winner].copy(),
                    sensors[winner].copy(),
                    params[winner].copy(),
                    iteration,
                    winner,
                )
            if iteration == 0:
                first_proof = branch_proof(
                    controller, targets[winner], states[winner], sensors[winner]
                )
                projected = (rollout_wall + cost_wall) * config.rounds
                benchmark = {
                    "one_full_batch_rollout_s": rollout_wall,
                    "one_full_batch_cost_s": cost_wall,
                    "projected_64_round_seconds": projected,
                    "first_branch_proof": first_proof,
                    "first_batch": report,
                    "search_update_performed": False,
                }
                write_json(args.out_dir / "benchmark.json", benchmark)
                if args.mode == "benchmark":
                    return
                if projected + (time.monotonic() - started) > config.wall_budget_s - 15:
                    raise RuntimeError(
                        "Full-cost benchmark exceeds fixed 300s gate; no CEM update or actual winner execution"
                    )
            search_rounds.append(report)
            write_json(args.out_dir / "search_rounds.json", search_rounds)
            mean = params[elites].mean(0)
            std = np.clip(params[elites].std(0), config.sigma_floor_rad, config.sigma_rad)
            if iteration % 4 == 0 or iteration == config.rounds - 1:
                print(
                    json.dumps(
                        {
                            "round": iteration,
                            "best_full_cost": best[0],
                            "current_terms": report["best_terms"],
                            "wall_s": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        if (
            time.monotonic() - started
            > config.wall_budget_s - config.winner_proof_and_execution_reserve_s
        ):
            raise TimeoutError(
                "Fixed 300s budget: insufficient reserve for winner proof and actual execution"
            )
        winner_proof = branch_proof(controller, best[1], best[2], best[3])
        write_json(args.out_dir / "winner_branch_proof.json", winner_proof)
        np.savez_compressed(
            args.out_dir / "winner_evaluated_branch.npz",
            action=best[1],
            fullphysics=best[2],
            sensors=best[3],
            parameters=best[4],
            best_cost=np.asarray(best[0]),
            round=np.asarray(best[5]),
            candidate=np.asarray(best[6]),
        )
        if time.monotonic() - started > config.wall_budget_s - config.actual_execution_reserve_s:
            raise TimeoutError("Fixed 300s budget: insufficient reserve before actual execution")
        result, arrays, records = execute_winner(
            controller, prefix, prefix_records, best[1], best[2], config
        )
        result.update(
            wall_seconds=time.monotonic() - started,
            frozen_manifest=manifest,
            completed_search_rounds=len(search_rounds),
            winner_cost=best[0],
            winner_round=best[5],
            winner_candidate=best[6],
        )
        np.savez_compressed(
            args.out_dir / "actual_rollout.npz",
            **arrays,
            time_video=arrays["time_physics"] * 1.5,
            control_dt=np.asarray(config.control_dt),
            robot_joint_names=np.asarray(G1_JOINT_NAMES),
            board_joint_names=np.asarray(BOARD_JOINT_NAMES),
            verified_teacher=np.asarray(False),
        )
        write_json(args.out_dir / "result.json", result)
        write_json(args.out_dir / "contacts.json", records)
        for name, source in sources.items():
            if digest(source["path"]) != source["sha256"]:
                raise RuntimeError(f"Input {name} changed during execution")
        write_json(
            args.out_dir / "output_hashes.json",
            {path.name: digest(path) for path in sorted(args.out_dir.iterdir()) if path.is_file()},
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    except (RuntimeError, ValueError, TimeoutError) as error:
        if best is not None:
            np.savez_compressed(
                args.out_dir / "best_so_far_evaluated_branch.npz",
                action=best[1],
                fullphysics=best[2],
                sensors=best[3],
                parameters=best[4],
                best_cost=np.asarray(best[0]),
                round=np.asarray(best[5]),
                candidate=np.asarray(best[6]),
                verified_teacher=np.asarray(False),
            )
        write_json(
            args.out_dir / "failure.json",
            {
                "error": str(error),
                "actual_endpoint_s": float(controller.data.time),
                "completed_search_rounds": len(search_rounds),
                "main_actual_continuation_executed": controller.data.time > 2.60 + 1e-9,
                "main_fingerprint": fingerprint(controller.data, controller.model),
                "best_evaluated_cost": float(best[0]) if best else None,
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        controller.pool.close()


if __name__ == "__main__":
    main()
