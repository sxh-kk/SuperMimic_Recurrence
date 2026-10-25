"""Fixed native-physics momentum/release data exploration; no learned policy.

One original initialization replays the 60 archive to 3.12 s. Scratch branches
share that hot state. Only one complete, evaluated 29-PD-target suffix may be
executed. Exploration grade is never an Ollie success certificate.
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

from scripts.local import optimize_ollie_contact_sequence as base
from scripts.local.continue_ollie_sampling_mpc import fingerprint
from scripts.local.evaluate_ollie_policy import outcome
from scripts.local.optimize_ollie_pop_events import replay_prefix, rolling_mean
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_JOINT_NAMES, BOARD_XML, make_scene_spec

SOURCE_SHA = "43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d"
SCHEMA = "supermimic.ollie.momentum_release.v1"


@dataclass(frozen=True)
class Config:
    prefix_controls: int = 156
    horizon: int = 101
    candidates: int = 256
    rounds: int = 64
    elites: int = 16
    dimensions: int = 15
    knot_times_s: tuple = (0, 0.02, 0.04, 0.08, 0.12, 0.18, 0.26, 0.38, 0.54, 0.76, 1.04, 1.42, 2.0)
    sigma_rad: float = 0.25
    sigma_floor_rad: float = 0.025
    seed: int = 76
    threads: int = 16
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    wall_budget_s: float = 300.0
    winner_proof_and_execution_reserve_s: float = 15.0
    actual_execution_reserve_s: float = 5.0
    foot_deck_friction: None = None
    selection: str = "same_lexicographic_grade_then_cost_all_64_rounds"


digest, write_json = base.digest, base.write_json
DEFAULT_CONFIG = Config()


def physical_compare(original, observed):
    """Only site/sensor/name metadata may differ; every physical array is exact."""
    compared, ignored = [], []
    for name in dir(original):
        if name.startswith(("_", "sensor_", "site_", "name_")) or name in ("names", "names_map"):
            continue
        value = getattr(original, name)
        if isinstance(value, np.ndarray):
            compared.append(name)
            if not np.array_equal(value, getattr(observed, name)):
                raise RuntimeError(f"Read-only observer changed physical array {name}")
    for name in dir(original.opt):
        if name.startswith("_"):
            continue
        value = getattr(original.opt, name)
        if isinstance(value, (np.ndarray, int, float)):
            compared.append("opt." + name)
            if not np.array_equal(value, getattr(observed.opt, name)):
                raise RuntimeError(f"Read-only observer changed option {name}")
    for name in (
        "nq",
        "nv",
        "nu",
        "na",
        "neq",
        "nmocap",
        "nplugin",
        "nuserdata",
        "ngeom",
        "nbody",
        "npair",
    ):
        if getattr(original, name) != getattr(observed, name):
            raise RuntimeError(f"Read-only observer changed {name}")
    if observed.nsite != original.nsite + 2:
        raise RuntimeError("Expected exactly two additional observation-only sites")
    # Appended sites can reorder site IDs by body. Compare every original site
    # by name, rather than silently exempting their positions/frames.
    for old in range(original.nsite):
        name = original.site(old).name
        new = observed.site(name).id
        for key in (
            "site_type",
            "site_bodyid",
            "site_pos",
            "site_quat",
            "site_size",
            "site_rgba",
            "site_group",
        ):
            if not np.array_equal(getattr(original, key)[old], getattr(observed, key)[new]):
                raise RuntimeError(f"Original site changed: {name}/{key}")
        ignored.append({"original_site": name, "name_mapped_arrays_exact": True})
    return {
        "physical_arrays_and_options_exact": True,
        "compared_fields": compared,
        "original_sites": ignored,
        "added_sites": 2,
        "sites_have_no_mass_or_collision": True,
    }


def add_sensors(controller, config):
    """65's native observations plus two whole-foot TOUCH sites; no dynamics edit."""
    spec = make_scene_spec(timestep=config.physics_dt, foot_deck_friction=None)
    controller.sensor_descriptions = {}

    def pose(name, objtype, objname):
        for suffix, kind in (
            ("pos", mujoco.mjtSensor.mjSENS_FRAMEPOS),
            ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ):
            spec.add_sensor(
                name="seq_" + name + suffix, type=kind, objtype=objtype, objname=objname
            )
            controller.sensor_descriptions[name + suffix] = (kind, objtype, objname)

    for name, body in (("root", "pelvis"), ("board", "board")):
        pose(name, mujoco.mjtObj.mjOBJ_XBODY, body)
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
        raise RuntimeError("Expected all nine physical board geometries")
    for name in controller.board_geoms:
        pose(name + "_", mujoco.mjtObj.mjOBJ_GEOM, name)
    found = 1 << int(mujoco.mjtConDataField.mjCONDATA_FOUND)
    force_field = 1 << int(mujoco.mjtConDataField.mjCONDATA_FORCE)
    fields = found | force_field
    for name in ("TORQUE", "POS"):
        fields |= 1 << int(getattr(mujoco.mjtConDataField, "mjCONDATA_" + name))
    controller.pairs, controller.foot_geometry, site_manifest = {}, {}, {}

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
            raise RuntimeError("Expected seven authored capsules on each foot body")
        body = controller.model.body(int(bodies[0])).name
        pose(side + "_body_", mujoco.mjtObj.mjOBJ_XBODY, body)
        center, size = controller.model.geom_pos[ids], controller.model.geom_size[ids, :2]
        axis = base.rotate(controller.model.geom_quat[ids], np.array([0.0, 0.0, 1.0]))
        controller.foot_geometry[side] = {
            "body": int(bodies[0]),
            "geoms": ids,
            "center": center.copy(),
            "axis": axis,
            "size": size.copy(),
        }
        extent = abs(axis) * size[:, 1:2] + size[:, :1]
        low, high = (center - extent).min(0) - 0.015, (center + extent).max(0) + 0.015
        site = "release_" + side + "_normal_region"
        spec.body(body).add_site(
            name=site,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(low + high) / 2,
            size=(high - low) / 2,
            rgba=[0, 0, 0, 0],
        )
        spec.add_sensor(
            name="seq_" + side + "_normal_load",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site,
        )
        site_manifest[side] = {
            "body": body,
            "site": site,
            "local_aabb_m": [low.tolist(), high.tolist()],
            "padding_m": 0.015,
            "all_body_contacts_including_ground_self": True,
        }
        for box in ("center", "tail", "nose"):
            name = side + "_" + box
            contact(name, "board_" + box, body, mujoco.mjtObj.mjOBJ_XBODY)
            controller.pairs[name] = (controller.model.geom("board_" + box).id, set(ids))
    for name in controller.board_geoms:
        contact("ground_" + name, "ground", name)
        controller.pairs["ground_" + name] = (
            controller.model.geom("ground").id,
            {controller.model.geom(name).id},
        )
    # A single real contact's normal cannot be manufactured by cancellation or
    # a sum of subthreshold contacts. Maxforce selects by full wrench norm;
    # its Fn>=.1 is conservative evidence of a genuine canonical tail contact.
    spec.add_sensor(
        name="seq_tail_strongest",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_GEOM,
        objname="ground",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="board_tail",
        intprm=[found | force_field, 2, 1],
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
    for name in (
        *controller.sensor_descriptions,
        *controller.pairs,
        "depth",
        "robot_ground",
        "left_normal_load",
        "right_normal_load",
        "tail_strongest",
    ):
        sensor = controller.branch_model.sensor("seq_" + name).id
        begin = int(controller.branch_model.sensor_adr[sensor])
        controller.slices[name] = slice(
            begin, begin + int(controller.branch_model.sensor_dim[sensor])
        )
    for name in (*controller.pairs, "left_normal_load", "right_normal_load", "tail_strongest"):
        sensor = controller.branch_model.sensor("seq_" + name).id
        if controller.branch_model.sensor_needstage[sensor] != mujoco.mjtStage.mjSTAGE_ACC:
            raise RuntimeError("Contact/TOUCH must be acceleration-stage observations")
    return {
        **physical_compare(controller.model, controller.branch_model),
        "touch_sites": site_manifest,
    }


def foot_forces(controller, data):
    """Read the executed solve, before any extra forward; never write a state."""
    model, observer = controller.model, controller.observer
    result = {
        side: {"all_normal_n": 0.0, "board_normal_n": 0.0, "board_on_foot_fz_n": 0.0}
        for side in ("left", "right")
    }
    local = np.empty(6)
    tail_normals, tail_norms = [], []
    tail, ground = model.geom("board_tail").id, model.geom("ground").id
    for i, c in enumerate(data.contact[: data.ncon]):
        a, b = map(int, c.geom)
        mujoco.mj_contactForce(model, data, i, local)
        fn = max(float(local[0]), 0.0)
        if {a, b} == {tail, ground}:
            tail_normals.append(fn)
            tail_norms.append(float(np.linalg.norm(local[:3])))
        for side, feet in observer.feet.items():
            if a not in feet and b not in feet:
                continue
            result[side]["all_normal_n"] += fn
            other = b if a in feet else a
            if other in observer.board:
                result[side]["board_normal_n"] += fn
                fz = float((np.asarray(c.frame).reshape(3, 3).T @ local[:3])[2])
                result[side]["board_on_foot_fz_n"] += -fz if a in feet else fz
    strongest = tail_normals[int(np.argmax(tail_norms))] if tail_norms else 0.0
    return {
        **{f"{side}_{key}": value for side, row in result.items() for key, value in row.items()},
        "tail_ground_normal_sum_n": sum(tail_normals),
        "tail_strongest_normal_n": strongest,
    }


def continuation_regression(controller, archive, config=DEFAULT_CONFIG):
    """Replay every remaining archived control on scratch, never on main."""
    saved = json.loads((controller.prefix_path.parent / "contacts.json").read_text())
    before = fingerprint(controller.data, controller.model)
    scratch, maxima, records = (
        copy.copy(controller.data),
        {"qpos": 0.0, "qvel": 0.0, "state": 0.0},
        0,
    )
    for frame in range(config.prefix_controls, len(archive["action"])):
        scratch.ctrl[controller.index.robot_actuators] = archive["action"][frame]
        for substep in range(8):
            mujoco.mj_step(controller.model, scratch)
            mujoco.mj_forward(controller.model, scratch)
            row = controller.observer.observe(scratch)
            source = saved[frame * 8 + substep + 1]
            if any(row[key] != source[key] for key in row):
                raise RuntimeError(
                    f"Old60 continuation Observer regression differs at {frame}/{substep}"
                )
            records += 1
        for name, actual in (
            ("qpos", scratch.qpos),
            ("qvel", scratch.qvel),
            ("state", controller.indices.pack(scratch.qpos, scratch.qvel).numpy()),
        ):
            maxima[name] = max(maxima[name], float(abs(actual - archive[name][frame + 1]).max()))
            if not np.array_equal(actual, archive[name][frame + 1]):
                raise RuntimeError(f"Old60 continuation {name} differs at {frame + 1}")
    if fingerprint(controller.data, controller.model) != before:
        raise RuntimeError("Continuation regression mutated the real hot state")
    return {
        "controls": len(archive["action"]) - config.prefix_controls,
        "observer_rows": records,
        "all_observer_fields_exact": True,
        "maximum_errors": maxima,
        "endpoint_s": float(scratch.time),
        "main_unchanged": True,
    }


def prepare_priors(controller, source, extra, config):
    grid = np.arange(config.horizon) * config.control_dt
    start = controller.data.ctrl[controller.index.robot_actuators].copy()
    kp = controller.model.actuator_gainprm[controller.index.robot_actuators, 0]
    standing = controller.reference.robot_state[0, 7:36] + controller.ff_tau[0] / kp
    standing = np.clip(standing, *controller.target_limits.T)
    blend = base.smoothstep((grid - 0.06) / (1.42 - 0.06))[:, None]
    nominal = start * (1 - blend) + standing * blend
    old = source["action"][config.prefix_controls :].copy()
    old_controls = len(old)
    old_duration = old_controls * config.control_dt
    prior = np.repeat(old[-1:], config.horizon, axis=0)
    prior[:old_controls] = old
    blend = base.smoothstep((grid - old_duration) / (1.42 - old_duration))[:, None]
    prior = prior * (1 - blend) + standing * blend
    priors = {f"old60_{old_controls}controls_then_standing_not_teacher": prior}
    if extra is not None:
        q6 = np.asarray(extra["candidate_right_leg_q6"])
        tuck = prior.copy()
        # Only six robot targets are used. No IK root/board/qpos is copied.
        driven = prior[:, 6:12].copy()
        rise = base.smoothstep((grid - 0.02) / (0.08 - 0.02))[:, None]
        tuck[:, 6:12] = driven * (1 - rise) + q6 * rise
        restore = base.smoothstep((grid - 0.30) / (1.42 - 0.30))[:, None]
        tuck[:, 6:12] = tuck[:, 6:12] * (1 - restore) + standing[6:12] * restore
        priors["fixed_rear6_kinematic_tuck_not_teacher"] = tuck
    priors = {name: np.clip(value, *controller.target_limits.T) for name, value in priors.items()}
    return (
        nominal,
        priors,
        {
            "standing_target": standing.tolist(),
            "nominal": "last actual target held +.06, quintic standing by +1.42",
            "legacy": "exact14 old60 targets to3.40, quintic standing +.28→+1.42"
            if old_controls == 14 and config.control_dt == 0.02
            else f"exact{old_controls} old60 targets, quintic standing +{old_duration:.2f}→+1.42",
            "tuck": "other23 legacy targets; right6 drive to+.02, quintic tuck by+.08, hold to+.30, quintic standing by+1.42"
            if extra
            else None,
            "is_teacher": False,
            "root_or_board_targets_used": False,
        },
    )


def momentum_cost(controller, states, sensors, targets, config):
    """One fixed grade and continuous cost for every round; no task certification."""
    sl, m, ix = controller.slices, controller.model, controller.index
    count, samples = states.shape[:2]
    times = np.arange(samples) * config.physics_dt
    initial = np.broadcast_to(controller.state(), (count, 1, states.shape[-1]))
    pre = np.concatenate((initial, states[:, :-1]), axis=1)
    bp, bq = sensors[..., sl["boardpos"]], sensors[..., sl["boardquat"]]
    rp, rq = sensors[..., sl["rootpos"]], sensors[..., sl["rootquat"]]
    bu = 1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)
    ru = 1 - 2 * (rq[..., 1] ** 2 + rq[..., 2] ** 2)
    lower = np.stack(
        [
            base.envelope_z(
                m,
                m.geom(name).id,
                sensors[..., sl[name + "_pos"]],
                sensors[..., sl[name + "_quat"]],
            )
            for name in controller.board_geoms
        ],
        axis=-1,
    ).min(-1)
    floor = np.stack(
        [
            np.maximum(sensors[..., sl["ground_" + name]][..., 3], 0)
            for name in controller.board_geoms
        ],
        -1,
    )
    normal = {
        side: np.maximum(sensors[..., sl[side + "_normal_load"]][..., 0], 0)
        for side in ("left", "right")
    }
    foot_fz = {
        side: sum(sensors[..., sl[side + "_" + box]][..., 3] for box in ("center", "tail", "nose"))
        for side in normal
    }
    load20 = {side: rolling_mean(value, 8) for side, value in normal.items()}
    tail_force = np.maximum(sensors[..., sl["ground_board_tail"]][..., 3], 0)
    tail_selected_fn = np.maximum(sensors[..., sl["tail_strongest"]][..., 1], 0)
    board_vz = sensors[..., sl["board_com_v"]][..., 2]
    robot_ground = sensors[..., sl["robot_ground"]][..., 0] > 0
    depth = np.maximum(-sensors[..., sl["depth"]][..., 1], 0)
    # Contact count rejects even subthreshold robot contacts: conservative with
    # respect to canonical .1N; it cannot promote a fallen robot's bouncing board.
    prefix_rows = controller.prefix_records
    prefix_safe = not any(
        row["unique_robot_geoms_ground"]
        or row["maximum_penetration_m"] > 0.01
        or row["board_upright_cosine"] <= 0
        for row in prefix_rows
    )
    safe = ~np.maximum.accumulate(robot_ground | (depth > 0.01) | (bu <= 0), axis=1)
    safe &= prefix_safe
    tail = (tail_selected_fn >= 0.1) & (board_vz > 0) & safe
    tail_before = np.concatenate(
        (np.zeros((count, 1), bool), np.maximum.accumulate(tail, 1)[:, :-1]), 1
    )
    flight = (lower > 0.005) & (floor.max(-1) < 0.1)
    flight40 = base.min_windows(flight.astype(np.int8), 16).astype(bool)
    clear40 = base.min_windows(lower, 16)
    # v3's 15mm peak is the global full-board minimum's maximum. For this
    # exploration label restrict that same definition to BEFORE this window's
    # end, so later post-fall bounce cannot supply the peak.
    peak_to_end = np.maximum.accumulate(lower, 1)[:, 15:]
    qualified = flight40 & tail_before[:, : samples - 15] & safe[:, 15:] & (peak_to_end >= 0.015)
    grade = np.where(qualified.any(1), 2, np.where(tail.any(1), 1, 0)).astype(np.int8)
    # All synchronized soft tasks use one selected tail anchor. They do not
    # require system COM ascent or a positive nose-drag vertical impulse.
    sync, anchor_id, flight_id = np.full(count, np.inf), np.zeros(count, int), np.zeros(count, int)
    names = (
        "tail_gap",
        "tail_force",
        "board_up_at_tail",
        "front_normal_release",
        "rear_normal_release",
        "flight_clearance",
        "flight_peak",
    )
    parts_best = {name: np.zeros(count) for name in names}
    flight_cost = 600 * (np.maximum(0.008 - clear40, 0) / 0.04) ** 2
    for anchor in np.flatnonzero((times >= 0.02) & (times <= 0.40)):
        before = max(0, anchor - 7)
        release_end = min(anchor + 33, load20["right"].shape[1])
        front = load20["left"][:, before:release_end].min(1)
        rear = load20["right"][:, anchor:release_end].min(1)
        tail_bottom = base.envelope_z(
            m,
            m.geom("board_tail").id,
            sensors[:, anchor, sl["board_tail_pos"]],
            sensors[:, anchor, sl["board_tail_quat"]],
        )
        end = min(anchor + 177, clear40.shape[1])
        eligible = flight_cost[:, anchor + 1 : end]
        chosen = np.argmin(eligible, 1)
        fs = anchor + 1 + chosen
        peak = lower[:, anchor + 1 : min(anchor + 193, samples)].max(1)
        parts = {
            "tail_gap": 300 * (np.maximum(tail_bottom, 0) / 0.068) ** 2,
            "tail_force": 200 * np.maximum(1 - tail_force[:, anchor] / 5, 0) ** 2,
            "board_up_at_tail": 200 * (np.maximum(0.5 - board_vz[:, anchor], 0) / 0.5) ** 2,
            "front_normal_release": 100 * np.minimum(front / 40, 5) ** 2,
            "rear_normal_release": 200 * np.minimum(rear / 40, 5) ** 2,
            "flight_clearance": eligible[np.arange(count), chosen],
            "flight_peak": 200 * (np.maximum(0.015 - peak, 0) / 0.015) ** 2,
        }
        value = sum(parts.values())
        improve = value < sync
        sync[improve], anchor_id[improve], flight_id[improve] = value[improve], anchor, fs[improve]
        for name, part in parts.items():
            parts_best[name][improve] = part[improve]
    q = pre[..., 1 + ix.robot_joint_qpos]
    limits = np.maximum(
        np.maximum(controller.limits[:, 0] - q, q - controller.limits[:, 1]), 0
    ).max(-1)
    lastq = states[:, -1, 1 + ix.robot_joint_qpos]
    lastlimit = np.maximum(
        np.maximum(controller.limits[:, 0] - lastq, lastq - controller.limits[:, 1]), 0
    ).max(-1)
    safety = 2000 * robot_ground.mean(1)
    safety += 500 * (depth.max(1) / 0.01) ** 2
    safety += 500 * (np.maximum(limits.max(1), lastlimit) / 0.05) ** 2
    safety += 500 * np.mean((np.maximum(-bu, 0) / 0.5) ** 2, 1)
    final = times >= config.horizon * config.control_dt - 0.3
    velocity = pre[..., 1 + m.nq :]
    rv = np.linalg.norm(velocity[..., ix.robot_qvel : ix.robot_qvel + 3], axis=-1)
    rw = np.linalg.norm(velocity[..., ix.robot_qvel + 3 : ix.robot_qvel + 6], axis=-1)
    bv = np.linalg.norm(velocity[..., ix.board_qvel : ix.board_qvel + 3], axis=-1)
    bw = np.linalg.norm(velocity[..., ix.board_qvel + 3 : ix.board_qvel + 6], axis=-1)
    support = 600 * np.mean(
        sum((1 - np.clip(foot_fz[side][:, final] / 10, 0, 1)) ** 2 for side in normal), 1
    )
    wi = [i for i, name in enumerate(controller.board_geoms) if name.startswith("wheel_")]
    support += 600 * np.mean((1 - np.clip(floor[:, final][:, :, wi], 0, 1)) ** 2, (1, 2))
    terminal = 300 * np.mean(
        (np.maximum(0.8 - ru[:, final], 0) / 0.2) ** 2
        + (np.maximum(0.95 - bu[:, final], 0) / 0.2) ** 2
        + (np.maximum(0.55 - (rp - bp)[:, final, 2], 0) / 0.2) ** 2,
        1,
    )
    terminal += 300 * np.mean(
        (np.maximum(rv[:, final] - 0.3, 0) / 0.3) ** 2
        + (np.maximum(rw[:, final] - 0.8, 0) / 0.8) ** 2
        + (np.maximum(bv[:, final] - 0.35, 0) / 0.35) ** 2
        + (np.maximum(bw[:, final] - 0.8, 0) / 0.8) ** 2,
        1,
    )
    catch = np.empty(count)
    for c in range(count):
        begin = min(flight_id[c] + 16, samples - 40)
        end = min(begin + 240, samples)
        quality = sum((1 - np.clip(foot_fz[side][c, begin:end] / 10, 0, 1)) ** 2 for side in normal)
        catch[c] = 200 * rolling_mean(quality[None], 40).min()
    drift = 10 * np.mean(
        np.sum(((bp[..., :2] - controller.data.xpos[ix.board_body, :2]) / 0.3) ** 2, -1), 1
    )
    previous = controller.data.ctrl[ix.robot_actuators]
    delta = np.diff(np.concatenate((np.broadcast_to(previous, (count, 1, 29)), targets), 1), axis=1)
    smooth = 0.1 * np.mean((delta / 0.25) ** 2, (1, 2))
    terms = {
        **parts_best,
        "physical_severity": safety,
        "catch_support_proxy": catch,
        "final_support_proxy": support,
        "final_stability": terminal,
        "board_xy_drift": drift,
        "target_change": smooth,
    }
    cost = sum(terms.values())
    valid = np.isfinite(states).all((1, 2)) & np.isfinite(sensors).all((1, 2)) & np.isfinite(cost)
    prefix_time = config.prefix_controls * config.control_dt
    episode_end = (config.prefix_controls + config.horizon) * config.control_dt
    valid &= np.isclose(states[:, -1, 0], episode_end, rtol=0, atol=1e-9)
    grade = np.where(valid, grade, -1)
    metrics = {
        "anchor_time_s": prefix_time + times[anchor_id],
        "board_com_vz_at_anchor_m_s": board_vz[np.arange(count), anchor_id],
        "maximum_whole_board_clearance_m": lower.max(1),
        "maximum_penetration_m": depth.max(1),
        "robot_ground_fraction": robot_ground.mean(1),
        "maximum_actual_joint_limit_violation_rad": np.maximum(limits.max(1), lastlimit),
        "first_qualified_flight_start_s": np.where(
            qualified.any(1), prefix_time + np.argmax(qualified, 1) * config.physics_dt, -1
        ),
        "tail_selected_contact_max_fn_n": tail_selected_fn.max(1),
        "minimum_front_20ms_touch_n": load20["left"].min(1),
        "minimum_rear_20ms_touch_n": load20["right"].min(1),
    }
    return grade, np.where(valid, cost, np.inf), terms, metrics


def normal_proof(controller, targets, sensors, config):
    """Independent all-contact normal audit, at the executed solve's time stage."""
    before = fingerprint(controller.data, controller.model)
    scratch, rows = copy.copy(controller.data), []
    touch_error, tail_error = {side: 0.0 for side in ("left", "right")}, 0.0
    cursor = 0
    for target in targets:
        scratch.ctrl[controller.index.robot_actuators] = target
        for _ in range(8):
            mujoco.mj_step(controller.model, scratch)
            force = foot_forces(controller, scratch)
            for side, old_error in touch_error.items():
                sensed = float(sensors[cursor, controller.slices[side + "_normal_load"]][0])
                touch_error[side] = max(old_error, abs(sensed - force[side + "_all_normal_n"]))
            sensed_tail = float(sensors[cursor, controller.slices["tail_strongest"]][1])
            tail_error = max(tail_error, abs(sensed_tail - force["tail_strongest_normal_n"]))
            rows.append(
                {
                    "stage": "execution_solve_before_extra_forward",
                    "interval_start_s": float(scratch.time - config.physics_dt),
                    "interval_end_s": float(scratch.time),
                    **force,
                }
            )
            mujoco.mj_forward(controller.model, scratch)
            cursor += 1
    if fingerprint(controller.data, controller.model) != before or tail_error > 1e-7:
        raise RuntimeError("Native strongest tail/unchanged-main normal proof failed")
    # TOUCH may exclude a contact whose point and normal ray miss the region.
    # It remains non-cancelling normal guidance, never measured zero-load proof.
    return {
        "substeps": cursor,
        "main_unchanged": True,
        "strongest_tail_fn_error_n": tail_error,
        "touch_vs_all_foot_normal_max_error_n": touch_error,
        "touch_all_normals_match_within_1e_minus7": all(e <= 1e-7 for e in touch_error.values()),
        "release_certification_uses_real_contact_fn_not_touch_or_net_fz": True,
    }, rows


def execute_winner(controller, source, prefix_records, targets, predicted, config):
    """65's complete/partial persistence, plus same-step execution normal loads."""
    m, data, ix = controller.model, controller.data, controller.index
    records, forces = list(prefix_records), []
    prefix_controls = config.prefix_controls
    arrays = {
        name: [v.copy() for v in source[name][: prefix_controls + 1]]
        for name in ("state", "qpos", "qvel")
    }
    actions = list(source["action"][:prefix_controls].copy())
    times = list(source["time_physics"][: prefix_controls + 1])
    failure, partial, maximum = None, None, 0.0
    committed_control_count = config.prefix_controls
    for frame, target in enumerate(targets):
        advanced = 0
        control_start = float(data.time)
        try:
            data.ctrl[ix.robot_actuators] = target
            for substep in range(8):
                mujoco.mj_step(m, data)
                advanced += 1
                forces.append(
                    {
                        "stage": "execution_solve_before_extra_forward",
                        "interval_start_s": float(data.time - config.physics_dt),
                        "interval_end_s": float(data.time),
                        **foot_forces(controller, data),
                    }
                )
                mujoco.mj_forward(m, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Native numerical failure")
                row = controller.observer.observe(data)
                records.append(row)
                if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                    raise RuntimeError("Unexpected external force")
                maximum = max(
                    maximum, float(abs(controller.state() - predicted[frame * 8 + substep]).max())
                )
            # Build every snapshot before mutating the regular boundary lists.
            # A KeyboardInterrupt during any append is rolled back below to
            # the last fully committed transaction; raw actual progress stays.
            snapshots = {
                "action": target.copy(),
                "state": controller.indices.pack(data.qpos, data.qvel).numpy().copy(),
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "time": float(data.time),
            }
            actions.append(snapshots["action"])
            for name, values in arrays.items():
                values.append(snapshots[name])
            times.append(snapshots["time"])
            committed_control_count = config.prefix_controls + frame + 1
            if maximum > 1e-10:
                raise RuntimeError("Actual differs from evaluated winner")
        except (RuntimeError, ValueError, KeyboardInterrupt) as error:
            failure = str(error) or type(error).__name__
            # An interrupt can arrive immediately after a native call returned.
            advanced = max(advanced, round((float(data.time) - control_start) / config.physics_dt))
            del actions[committed_control_count:]
            for values in arrays.values():
                del values[committed_control_count + 1 :]
            del times[committed_control_count + 1 :]
            if advanced and committed_control_count < config.prefix_controls + frame + 1:
                partial = {
                    "control_step": prefix_controls + frame,
                    "target": target.tolist(),
                    "native_substeps_advanced": advanced,
                    "actual_time_s": float(data.time),
                    "omitted_from_regular_20ms_boundary_arrays": True,
                }
            break
    complete = committed_control_count == 257 and abs(data.time - 5.14) < 1e-9 and failure is None
    result = outcome(records, full_reference=complete, numerical_failure=failure)
    result.update(
        schema=SCHEMA,
        initializations=1,
        phase_resets=0,
        root_board_writes_after_initialization=0,
        board_actuators=0,
        actual_control_dim=29,
        external_forces_applied=False,
        actual_endpoint_s=float(data.time),
        partial_control=partial,
        actual_prediction_max_error=maximum,
        committed_control_count=committed_control_count,
        verified_teacher=False,
        supermimic_training_performed=False,
        foot_deck_friction=None,
        explicit_contact_pairs=int(m.npair),
    )
    packed = {name: np.asarray(value) for name, value in arrays.items()}
    packed.update(
        action=np.asarray(actions),
        time_physics=np.asarray(times),
        raw_terminal_qpos=data.qpos.copy(),
        raw_terminal_qvel=data.qvel.copy(),
        raw_terminal_time_s=np.asarray(data.time),
    )
    return result, packed, records, forces


def release_summary(rows):
    """Report actual joint 20ms normal windows; never infer unload from net Fz."""
    if len(rows) < 8:
        return {"complete_20ms_windows": 0, "stage": "execution_solve_before_extra_forward"}
    joint = np.maximum(
        np.asarray([row["left_all_normal_n"] for row in rows]),
        np.asarray([row["right_all_normal_n"] for row in rows]),
    )
    # Conservative joint low load: every sample of BOTH feet, not the mean of
    # opposing wrenches or separate minima at unrelated times.
    joint20 = np.maximum.reduce([joint[k : len(joint) - 7 + k] for k in range(8)])
    best = int(np.argmin(joint20))
    tail = next((i for i, row in enumerate(rows) if row["tail_strongest_normal_n"] >= 0.1), None)
    near = np.zeros(len(joint20), bool)
    if tail is not None:
        tail_time = rows[tail]["interval_start_s"]
        near = np.asarray(
            [
                tail_time - 0.02 <= row["interval_start_s"] <= tail_time + 0.08
                for row in rows[: len(joint20)]
            ]
        )
    within = np.flatnonzero(near)
    near_best = int(within[np.argmin(joint20[within])]) if len(within) else None

    def window(k):
        selected = rows[k : k + 8]
        return {
            "start_s": selected[0]["interval_start_s"],
            "end_s": selected[-1]["interval_end_s"],
            "maximum_either_foot_all_normal_n": float(joint20[k]),
            "both_feet_every_sample_below40n": bool(joint20[k] <= 40),
            **{
                name: float(np.mean([row[name] for row in selected]))
                for name in (
                    "left_all_normal_n",
                    "right_all_normal_n",
                    "left_board_normal_n",
                    "right_board_normal_n",
                    "left_board_on_foot_fz_n",
                    "right_board_on_foot_fz_n",
                )
            },
        }

    return {
        "stage": "execution_solve_before_extra_forward",
        "complete_20ms_windows": len(joint20),
        "strongest_tail_first_interval_s": rows[tail]["interval_start_s"]
        if tail is not None
        else None,
        "global_best_joint20ms": window(best),
        "tail_near_best_joint20ms": window(near_best) if near_best is not None else None,
        "normal_sums_include_all_force_magnitudes_without_threshold": True,
        "is_success_certificate": False,
    }


def model_sources():
    """Freeze the authored board plus generated original scene and mesh bytes."""
    spec = make_scene_spec(timestep=0.0025, foot_deck_friction=None)
    meshes = {}
    for mesh in spec.meshes:
        path = Path(spec.modelfiledir) / spec.meshdir / mesh.file
        if mesh.file:
            meshes[str(path.resolve())] = digest(path)
    return {
        "board_xml": {"path": str(BOARD_XML.resolve()), "sha256": digest(BOARD_XML)},
        "canonical_scene_xml_sha256": hashlib.sha256(spec.to_xml().encode()).hexdigest(),
        "mesh_sha256": meshes,
    }


def save_branch(path, candidate):
    np.savez_compressed(
        path,
        action=candidate[2],
        fullphysics=candidate[3],
        sensors=candidate[4],
        parameters=candidate[5],
        grade=np.asarray(candidate[0]),
        cost=np.asarray(candidate[1]),
        round=np.asarray(candidate[6]),
        candidate=np.asarray(candidate[7]),
        verified_teacher=np.asarray(False),
    )


def build_argument_parser():
    """Common frozen-input/observation/IO options for a bounded experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "benchmark", "run"), default="inspect")
    parser.add_argument(
        "--source", type=Path, default=Path("logs/ollie_pop_events_preflight_02/actual_rollout.npz")
    )
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
    parser.add_argument("--extra-prior", type=Path)
    parser.add_argument("--extra-prior-sha256")
    parser.add_argument("--extra-prior-result", type=Path)
    parser.add_argument("--extra-prior-result-sha256")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def run_experiment(
    args,
    config,
    cost_function=momentum_cost,
    prior_factory=prepare_priors,
    source_sha=SOURCE_SHA,
    schema=SCHEMA,
    manifest_extra=None,
    elite_selector=None,
):
    """Run the shared native protocol with explicitly supplied cost and priors.

    ``cost_function(controller, states, sensors, targets, config)`` returns
    ``(grades, costs, terms, metrics)``. ``prior_factory`` has the same signature
    and return contract as ``prepare_priors``; its first prior seeds the mean.
    Args are the Namespace from ``build_argument_parser`` (possibly extended).
    ``manifest_extra`` records caller provenance without replacing frozen data.
    The caller owns configuration approval; this API never starts an extra run.
    """
    parser = build_argument_parser()
    if args.out_dir.exists():
        parser.error("A fresh output directory is required")
    started = time.monotonic()
    sources = {}
    for name, expected in (
        ("source", source_sha),
        ("reference", base.REFERENCE_HASH),
        ("feedforward", base.FEEDFORWARD_HASH),
    ):
        path = getattr(args, name)
        if digest(path) != expected:
            raise ValueError(f"Frozen {name} SHA differs")
        sources[name] = {"path": str(path.resolve()), "sha256": expected}
    contacts = args.source.parent / "contacts.json"
    sources["source_contacts"] = {"path": str(contacts.resolve()), "sha256": digest(contacts)}
    extra = None
    if any(
        (
            args.extra_prior,
            args.extra_prior_sha256,
            args.extra_prior_result,
            args.extra_prior_result_sha256,
        )
    ):
        if not all(
            (
                args.extra_prior,
                args.extra_prior_sha256,
                args.extra_prior_result,
                args.extra_prior_result_sha256,
            )
        ):
            parser.error("Explicit prior requires NPZ/result paths and both frozen hashes")
        for name in ("extra_prior", "extra_prior_result"):
            path, expected = getattr(args, name), getattr(args, name + "_sha256")
            if digest(path) != expected:
                raise ValueError("Explicit kinematic prior hash mismatch")
            sources[name] = {"path": str(path.resolve()), "sha256": expected}
        extra = json.loads(args.extra_prior_result.read_text())
        if not (
            extra["static_geometry_pass"]
            and not extra["is_teacher"]
            and extra["source_sha256"] == source_sha
            and extra["source_control_frame"] == config.prefix_controls
            and extra["all_other_qpos_exact"]
            and tuple(extra["joint_names29"]) == G1_JOINT_NAMES
        ):
            raise ValueError("Kinematic prior provenance/geometry gate failed")
        q6 = np.asarray(extra["candidate_right_leg_q6"])
        with np.load(args.extra_prior, allow_pickle=False) as data:
            # Read only this six-joint value, never apply saved whole qpos.
            if "target29" in data.files:
                saved6 = data["target29"][6:12]
            elif "right_leg_q6" in data.files:
                saved6 = data["right_leg_q6"]
            else:
                raise ValueError("Explicit prior lacks a supported right6 target")
            if q6.shape != (6,) or not np.isfinite(q6).all() or not np.array_equal(saved6, q6):
                raise ValueError("NPZ/result right6 mismatch")
    with np.load(args.source, allow_pickle=False) as data:
        source = {name: data[name].copy() for name in data.files}
    shapes = {"state": (171, 92), "qpos": (171, 47), "qvel": (171, 45), "action": (170, 29)}
    if any(
        source[name].shape != shape or not np.isfinite(source[name]).all()
        for name, shape in shapes.items()
    ):
        raise ValueError("Expected original60 full170control archive")
    if (
        tuple(source["robot_joint_names"]) != G1_JOINT_NAMES
        or tuple(source["board_joint_names"]) != BOARD_JOINT_NAMES
    ):
        raise ValueError("Source joint ordering differs")
    if not np.array_equal(source["time_physics"], np.arange(171) * 0.02):
        raise ValueError("Source timestamps differ")
    dependency_paths = (
        Path(base.__file__),
        Path("scripts/local/optimize_ollie_pop_events.py"),
        Path("scripts/local/track_ollie_sampling_mpc.py"),
        Path("scripts/local/evaluate_ollie_policy.py"),
        Path("src/supermimic/ollie/physics.py"),
    )
    manifest = {
        "schema": schema + ".manifest",
        "mode": args.mode,
        "config": asdict(config),
        "sources": sources,
        "script_sha256": digest(__file__),
        "mujoco_version": mujoco.__version__,
        "dependencies": {str(p.resolve()): digest(p) for p in dependency_paths},
        "canonical_assets": model_sources(),
        "physical_model": "canonical default foot mu=.6; no explicit pairs, unchanged board and29PD",
        "grades_are_exploration_only": True,
        "is_teacher": False,
        "touch_normal_is_guidance_only": True,
        "force_stage": "execution after mj_step before extra mj_forward",
        "sensor_pose_stage": "preintegration geometry and COMv; same-control execution force",
        "observer_stage": "canonical post mj_forward .1N; original strict v3 unchanged",
        "selection": "highest grade, lowest fixed cost; one evaluated complete winner; preserve best each grade",
    }
    if manifest_extra is not None:
        manifest["experiment"] = manifest_extra
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "frozen_manifest.json", manifest)
    if args.mode == "inspect":
        print(json.dumps(manifest, allow_nan=False))
        return
    controller = NativeSamplingMPC(
        args.reference, args.feedforward, seed=config.seed, foot_deck_friction=None
    )
    controller.prefix_path, controller.horizon = args.source, config.horizon
    best, per_grade, reports = None, {}, []
    try:
        prefix_records, proof = replay_prefix(controller, source, config)
        controller.prefix_records = prefix_records
        write_json(args.out_dir / "prefix_proof.json", proof)
        write_json(
            args.out_dir
            / f"old60_{len(source['action']) - config.prefix_controls}control_regression.json",
            continuation_regression(controller, source, config),
        )
        if (
            controller.model.npair
            or controller.model.neq
            or controller.model.na
            or controller.model.nmocap
        ):
            raise RuntimeError(
                "Expected original unconstrained passive board and default contact model"
            )
        model_proof = add_sensors(controller, config)
        nominal, priors, metadata = prior_factory(controller, source, extra, config)
        write_json(
            args.out_dir / "model_and_prior_proof.json", {"model": model_proof, "prior": metadata}
        )
        np.savez_compressed(args.out_dir / "target_priors.npz", nominal=nominal, **priors)
        mean = base.fit_knots(next(iter(priors.values())), nominal, config)
        std = np.full_like(mean, config.sigma_rad)
        for iteration in range(config.rounds):
            if (
                time.monotonic() - started
                >= config.wall_budget_s - config.winner_proof_and_execution_reserve_s
            ):
                raise TimeoutError(
                    "300s budget: no new batch with less than15s proof/execution reserve"
                )
            params = controller.rng.normal(mean, std, (config.candidates, *mean.shape))
            params[0], params[1] = mean, 0
            targets = base.interpolate_knots(params, nominal, config)
            labels, slot = {0: "evaluated_current_mean", 1: "standing_nominal"}, 2
            for name, prior in priors.items():
                targets[slot], params[slot], labels[slot] = (
                    prior,
                    base.fit_knots(prior, nominal, config),
                    name,
                )
                slot += 1
            for grade_id in sorted(per_grade):
                candidate = per_grade[grade_id]
                targets[slot], params[slot], labels[slot] = (
                    candidate[2],
                    candidate[5],
                    f"evaluated_best_grade{grade_id}",
                )
                slot += 1
            targets = np.clip(targets, *controller.target_limits.T)
            tick = time.monotonic()
            states, sensors = controller.rollout(targets)
            rollout_wall = time.monotonic() - tick
            tick = time.monotonic()
            grades, costs, terms, metrics = cost_function(
                controller, states, sensors, targets, config
            )
            cost_wall = time.monotonic() - tick
            order = np.lexsort((costs, -grades))
            order = order[np.isfinite(costs[order]) & (grades[order] >= 0)]
            if not len(order):
                raise RuntimeError("No finite complete branch")
            winner = int(order[0])
            elites = (
                order[: config.elites]
                if elite_selector is None
                else np.asarray(elite_selector(order, grades, costs, config), dtype=int)
            )
            if (
                not 1 <= len(elites) <= config.elites
                or len(np.unique(elites)) != len(elites)
                or not np.isin(elites, order).all()
            ):
                raise ValueError("Elite selector must return unique finite evaluated candidates")
            for grade_id in (0, 1, 2):
                selected = order[grades[order] == grade_id]
                if not len(selected):
                    continue
                j = int(selected[0])
                if grade_id not in per_grade or costs[j] < per_grade[grade_id][1]:
                    per_grade[grade_id] = (
                        grade_id,
                        float(costs[j]),
                        targets[j].copy(),
                        states[j].copy(),
                        sensors[j].copy(),
                        params[j].copy(),
                        iteration,
                        j,
                    )
                    save_branch(
                        args.out_dir / f"best_grade{grade_id}_evaluated_branch.npz",
                        per_grade[grade_id],
                    )
            best = per_grade[max(per_grade)]
            report = {
                "round": iteration,
                "rollout_wall_s": rollout_wall,
                "cost_wall_s": cost_wall,
                "grade": grades.tolist(),
                "cost": [float(v) if np.isfinite(v) else None for v in costs],
                "terms": {
                    name: [float(v) if np.isfinite(v) else None for v in values]
                    for name, values in terms.items()
                },
                "metrics": {
                    name: [float(v) if np.isfinite(v) else None for v in values]
                    for name, values in metrics.items()
                },
                "elite_ids": elites.tolist(),
                "special_candidate_labels": labels,
                "best_candidate": winner,
                "best_grade": int(grades[winner]),
                "best_cost": float(costs[winner]),
                "global_best_grade": best[0],
                "global_best_cost": best[1],
            }
            reports.append(report)
            write_json(args.out_dir / "search_rounds.json", reports)
            if iteration == 0:
                if (
                    time.monotonic() - started
                    > config.wall_budget_s - config.winner_proof_and_execution_reserve_s
                ):
                    raise TimeoutError("Insufficient first-proof reserve")
                first = base.branch_proof(
                    controller, targets[winner], states[winner], sensors[winner]
                )
                normals, force_rows = normal_proof(
                    controller, targets[winner], sensors[winner], config
                )
                write_json(args.out_dir / "first_branch_force_audit.json", force_rows)
                projected = (rollout_wall + cost_wall) * config.rounds
                benchmark = {
                    "one_full_batch_rollout_s": rollout_wall,
                    "one_full_batch_cost_s": cost_wall,
                    "projected_64_round_seconds": projected,
                    "first_branch_proof": first,
                    "first_branch_normal_proof": normals,
                    "first_batch": report,
                    "search_update_performed": False,
                }
                write_json(args.out_dir / "benchmark.json", benchmark)
                if args.mode == "benchmark":
                    return
                if projected + (time.monotonic() - started) > config.wall_budget_s - 15:
                    raise RuntimeError(
                        "Fixed budget gate failed; no distribution update or actual suffix"
                    )
            mean = params[elites].mean(0)
            std = np.clip(params[elites].std(0), config.sigma_floor_rad, config.sigma_rad)
            if iteration % 16 == 0 or iteration == config.rounds - 1:
                print(
                    json.dumps(
                        {
                            "round": iteration,
                            "best_grade": best[0],
                            "best_cost": best[1],
                            "wall_s": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        if (
            time.monotonic() - started
            > config.wall_budget_s - config.winner_proof_and_execution_reserve_s
        ):
            raise TimeoutError("Insufficient reserve after final round; no actual suffix")
        proof = base.branch_proof(controller, best[2], best[3], best[4])
        normals, predicted_force = normal_proof(controller, best[2], best[4], config)
        write_json(args.out_dir / "winner_branch_proof.json", {"native": proof, "normals": normals})
        write_json(args.out_dir / "winner_execution_force_audit.json", predicted_force)
        save_branch(args.out_dir / "winner_evaluated_branch.npz", best)
        if time.monotonic() - started > config.wall_budget_s - config.actual_execution_reserve_s:
            raise TimeoutError("Insufficient actual execution reserve")
        result, arrays, records, forces = execute_winner(
            controller, source, prefix_records, best[2], best[3], config
        )
        result.update(
            schema=schema,
            frozen_manifest=manifest,
            winner_exploration_grade=best[0],
            winner_cost=best[1],
            winner_round=best[6],
            winner_candidate=best[7],
            completed_search_rounds=len(reports),
            wall_seconds=time.monotonic() - started,
            release_guidance_is_teacher=False,
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
        write_json(args.out_dir / "contacts.json", records)
        write_json(args.out_dir / "actual_execution_forces.json", forces)
        write_json(args.out_dir / "actual_release_windows.json", release_summary(forces))
        write_json(args.out_dir / "result.json", result)
        for name, source_info in sources.items():
            if digest(source_info["path"]) != source_info["sha256"]:
                raise RuntimeError(f"Input changed: {name}")
        write_json(
            args.out_dir / "output_hashes.json",
            {p.name: digest(p) for p in args.out_dir.iterdir() if p.is_file()},
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    except (RuntimeError, ValueError, TimeoutError, KeyboardInterrupt) as error:
        if best is not None:
            save_branch(args.out_dir / "best_so_far_evaluated_branch.npz", best)
        write_json(
            args.out_dir / "failure.json",
            {
                "error": str(error) or type(error).__name__,
                "actual_endpoint_s": float(controller.data.time),
                "completed_search_rounds": len(reports),
                "main_actual_continuation_executed": controller.data.time
                > config.prefix_controls * config.control_dt + 1e-9,
                "main_fingerprint": fingerprint(controller.data, controller.model),
                "wall_seconds": time.monotonic() - started,
                "best_evaluated_grade": best[0] if best else None,
                "best_evaluated_cost": best[1] if best else None,
            },
        )
        raise
    finally:
        controller.pool.close()


def main():
    run_experiment(build_argument_parser().parse_args(), Config())


if __name__ == "__main__":
    main()
