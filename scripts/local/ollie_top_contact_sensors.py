"""Read-only, per-contact native deck support with canonical top-face semantics.

The base builder is a static copy of76's sensor builder, extended at compile
only; no old source or global factory is patched. All physical arrays/options
are compared against the original model. No force, root or board is written.

CONTACT NONE slot: [found, Fn, Ft1, Ft2, pos_xyz, normal_xyz]. FORCE is local;
POS/NORMAL are world quantities. The sensor direction is board box→foot.
Ordinary mj_step exposes preintegration geometry and that step's solve force;
a following mj_forward exposes the new pose and recomputed force. Compare each
stage separately. Post-forward canonical Observer remains unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local import optimize_ollie_momentum_release as protocol
from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_JOINT_NAMES, make_scene_spec

base, physical_compare = protocol.base, protocol.physical_compare
SCHEMA = "supermimic.ollie.native_top_contacts.v1"
SLOTS = 32
SLOT_DIM = 10
GROUND_SLOTS = 64
GROUND_SLOT_DIM = 4
FORCE_THRESHOLD_N = 0.1
SIDES = ("left", "right")
BOXES = ("center", "tail", "nose")


def _append_top_spec(spec, controller):
    fields = sum(
        1 << int(getattr(mujoco.mjtConDataField, "mjCONDATA_" + name))
        for name in ("FOUND", "FORCE", "POS", "NORMAL")
    )
    if fields != 51:
        raise RuntimeError("Unexpected CONTACT data-field enum layout")
    for side in SIDES:
        body = controller.model.body(controller.foot_geometry[side]["body"]).name
        for box in BOXES:
            spec.add_sensor(
                name=f"top_{side}_{box}",
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname="board_" + box,
                reftype=mujoco.mjtObj.mjOBJ_XBODY,
                refname=body,
                intprm=[fields, 0, SLOTS],
            )
    spec.add_sensor(
        name="top_robot_ground",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="pelvis",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="ground",
        intprm=[3, 0, GROUND_SLOTS],
    )


def _bind_top_slices(controller):
    model = controller.branch_model
    controller.top_slices = {}
    for side in SIDES:
        for box in BOXES:
            key = side + "_" + box
            sensor = model.sensor("top_" + key).id
            if (
                model.sensor_dim[sensor] != SLOTS * SLOT_DIM
                or model.sensor_needstage[sensor] != mujoco.mjtStage.mjSTAGE_ACC
                or not np.array_equal(model.sensor_intprm[sensor], [51, 0, SLOTS])
            ):
                raise RuntimeError("Per-contact native sensor schema/stage mismatch")
            start = int(model.sensor_adr[sensor])
            controller.top_slices[key] = slice(start, start + SLOTS * SLOT_DIM)
    sensor = model.sensor("top_robot_ground").id
    if (
        model.sensor_dim[sensor] != GROUND_SLOTS * GROUND_SLOT_DIM
        or model.sensor_needstage[sensor] != mujoco.mjtStage.mjSTAGE_ACC
        or not np.array_equal(model.sensor_intprm[sensor], [3, 0, GROUND_SLOTS])
    ):
        raise RuntimeError("Native robot-ground sensor schema/stage mismatch")
    start = int(model.sensor_adr[sensor])
    controller.top_robot_ground_slice = slice(start, start + GROUND_SLOTS * GROUND_SLOT_DIM)
    controller.top_sensor_metadata = {
        "schema": SCHEMA,
        "slots_per_side_box": SLOTS,
        "slot_dimension": SLOT_DIM,
        "additional_dimensions": len(SIDES) * len(BOXES) * SLOTS * SLOT_DIM
        + GROUND_SLOTS * GROUND_SLOT_DIM,
        "total_dimensions": int(model.nsensordata),
        "slot_fields": [
            "found",
            "Fn",
            "Ft1",
            "Ft2",
            "pos_x",
            "pos_y",
            "pos_z",
            "normal_x",
            "normal_y",
            "normal_z",
        ],
        "force_frame": "contact-local; Fn is slot[1]",
        "pose_normal_frame": "world",
        "normal_direction": "deck-box toward foot subtree",
        "reduce": "NONE",
        "force_threshold_n": FORCE_THRESHOLD_N,
        "top_rule": "local_z>=0 and dot(normal_on_foot,surface_up)>.5; sum Fn*max(normal_world_z,0)",
        "top_supported_rule": "canonical per-foot upward support strictly>1N",
        "underside_rule": "local_z<0 and dot(normal_on_foot,surface_up)<-.5",
        "side_or_other_rule": "active contacts excluded from top and underside rules",
        "overflow_rule": "found-total>32 yields overflow=true and valid=false; support NaN",
        "slices": {name: [sl.start, sl.stop] for name, sl in controller.top_slices.items()},
        "robot_ground": {
            "slots": GROUND_SLOTS,
            "slot_dimension": GROUND_SLOT_DIM,
            "fields": ["found", "Fn", "Ft1", "Ft2"],
            "reduce": "NONE",
            "slice": [
                controller.top_robot_ground_slice.start,
                controller.top_robot_ground_slice.stop,
            ],
            "active_rule": "any real robot/ground contact Fn>=.1N; CONTACT local Fn",
            "overflow_rule": "matched-total>64 invalidates that sample only",
        },
    }


def add_top_sensors(controller, config):
    """Mirror frozen76 observations and append six per-contact top sensors."""
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
    _append_top_spec(spec, controller)
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
    _bind_top_slices(controller)
    return {
        "top_contact_observations": controller.top_sensor_metadata,
        **physical_compare(controller.model, controller.branch_model),
        "touch_sites": site_manifest,
    }


def top_support(controller, sensors):
    """Return sparse per-contact classification with the sensors' leading shape.

    Scalars and any batch/time rank are accepted (..., nsensordata). Overflow
    and nonfinite observations invalidate only those leading elements. Counts
    describe these three deck boxes; they never stand in for actual support.
    """
    values = np.asarray(sensors)
    if values.ndim < 1 or values.shape[-1] != controller.branch_model.nsensordata:
        raise ValueError("Expected full native branch sensordata as final dimension")
    shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    count = len(flat)
    invalid, overflow = np.zeros(count, bool), np.zeros(count, bool)
    result = {}
    ground = flat[:, controller.top_robot_ground_slice].reshape(
        count, GROUND_SLOTS, GROUND_SLOT_DIM
    )
    ground_count = ground[:, 0, 0]
    ground_overflow = ground_count > GROUND_SLOTS
    ground_invalid = ~np.isfinite(ground_count) | ~np.isfinite(ground[..., 1]).all(-1)
    ground_present = ((ground[..., 0] > 0) & (ground[..., 1] >= FORCE_THRESHOLD_N)).any(-1)
    invalid |= ground_invalid
    overflow |= ground_overflow
    for side in SIDES:
        support, upper_fn, underside_fn, other_fn = (np.zeros(count) for _ in range(4))
        active_count, top_count = np.zeros(count, int), np.zeros(count, int)
        matched_count, side_overflow = np.zeros(count), np.zeros(count, bool)
        for box in BOXES:
            block = flat[:, controller.top_slices[side + "_" + box]].reshape(count, SLOTS, SLOT_DIM)
            # Every found slot repeats the total matched count. Its first slot
            # suffices to detect truncation even when all32 output slots fill.
            found = block[:, 0, 0]
            matched_count += found
            side_overflow |= found > SLOTS
            invalid |= ~np.isfinite(found) | ~np.isfinite(block[..., 1]).all(-1)
            active = (block[..., 0] > 0) & (block[..., 1] >= FORCE_THRESHOLD_N)
            rows, slots = np.nonzero(active)
            if not len(rows):
                continue
            contacts = block[rows, slots]
            pose = flat[rows, controller.slices["board_" + box + "_pos"]]
            quat = flat[rows, controller.slices["board_" + box + "_quat"]]
            finite = (
                np.isfinite(contacts).all(-1)
                & np.isfinite(pose).all(-1)
                & np.isfinite(quat).all(-1)
            )
            if not finite.all():
                invalid[np.unique(rows[~finite])] = True
                rows, contacts, pose, quat = (
                    rows[finite],
                    contacts[finite],
                    pose[finite],
                    quat[finite],
                )
            if not len(rows):
                continue
            fn, position, normal = contacts[:, 1], contacts[:, 4:7], contacts[:, 7:10]
            local_z = base.inverse_rotate(quat, position - pose)[:, 2]
            up = base.rotate(quat, np.array([0.0, 0.0, 1.0]))
            alignment = np.sum(normal * up, -1)
            upper = (local_z >= 0) & (alignment > 0.5)
            underside = (local_z < 0) & (alignment < -0.5)
            other = ~(upper | underside)
            support += np.bincount(
                rows[upper], fn[upper] * np.maximum(normal[upper, 2], 0), minlength=count
            )
            upper_fn += np.bincount(rows[upper], fn[upper], minlength=count)
            underside_fn += np.bincount(rows[underside], fn[underside], minlength=count)
            other_fn += np.bincount(rows[other], fn[other], minlength=count)
            active_count += np.bincount(rows, minlength=count)
            top_count += np.bincount(rows[upper], minlength=count)
        overflow |= side_overflow
        result.update(
            {
                side + "_top_n": support,
                side + "_upper_face_normal_n": upper_fn,
                side + "_underside_normal_n": underside_fn,
                side + "_side_or_other_normal_n": other_fn,
                side + "_side_or_under_n": underside_fn + other_fn,
                side + "_active_contacts": active_count,
                side + "_upper_face_contacts": top_count,
                side + "_matched_contacts": matched_count,
                side + "_overflow": side_overflow,
            }
        )
    valid = ~(invalid | overflow)
    for side in SIDES:
        for suffix in (
            "top_n",
            "upper_face_normal_n",
            "underside_normal_n",
            "side_or_other_normal_n",
            "side_or_under_n",
        ):
            key = side + "_" + suffix
            result[key] = np.where(valid, result[key], np.nan)
        result[side + "_top_supported_above_1n"] = valid & (result[side + "_top_n"] > 1.0)
        result[side + "_foot_deck_upward_support_n"] = result[side + "_top_n"]
    result.update(
        robot_ground_above_threshold=ground_present,
        robot_ground_overflow=ground_overflow,
        robot_ground_matched_contacts=ground_count,
        overflow=overflow,
        invalid=invalid,
        valid=valid,
    )
    return {name: value.reshape(shape) for name, value in result.items()}


vectorized_top_support = top_support


def canonical_top_proof(controller, data):
    """Check one branch MjData's current stage against the original Observer.

    No forward/step is called. The caller chooses execution-solve or post-
    forward stage; comparing stages against one another is not permitted.
    """
    from scripts.local.evaluate_ollie_policy import ContactObserver
    from supermimic.ollie.physics import SceneIndex

    if not hasattr(controller, "_top_canonical_observer"):
        controller._top_canonical_observer = ContactObserver(
            controller.branch_model,
            SceneIndex.from_model(controller.branch_model),
            FORCE_THRESHOLD_N,
        )
    expected = controller._top_canonical_observer.observe(data)
    actual = top_support(controller, data.sensordata)
    if not bool(actual["valid"]):
        raise RuntimeError("Selected branch has overflowing/nonfinite top observations")
    errors = {
        side: abs(float(actual[side + "_top_n"]) - expected[side + "_foot_deck_upward_support_n"])
        for side in SIDES
    }
    if max(errors.values()) > 1e-7:
        raise RuntimeError(f"Canonical top-support proof failed: {errors}")
    if bool(actual["robot_ground_above_threshold"]) != bool(expected["unique_robot_geoms_ground"]):
        raise RuntimeError("Canonical robot-ground contact proof failed")
    return {
        "absolute_support_error_n": errors,
        "overflow": False,
        "same_data_same_stage": True,
        "no_forward_or_step_called": True,
        "robot_ground_above_threshold_exact": True,
    }
