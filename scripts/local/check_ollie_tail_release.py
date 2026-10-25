"""One fixed finite-nose lifting probe with an external mechanical foot fixture.

This is not a robot rollout, a controller teacher, or a successful Ollie. Only
the three fixture slide actuators supply energy; the canonical board is passive.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from supermimic.ollie.physics import (
    BOARD_JOINT_NAMES,
    BOARD_REST_HEIGHT,
    BOARD_XML,
    SceneIndex,
    compile_scene,
    make_board_spec,
)

DT = .0025
DURATION = .4
KP = 10000.
KD = 200.
FORCE_CAP = 250.


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_scalar) + "\n")


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def quintic(value):
    return value**3 * (10 - 15 * value + 6 * value**2)


def prescribed_target(t):
    """The sole profile, fixed before the first trial; slide coordinates are world XYZ."""
    knots = ((0., [0., 0., 0.]), (.04, [-.025, 0., 0.]),
             (.24, [-.12, 0., .2]), (.32, [.08, 0., .2]), (.4, [.08, 0., .2]))
    for (a, x), (b, y) in itertools.pairwise(knots):
        if t <= b + 1e-12:
            u = quintic(float(np.clip((t - a) / (b - a), 0, 1)))
            return (1 - u) * np.asarray(x) + u * np.asarray(y)
    return np.asarray(knots[-1][1], dtype=float)


def geom_bounds(model, data, gid):
    rotation = data.geom_xmat[gid].reshape(3, 3)
    kind, size = model.geom_type[gid], model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = abs(rotation) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = rotation[:, 2]
        extent = size[1] * abs(axis) + size[0] * np.sqrt(np.maximum(0., 1 - axis**2))
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * abs(rotation[:, 2])
    else:
        raise ValueError(f"unsupported geometry {kind}")
    return data.geom_xpos[gid] - extent, data.geom_xpos[gid] + extent


def make_fixture_scene():
    source, _, _ = compile_scene(include_robot=True)
    original, original_data, original_index = compile_scene(include_robot=False)
    nose = original.geom("board_nose").id
    nose_rotation = original_data.geom_xmat[nose].reshape(3, 3)
    nose_end = original_data.geom_xpos[nose] + nose_rotation[:, 0] * original.geom_size[nose, 0]
    nose_max = geom_bounds(original, original_data, nose)[1]
    yaw_quat = np.array([2**-.5, 0., 0., 2**-.5])
    yaw_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    lower, upper = [], []
    for i in range(1, 8):
        gid = source.geom(f"left_foot{i}_collision").id
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, source.geom_quat[gid])
        rotation = yaw_rotation @ rotation.reshape(3, 3)
        center = yaw_rotation @ source.geom_pos[gid]
        extent = source.geom_size[gid, 0] + source.geom_size[gid, 1] * abs(rotation[:, 2])
        lower.append(center - extent)
        upper.append(center + extent)
    lower, upper = np.min(lower, axis=0), np.max(upper, axis=0)
    origin = np.array([nose_max[0] + .001 - lower[0],
                       -.5 * (lower[1] + upper[1]), nose_end[2] - .5 * (lower[2] + upper[2])])
    spec = make_board_spec(timestep=DT)
    body_id = source.body("left_ankle_roll_link").id
    body = spec.worldbody.add_body(
        name="mechanical_foot_fixture", pos=origin, quat=yaw_quat,
        mass=float(source.body_mass[body_id]), ipos=source.body_ipos[body_id].tolist(),
        iquat=source.body_iquat[body_id].tolist(), inertia=source.body_inertia[body_id].tolist(),
        explicitinertial=True,
    )
    for i, axis in enumerate(np.eye(3)):
        name = f"fixture_slide_{i}"
        body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE,
                       axis=yaw_rotation.T @ axis, limited=False, damping=0, armature=0)
        spec.add_actuator(
            name=f"external_fixture_servo_{i}", target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[KP] + [0.] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -KP, -KD] + [0.] * 7,
            forcelimited=True, forcerange=[-FORCE_CAP, FORCE_CAP],
        )
    properties = {}
    for i in range(1, 8):
        name = f"left_foot{i}_collision"
        gid = source.geom(name).id
        body.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            pos=source.geom_pos[gid], quat=source.geom_quat[gid], size=source.geom_size[gid],
            mass=0, contype=int(source.geom_contype[gid]), conaffinity=int(source.geom_conaffinity[gid]),
            condim=int(source.geom_condim[gid]), priority=int(source.geom_priority[gid]),
            friction=source.geom_friction[gid], solref=source.geom_solref[gid],
            solimp=source.geom_solimp[gid], solmix=float(source.geom_solmix[gid]),
            margin=float(source.geom_margin[gid]), gap=float(source.geom_gap[gid]),
            rgba=[.2, .4, .8, 1],
        )
        properties[name] = {field: getattr(source, "geom_" + field)[gid].tolist()
                            for field in ("type", "pos", "quat", "size", "friction", "solref", "solimp",
                                          "solmix", "priority", "condim", "contype", "conaffinity", "margin", "gap")}
    model = spec.compile()
    data = mujoco.MjData(model)
    index = SceneIndex.from_model(model)
    data.qpos[index.board_qpos + 2] = BOARD_REST_HEIGHT
    mujoco.mj_forward(model, data)
    assert model.nu == 3 and model.neq == model.nmocap == 0
    assert not any(model.jnt_bodyid[j] == index.board_body for j in model.actuator_trnid[:, 0])
    # Confirm every canonical board and wheel material/inertia/geometry property
    # survived addition of the independent fixture, without altering the XML.
    board_equal = {}
    for bid in range(1, original.nbody):
        new = model.body(original.body(bid).name).id
        for field in ("mass", "inertia", "ipos", "iquat", "pos", "quat"):
            assert np.array_equal(getattr(original, "body_" + field)[bid], getattr(model, "body_" + field)[new])
    for gid in range(original.ngeom):
        new = model.geom(original.geom(gid).name).id
        for field in ("type", "size", "pos", "quat", "friction", "solref", "solimp", "priority", "condim"):
            assert np.array_equal(getattr(original, "geom_" + field)[gid], getattr(model, "geom_" + field)[new])
        board_equal[original.geom(gid).name] = True
    for name in BOARD_JOINT_NAMES:
        assert not any(model.actuator_trnid[:, 0] == model.joint(name).id)
    for name, fields in properties.items():
        gid = model.geom(name).id
        for field, value in fields.items():
            assert np.array_equal(getattr(model, "geom_" + field)[gid], np.asarray(value))
    foot_bounds = [geom_bounds(model, data, model.geom(name).id) for name in properties]
    metadata = {
        "foot_geometry_material_exactly_copied_from_g1": properties,
        "foot_original_body_mass_kg": float(source.body_mass[body_id]),
        "fixture_origin_world_m": origin.tolist(), "fixture_yaw_deg": 90.,
        "initial_nose_end_face_center_world_m": nose_end.tolist(),
        "initial_foot_aabb_min_world_m": np.min([v[0] for v in foot_bounds], axis=0).tolist(),
        "initial_foot_aabb_max_world_m": np.max([v[1] for v in foot_bounds], axis=0).tolist(),
        "initial_aabb_x_gap_m": float(min(v[0][0] for v in foot_bounds) - nose_max[0]),
        "canonical_board_properties_exact": board_equal,
        "board_mass_kg": float(original.body_subtreemass[original_index.board_body]),
        "initial_all_contact_maximum_penetration_m": max([0.] + [-float(c.dist) for c in data.contact]),
    }
    return model, data, index, metadata, spec.to_xml()


def mechanical_energy(model, data, bodies):
    kinetic = 0.
    for bid in bodies:
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, linear, angular, bid)
        velocity, omega = linear @ data.qvel, angular @ data.qvel
        rotation = data.ximat[bid].reshape(3, 3)
        tensor = rotation @ np.diag(model.body_inertia[bid]) @ rotation.T
        kinetic += .5 * (model.body_mass[bid] * (velocity @ velocity) + omega @ tensor @ omega)
    potential = -sum(model.body_mass[bid] * (data.xipos[bid] @ model.opt.gravity) for bid in bodies)
    return float(kinetic), float(potential)


def snapshot(model, data, index, board_geoms, board_bodies, slide_qpos, slide_qvel):
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jac, index.board_body)
    mass = model.body_subtreemass[index.board_body]
    com = sum(model.body_mass[b] * data.xipos[b] for b in board_bodies) / mass
    kinetic, potential = mechanical_energy(model, data, board_bodies)
    fixture_k, fixture_u = mechanical_energy(model, data, [model.body("mechanical_foot_fixture").id])
    return {
        "time_s": float(data.time), "board_com_world_m": com.tolist(),
        "board_com_velocity_world_m_s": (jac @ data.qvel).tolist(),
        "board_kinetic_j": kinetic, "board_gravity_potential_j": potential,
        "fixture_kinetic_j": fixture_k, "fixture_gravity_potential_j": fixture_u,
        "board_mechanical_energy_j": kinetic + potential,
        "total_mechanical_energy_j": kinetic + potential + fixture_k + fixture_u,
        "whole_board_lowest_point_m": min(float(geom_bounds(model, data, g)[0][2]) for g in board_geoms),
        "tail_lowest_point_m": float(geom_bounds(model, data, model.geom("board_tail").id)[0][2]),
        "fixture_actual_slide_world_m": data.qpos[slide_qpos].tolist(),
        "fixture_actual_velocity_world_m_s": data.qvel[slide_qvel].tolist(),
        "board_qpos": data.qpos[index.board_qpos:index.board_qpos + 7].tolist(),
        "board_qvel": data.qvel[index.board_qvel:index.board_qvel + 6].tolist(),
        "all_contact_maximum_penetration_m": max([0.] + [-float(c.dist) for c in data.contact]),
    }


def longest_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best * DT


def run(out_dir):
    begin = time.monotonic()
    mujoco.set_mjcb_control(None)
    xml_hash = hashlib.sha256(BOARD_XML.read_bytes()).hexdigest()
    model, data, index, metadata, xml = make_fixture_scene()
    (out_dir / "mechanical_fixture_scene.xml").write_text(xml)
    board_bodies = [index.board_body] + [model.body(name.removesuffix("_joint")).id for name in BOARD_JOINT_NAMES]
    board_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in board_bodies]
    feet = {model.geom(f"left_foot{i}_collision").id for i in range(1, 8)}
    slide_ids = np.array([model.joint(f"fixture_slide_{i}").id for i in range(3)])
    slide_qpos, slide_qvel = model.jnt_qposadr[slide_ids], model.jnt_dofadr[slide_ids]
    states = [snapshot(model, data, index, board_geoms, board_bodies, slide_qpos, slide_qvel)]
    contacts, intervals, qpos, qvel, actions = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    assert metadata["initial_all_contact_maximum_penetration_m"] < 1e-12
    for _ in range(round(DURATION / DT)):
        assert time.monotonic() - begin < 120, "component wall budget exhausted"
        start = float(data.time)
        # Use interval-start targets, without writing any qpos/qvel after reset.
        target = prescribed_target(start)
        data.ctrl[:] = target
        before_qpos = data.qpos.copy()
        mujoco.mj_step(model, data)
        executed_force = data.actuator_force.copy()
        step_work = executed_force * (data.qpos[slide_qpos] - before_qpos[slide_qpos])
        board_force = np.zeros(3)
        board_foot_normal, board_foot_tangent = np.zeros(3), np.zeros(3)
        ground_force = np.zeros(3)
        wheel_contacts = set()
        tail_active = False
        max_penetration = 0.
        for cid, contact in enumerate(data.contact[:data.ncon]):
            g0, g1 = map(int, contact.geom)
            frame = contact.frame.reshape(3, 3)
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, cid, force)
            names = [model.geom(g).name for g in (g0, g1)]
            max_penetration = max(max_penetration, -float(contact.dist))
            board_gid = g0 if g0 in board_geoms else g1 if g1 in board_geoms else None
            normal = np.zeros(3)
            tangent = np.zeros(3)
            board_contact_force = np.zeros(3)
            is_foot_board = board_gid is not None and (g0 in feet or g1 in feet)
            if board_gid is not None:
                sign = 1 if board_gid == g1 else -1
                normal = sign * frame[0] * force[0]
                tangent = sign * (frame[1] * force[1] + frame[2] * force[2])
                board_contact_force = normal + tangent
                if is_foot_board:
                    board_force += board_contact_force
                    board_foot_normal += normal
                    board_foot_tangent += tangent
                if "ground" in names:
                    ground_force += board_contact_force
                    if force[0] > .1:
                        if model.geom(board_gid).name.startswith("wheel_"):
                            wheel_contacts.add(model.geom(board_gid).name)
                        tail_active |= "board_tail" in names
            contacts.append({
                "interval_start_s": start, "stage": "after_mj_step_before_extra_forward",
                "geoms": names, "position_world_m": contact.pos.tolist(),
                "frame_rows_world": frame.tolist(), "distance_m": float(contact.dist),
                "native_force_6d": force.tolist(), "dim": int(contact.dim),
                "friction_5d": contact.friction.tolist(), "solref": contact.solref.tolist(),
                "solimp": contact.solimp.tolist(), "is_foot_board": is_foot_board,
                "force_on_board_world_n": board_contact_force.tolist(),
                "normal_force_on_board_world_n": normal.tolist(),
                "tangent_force_on_board_world_n": tangent.tolist(),
            })
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert not (np.any(data.qfrc_applied) or np.any(data.xfrc_applied))
        assert np.max(abs(executed_force)) <= FORCE_CAP + 1e-9
        # Position/contact geometry is post-integration only after this forward.
        mujoco.mj_forward(model, data)
        state = snapshot(model, data, index, board_geoms, board_bodies, slide_qpos, slide_qvel)
        intervals.append({
            "start_s": start, "end_s": float(data.time),
            "target_fixture_slide_world_m": target.tolist(),
            "executed_fixture_actuator_force_n": executed_force.tolist(),
            "fixture_actuator_work_per_axis_j": step_work.tolist(),
            "foot_force_on_board_world_n": board_force.tolist(),
            "foot_normal_force_on_board_world_n": board_foot_normal.tolist(),
            "foot_tangent_force_on_board_world_n": board_foot_tangent.tolist(),
            "ground_force_on_board_world_n": ground_force.tolist(),
            "unique_active_ground_wheels": sorted(wheel_contacts),
            "tail_ground_above_point1n": bool(tail_active),
            "board_ground_force_above_point1n": bool(wheel_contacts or tail_active or np.linalg.norm(ground_force) > .1),
            "maximum_preintegration_contact_penetration_m": max_penetration,
            "postintegration_whole_board_lowest_point_m": state["whole_board_lowest_point_m"],
        })
        states.append(state)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        actions.append(target)
    work = np.asarray([v["fixture_actuator_work_per_axis_j"] for v in intervals])
    foot_forces = np.asarray([v["foot_force_on_board_world_n"] for v in intervals])
    ground_forces = np.asarray([v["ground_force_on_board_world_n"] for v in intervals])
    mass = metadata["board_mass_kg"]
    momentum_change = mass * (np.asarray(states[-1]["board_com_velocity_world_m_s"]) - states[0]["board_com_velocity_world_m_s"])
    expected_impulse = DT * np.sum(foot_forces + ground_forces + mass * model.opt.gravity, axis=0)
    maximum_penetration = max(max(v["all_contact_maximum_penetration_m"] for v in states),
                              max(v["maximum_preintegration_contact_penetration_m"] for v in intervals))
    positive_fz_impulse = float(DT * np.maximum(foot_forces[:, 2], 0).sum())
    flight = [not r["board_ground_force_above_point1n"] and r["postintegration_whole_board_lowest_point_m"] > .005
              for r in intervals]
    geometry_flight = [r["postintegration_whole_board_lowest_point_m"] > .005 for r in intervals]
    summary = {
        "scope": "external mechanical fixture component test; no G1 execution, teacher, or Ollie success claim",
        "configuration_count": 1, "saved_component_episodes": 1, "profile_changed_after_observation": False,
        "canonical_board_xml_sha256": xml_hash,
        "canonical_board_xml_unchanged": hashlib.sha256(BOARD_XML.read_bytes()).hexdigest() == xml_hash,
        "mujoco_version": mujoco.__version__, "dt_s": DT, "duration_s": DURATION,
        "wall_s": time.monotonic() - begin, "initializations": 1,
        "post_initialization_state_writes": 0, "external_qfrc_or_xfrc_applied": False,
        "board_actuators": 0, "fixture_actuators": model.nu,
        "equality_constraints": model.neq, "mocap_bodies": model.nmocap,
        "fixture_servo_kp_n_per_m": KP, "fixture_servo_kd_ns_per_m": KD,
        "fixture_force_cap_per_axis_n": FORCE_CAP,
        "fixture_power_source_note": "External 3-slide test equipment; its force, stiffness and fixed orientation are not a claim about G1 capability.",
        "metadata": metadata,
        "profile_knots_time_xyz_m": [[t, prescribed_target(t).tolist()] for t in (0., .04, .24, .32, .4)],
        "samples": len(states), "executed_intervals": len(intervals),
        "maximum_all_geometry_penetration_m": maximum_penetration,
        "normal_contact_quality_pass_le_10mm": maximum_penetration <= .01,
        "foot_board_active_intervals": sum(np.linalg.norm(f) > .1 for f in foot_forces),
        "maximum_foot_upward_force_on_board_n": float(foot_forces[:, 2].max()),
        "positive_foot_vertical_impulse_ns": positive_fz_impulse,
        "net_foot_force_impulse_world_ns": (DT * foot_forces.sum(axis=0)).tolist(),
        "normal_foot_vertical_impulse_ns": DT * sum(r["foot_normal_force_on_board_world_n"][2] for r in intervals),
        "tangential_foot_vertical_impulse_ns": DT * sum(r["foot_tangent_force_on_board_world_n"][2] for r in intervals),
        "ground_force_impulse_world_ns": (DT * ground_forces.sum(axis=0)).tolist(),
        "board_com_momentum_change_world_ns": momentum_change.tolist(),
        "force_impulse_momentum_balance_error_world_ns": (momentum_change - expected_impulse).tolist(),
        "fixture_signed_actuator_work_j": float(work.sum()),
        "fixture_positive_actuator_work_j": float(np.maximum(work, 0).sum()),
        "fixture_negative_actuator_work_j": float(np.minimum(work, 0).sum()),
        "fixture_signed_work_per_axis_j": work.sum(axis=0).tolist(),
        "fixture_peak_executed_force_n": float(max(abs(np.asarray(r["executed_fixture_actuator_force_n"])).max() for r in intervals)),
        "board_mechanical_energy_change_j": states[-1]["board_mechanical_energy_j"] - states[0]["board_mechanical_energy_j"],
        "total_mechanical_energy_change_j": states[-1]["total_mechanical_energy_j"] - states[0]["total_mechanical_energy_j"],
        "actuator_work_minus_total_mechanical_energy_change_j": float(work.sum()) - (states[-1]["total_mechanical_energy_j"] - states[0]["total_mechanical_energy_j"]),
        "work_note": "Discrete executed generalized force times actual slide displacement, including gravity compensation by the fixture. Work-energy residual includes compliant-contact/friction and numerical dissipation; it is not unaccounted direct board actuation.",
        "maximum_board_com_upward_velocity_m_s": max(r["board_com_velocity_world_m_s"][2] for r in states),
        "maximum_whole_board_lowest_point_m": max(r["whole_board_lowest_point_m"] for r in states),
        "longest_geometry_clearance_above_5mm_s": longest_run(geometry_flight),
        "longest_no_ground_force_and_clearance_above_5mm_s": longest_run(flight),
        "tail_ground_active_duration_s": DT * sum(r["tail_ground_above_point1n"] for r in intervals),
        "positive_vertical_force_component_observed": bool(positive_fz_impulse > 0),
        "net_upward_contact_mechanism_positive_example": bool(maximum_penetration <= .01 and foot_forces[:, 2].sum() > 0),
        "whole_board_flight_component_positive_example": bool(maximum_penetration <= .01 and longest_run(flight) >= .04),
        "interpretation_limits": [
            "A finite force actuator, orientation guide and prescribed path are external test equipment, not the robot.",
            "Positive upward impulse and whole-board flight are separate observations, neither proves a G1 Ollie.",
            "A failed fixed path does not prove every admissible nose contact or robot control must fail.",
            "No real shoe/grip-tape, deck flexibility or material parameters were calibrated.",
            "No retries, parameter search, hidden board force, intermediate pose reset or success promotion are permitted.",
        ],
    }
    assert summary["canonical_board_xml_unchanged"]
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "states.json", states)
    save_json(out_dir / "intervals.json", intervals)
    save_json(out_dir / "contacts_execution_stage.json", contacts)
    np.savez_compressed(out_dir / "fixture_rollout.npz", qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                        action=np.asarray(actions), time_s=np.arange(len(states)) * DT,
                        metadata_json=np.asarray(json.dumps(summary, default=json_scalar)))
    print(json.dumps({key: value for key, value in summary.items() if key != "metadata"}, indent=2, default=json_scalar))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        parser.error("choose a fresh directory; the single trial must not overwrite earlier results")
    out_dir.mkdir(parents=True)
    run(out_dir)


if __name__ == "__main__":
    main()
