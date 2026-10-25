"""Replay the immutable 60 episode and audit finite-nose contact force bounds.

The translated-foot probe is geometry only: no full-body IK, no reachability or
force realization claim, and no simulated control sequence is optimized here.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.ollie.physics import BOARD_JOINT_NAMES, BOARD_XML, SceneIndex, make_scene_spec
from supermimic.ollie.state import JointStateIndices

ACTUAL_HASH = "43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def native_pyramid_ray_proof():
    """Check the four condim=3 rays using MuJoCo's C API with explicit dim=3."""
    library_path = next(Path(mujoco.__file__).parent.glob("libmujoco.so*"))
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_double)
    decode = library.mju_decodePyramid
    decode.argtypes = [pointer, pointer, pointer, ctypes.c_int]
    decode.restype = None
    mu = np.array([0.6, 0.6, 0.005, 0.0001, 0.0001], dtype=np.float64)
    rays = []
    for weights in np.eye(4):
        force = np.zeros(6)
        decode(
            force.ctypes.data_as(pointer),
            weights.ctypes.data_as(pointer),
            mu.ctypes.data_as(pointer),
            3,
        )
        rays.append(force.tolist())
    expected = np.array([[1, 0.6, 0], [1, -0.6, 0], [1, 0, 0.6], [1, 0, -0.6]])
    error = float(abs(np.asarray(rays)[:, :3] - expected).max())
    if error:
        raise RuntimeError("Native pyramid rays do not match the stated contact bound")
    return {
        "dim": 3,
        "mu": mu.tolist(),
        "native_force_rays": rays,
        "maximum_error": error,
        "native_library": str(library_path.resolve()),
        "native_library_sha256": sha256(library_path),
    }


def contact_row(model, data, contact_id, feet, boxes):
    contact = data.contact[contact_id]
    first, second = map(int, contact.geom)
    foot = first if first in feet else second if second in feet else None
    box = second if first in feet else first if second in feet else None
    if foot is None or box not in boxes:
        return None
    # MuJoCo's native frame points geom0 -> geom1. Force acts on geom1.
    sign = -1 if foot == first else 1
    frame = contact.frame.reshape(3, 3)
    normal = sign * frame[0]
    local_force = np.empty(6)
    mujoco.mj_contactForce(model, data, contact_id, local_force)
    board_force = -sign * (frame.T @ local_force[:3])
    # For dim=3 the pyramidal cross-section is the convex hull of the four
    # decoded rays; torsional/rolling coordinates in dim>3 add no linear force.
    bound = -normal[2] + max(
        contact.friction[0] * abs(frame[1, 2]), contact.friction[1] * abs(frame[2, 2])
    )
    box_rotation = data.geom_xmat[box].reshape(3, 3)
    return {
        "time_physics_s": float(data.time),
        "side": feet[foot],
        "box": boxes[box],
        "native_geom_order": [model.geom(first).name, model.geom(second).name],
        "foot_geom": model.geom(foot).name,
        "board_geom": model.geom(box).name,
        "condim": int(contact.dim),
        "normal_on_foot_world": normal.tolist(),
        "tangent1_native_world": frame[1].tolist(),
        "tangent2_native_world": frame[2].tolist(),
        "native_contact_friction": contact.friction.tolist(),
        "normal_force_n": float(local_force[0]),
        "active_at_fn_ge_0_1n": bool(local_force[0] >= 0.1),
        "actual_force_on_board_world_n": board_force.tolist(),
        "max_board_vertical_force_per_normal_force": float(bound),
        "actual_board_fz_bound_violation_n": float(max(board_force[2] - bound * local_force[0], 0)),
        "contact_position_world_m": contact.pos.tolist(),
        "contact_position_board_geom_local_m": (
            box_rotation.T @ (contact.pos - data.geom_xpos[box])
        ).tolist(),
        "signed_contact_distance_m": float(contact.dist),
    }


def summarize(rows, start_time, active_only):
    selected = [
        r
        for r in rows
        if r["time_physics_s"] >= start_time - 1e-8
        and (not active_only or r["active_at_fn_ge_0_1n"])
    ]
    groups = {}
    for side in ("left", "right"):
        for box in ("center", "tail", "nose"):
            group = [r for r in selected if r["side"] == side and r["box"] == box]
            value = {
                "contact_occurrences": len(group),
                "distinct_physical_times": len({r["time_physics_s"] for r in group}),
            }
            if group:
                value.update(
                    normal_on_foot_z_range=[
                        min(r["normal_on_foot_world"][2] for r in group),
                        max(r["normal_on_foot_world"][2] for r in group),
                    ],
                    contact_mu_values=sorted({tuple(r["native_contact_friction"]) for r in group}),
                    maximum_upward_board_force_capability_per_fn=max(
                        r["max_board_vertical_force_per_normal_force"] for r in group
                    ),
                    maximum_actual_board_fz_n=max(
                        r["actual_force_on_board_world_n"][2] for r in group
                    ),
                    positive_vertical_capability_occurrences=sum(
                        r["max_board_vertical_force_per_normal_force"] > 0 for r in group
                    ),
                    maximum_realized_cone_bound_violation_n=max(
                        r["actual_board_fz_bound_violation_n"] for r in group
                    ),
                )
            groups[side + "_" + box] = value
    return {
        "start_time_s": start_time,
        "active_force_threshold_n": 0.1 if active_only else None,
        "count_semantics": "One occurrence per native contact per sampled physical time; not a count of unique contacts or forces summed across time.",
        "groups": groups,
    }


def replay(model, index, archive, source_records):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = archive["qpos"][0], archive["qvel"][0]
    data.ctrl[index.robot_actuators] = archive["qpos"][0, index.robot_joint_qpos]
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, index, 0.1)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    feet = {
        model.geom(f"{side}_foot{i}_collision").id: side
        for side in ("left", "right")
        for i in range(1, 8)
    }
    boxes = {model.geom("board_" + box).id: box for box in ("center", "tail", "nose")}
    rows, samples, errors = [], [], {"qpos": 0.0, "qvel": 0.0, "state92": 0.0}
    geometry_peak = {
        "nose_up_rad": -np.inf,
        "nose_up_time_s": 0.0,
        "nose_top_normal_tilt_rad": -np.inf,
        "nose_top_tilt_time_s": 0.0,
    }

    def observe():
        sample = observer.observe(data)
        expected = source_records[len(samples)]
        if sample != expected:
            raise RuntimeError(f"Observer differs at sample {len(samples)}")
        samples.append(sample)
        if data.time >= 2.6 - 1e-8:
            rotation = data.xmat[index.board_body].reshape(3, 3)
            nose_up = float(np.arctan2(rotation[2, 0], rotation[2, 2]))
            tilt = float(np.arccos(np.clip(data.geom_xmat[model.geom("board_nose").id, 8], -1, 1)))
            if nose_up > geometry_peak["nose_up_rad"]:
                geometry_peak.update(nose_up_rad=nose_up, nose_up_time_s=float(data.time))
            if tilt > geometry_peak["nose_top_normal_tilt_rad"]:
                geometry_peak.update(
                    nose_top_normal_tilt_rad=tilt, nose_top_tilt_time_s=float(data.time)
                )
        for i in range(data.ncon):
            row = contact_row(model, data, i, feet, boxes)
            if row is not None:
                rows.append(row)

    observe()
    for frame, target in enumerate(archive["action"]):
        data.ctrl[index.robot_actuators] = target
        for _ in range(8):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            observe()
        for name, value, field in (
            ("qpos", data.qpos, "qpos"),
            ("qvel", data.qvel, "qvel"),
            ("state92", indices.pack(data.qpos, data.qvel).numpy(), "state"),
        ):
            error = float(abs(value - archive[field][frame + 1]).max())
            errors[name] = max(errors[name], error)
            if error:
                raise RuntimeError(f"Actual replay {name} differs at frame {frame + 1}")
    if len(samples) != len(source_records):
        raise RuntimeError("Physical sample count mismatch")
    proof = {
        "initializations": 1,
        "phase_resets": 0,
        "actions": len(archive["action"]),
        "boundaries": len(archive["qpos"]),
        "physical_observer_samples": len(samples),
        "maximum_absolute_error": errors,
        "all_original_observer_fields_exact": True,
        "maximum_external_force_norm": max(
            max(r["external_generalized_force_norm"], r["external_body_force_norm"])
            for r in samples
        ),
        "endpoint_time_s": float(data.time),
        "pop_phase_geometry_peak": geometry_peak,
    }
    return proof, rows, samples


def format_numbers(value):
    return " ".join(f"{number:.17g}" for number in np.asarray(value).ravel())


def probe_xml(model, data, foot_body, nose, translation):
    """Preserve authored shapes/frames/materials; deliberately omit the robot legs."""
    root = ET.Element("mujoco", model="geometry_only_finite_nose_probe")
    ET.SubElement(root, "option", gravity="0 0 -9.81", cone="pyramidal", timestep="0.0025")
    world = ET.SubElement(root, "worldbody")
    nq = np.empty(4)
    mujoco.mju_mat2Quat(nq, data.geom_xmat[nose])

    def geom_attributes(geom):
        return {
            "name": model.geom(geom).name,
            "pos": format_numbers(model.geom_pos[geom]),
            "quat": format_numbers(model.geom_quat[geom]),
            "friction": format_numbers(model.geom_friction[geom]),
            "priority": str(int(model.geom_priority[geom])),
            "condim": str(int(model.geom_condim[geom])),
            "contype": str(int(model.geom_contype[geom])),
            "conaffinity": str(int(model.geom_conaffinity[geom])),
            "margin": str(float(model.geom_margin[geom])),
            "gap": str(float(model.geom_gap[geom])),
            "solref": format_numbers(model.geom_solref[geom]),
            "solimp": format_numbers(model.geom_solimp[geom]),
        }

    attrs = geom_attributes(nose)
    attrs.update(
        type="box",
        size=format_numbers(model.geom_size[nose]),
        pos=format_numbers(data.geom_xpos[nose]),
        quat=format_numbers(nq),
    )
    ET.SubElement(world, "geom", **attrs)
    foot = ET.SubElement(
        world,
        "body",
        name="isolated_left_foot",
        pos=format_numbers(data.xpos[foot_body] + translation),
        quat=format_numbers(data.xquat[foot_body]),
    )
    ET.SubElement(foot, "freejoint")
    ET.SubElement(
        foot,
        "inertial",
        mass=str(float(model.body_mass[foot_body])),
        pos=format_numbers(model.body_ipos[foot_body]),
        quat=format_numbers(model.body_iquat[foot_body]),
        diaginertia=format_numbers(model.body_inertia[foot_body]),
    )
    for i in range(1, 8):
        geom = model.geom(f"left_foot{i}_collision").id
        if model.geom_bodyid[geom] != foot_body:
            raise ValueError("Foot geoms must share the authored foot body")
        attrs = geom_attributes(geom)
        attrs.update(type="capsule", size=format_numbers(model.geom_size[geom, :2]))
        ET.SubElement(foot, "geom", **attrs)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def closest_and_probe(model, index, archive, output):
    frame = 145
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = archive["qpos"][frame], archive["qvel"][frame]
    mujoco.mj_forward(model, data)
    nose = model.geom("board_nose").id
    candidates = []
    for i in range(1, 8):
        geom = model.geom(f"left_foot{i}_collision").id
        line = np.empty(6)
        gap = float(mujoco.mj_geomDistance(model, data, nose, geom, 2.0, line))
        candidates.append((gap, geom, line.copy()))
    gap, geom, line = min(candidates, key=lambda row: row[0])
    if gap <= 0:
        raise ValueError("This frozen probe expects a positive original nose separation")
    direction = (line[3:] - line[:3]) / gap
    foot_body = int(model.geom_bodyid[geom])
    translation = -direction * (gap + 0.00005)
    text = probe_xml(model, data, foot_body, nose, translation)
    (output / "geometry_only_probe.xml").write_text(text)
    probe = mujoco.MjModel.from_xml_string(text)
    scratch = mujoco.MjData(probe)
    mujoco.mj_forward(probe, scratch)
    feet = {probe.geom(f"left_foot{i}_collision").id: "left" for i in range(1, 8)}
    contacts = [
        contact_row(probe, scratch, i, feet, {probe.geom("board_nose").id: "nose"})
        for i in range(scratch.ncon)
    ]
    # Only geometry/cone data are meaningful in this isolated forward call.
    # Its unsupported free-foot solver force is not an actual controlled force.
    excluded = {
        "time_physics_s",
        "normal_force_n",
        "active_at_fn_ge_0_1n",
        "actual_force_on_board_world_n",
        "actual_board_fz_bound_violation_n",
    }
    contacts = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in contacts
        if row is not None
    ]
    if not contacts:
        raise RuntimeError("Finite-edge geometric closure produced no native contact")
    local = data.geom_xmat[nose].reshape(3, 3).T @ (line[:3] - data.geom_xpos[nose])
    return {
        "source_frame": frame,
        "source_time_physics_s": float(archive["time_physics"][frame]),
        "original_gap_m": gap,
        "nearest_original_capsule": model.geom(geom).name,
        "original_nose_witness_world_m": line[:3].tolist(),
        "original_foot_witness_world_m": line[3:].tolist(),
        "original_nose_witness_local_m": local.tolist(),
        "original_separation_direction_is_not_an_actual_contact_normal": True,
        "original_separation_direction_nose_to_foot": direction.tolist(),
        "nose_witness_surface": "x_min/z_max finite edge (back end/top), not top-plane interior",
        "isolated_whole_foot_translation_world_m": translation.tolist(),
        "deliberate_probe_overlap_m": 0.00005,
        "native_probe_contacts": contacts,
        "maximum_probe_board_vertical_force_per_fn": max(
            r["max_board_vertical_force_per_normal_force"] for r in contacts
        ),
        "canonical_geometry_material_priority_preserved": True,
        "scope_limits": {
            "full_body_ik": False,
            "leg_reachability_certified": False,
            "whole_robot_collision_certified": False,
            "actual_episode_execution": False,
            "contact_force_realization_tested": False,
            "control_search": False,
            "integration_steps": 0,
        },
        "warning": "Positive force-cone capability is a local geometric possibility only; it does not prove a reachable whole-robot drag or an executable Ollie.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actual", type=Path, default=Path("logs/ollie_pop_events_preflight_02/actual_rollout.npz")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/ollie_nose_contact_audit_01"))
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory must be new")
    if sha256(args.actual) != ACTUAL_HASH:
        raise ValueError("Audit is restricted to the immutable actual 60 rollout")
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("Control callbacks are not permitted")
    with np.load(args.actual, allow_pickle=False) as data:
        archive = {name: data[name].copy() for name in data.files}
    source_path = args.actual.parent / "contacts.json"
    original_records = json.loads(source_path.read_text())
    model = make_scene_spec(timestep=0.0025).compile()
    if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError("This exact bound requires the unchanged pyramidal model")
    index = SceneIndex.from_model(model)
    proof, contacts, samples = replay(model, index, archive, original_records)
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "actual_contact_occurrences.jsonl").open("w") as file:
        for row in contacts:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    probe = closest_and_probe(model, index, archive, args.out_dir)
    max_tilt = max(float(np.arccos(np.clip(row["board_upright_cosine"], -1, 1))) for row in samples)
    kick = 0.20943951
    # This top-plane expression assumes pure board pitch and favorable friction;
    # the circular cone is an upper bound for the narrower pyramidal cone.
    top_tilt = proof["pop_phase_geometry_peak"]["nose_up_rad"] + kick
    result = {
        "schema": "supermimic.ollie.finite_nose_contact_audit.v1",
        "inputs": {
            "actual": str(args.actual.resolve()),
            "actual_sha256": ACTUAL_HASH,
            "contacts_sha256": sha256(source_path),
            "canonical_board_xml_sha256": sha256(BOARD_XML),
            "script_sha256": sha256(__file__),
            "mujoco_version": mujoco.__version__,
        },
        "replay_proof": proof,
        "cone_type": "pyramidal",
        "pyramid_native_ray_proof": native_pyramid_ray_proof(),
        "actual_contacts": {
            "all_episode_all_registered": summarize(contacts, 0.0, False),
            "all_episode_fn_ge_0_1n": summarize(contacts, 0.0, True),
            "pop_phase_all_registered": summarize(contacts, 2.6, False),
            "pop_phase_fn_ge_0_1n": summarize(contacts, 2.6, True),
        },
        "material_examples": {
            name: {
                "geom_friction": model.geom_friction[model.geom(name).id].tolist(),
                "priority": int(model.geom_priority[model.geom(name).id]),
            }
            for name in ("left_foot6_collision", "board_center", "board_tail", "board_nose")
        },
        "top_plane_conditional_bound": {
            "mu": 0.6,
            "phase_peak_board_tilt_rad": max_tilt,
            "measured_nose_top_normal_max_tilt_rad": proof["pop_phase_geometry_peak"][
                "nose_top_normal_tilt_rad"
            ],
            "pure_pitch_nose_top_tilt_deg": top_tilt * 180 / np.pi,
            "circular_upper_bound_board_fz_per_fn": float(
                -np.cos(top_tilt) + 0.6 * np.sin(top_tilt)
            ),
            "necessary_top_tilt_for_positive_board_fz_deg": float(np.arctan(1 / 0.6) * 180 / np.pi),
            "pure_board_pitch_threshold_with_12deg_nose_deg": float(
                np.arctan(1 / 0.6) * 180 / np.pi - 12
            ),
            "assumption": "Top-plane interior and pure nose-up pitch; not all finite faces/edges or all reachable poses.",
        },
        "geometry_only_edge_probe": probe,
        "conclusion": "Actual pop-phase contacts only push the board down. A native finite-edge geometric probe admits a positive upward force-cone bound at mu=.6, so the top-plane no-lift condition cannot establish whole-model impossibility. Full-body reachability and actual sliding/force realization remain untested.",
        "is_control_success": False,
        "is_eligible_teacher": False,
        "physics_or_canonical_asset_modified": False,
    }
    save_json(args.out_dir / "result.json", result)
    save_json(
        args.out_dir / "output_hashes.json",
        {
            p.name: sha256(p)
            for p in args.out_dir.iterdir()
            if p.is_file() and p.name != "output_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                "replay": proof,
                "pop_contact_groups": result["actual_contacts"]["pop_phase_fn_ge_0_1n"]["groups"],
                "probe_max_board_fz_per_fn": probe["maximum_probe_board_vertical_force_per_fn"],
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
