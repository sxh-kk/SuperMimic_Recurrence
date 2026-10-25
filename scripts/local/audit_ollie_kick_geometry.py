"""Compare two fixed kick-angle hypotheses using geometry, without integration.

The 20-degree hypothesis keeps the original inner/lower kick attachment point
and component masses. It is a separate board-only model, not a calibrated board
or an executed Ollie. The friction bound is an outer cone necessary condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.optimize import brentq

from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_JOINT_NAMES, BOARD_XML, make_board_spec


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_matrix(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def geom_bottom(model, data, geom):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        extent = abs(rotation[2]) @ model.geom_size[geom]
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        z = rotation[2, 2]
        radius, halfwidth = model.geom_size[geom, :2]
        extent = radius * np.sqrt(max(0, 1 - z * z)) + halfwidth * abs(z)
    else:
        raise ValueError("unexpected board collision geometry")
    return float(data.geom_xpos[geom, 2] - extent)


def evaluate(model, original_gamma):
    tail, nose = model.geom("board_tail").id, model.geom("board_nose").id
    rear = model.body("wheel_rear_left").id
    wheel = model.geom("wheel_rear_left_collision").id
    radius = model.geom_size[wheel, 0]
    rear_center = model.body_pos[rear] + model.geom_pos[wheel]
    free = model.joint("board_free").id
    qadr = model.jnt_qposadr[free]
    tail_rotation = quat_matrix(model.geom_quat[tail])
    actual_gamma = float(np.rad2deg(np.arctan2(tail_rotation[0, 2], tail_rotation[0, 0])))
    if abs(actual_gamma - original_gamma) > 1e-6:
        raise RuntimeError("compiled kick angle differs from the declared hypothesis")

    def lower(phi):
        rotation = rotation_y(-phi)
        root_z = radius - (rotation @ rear_center)[2]
        return (
            root_z
            + (rotation @ model.geom_pos[tail])[2]
            - (abs((rotation @ tail_rotation)[2]) @ model.geom_size[tail])
        )

    phi = brentq(lower, 0, np.deg2rad(70), xtol=1e-14)
    data = mujoco.MjData(model)
    data.qpos[qadr : qadr + 3] = [0, 0, radius - (rotation_y(-phi) @ rear_center)[2]]
    data.qpos[qadr + 3 : qadr + 7] = [np.cos(phi / 2), 0, -np.sin(phi / 2), 0]
    mujoco.mj_forward(model, data)
    # This only sets hypothetical poses for geometry. It never mj_steps.
    native_tail = geom_bottom(model, data, tail)
    distance = mujoco.mj_geomDistance(model, data, tail, model.geom("ground").id, 1.0, None)
    nose_normal = data.geom_xmat[nose].reshape(3, 3)[:, 2]
    tangent_z = np.sqrt(max(0, 1 - nose_normal[2] ** 2))
    names = BOARD_GEOM_NAMES + tuple(
        name.removesuffix("_joint") + "_collision" for name in BOARD_JOINT_NAMES
    )
    clearances = {name: geom_bottom(model, data, model.geom(name).id) for name in names}
    if abs(native_tail) > 1e-10 or abs(distance - native_tail) > 1e-8:
        raise RuntimeError("analytic tail threshold and native geometry distance disagree")
    return {
        "kick_angle_deg": float(original_gamma),
        "compiled_tail_kick_angle_deg": actual_gamma,
        "first_tail_touch_board_nose_up_deg": float(np.rad2deg(phi)),
        "nose_top_face_world_tilt_deg": float(np.rad2deg(np.arccos(nose_normal[2]))),
        "top_face_mu_strictly_greater_than_for_positive_upward_force": float(
            nose_normal[2] / tangent_z
        ),
        "top_face_circular_outer_cone_fz_over_fn_upper_bound": {
            str(mu): float(-nose_normal[2] + mu * tangent_z) for mu in (0.6, 0.9)
        },
        "native_tail_clearance_m": native_tail,
        "native_geom_distance_tail_ground_m": float(distance),
        "all_nine_geometry_clearances_m": clearances,
        "root_position_for_geometry_only_m": data.qpos[qadr : qadr + 3].tolist(),
        "tail_local_center_m": model.geom_pos[tail].tolist(),
        "nose_local_center_m": model.geom_pos[nose].tolist(),
        "board_total_mass_kg": float(model.body_subtreemass[model.body("board").id]),
        "board_body_principal_inertia_kg_m2": model.body_inertia[model.body("board").id].tolist(),
        "full_ollie_success_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a fresh output directory")
    source_sha = hashlib.sha256(BOARD_XML.read_bytes()).hexdigest()
    source = make_board_spec().compile()
    alternate_spec = make_board_spec()
    changed = {}
    for name, sign in (("board_tail", 1), ("board_nose", -1)):
        geom = source.geom(name).id
        inner_lower = np.array([sign * source.geom_size[geom, 0], 0, -source.geom_size[geom, 2]])
        attachment = source.geom_pos[geom] + quat_matrix(source.geom_quat[geom]) @ inner_lower
        angle = sign * np.deg2rad(20)
        target = alternate_spec.geom(name)
        target.pos = attachment - rotation_y(angle) @ inner_lower
        # The authored XML uses Euler orientation; changing quat alone leaves
        # that alternative representation active in MjSpec.
        target.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        target.quat = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
        changed[name] = {
            "fixed_inner_lower_attachment_m": attachment.tolist(),
            "candidate_local_center_m": target.pos.tolist(),
        }
    alternate = alternate_spec.compile()
    for name, sign in (("board_tail", 1), ("board_nose", -1)):
        geom = alternate.geom(name).id
        rotation = quat_matrix(alternate.geom_quat[geom])
        expected = rotation_y(sign * np.deg2rad(20))
        inner_lower = np.array(
            [sign * alternate.geom_size[geom, 0], 0, -alternate.geom_size[geom, 2]]
        )
        attachment = alternate.geom_pos[geom] + rotation @ inner_lower
        if (
            np.max(abs(rotation - expected)) > 1e-12
            or np.max(abs(attachment - changed[name]["fixed_inner_lower_attachment_m"])) > 1e-12
        ):
            raise RuntimeError("compiled candidate rotation or fixed attachment changed")
    if not np.array_equal(source.body_mass, alternate.body_mass):
        raise RuntimeError("component masses changed")
    results = {"canonical_12deg": evaluate(source, 12), "hypothesis_20deg": evaluate(alternate, 20)}
    if hashlib.sha256(BOARD_XML.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("canonical board file changed")
    report = {
        "schema": "supermimic.ollie.kick_geometry_comparison.v1",
        "canonical_board_sha256": source_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_definition": changed,
        "scope": "board-only kinematic geometry; rear wheels support; pure pitch; no integration/robot/control/training",
        "integration_steps": 0,
        "component_masses_exact": True,
        "inertia_note": "Changing kick geometry recomputes its actual composite inertia; it is reported rather than frozen to the old geometry.",
        "results": results,
        "limits": [
            "Positive outer-cone bound does not certify native pyramidal contact force, shoe reach or sustained contact.",
            "Negative bound rules out upward force from that top face at this pose, not from every finite edge/contact configuration.",
            "No measured kick angle, friction or restitution was inferred from the source video.",
            "20 degrees is a literature hypothesis, not a selected production model or new successful trajectory.",
        ],
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
