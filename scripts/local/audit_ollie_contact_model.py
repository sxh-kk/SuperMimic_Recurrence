"""Audit effective contacts and run isolated friction sensitivity branches.

Canonical assets are never edited. Explicit contact pairs are diagnostic models,
not calibrated replacements. Counterfactual branches do not certify an Ollie.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local.evaluate_ollie_policy import ContactObserver, continuous_windows
from supermimic.ollie.physics import BOARD_XML, SceneIndex, compile_scene, make_scene_spec

TRAJECTORY_SHA = "43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(records):
    times = np.array([r["time_physics"] for r in records])
    airborne = np.array(
        [
            r["unique_board_geoms_ground"] == 0 and r["minimum_board_clearance_m"] > 0.005
            for r in records
        ]
    )
    return {
        "endpoint_s": records[-1]["time_physics"],
        "first_robot_ground_s": next(
            (r["time_physics"] for r in records if r["unique_robot_geoms_ground"]), None
        ),
        "tail_windows": continuous_windows(np.array([r["tail_ground"] for r in records]), times),
        "whole_board_max_clearance_m": max(r["minimum_board_clearance_m"] for r in records),
        "flight_windows": continuous_windows(airborne, times),
        "maximum_penetration_m": max(r["maximum_penetration_m"] for r in records),
        "minimum_board_upright": min(r["board_upright_cosine"] for r in records),
        "external_forces_zero": all(
            r["external_body_force_norm"] == 0 and r["external_generalized_force_norm"] == 0
            for r in records
        ),
        "does_not_certify_full_ollie": True,
    }


def pairs_model(mu):
    baseline = make_scene_spec(timestep=0.0025).compile()
    model = make_scene_spec(timestep=0.0025, foot_deck_friction=mu).compile()
    for key in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_priority",
        "dof_damping",
        "dof_armature",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_trnid",
        "jnt_range",
    ):
        if not np.array_equal(getattr(model, key), getattr(baseline, key)):
            raise RuntimeError(f"Pair-only diagnostic changed {key}")
    return model


def contact_rows(model, data, feet, deck):
    rows = []
    for i, c in enumerate(data.contact[: data.ncon]):
        a, b = map(int, c.geom)
        side = next(
            (s for s, ids in feet.items() if (a in ids and b in deck) or (b in ids and a in deck)),
            None,
        )
        if side is None:
            continue
        value = np.empty(6)
        mujoco.mj_contactForce(model, data, i, value)
        if value[0] <= 0.1:
            continue
        sign = 1 if b in deck else -1
        frame = c.frame.reshape(3, 3)
        normal = sign * frame[0]
        force = sign * (frame.T @ value[:3])
        # Circular Coulomb cone is an outer bound for a pyramidal cone. A
        # positive bound is only permission, not proof that this force occurs.
        bound = float(normal[2] + c.friction[0] * math.sqrt(max(0, 1 - normal[2] ** 2)))
        rows.append(
            {
                "time_s": float(data.time),
                "side": side,
                "geoms": [model.geom(a).name, model.geom(b).name],
                "friction": c.friction.tolist(),
                "condim": int(c.dim),
                "solref": c.solref.tolist(),
                "solimp": c.solimp.tolist(),
                "normal_on_board": normal.tolist(),
                "force_on_board_world_n": force.tolist(),
                "normal_force_n": float(value[0]),
                "maximum_vertical_force_per_normal_force_outer_bound": bound,
            }
        )
    return rows


def foot_ground_parameter_probe(model, initial_qpos):
    """Geometry-only probe, outside every actual episode and counterfactual."""
    data = mujoco.MjData(model)
    index = SceneIndex.from_model(model)
    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)
    feet = {model.geom(f"{s}_foot{i}_collision").id for s in ("left", "right") for i in range(1, 8)}
    minimum = min(
        data.geom_xpos[g, 2]
        - model.geom_size[g, 1] * abs(data.geom_xmat[g, 8])
        - model.geom_size[g, 0]
        for g in feet
    )
    data.qpos[index.board_qpos] += 10  # Remove the board from this isolated probe.
    data.qpos[index.robot_qpos + 2] -= minimum + 0.001
    mujoco.mj_forward(model, data)
    ground = model.geom("ground").id
    values = [
        c.friction.tolist()
        for c in data.contact[: data.ncon]
        if (int(c.geom[0]) in feet and int(c.geom[1]) == ground)
        or (int(c.geom[1]) in feet and int(c.geom[0]) == ground)
    ]
    if not values or any(v[0] != 0.6 for v in values):
        raise RuntimeError("Foot-ground friction changed with a foot-deck-only override")
    return {
        "effective_friction": values[0],
        "contact_count": len(values),
        "scope": "geometry-only parameter probe; displaced poses are not an executed episode",
    }


def run(model, archive, *, start_frame=0, hot_state=None, compare=False):
    data = mujoco.MjData(model)
    index = SceneIndex.from_model(model)
    if hot_state is None:
        data.qpos[:], data.qvel[:] = archive["qpos"][0], archive["qvel"][0]
        data.ctrl[index.robot_actuators] = archive["qpos"][0, index.robot_joint_qpos]
    else:
        mujoco.mj_setState(model, data, hot_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, index, 0.1)
    records = [observer.observe(data)]
    feet = {
        s: {model.geom(f"{s}_foot{i}_collision").id for i in range(1, 8)} for s in ("left", "right")
    }
    deck = {model.geom("board_" + s).id for s in ("center", "tail", "nose")}
    contact_data = contact_rows(model, data, feet, deck)
    errors = {"qpos": 0.0, "qvel": 0.0}
    hot = None
    for frame in range(start_frame, len(archive["action"])):
        data.ctrl[index.robot_actuators] = archive["action"][frame]
        for _ in range(8):
            before = data.time
            mujoco.mj_step(model, data)
            if abs(data.time - before - 0.0025) > 1e-10:
                raise RuntimeError("Physical time discontinuity")
            mujoco.mj_forward(model, data)
            records.append(observer.observe(data))
            contact_data.extend(contact_rows(model, data, feet, deck))
        if compare:
            for name, previous in errors.items():
                errors[name] = max(
                    previous, float(abs(getattr(data, name) - archive[name][frame + 1]).max())
                )
        if frame + 1 == 130:
            hot = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, hot, mujoco.mjtState.mjSTATE_INTEGRATION)
    return summary(records), contact_data, errors, hot, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("logs/ollie_pop_events_preflight_02/actual_rollout.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a new output directory")
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("Control callbacks are not permitted during native replay")
    if digest(args.trajectory) != TRAJECTORY_SHA:
        raise ValueError("Expected frozen 60 actual trajectory")
    with np.load(args.trajectory, allow_pickle=False) as z:
        archive = {key: z[key].copy() for key in z.files}
    args.out_dir.mkdir(parents=True)
    original_xml_hash = digest(BOARD_XML)
    model, _, _ = compile_scene(timestep=0.0025)
    baseline, contacts, errors, hot, _ = run(model, archive, compare=True)
    if any(errors.values()):
        raise RuntimeError(f"Canonical native replay differs: {errors}")
    if hot is None or model.na or model.neq or model.nu != 29:
        raise RuntimeError("Unexpected model or missing exact 2.60s hot snapshot")
    parameters = {}
    for name in (
        "left_foot1_collision",
        "right_foot1_collision",
        "board_center",
        "board_tail",
        "board_nose",
        "ground",
    ):
        i = model.geom(name).id
        parameters[name] = {
            "friction": model.geom_friction[i].tolist(),
            "priority": int(model.geom_priority[i]),
            "solref": model.geom_solref[i].tolist(),
            "solimp": model.geom_solimp[i].tolist(),
            "margin": float(model.geom_margin[i]),
            "gap": float(model.geom_gap[i]),
        }
    variants = []
    for mu in (0.6, 0.9, 1.2):
        changed = pairs_model(mu)
        if mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION) != mujoco.mj_stateSize(
            changed, mujoco.mjtState.mjSTATE_INTEGRATION
        ):
            raise RuntimeError("Diagnostic pair model changed state layout")
        full, full_contacts, full_errors, _, _ = run(changed, archive, compare=True)
        branch, branch_contacts, branch_errors, _, _ = run(
            changed, archive, start_frame=130, hot_state=hot, compare=True
        )
        if mu == 0.6 and (any(full_errors.values()) or any(branch_errors.values())):
            raise RuntimeError("Explicit baseline pairs changed original dynamics")
        for rows in (full_contacts, branch_contacts):
            if any(abs(r["friction"][0] - mu) > 1e-12 for r in rows):
                raise RuntimeError("Explicit pair friction not applied")
        variants.append(
            {
                "explicit_foot_deck_mu": mu,
                "foot_deck_pairs": int(changed.npair),
                "from_same_original_initial_state_and_same_controls": full,
                "counterfactual_from_original_2p60_hot_state": branch,
                "full_state_deviation_from_original": full_errors,
                "branch_state_deviation_from_original": branch_errors,
                "foot_ground_parameter_probe": foot_ground_parameter_probe(
                    changed, archive["qpos"][0]
                ),
                "counterfactual_scope": "Model parameter changes at branch origin; not a continuous original-model episode or retrained controller",
                "calibrated_material": False,
            }
        )
    observed = sorted({tuple(r["friction"]) for r in contacts})
    report = {
        "schema": "supermimic.ollie.contact_model_audit.v1",
        "trajectory": str(args.trajectory.resolve()),
        "trajectory_sha256": TRAJECTORY_SHA,
        "canonical_assets_edited": False,
        "canonical_board_xml_sha256": original_xml_hash,
        "canonical_replay_errors": errors,
        "baseline": baseline,
        "geom_parameters": parameters,
        "observed_native_foot_deck_friction": observed,
        "observed_native_foot_deck_solref": sorted({tuple(r["solref"]) for r in contacts}),
        "official_contact_rule": "https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters",
        "top_surface_cone_example": {
            "board_nose_up_deg": 30.219872,
            "authored_nose_rise_deg": 12,
            "surface_tilt_deg": 42.219872,
            "mu": 0.6,
            "maximum_vertical_force_per_normal_force": -math.cos(math.radians(42.219872))
            + 0.6 * math.sin(math.radians(42.219872)),
            "minimum_surface_tilt_for_upward_force_deg": math.degrees(math.atan(1 / 0.6)),
            "restriction": "Planar top-face contact only. Does not exclude side/edge contacts, larger pitch, or different motion. Not a general impossibility proof.",
        },
        "friction_sensitivity_same_saved_controls": variants,
        "real_material_parameters_measured": False,
        "does_not_certify_full_ollie": True,
    }
    (args.out_dir / "native_foot_deck_contacts.json").write_text(
        json.dumps(contacts, indent=2) + "\n"
    )
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if digest(BOARD_XML) != original_xml_hash or digest(args.trajectory) != TRAJECTORY_SHA:
        raise RuntimeError("Canonical input changed during audit")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
