"""Bounded fixed-control capsule-versus-AABB-sole native physics diagnostic.

No actions are optimized. The candidate retains original body inertias and
materials and receives controls only if its unchanged initial pose is valid.
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

from scripts.local.evaluate_ollie_policy import ContactObserver, outcome
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_XML, SceneIndex, make_scene_spec

SOURCE_SHA = "8b602ad7f88266ba831c71df6c0f25bc84366a3af13427166e559f45d4470cb7"
CONTACT_SHA = "35a73e6aed6762289ad2a0a1e4d5a496ac97e0ce97b2cac1938ff264c895bd24"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def axis(quaternion):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return rotation.reshape(3, 3)[:, 2]


def capsule_union_aabb(model, ids):
    """Exact local coordinate extrema of rotated capsule line segments+spheres."""
    if np.any(model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_CAPSULE):
        raise ValueError("Baseline foot must contain seven authored capsules")
    if len(set(model.geom_bodyid[ids])) != 1:
        raise ValueError("All seven foot shapes must use the same local body frame")
    centers = model.geom_pos[ids]
    directions = np.stack([axis(q) for q in model.geom_quat[ids]])
    half_length, radius = model.geom_size[ids, 1:2], model.geom_size[ids, 0:1]
    endpoints = np.stack(
        (centers - half_length * directions, centers + half_length * directions), axis=1
    )
    lower = (endpoints.min(axis=1) - radius).min(axis=0)
    upper = (endpoints.max(axis=1) + radius).max(axis=0)
    return lower, upper, endpoints


def candidate_model(baseline):
    spec = make_scene_spec(timestep=0.0025, foot_deck_friction=0.9)
    disabled, geometry = set(), {}
    for side in ("left", "right"):
        names = [f"{side}_foot{i}_collision" for i in range(1, 8)]
        ids = [baseline.geom(name).id for name in names]
        lower, upper, endpoints = capsule_union_aabb(baseline, ids)
        center, half_size = (lower + upper) / 2, (upper - lower) / 2
        kept = spec.geom(names[0])
        kept.type = mujoco.mjtGeom.mjGEOM_BOX
        kept.fromto = np.full(6, np.nan)
        kept.pos, kept.quat, kept.size = center, [1, 0, 0, 0], half_size
        for name in names[1:]:
            geom = spec.geom(name)
            geom.contype, geom.conaffinity = 0, 0
            disabled.add(name)
        geometry[side] = {
            "body": baseline.body(int(baseline.geom_bodyid[ids[0]])).name,
            "original_geom_names": names,
            "kept_geom_name": names[0],
            "capsule_segment_endpoints_local_m": endpoints.tolist(),
            "capsule_radii_m": baseline.geom_size[ids, 0].tolist(),
            "union_aabb_lower_local_m": lower.tolist(),
            "union_aabb_upper_local_m": upper.tolist(),
            "box_center_local_m": center.tolist(),
            "box_half_size_m": half_size.tolist(),
            "interpretation": "axis-aligned bounding box contains original union; fills its gaps/corners; not the same collision surface",
        }
    deleted = []
    for pair in list(spec.pairs):
        if pair.geomname1 in disabled or pair.geomname2 in disabled:
            deleted.append(pair.name)
            spec.delete(pair)
    if len(deleted) != 36:
        raise RuntimeError("Expected exactly 36 disabled-capsule explicit pairs removed")
    candidate = spec.compile()
    for side, row in geometry.items():
        geom = candidate.geom(row["kept_geom_name"]).id
        # Verify the compiled box, not just mutable MjSpec properties.
        if (
            candidate.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX
            or not np.array_equal(candidate.geom_quat[geom], [1, 0, 0, 0])
            or not np.array_equal(candidate.geom_pos[geom], row["box_center_local_m"])
            or not np.array_equal(candidate.geom_size[geom], row["box_half_size_m"])
        ):
            raise RuntimeError(f"Compiled {side} sole differs from its exact union AABB")
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        vertices = candidate.geom_pos[geom] + vertices * candidate.geom_size[geom]
        error = max(
            abs(vertices.min(0) - row["union_aabb_lower_local_m"]).max(),
            abs(vertices.max(0) - row["union_aabb_upper_local_m"]).max(),
        )
        if error > 1e-15:
            raise RuntimeError("Candidate box 8-vertex extrema fail exact AABB check")
        row["compiled_box_vertex_envelope_error_m"] = float(error)
    for name in disabled:
        geom = candidate.geom(name).id
        if candidate.geom_contype[geom] or candidate.geom_conaffinity[geom]:
            raise RuntimeError("A disabled source capsule remains collision-enabled")
    for first, second in zip(candidate.pair_geom1, candidate.pair_geom2, strict=True):
        if (
            candidate.geom(int(first)).name in disabled
            or candidate.geom(int(second)).name in disabled
        ):
            raise RuntimeError("An explicit pair forces a disabled capsule into collision")
    return candidate, {
        "feet": geometry,
        "deleted_pairs": deleted,
        "remaining_explicit_pairs": int(candidate.npair),
    }


def invariant_proof(original, candidate):
    checked, differences = [], []
    # Collision-derived BVHs/bounds may change. All physical body/joint/drive
    # parameters must remain identical rather than being silently refitted.
    prefixes = ("jnt_", "dof_", "actuator_", "eq_")
    explicit = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_gravcomp",
        "body_pos",
        "body_quat",
        "body_parentid",
        "body_subtreemass",
        "qpos0",
    )
    for name in dir(original):
        value = getattr(original, name)
        if isinstance(value, np.ndarray) and (name.startswith(prefixes) or name in explicit):
            checked.append(name)
            if not np.array_equal(value, getattr(candidate, name)):
                differences.append(name)
    for name in dir(original.opt):
        if name.startswith("_"):
            continue
        value = getattr(original.opt, name)
        if isinstance(value, (np.ndarray, int, float)):
            checked.append("opt." + name)
            if not np.array_equal(value, getattr(candidate.opt, name)):
                differences.append("opt." + name)
    foot_names = {f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)}
    unchanged = [i for i in range(original.ngeom) if original.geom(i).name not in foot_names]
    for name in dir(original):
        value = getattr(original, name)
        if (
            name.startswith("geom_")
            and isinstance(value, np.ndarray)
            and value.ndim
            and value.shape[0] == original.ngeom
        ):
            checked.append(name + "[non-foot]")
            if not np.array_equal(value[unchanged], getattr(candidate, name)[unchanged]):
                differences.append(name + "[non-foot]")
    materials = (
        "geom_condim",
        "geom_friction",
        "geom_priority",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "geom_solmix",
        "geom_bodyid",
    )
    for name in materials:
        checked.append(name)
        if not np.array_equal(getattr(original, name), getattr(candidate, name)):
            differences.append(name)
    for count in ("nq", "nv", "nu", "neq", "nmocap", "na", "nbody", "ngeom"):
        if getattr(original, count) != getattr(candidate, count):
            differences.append(count)
    for pair_id in range(candidate.npair):
        name = candidate.pair(pair_id).name
        original_id = original.pair(name).id
        for field in dir(original):
            value = getattr(original, field)
            if (
                field.startswith("pair_")
                and isinstance(value, np.ndarray)
                and value.ndim
                and value.shape[0] == original.npair
            ):
                checked.append(field + "[" + name + "]")
                if not np.array_equal(value[original_id], getattr(candidate, field)[pair_id]):
                    differences.append(field + "[" + name + "]")
    if differences:
        raise RuntimeError(f"Shape-only candidate changed forbidden parameters: {differences}")
    return {
        "all_required_fields_exact": True,
        "checked": checked,
        "board_ground_and_nonfoot_geometry_unchanged": True,
        "materials_and_all_body_inertias_unchanged": True,
    }


def initialize(model, source):
    index = SceneIndex.from_model(model)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = source["qpos"][0], source["qvel"][0]
    data.ctrl[index.robot_actuators] = source["qpos"][0, index.robot_joint_qpos]
    mujoco.mj_forward(model, data)
    return data, index, ContactObserver(model, index, 0.1)


def initial_gate(model, data, index, observer):
    row = observer.observe(data)
    ground = model.geom("ground").id
    unsafe = []
    for contact in data.contact[: data.ncon]:
        first, second = map(int, contact.geom)
        robot_first = first not in observer.board and first != ground
        robot_second = second not in observer.board and second != ground
        if (robot_first and robot_second) or (
            ground in (first, second) and (robot_first or robot_second)
        ):
            unsafe.append(
                {
                    "geoms": [model.geom(first).name, model.geom(second).name],
                    "signed_distance_m": float(contact.dist),
                }
            )
    actual_q = data.qpos[index.robot_joint_qpos]
    ranges = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]]
    joints_legal = bool(np.all(actual_q >= ranges[:, 0]) and np.all(actual_q <= ranges[:, 1]))
    passed = row["maximum_penetration_m"] <= 0.002 and joints_legal and not unsafe
    return {
        "passed": passed,
        "maximum_initial_penetration_m": row["maximum_penetration_m"],
        "initial_joints_legal": joints_legal,
        "robot_self_or_ground_contacts": unsafe,
        "initial_root_or_board_adjustments": 0,
    }, row


def forces(model, data, observer, *, initial=False):
    result = {
        "time_physics": float(data.time),
        "left_nose_force_on_board_z_n": 0.0,
        "left_nose_active_contacts": 0,
        "right_tail_force_on_board_z_n": 0.0,
        "sample_stage": "initial_static_forward"
        if initial
        else "after_mj_step_before_extra_forward",
        "interval_start_s": float(data.time) if initial else float(data.time - 0.0025),
        "interval_end_s": float(data.time),
        "included_in_execution_impulse": not initial,
    }
    nose, tail = model.geom("board_nose").id, model.geom("board_tail").id
    local = np.empty(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        first, second = map(int, c.geom)
        side, box = None, None
        for label, surface in (("left", nose), ("right", tail)):
            if first in observer.feet[label] and second == surface:
                side, box, sign = label, surface, 1
            elif second in observer.feet[label] and first == surface:
                side, box, sign = label, surface, -1
        if side is None:
            continue
        mujoco.mj_contactForce(model, data, i, local)
        board_force = sign * (c.frame.reshape(3, 3).T @ local[:3])
        name = "left_nose" if box == nose else "right_tail"
        result[name + "_force_on_board_z_n"] += float(board_force[2])
        if box == nose and local[0] >= 0.1:
            result["left_nose_active_contacts"] += 1
    return result


def replay(model, source, folder, *, expected=None):
    folder.mkdir()
    data, index, observer = initialize(model, source)
    gate, initial = initial_gate(model, data, index, observer)
    save(folder / "initial_gate.json", gate)
    poses, velocities, actions, records, force_rows = (
        [data.qpos.copy()],
        [data.qvel.copy()],
        [],
        [initial],
        [forces(model, data, observer, initial=True)],
    )
    failure, partial, max_error = None, None, {"qpos": 0.0, "qvel": 0.0}
    if expected is not None and initial != expected[0]:
        raise RuntimeError("Baseline initial Observer differs from frozen 65")
    if not gate["passed"] and expected is None:
        result = {
            "status": "initial-invalid",
            "controls_executed": 0,
            "initial_gate": gate,
            "outcome": "NOT_EXECUTED",
            "verified_teacher": False,
        }
    else:
        for step, target in enumerate(source["action"]):
            advanced, committed = 0, False
            try:
                data.ctrl[index.robot_actuators] = target
                for _ in range(8):
                    previous = float(data.time)
                    mujoco.mj_step(model, data)
                    advanced += 1
                    if not np.isclose(data.time, previous + 0.0025, rtol=0, atol=1e-10):
                        raise RuntimeError("Native time discontinuity")
                    # mj_step leaves the executed interval's contact/frame/force;
                    # the extra forward below would compute a new diagnostic solve.
                    force_rows.append(forces(model, data, observer))
                    mujoco.mj_forward(model, data)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise RuntimeError("Native nonfinite state")
                    row = observer.observe(data)
                    records.append(row)
                    if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                        raise RuntimeError("Unexpected external force")
                    if expected is not None and row != expected[len(records) - 1]:
                        raise RuntimeError("Baseline 400Hz Observer differs from frozen 65")
                actions.append(target.copy())
                poses.append(data.qpos.copy())
                velocities.append(data.qvel.copy())
                committed = True
                if expected is not None:
                    for name, value in (("qpos", data.qpos), ("qvel", data.qvel)):
                        error = float(abs(value - source[name][step + 1]).max())
                        max_error[name] = max(max_error[name], error)
                        if error:
                            raise RuntimeError(f"Baseline {name} differs from frozen 65")
            except RuntimeError as error:
                failure = str(error)
                if advanced and not committed:
                    partial = {
                        "control_step": step,
                        "target": target.tolist(),
                        "native_substeps_advanced": advanced,
                    }
                break
        complete = len(actions) == 257 and failure is None and abs(data.time - 5.14) < 1e-9
        result = outcome(records, full_reference=complete, numerical_failure=failure)
        result.update(
            status="complete" if complete else "partial-failure",
            controls_executed=len(actions),
            initial_gate=gate,
            partial_control=partial,
            actual_endpoint_s=float(data.time),
            one_initialization=True,
            phase_resets=0,
            root_board_writes_after_initialization=0,
            board_actuators=0,
            verified_teacher=False,
            baseline_replay_error=max_error if expected is not None else None,
            force_sampling_stage="after mj_step, before extra mj_forward; initial static sample excluded",
            left_nose_upward_impulse_ns=float(
                sum(max(r["left_nose_force_on_board_z_n"], 0) for r in force_rows[1:]) * 0.0025
            ),
            left_nose_active_seconds=float(
                sum(r["left_nose_active_contacts"] > 0 for r in force_rows[1:]) * 0.0025
            ),
        )
    np.savez_compressed(
        folder / "actual_rollout.npz",
        qpos=np.asarray(poses),
        qvel=np.asarray(velocities),
        action=np.asarray(actions).reshape(-1, 29),
        time_physics=np.arange(len(poses)) * 0.02,
        raw_terminal_qpos=data.qpos.copy(),
        raw_terminal_qvel=data.qvel.copy(),
        raw_terminal_time_s=np.asarray(data.time),
    )
    save(folder / "contacts.json", records)
    save(folder / "force_samples_400hz.json", force_rows)
    save(folder / "result.json", result)
    if expected is not None and (failure or len(records) != len(expected)):
        raise RuntimeError("Baseline exact replay failed; candidate not executed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "run"), default="inspect")
    parser.add_argument(
        "--source", type=Path, default=Path("logs/ollie_contact_sequence_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--source-contacts", type=Path, default=Path("logs/ollie_contact_sequence_01/contacts.json")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory must be fresh")
    if sha(args.source) != SOURCE_SHA or sha(args.source_contacts) != CONTACT_SHA:
        raise ValueError("Frozen 65 inputs differ")
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("Global control callback is forbidden")
    with np.load(args.source, allow_pickle=False) as a:
        source = {name: a[name].copy() for name in ("qpos", "qvel", "action")}
        if tuple(a["robot_joint_names"]) != G1_JOINT_NAMES:
            raise ValueError("Source robot joint order differs")
    if any(
        source[name].shape != shape or not np.isfinite(source[name]).all()
        for name, shape in (("qpos", (258, 47)), ("qvel", (258, 45)), ("action", (257, 29)))
    ):
        raise ValueError("Source is not the complete 65 episode")
    expected = json.loads(args.source_contacts.read_text())
    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema": "supermimic.ollie.sole_collision_comparison.v1",
        "mode": args.mode,
        "script_sha256": sha(__file__),
        "source": str(args.source.resolve()),
        "source_sha256": SOURCE_SHA,
        "source_contacts": str(args.source_contacts.resolve()),
        "source_contacts_sha256": CONTACT_SHA,
        "canonical_board_xml": str(BOARD_XML.resolve()),
        "canonical_board_sha256": sha(BOARD_XML),
        "canonical_physics_source": str(Path(make_scene_spec.__code__.co_filename).resolve()),
        "canonical_physics_sha256": sha(make_scene_spec.__code__.co_filename),
        "mu": 0.9,
        "control_dt": 0.02,
        "physics_dt": 0.0025,
        "control_dim": 29,
        "fixed_controls": 257,
        "search_or_training": False,
    }
    save(args.out_dir / "frozen_manifest.json", manifest)
    try:
        original = make_scene_spec(timestep=0.0025, foot_deck_friction=0.9).compile()
        original_index = SceneIndex.from_model(original)
        target_range = original.jnt_range[original.dof_jntid[original_index.robot_joint_qvel]]
        if np.any(source["action"] < target_range[:, 0] + 0.005) or np.any(
            source["action"] > target_range[:, 1] - 0.005
        ):
            raise RuntimeError("Frozen controls violate the original target interval")
        candidate, geometry = candidate_model(original)
        invariant = invariant_proof(original, candidate)
        save(args.out_dir / "geometry.json", geometry)
        save(args.out_dir / "invariants.json", invariant)
        data, index, observer = initialize(candidate, source)
        gate, _ = initial_gate(candidate, data, index, observer)
        save(args.out_dir / "candidate_initial_inspection.json", gate)
        if args.mode == "run":
            baseline = replay(
                original, source, args.out_dir / "baseline_capsules", expected=expected
            )
            trial = replay(candidate, source, args.out_dir / "candidate_aabb_box")
            save(
                args.out_dir / "comparison.json",
                {
                    "baseline": baseline,
                    "candidate": trial,
                    "interpretation": "same actions only; model sensitivity diagnostic, not a trained teacher or global feasibility proof",
                },
            )
        if (
            sha(args.source) != SOURCE_SHA
            or sha(args.source_contacts) != CONTACT_SHA
            or sha(BOARD_XML) != manifest["canonical_board_sha256"]
        ):
            raise RuntimeError("An immutable input changed")
        save(
            args.out_dir / "output_hashes.json",
            {
                str(p.relative_to(args.out_dir)): sha(p)
                for p in args.out_dir.rglob("*")
                if p.is_file()
            },
        )
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "candidate_initial_gate": gate,
                    "out_dir": str(args.out_dir.resolve()),
                }
            )
        )
    except Exception as error:
        trial_result = args.out_dir / "candidate_aabb_box/result.json"
        trial_controls = (
            json.loads(trial_result.read_text())["controls_executed"]
            if trial_result.exists()
            else 0
        )
        save(
            args.out_dir / "failure.json",
            {
                "error": str(error),
                "candidate_controls_executed": trial_controls,
                "scope": "Inspect/preflight/baseline failure stops further execution",
            },
        )
        raise


if __name__ == "__main__":
    main()
