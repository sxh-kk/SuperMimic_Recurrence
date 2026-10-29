"""Fixed warm-state PD interventions spanning actual90 Ollie contact events.

The original default mu=.6 model, passive board, actuators and complete source
replay are unchanged. Fractional-control anchors begin a NEW 20 ms control
clock: their baseline is not an exact continuation of the source clock.
No world-model fit, actor update or successful-teacher promotion is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local.collect_ollie_mu09_responses import (
    NativeLabels,
    digest,
    fingerprint,
    load_archive,
    write_json,
)
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.gpu_contact_observer import METRIC_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, BOARD_XML, compile_scene
from supermimic.ollie.state import JointStateIndices

SOURCE_SHA = "ac6effe28909ca875149e6f94cdb17b3b59b0152a4960dfff4ab7e38beb35526"
SCHEMA = "supermimic.ollie.default_phase_warm_responses.v1"
DT, CONTROL_DT, HORIZON, PULSE, BUDGET = 0.0025, 0.02, 8, 0.03, 120.0
JOINTS = tuple(
    name
    for name in G1_JOINT_NAMES
    if any(part in name for part in ("_hip_", "_knee_", "_ankle_", "waist_"))
)
PHASE_FLAGS = (
    "tail_contact",
    "whole_board_air_above5mm",
    "air_rising_com",
    "air_descending_com",
    "board_ground",
    "left_top_above1n",
    "right_top_above1n",
    "both_top_above1n",
    "robot_ground",
)
MI = {name: i for i, name in enumerate(METRIC_NAMES)}


def pick_three(indices):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 3:
        raise ValueError("Frozen phase needs three distinct actual native states")
    return indices[[0, len(indices) // 2, -1]].tolist()


def anchor_plan(contacts, result):
    """Use saved actual400Hz contact/geometry events, never source-video time."""
    first_tail = next(i for i, row in enumerate(contacts) if row["tail_ground"])
    flight_start_s = result["flight_windows"][0]["start_s"]
    start = next(
        i for i, row in enumerate(contacts) if abs(row["time_physics"] - flight_start_s) < 1e-10
    )
    landed = next(
        i for i in range(start + 1, len(contacts)) if contacts[i]["unique_board_geoms_ground"] > 0
    )
    front = next(
        i
        for i in range(landed, len(contacts))
        if contacts[i]["left_foot_deck_upward_support_n"] > 1
    )
    height = np.asarray([row["board_root_height_m"] for row in contacts])
    # Geometry-derived derivative selects the frozen anchors. Actual native
    # COM velocity is independently saved and checked during real collection.
    vz = np.gradient(height, DT)
    preparation = [
        i
        for i in range(first_tail - 240, first_tail - 80)
        if contacts[i]["unique_wheels_ground"] == 4
        and contacts[i]["left_foot_deck_upward_support_n"] > 1
        and contacts[i]["right_foot_deck_upward_support_n"] > 1
        and not contacts[i]["tail_ground"]
        and contacts[i]["unique_robot_geoms_ground"] == 0
    ]
    air = [
        i
        for i in range(start, landed)
        if contacts[i]["minimum_board_clearance_m"] > 0.005
        and contacts[i]["unique_board_geoms_ground"] == 0
    ]
    phases = (
        ("preparation_supported", pick_three(preparation)),
        ("tail_impact_neighborhood", list(range(first_tail - 2, first_tail + 1))),
        ("air_rising", pick_three([i for i in air if vz[i] > 0])),
        ("air_descending", pick_three([i for i in air if vz[i] <= 0])),
        ("first_board_landing_neighborhood", list(range(landed, landed + 3))),
        ("late_front_support_neighborhood", list(range(front - 1, front + 2))),
    )
    rows = []
    for role, indices in phases:
        for local, step in enumerate(indices):
            row = contacts[step]
            rows.append(
                {
                    "native_step": step,
                    "time_native_s": row["time_physics"],
                    "role": role,
                    "role_index": local,
                    "heldout": local == 1,
                    "source_action_frame": step // 8,
                    "new_control_clock_shift_s": (step % 8) * DT,
                    "actual_source_tail_ground": row["tail_ground"],
                    "actual_source_left_top_n": row["left_foot_deck_upward_support_n"],
                    "actual_source_right_top_n": row["right_foot_deck_upward_support_n"],
                    "actual_source_board_ground": row["unique_board_geoms_ground"] > 0,
                }
            )
    if len({row["native_step"] for row in rows}) != 18:
        raise ValueError("Phase anchors must be distinct actual hot states")
    return rows


def phase_flags(metrics):
    air = (metrics[..., MI["minimum_board_clearance_m"]] > 0.005) & (
        metrics[..., MI["any_board_ground"]] == 0
    )
    left = metrics[..., MI["left_foot_deck_upward_support_n"]] > 1
    right = metrics[..., MI["right_foot_deck_upward_support_n"]] > 1
    return np.stack(
        (
            metrics[..., MI["tail_ground"]] > 0,
            air,
            air & (metrics[..., MI["board_com_vz_m_s"]] > 0),
            air & (metrics[..., MI["board_com_vz_m_s"]] <= 0),
            metrics[..., MI["any_board_ground"]] > 0,
            left,
            right,
            left & right,
            metrics[..., MI["any_robot_ground"]] > 0,
        ),
        axis=-1,
    )


def physics_digest(model):
    """Bind physical arrays; sensor storage and MjData are not part of physics."""
    fields = (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_priority",
        "geom_contype",
        "geom_conaffinity",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    )
    pieces = [np.asarray(getattr(model, name)).tobytes() for name in fields]
    pieces.append(np.asarray(model.opt.gravity).tobytes())
    pieces.append(
        np.asarray([model.opt.timestep, model.opt.iterations, model.opt.integrator]).tobytes()
    )
    return hashlib.sha256(b"".join(pieces)).hexdigest()


def make_manifest(args, source, contacts, result):
    if source["action"].shape != (257, 29) or source["state"].shape != (258, 92):
        raise ValueError("Expected complete actual90 archive")
    if bool(source["verified_teacher"]) or len(contacts) != 2057:
        raise ValueError("Expected the actual failed90 complete400Hz record")
    if result["criteria_version"] != "whole_board_geometry_top_support_v3":
        raise ValueError("Original strict criterion required")
    model, _, _ = compile_scene()
    if (model.nq, model.nv, model.nu, model.neq, model.npair, model.nmocap, model.na) != (
        47,
        45,
        29,
        0,
        0,
        0,
        0,
    ):
        raise ValueError("Original passive default model required")
    plan = anchor_plan(contacts, result)
    return {
        "schema": SCHEMA,
        "source": str(args.source.resolve()),
        "source_sha256": SOURCE_SHA,
        "source_action_sha256": hashlib.sha256(source["action"].tobytes()).hexdigest(),
        "source_contacts_sha256": digest(args.source.parent / "contacts.json"),
        "canonical_board_xml_sha256": digest(BOARD_XML),
        "physics_array_sha256": physics_digest(model),
        "collector_sha256": digest(__file__),
        "native_label_implementation_sha256": digest(
            Path(__file__).with_name("collect_ollie_mu09_responses.py")
        ),
        "backend": "mujoco_native_float64",
        "mujoco_version": mujoco.__version__,
        "foot_deck_friction": 0.6,
        "explicit_contact_pairs": 0,
        "original_model_gains_caps_gravity_and_board_unchanged": True,
        "model_material_is_unvalidated_engineering_assumption": True,
        "anchor_plan": plan,
        "branches_per_anchor": 31,
        "expected_branches": 558,
        "expected_transitions": 4464,
        "extra_same_action_repeat_branches": 54,
        "horizon_controls": HORIZON,
        "control_dt": CONTROL_DT,
        "physics_dt": DT,
        "first_control_pulse_rad": PULSE,
        "joint_names": JOINTS,
        "robot_joint_names": G1_JOINT_NAMES,
        "board_joint_names": BOARD_JOINT_NAMES,
        "control_contract": "Eight20ms intervals start at each actual native hot anchor; baseline[h]=source_action[floor(native_step/8)+h]. Original target-switch clock shifts0..17.5ms and baseline is not an exact source continuation.",
        "split_rule": "Each phase role has3 anchors: first and last train, middle heldout; all31 sibling branches stay in one split.",
        "split_limitation": "Nearby states and overlapping horizons from one failed trajectory; local interpolation only, not independent episode/skill generalization.",
        "role_is_not_contact_label": True,
        "phase_label_contract": "Actual400Hz post-forward Fn>=.1N contacts, true top>1N, all9geometry>5mm and native Jacobian COMVZ. Tail/landing/front-support neighborhood roles are not forced labels.",
        "metric_names": METRIC_NAMES,
        "phase_flag_names": PHASE_FLAGS,
        "contact_label_stage": "native mj_step then native mj_forward at every physical substep",
        "post_forward_force_is_not_execution_impulse": True,
        "torque_stage": "qfrc_actuator immediately after mj_step, before extra mj_forward",
        "warm_state": "copy.copy(MjData) from continuous original90 replay, including solver/contact/ctrl history; no cold endpoint initialization",
        "wall_budget_s": BUDGET,
        "training_updates": 0,
        "actor_updates": 0,
        "board_bptt_allowed": False,
        "verified_teacher": False,
    }


def execute_branch(model, hot, indices, labels, index, targets, deadline):
    d = copy.copy(hot)
    values = {
        "state": [indices.pack(d.qpos, d.qvel).numpy()],
        "qpos": [d.qpos.copy()],
        "qvel": [d.qvel.copy()],
        "metrics": [],
        "torque": [],
    }
    valid = np.zeros(HORIZON, dtype=bool)
    failure = None
    for h, target in enumerate(targets):
        if time.monotonic() >= deadline:
            failure = "wall_budget_before_next_control"
            break
        d.ctrl[index.robot_actuators] = target
        sub, torque = [], []
        for _ in range(8):
            before = float(d.time)
            mujoco.mj_step(model, d)
            if abs(d.time - before - DT) > 1e-10 or not (
                np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            ):
                failure = "nonfinite_or_native_automatic_reset"
                break
            if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
                raise ValueError("External forces are forbidden")
            torque.append(d.qfrc_actuator[index.robot_joint_qvel].copy())
            mujoco.mj_forward(model, d)
            sub.append(labels(d))
        if failure:
            break
        valid[h] = True
        values["state"].append(indices.pack(d.qpos, d.qvel).numpy())
        values["qpos"].append(d.qpos.copy())
        values["qvel"].append(d.qvel.copy())
        values["metrics"].append(sub)
        values["torque"].append(torque)
    return {
        **{key: np.asarray(value) for key, value in values.items()},
        "transition_valid": valid,
        "failure": failure,
    }


def response_summary(fields, shared):
    responses = []
    for group in range(18):
        for j, name in enumerate(JOINTS):
            pos, neg = group * 31 + 1 + 2 * j, group * 31 + 2 + 2 * j
            for h in (1, 8):
                if not all(shared["valid_complete"][i] for i in (pos, neg)):
                    responses.append(
                        {
                            "group": group,
                            "role": str(shared["anchor_role"][pos]),
                            "native_step": int(shared["anchor_native_step"][pos]),
                            "heldout": bool(shared["heldout_mask"][pos]),
                            "joint": name,
                            "horizon": h,
                            "valid_complete": False,
                        }
                    )
                    continue
                responses.append(
                    {
                        "group": group,
                        "role": str(shared["anchor_role"][pos]),
                        "native_step": int(shared["anchor_native_step"][pos]),
                        "heldout": bool(shared["heldout_mask"][pos]),
                        "joint": name,
                        "horizon": h,
                        "board_root_vz_pos_minus_neg_m_s": float(
                            fields["state"][pos, h, 84] - fields["state"][neg, h, 84]
                        ),
                        "board_com_vz_pos_minus_neg_m_s": float(
                            fields["post_forward_metrics"][pos, h - 1, -1, MI["board_com_vz_m_s"]]
                            - fields["post_forward_metrics"][neg, h - 1, -1, MI["board_com_vz_m_s"]]
                        ),
                    }
                )
    quantiles = {}
    for held in (False, True):
        for h in (1, 8):
            vals = np.array(
                [
                    abs(row["board_root_vz_pos_minus_neg_m_s"])
                    for row in responses
                    if row["heldout"] == held
                    and row["horizon"] == h
                    and "board_root_vz_pos_minus_neg_m_s" in row
                ]
            )
            finite = vals[np.isfinite(vals)]
            quantiles[f"{'heldout' if held else 'train'}_H{h}"] = {
                "pairs": len(vals),
                "finite_pairs": len(finite),
                "median_p90_max_m_s": np.quantile(finite, [0.5, 0.9, 1]).tolist()
                if len(finite)
                else None,
            }
    return responses, quantiles


def run(args, source, contacts, manifest):
    start = time.monotonic()
    deadline = start + BUDGET
    if mujoco.get_mjcb_control() is not None:
        raise ValueError("No control callback allowed")
    model, data, index = compile_scene()
    if physics_digest(model) != manifest["physics_array_sha256"]:
        raise ValueError("Physical model changed since inspect")
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    labels = NativeLabels(model, index)
    # Only initialization is the unique original q0. All later snapshots are
    # copied from this continuously executed scene, never restored cold.
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = source["qpos"][0], source["qvel"][0]
    data.ctrl[index.robot_actuators] = source["state"][0, 7:36]
    mujoco.mj_forward(model, data)
    snapshots, anchor_metrics = {}, {}
    required = {row["native_step"] for row in manifest["anchor_plan"]}
    errors = {"qpos": 0.0, "qvel": 0.0, "state": 0.0}
    for step in range(2057):
        observed = labels.observer.observe(data)
        if observed != contacts[step]:
            raise ValueError(f"Original actual90 Observer mismatch at native step{step}")
        if step % 8 == 0:
            frame = step // 8
            current = {
                "qpos": data.qpos,
                "qvel": data.qvel,
                "state": indices.pack(data.qpos, data.qvel).numpy(),
            }
            for key, value in current.items():
                errors[key] = max(errors[key], float(abs(value - source[key][frame]).max()))
        if step in required:
            snapshots[step] = copy.copy(data)
            anchor_metrics[step] = labels(data)
        if step == 2056:
            break
        if step % 8 == 0:
            data.ctrl[index.robot_actuators] = source["action"][step // 8]
        before = float(data.time)
        mujoco.mj_step(model, data)
        if (
            abs(data.time - before - DT) > 1e-10
            or np.any(data.qfrc_applied)
            or np.any(data.xfrc_applied)
        ):
            raise ValueError("Source replay reset or external force")
        mujoco.mj_forward(model, data)
    if max(errors.values()) != 0 or len(snapshots) != 18:
        raise ValueError("Original source replay or hot snapshot mismatch")
    write_json(
        args.out_dir / "baseline_replay.json",
        {
            "maximum_errors": errors,
            "canonical_observer_records_exact": 2057,
            "unique_initializations": 1,
            "intermediate_resets": 0,
            "source_full_controls": 257,
            "physical_arrays_sha256_before": physics_digest(model),
        },
    )
    limits = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]] + np.array([0.005, -0.005])
    episodes, qa, anchors = [], [], []
    warm_states, warm_acc, warm_ctrl = [], [], []
    for group, row in enumerate(manifest["anchor_plan"]):
        step = row["native_step"]
        hot = snapshots[step]
        hot_hash = fingerprint(model, hot)
        source_frame = step // 8
        base = source["action"][source_frame : source_frame + HORIZON].copy()
        baseline_id = len(episodes)
        descriptors = [(-1, 0)] + [
            (G1_JOINT_NAMES.index(name), sign) for name in JOINTS for sign in (1, -1)
        ]
        for joint, sign in descriptors:
            targets, requested = base.copy(), np.zeros(29)
            if joint >= 0:
                requested[joint] = sign * PULSE
                targets[0, joint] = np.clip(
                    targets[0, joint] + requested[joint], limits[joint, 0], limits[joint, 1]
                )
            episode = execute_branch(model, hot, indices, labels, index, targets, deadline)
            episode.update(
                action=targets,
                actual_delta=targets - base,
                requested_delta=requested,
                group=group,
                anchor=step,
                role=row["role"],
                heldout=row["heldout"],
                joint=joint,
                sign=sign,
                baseline_id=baseline_id,
            )
            episodes.append(episode)
            if joint == -1 or joint == descriptors[1][0]:
                repeat = execute_branch(model, hot, indices, labels, index, targets, deadline)
                complete = episode["failure"] is None and repeat["failure"] is None
                differences = (
                    {
                        key: float(abs(episode[key] - repeat[key]).max())
                        for key in ("state", "qpos", "qvel", "metrics", "torque")
                    }
                    if complete
                    else None
                )
                qa.append(
                    {
                        "anchor_native_step": step,
                        "joint": joint,
                        "sign": sign,
                        "complete": complete,
                        "maximum_repeat_errors": differences,
                    }
                )
        unchanged = fingerprint(model, hot) == hot_hash
        if not unchanged:
            raise ValueError("Branch altered its source hot state")
        anchors.append(
            {
                **row,
                "warm_snapshot_fingerprint": hot_hash,
                "source_snapshot_unchanged": unchanged,
                "actual_anchor_metrics": dict(
                    zip(METRIC_NAMES, anchor_metrics[step].tolist(), strict=True)
                ),
                "actual_anchor_phase_flags": dict(
                    zip(PHASE_FLAGS, phase_flags(anchor_metrics[step]).tolist(), strict=True)
                ),
            }
        )
        state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, hot, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        warm_states.append(state)
        warm_acc.append(hot.qacc_warmstart.copy())
        warm_ctrl.append(hot.ctrl.copy())
    count = len(episodes)
    if count != 558 or len(qa) != 54:
        raise ValueError("Fixed branch count changed")
    fields = {
        "state": np.full((count, 9, 92), np.nan),
        "qpos": np.full((count, 9, 47), np.nan),
        "qvel": np.full((count, 9, 45), np.nan),
        "post_forward_metrics": np.full((count, 8, 8, 49), np.nan),
        "execution_actuator_torque_nm": np.full((count, 8, 8, 29), np.nan),
    }
    for i, e in enumerate(episodes):
        for key in ("state", "qpos", "qvel"):
            fields[key][i, : len(e[key])] = e[key]
        fields["post_forward_metrics"][i, : len(e["metrics"])] = e["metrics"]
        fields["execution_actuator_torque_nm"][i, : len(e["torque"])] = e["torque"]
    shared = {
        "branch_id": np.arange(count),
        "group_id": np.array([e["group"] for e in episodes]),
        "anchor_native_step": np.array([e["anchor"] for e in episodes]),
        "anchor_source_action_frame": np.array([e["anchor"] // 8 for e in episodes]),
        "anchor_role": np.array([e["role"] for e in episodes]),
        "joint_index": np.array([e["joint"] for e in episodes]),
        "sign": np.array([e["sign"] for e in episodes]),
        "paired_baseline_id": np.array([e["baseline_id"] for e in episodes]),
        "is_baseline": np.array([e["sign"] == 0 for e in episodes]),
        "heldout_mask": np.array([e["heldout"] for e in episodes]),
        "train_mask": np.array([not e["heldout"] for e in episodes]),
        "action": np.array([e["action"] for e in episodes]),
        "requested_delta": np.array([e["requested_delta"] for e in episodes]),
        "actual_delta": np.array([e["actual_delta"] for e in episodes]),
        "transition_valid": np.array([e["transition_valid"] for e in episodes]),
        "valid_complete": np.array([e["failure"] is None for e in episodes]),
        "metric_names": np.array(METRIC_NAMES),
        "phase_flag_names": np.array(PHASE_FLAGS),
        "phase_flags": phase_flags(fields["post_forward_metrics"]),
        "anchor_post_forward_metrics": np.array([anchor_metrics[e["anchor"]] for e in episodes]),
        "metadata_json": np.asarray(json.dumps(manifest)),
    }
    np.savez_compressed(args.out_dir / "paired_responses.npz", **fields, **shared)
    transitions = {
        "state": fields["state"][:, :-1].reshape(-1, 92),
        "next_state": fields["state"][:, 1:].reshape(-1, 92),
        "action": shared["action"].reshape(-1, 29),
        "contact_metrics": fields["post_forward_metrics"].reshape(-1, 8, 49),
        "execution_actuator_torque_nm": fields["execution_actuator_torque_nm"].reshape(-1, 8, 29),
        "phase_flags": shared["phase_flags"].reshape(-1, 8, len(PHASE_FLAGS)),
        "step": np.tile(np.arange(8), count),
        "valid": shared["transition_valid"].reshape(-1),
        "actual_delta": shared["actual_delta"].reshape(-1, 29),
        "metadata_json": shared["metadata_json"],
        "metric_names": shared["metric_names"],
        "phase_flag_names": shared["phase_flag_names"],
    }
    for key in (
        "branch_id",
        "group_id",
        "anchor_native_step",
        "anchor_role",
        "joint_index",
        "sign",
        "paired_baseline_id",
        "train_mask",
        "heldout_mask",
    ):
        transitions[key] = np.repeat(shared[key], 8)
    np.savez_compressed(args.out_dir / "transitions.npz", **transitions)
    np.savez_compressed(
        args.out_dir / "warm_snapshot_states.npz",
        integration_state=np.array(warm_states),
        qacc_warmstart=np.array(warm_acc),
        ctrl=np.array(warm_ctrl),
        native_step=np.array([row["native_step"] for row in manifest["anchor_plan"]]),
        note=np.array(
            "Collection used full copy.copy(MjData), not reconstruction from this diagnostic subset."
        ),
    )
    responses, quantiles = response_summary(fields, shared)
    write_json(args.out_dir / "paired_board_vz_responses.json", responses)
    coverage = {}
    flags = shared["phase_flags"]
    for split, mask in (("train", shared["train_mask"]), ("heldout", shared["heldout_mask"])):
        coverage[split] = {
            name: {
                "physical_samples": int(flags[mask, ..., i].sum()),
                "branches_any": int(flags[mask, ..., i].any(axis=(1, 2)).sum()),
            }
            for i, name in enumerate(PHASE_FLAGS)
        }
    maximum_repeat = {
        key: max((row["maximum_repeat_errors"][key] for row in qa if row["complete"]), default=None)
        for key in ("state", "qpos", "qvel", "metrics", "torque")
    }
    metrics = fields["post_forward_metrics"]
    quality = {
        "maximum_penetration_m": np.nanmax(metrics[..., MI["maximum_penetration_m"]], axis=(1, 2)),
        "maximum_actual_joint_limit_violation_rad": np.nanmax(
            metrics[..., MI["maximum_actual_joint_limit_violation_rad"]], axis=(1, 2)
        ),
        "robot_ground_any": flags[..., PHASE_FLAGS.index("robot_ground")].any(axis=(1, 2)),
        "true_both_top_any": flags[..., PHASE_FLAGS.index("both_top_above1n")].any(axis=(1, 2)),
    }
    quality["penetration_and_joint_quality"] = (quality["maximum_penetration_m"] <= 0.01) & (
        quality["maximum_actual_joint_limit_violation_rad"] <= 0.05
    )
    np.savez_compressed(args.out_dir / "quality.npz", **quality)
    baselines = shared["paired_baseline_id"]
    schema_audit = {
        "paired_initial_state_max_error": float(
            abs(fields["state"][:, 0] - fields["state"][baselines, 0]).max()
        ),
        "actual_delta_max_error": float(
            abs(shared["actual_delta"] - (shared["action"] - shared["action"][baselines])).max()
        ),
        "postpulse_delta_max_abs": float(abs(shared["actual_delta"][:, 1:]).max()),
        "target_limit_violation_max_rad": float(
            np.maximum(
                np.maximum(limits[:, 0] - shared["action"], shared["action"] - limits[:, 1]), 0
            ).max()
        ),
        "external_generalized_force_max_norm": float(
            np.nanmax(metrics[..., MI["external_generalized_force_norm"]])
        ),
        "external_body_force_max_norm": float(
            np.nanmax(metrics[..., MI["external_body_force_norm"]])
        ),
        "execution_torque_limit_violation_max_nm": float(
            np.nanmax(
                np.maximum(
                    abs(fields["execution_actuator_torque_nm"])
                    - np.max(abs(model.actuator_forcerange[index.robot_actuators]), axis=1),
                    0,
                )
            )
        ),
        "branch_split_disjoint": bool(np.all(shared["train_mask"] != shared["heldout_mask"])),
        "every_role_has_two_train_one_heldout_anchors": all(
            sum(row["role"] == role and not row["heldout"] for row in manifest["anchor_plan"]) == 2
            and sum(row["role"] == role and row["heldout"] for row in manifest["anchor_plan"]) == 1
            for role in np.unique(shared["anchor_role"])
        ),
    }
    write_json(args.out_dir / "schema_quality_audit.json", schema_audit)
    result = {
        "branches": count,
        "finite_complete_branches": int(shared["valid_complete"].sum()),
        "valid_transitions": int(shared["transition_valid"].sum()),
        "train_branches": int(shared["train_mask"].sum()),
        "heldout_branches": int(shared["heldout_mask"].sum()),
        "noise_repeat_branches": len(qa),
        "maximum_same_action_repeat_errors": maximum_repeat,
        "source_replay_maximum_errors": errors,
        "coverage": coverage,
        "paired_board_root_vz": quantiles,
        "clipped_pulse_branches": sum(
            not np.allclose(e["actual_delta"][0], e["requested_delta"], rtol=0, atol=1e-12)
            for e in episodes
        ),
        "maximum_penetration_m": float(np.nanmax(metrics[..., MI["maximum_penetration_m"]])),
        "maximum_actual_joint_limit_violation_rad": float(
            np.nanmax(metrics[..., MI["maximum_actual_joint_limit_violation_rad"]])
        ),
        "quality_failures_retained": True,
        "branch_failures": [e["failure"] for e in episodes],
        "physical_arrays_unchanged": physics_digest(model) == manifest["physics_array_sha256"],
        "quality_gate_failed_branches_retained": int(
            (~quality["penetration_and_joint_quality"]).sum()
        ),
        "schema_quality_audit": schema_audit,
        "wall_seconds": time.monotonic() - start,
        "wall_budget_s": BUDGET,
        "world_model_fit_updates": 0,
        "actor_updates": 0,
        "verified_teacher": False,
        "board_bptt_allowed": False,
    }
    write_json(args.out_dir / "anchor_snapshots.json", anchors)
    write_json(args.out_dir / "repeat_determinism.json", qa)
    write_json(args.out_dir / "result.json", result)
    output_names = (
        "paired_responses.npz",
        "transitions.npz",
        "warm_snapshot_states.npz",
        "quality.npz",
    )
    write_json(
        args.out_dir / "dataset_manifest.json",
        {
            **manifest,
            "files": {name: {"sha256": digest(args.out_dir / name)} for name in output_names},
        },
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "branch_failures"}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "run"), required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("logs/ollie_air_catch_landing_01/actual_rollout.npz")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.source) != SOURCE_SHA:
        parser.error("Requires frozen actual90 source archive")
    source = load_archive(args.source)
    contacts = json.loads((args.source.parent / "contacts.json").read_text())
    result = json.loads((args.source.parent / "result.json").read_text())
    manifest = json.loads(json.dumps(make_manifest(args, source, contacts, result)))
    path = args.out_dir / "manifest.json"
    if args.mode == "inspect":
        args.out_dir.mkdir(parents=True, exist_ok=False)
        write_json(path, manifest)
        print(json.dumps(manifest, indent=2))
    else:
        if not path.exists() or json.loads(path.read_text()) != manifest:
            parser.error("Run requires unchanged manifest from inspect")
        if any(args.out_dir.glob("*.npz")) or (args.out_dir / "baseline_replay.json").exists():
            parser.error("Never overwrite or rerun an existing fixed collection")
        run(args, source, contacts, manifest)


if __name__ == "__main__":
    main()
