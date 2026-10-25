"""Fixed native warm-state action interventions on the explicit mu=.9 model.

No training, counterfactual force, robot reset inside a branch, or teacher
certification is performed. Contact labels are post-forward observations,
not the actuator/contact forces used by the preceding integration.
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

from scripts.local.check_ollie_tail_release import geom_bounds
from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.gpu_contact_observer import METRIC_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, BOARD_XML, compile_scene
from supermimic.ollie.state import JointStateIndices

SOURCE_SHA = "8b602ad7f88266ba831c71df6c0f25bc84366a3af13427166e559f45d4470cb7"
ANCHORS = (120, 130, 140, 146, 149, 160, 240)
HELDOUT = (146, 160)
HORIZON = 8
PULSE = .06
DT = .0025
JOINTS = tuple(name for name in G1_JOINT_NAMES if any(p in name for p in ("_hip_", "_knee_", "_ankle_", "waist_")))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def fingerprint(model, data):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    pieces = [state, data.qacc, data.qacc_warmstart, data.ctrl, data.efc_force[:data.nefc]]
    for c in data.contact[:data.ncon]:
        pieces.extend((np.asarray(c.geom, dtype=float), c.pos, c.frame, np.asarray([c.dist, c.efc_address])))
    return hashlib.sha256(b"".join(np.asarray(v).tobytes() for v in pieces)).hexdigest()


class NativeLabels:
    def __init__(self, model, index):
        self.model, self.index = model, index
        self.observer = ContactObserver(model, index, .1)
        self.feet = {gid: side for side, ids in self.observer.feet.items() for gid in ids}
        self.board_names = tuple(model.geom(g).name for g in sorted(self.observer.board))

    def __call__(self, data):
        model, observer = self.model, self.observer
        values = observer.observe(data)
        values.update(
            any_board_ground=values["unique_board_geoms_ground"] > 0,
            any_robot_ground=values["unique_robot_geoms_ground"] > 0,
            minimum_wheel_geometry_clearance_m=values["minimum_wheel_clearance_m"],
            contact_count=data.ncon, overflow=0,
            left_board_fz_n=0., right_board_fz_n=0., front_nose_board_fz_n=0.,
            tail_ground_normal_force_n=0.,
        )
        for cid, contact in enumerate(data.contact[:data.ncon]):
            first, second = map(int, contact.geom)
            local = np.empty(6)
            mujoco.mj_contactForce(model, data, cid, local)
            global_force = contact.frame.reshape(3, 3).T @ local[:3]
            foot = first if first in self.feet else second if second in self.feet else None
            other = second if foot == first else first if foot is not None else None
            if foot is not None and other in observer.board:
                on_foot = global_force * (-1 if foot == first else 1)
                values[self.feet[foot] + "_board_fz_n"] += float(on_foot[2])
                if self.feet[foot] == "left" and model.geom(other).name == "board_nose":
                    values["front_nose_board_fz_n"] -= float(on_foot[2])
            if {first, second} == {observer.ground, model.geom("board_tail").id}:
                values["tail_ground_normal_force_n"] += max(float(local[0]), 0.)
        masses, velocities = [], []
        for name, body in (("robot", observer.robot_body), ("board", self.index.board_body)):
            jac = np.empty((3, model.nv))
            mujoco.mj_jacSubtreeCom(model, data, jac, body)
            values[name + "_com_vz_m_s"] = float((jac @ data.qvel)[2])
            masses.append(model.body_subtreemass[body])
            velocities.append(values[name + "_com_vz_m_s"])
        values["system_com_vz_m_s"] = float(np.dot(masses, velocities) / sum(masses))
        for name in self.board_names:
            values[name.removesuffix("_collision") + "_clearance_m"] = float(geom_bounds(model, data, model.geom(name).id)[0][2])
        return np.array([values[name] for name in METRIC_NAMES], dtype=np.float64)


def make_manifest(args, source):
    assert len(JOINTS) == 15 and len(METRIC_NAMES) == 49
    return {
        "schema": "supermimic.ollie.mu09_warm_responses.v1",
        "source": str(args.source.resolve()), "source_sha256": SOURCE_SHA,
        "source_action_sha256": hashlib.sha256(source["action"].tobytes()).hexdigest(),
        "action_sha256": hashlib.sha256(source["action"].tobytes()).hexdigest(),
        "action_sha256_scope": "frozen source257 targets, not the intervened branch tensor",
        "canonical_board_xml_sha256": digest(BOARD_XML),
        "board_xml_sha256": digest(BOARD_XML),
        "collector_source_sha256": digest(__file__),
        "backend": "mujoco_native_float64", "mujoco_version": mujoco.__version__,
        "foot_deck_friction": .9, "explicit_contact_pairs": 42,
        "model_material_is_unvalidated_engineering_assumption": True,
        "anchors": ANCHORS, "heldout_anchors": HELDOUT, "joint_names": JOINTS,
        "robot_joint_names": G1_JOINT_NAMES, "board_joint_names": BOARD_JOINT_NAMES,
        "branches_per_anchor": 31, "expected_branches": 217, "expected_transitions": 1736,
        "horizon_controls": HORIZON, "control_dt_s": .02, "physics_dt_s": DT,
        "control_dt": .02, "physics_dt": DT,
        "pulse_first_control_only_rad": PULSE, "force_threshold_n": .1,
        "metric_names": METRIC_NAMES,
        "contact_label_stage": "native mj_step then native mj_forward, observe all 8 physical substeps",
        "post_forward_force_is_not_execution_impulse": True,
        "force_direction": "left/right_board_fz is board-on-foot; front_nose_board_fz is left-foot-on-nose",
        "native_overflow_column": "zero: native dynamically allocated contacts, no Warp capacity flag",
        "warm_state": "mj_copyData via copy.copy(MjData), including solver warmstart/contact/ctrl history from unique continuous source prefix",
        "split_rule": "all branches of anchor146 and160 heldout, all other anchors train",
        "noise_qa": "repeat baseline and first-joint positive/negative once per anchor, excluded from main dataset",
        "wall_budget_s": 120, "training_updates": 0, "verified_teacher": False,
    }


def run(args, source, manifest):
    start = time.monotonic()
    assert mujoco.get_mjcb_control() is None
    model, data, index = compile_scene(foot_deck_friction=.9)
    assert model.npair == 42 and model.nu == 29 and model.neq == model.nmocap == model.na == 0
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    labels = NativeLabels(model, index)
    expected = json.loads((args.source.parent / "contacts.json").read_text())
    assert source["action"].shape == (257, 29) and len(expected) == 2057
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = source["qpos"][0], source["qvel"][0]
    data.ctrl[index.robot_actuators] = source["state"][0, 7:36]
    mujoco.mj_forward(model, data)
    snapshots, records, errors = {}, [], dict.fromkeys(("qpos", "qvel", "state"), 0.)
    def observe_baseline():
        row = labels.observer.observe(data)
        assert all(expected[len(records)][key] == value for key, value in row.items())
        records.append(row)
    observe_baseline()
    for frame in range(258):
        for key, value in (("qpos", data.qpos), ("qvel", data.qvel), ("state", indices.pack(data.qpos, data.qvel).numpy())):
            errors[key] = max(errors[key], float(abs(value - source[key][frame]).max()))
        if frame in ANCHORS:
            snapshots[frame] = copy.copy(data)
        if frame == 257:
            break
        data.ctrl[index.robot_actuators] = source["action"][frame]
        for _ in range(8):
            before = data.time
            mujoco.mj_step(model, data)
            assert abs(data.time - before - DT) < 1e-10
            mujoco.mj_forward(model, data)
            observe_baseline()
    assert max(errors.values()) == 0 and len(records) == 2057
    write_json(args.out_dir / "baseline_replay.json", {"maximum_errors": errors, "observer_records": len(records),
               "all_original_observer_fields_exact": True, "unique_initializations": 1, "intermediate_resets": 0})
    limits = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]] + np.array([.005, -.005])
    qa, episodes, anchor_records = [], [], []

    def execute(anchor, targets):
        assert time.monotonic() - start < 120
        d = copy.copy(snapshots[anchor])
        states = [indices.pack(d.qpos, d.qvel).numpy()]
        qpos, qvel, contact = [d.qpos.copy()], [d.qvel.copy()], []
        completed = np.zeros(HORIZON, dtype=bool)
        failure = None
        for step, target in enumerate(targets):
            d.ctrl[index.robot_actuators] = target
            substeps = []
            for _ in range(8):
                before = float(d.time)
                mujoco.mj_step(model, d)
                if abs(d.time - before - DT) > 1e-10 or not(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                    failure = "nonfinite or automatic reset/time discontinuity"
                    break
                assert not (np.any(d.qfrc_applied) or np.any(d.xfrc_applied))
                mujoco.mj_forward(model, d)
                substeps.append(labels(d))
            if failure:
                break
            completed[step] = True
            states.append(indices.pack(d.qpos, d.qvel).numpy())
            qpos.append(d.qpos.copy()); qvel.append(d.qvel.copy()); contact.append(substeps)
        return {"state": np.asarray(states), "qpos": np.asarray(qpos), "qvel": np.asarray(qvel),
                "metrics": np.asarray(contact), "completed": completed, "failure": failure,
                "raw_terminal_time_s": float(d.time)}

    for group, anchor in enumerate(ANCHORS):
        warm_hash = fingerprint(model, snapshots[anchor])
        base_targets = source["action"][anchor:anchor + HORIZON].copy()
        baseline_id = len(episodes)
        descriptors = [(-1, 0)] + [(G1_JOINT_NAMES.index(name), sign) for name in JOINTS for sign in (1, -1)]
        for joint, sign in descriptors:
            targets = base_targets.copy()
            requested = np.zeros(29)
            if joint >= 0:
                requested[joint] = sign * PULSE
                targets[0] = np.clip(targets[0] + requested, limits[:, 0], limits[:, 1])
            trial = execute(anchor, targets)
            if joint == -1:
                assert trial["failure"] is None
                for key in ("state", "qpos", "qvel"):
                    assert np.array_equal(trial[key], source[key][anchor:anchor + HORIZON + 1])
            trial.update(action=targets, actual_delta=targets - base_targets, requested_delta=requested,
                         anchor=anchor, group=group, joint=joint, sign=sign, baseline_id=baseline_id)
            episodes.append(trial)
            if joint == -1 or joint == descriptors[1][0]:
                repeat = execute(anchor, targets)
                differences = {key: float(abs(trial[key] - repeat[key]).max()) for key in ("state", "qpos", "qvel", "metrics")}
                qa.append({"anchor": anchor, "joint": joint, "sign": sign, "maximum_absolute_repeat_errors": differences})
        assert fingerprint(model, snapshots[anchor]) == warm_hash
        anchor_records.append({"anchor_frame": anchor, "time_s": float(snapshots[anchor].time),
                               "warm_snapshot_fingerprint": warm_hash, "source_snapshot_unchanged": True})
    assert len(episodes) == 217
    finite_complete = [e["failure"] is None and e["completed"].all() for e in episodes]
    # Native failures are preserved explicitly. Do not pretend padding is data.
    fields = {"state": np.full((217, 9, 92), np.nan), "qpos": np.full((217, 9, 47), np.nan),
              "qvel": np.full((217, 9, 45), np.nan), "post_forward_metrics": np.full((217, 8, 8, 49), np.nan)}
    for i, e in enumerate(episodes):
        for key in ("state", "qpos", "qvel"):
            fields[key][i, :len(e[key])] = e[key]
        fields["post_forward_metrics"][i, :len(e["metrics"])] = e["metrics"]
    anchors = np.array([e["anchor"] for e in episodes])
    shared = {
        "anchor_frame": anchors, "group_id": np.array([e["group"] for e in episodes]),
        "branch_id": np.arange(217), "joint_index": np.array([e["joint"] for e in episodes]),
        "joint_name": np.array([G1_JOINT_NAMES[e["joint"]] if e["joint"] >= 0 else "baseline" for e in episodes]),
        "sign": np.array([e["sign"] for e in episodes]),
        "paired_baseline_id": np.array([e["baseline_id"] for e in episodes]),
        "is_baseline": np.array([e["sign"] == 0 for e in episodes]),
        "train_mask": ~np.isin(anchors, HELDOUT), "heldout_mask": np.isin(anchors, HELDOUT),
        "valid_complete": np.asarray(finite_complete),
        "actual_delta": np.asarray([e["actual_delta"] for e in episodes]),
        "requested_delta": np.asarray([e["requested_delta"] for e in episodes]),
        "action": np.asarray([e["action"] for e in episodes]),
        "transition_valid": np.asarray([e["completed"] for e in episodes]),
        "metric_names": np.asarray(METRIC_NAMES),
        "metadata_json": np.asarray(json.dumps(manifest)),
    }
    np.savez_compressed(args.out_dir / "paired_responses.npz", **fields, **shared)
    transitions = {"state": fields["state"][:, :-1].reshape(-1, 92), "next_state": fields["state"][:, 1:].reshape(-1, 92),
                   "action": shared["action"].reshape(-1, 29), "contact_metrics": fields["post_forward_metrics"].reshape(-1, 8, 49),
                   "step": np.tile(np.arange(8), 217), "valid": shared["transition_valid"].reshape(-1),
                   "actual_delta": shared["actual_delta"].reshape(-1, 29), "metadata_json": shared["metadata_json"],
                   "metric_names": shared["metric_names"]}
    for key in ("anchor_frame", "group_id", "branch_id", "joint_index", "sign", "paired_baseline_id", "train_mask", "heldout_mask"):
        transitions[key] = np.repeat(shared[key], 8)
    np.savez_compressed(args.out_dir / "transitions.npz", **transitions)
    mi = {name: i for i, name in enumerate(METRIC_NAMES)}
    metrics = fields["post_forward_metrics"]
    quality = {
        "maximum_penetration_m": np.nanmax(metrics[..., mi["maximum_penetration_m"]], axis=(1, 2)),
        "maximum_actual_joint_limit_violation_rad": np.nanmax(metrics[..., mi["maximum_actual_joint_limit_violation_rad"]], axis=(1, 2)),
        "robot_ground_any": np.any(metrics[..., mi["any_robot_ground"]] > 0, axis=(1, 2)),
        "tail_ground_any": np.any(metrics[..., mi["tail_ground"]] > 0, axis=(1, 2)),
        "front_nose_net_lifting_any": np.any(metrics[..., mi["front_nose_board_fz_n"]] > .1, axis=(1, 2)),
        "whole_board_clearance_above5mm_any": np.any(metrics[..., mi["minimum_board_clearance_m"]] > .005, axis=(1, 2)),
    }
    quality["existing_penetration_and_joint_quality"] = (quality["maximum_penetration_m"] <= .01) & (quality["maximum_actual_joint_limit_violation_rad"] <= .05)
    np.savez_compressed(args.out_dir / "quality.npz", **quality)
    paired = []
    for group, anchor in enumerate(ANCHORS):
        for j, name in enumerate(JOINTS):
            base, pos, neg = group * 31, group * 31 + 1 + 2 * j, group * 31 + 2 + 2 * j
            dim = G1_JOINT_NAMES.index(name)
            denominator = shared["actual_delta"][pos, 0, dim] - shared["actual_delta"][neg, 0, dim]
            for h in (1, 8):
                for label, get in (("board_root_z_m", lambda e, h=h: fields["state"][e, h, 73]),
                                   ("board_root_vz_m_s", lambda e, h=h: fields["state"][e, h, 84]),
                                   ("board_com_vz_m_s", lambda e, h=h: metrics[e, h - 1, -1, mi["board_com_vz_m_s"]])):
                    if not all(finite_complete[e] for e in (base, pos, neg)):
                        paired.append({"anchor": anchor, "joint": name, "horizon": h,
                                       "observable": label, "valid_complete": False})
                        continue
                    plus, minus = float(get(pos) - get(base)), float(get(neg) - get(base))
                    paired.append({"anchor": anchor, "joint": name, "horizon": h, "observable": label,
                                   "positive_response": plus, "negative_response": minus,
                                   "positive_sign": int(np.sign(plus)), "negative_sign": int(np.sign(minus)),
                                   "actual_positive_minus_negative_action_rad": float(denominator),
                                   "secant_gain_per_rad": float((plus - minus) / denominator) if denominator > 1e-12 else None})
    summary = {
        "branches": 217, "transitions": int(shared["transition_valid"].sum()), "expected_transitions": 1736,
        "finite_complete_branches": int(sum(finite_complete)), "train_branches": int(shared["train_mask"].sum()),
        "heldout_branches": int(shared["heldout_mask"].sum()),
        "train_transitions": int((shared["transition_valid"] & shared["train_mask"][:, None]).sum()),
        "heldout_transitions": int((shared["transition_valid"] & shared["heldout_mask"][:, None]).sum()),
        "clipped_pulse_branches": sum(not np.allclose(e["actual_delta"][0], e["requested_delta"], rtol=0, atol=1e-12) for e in episodes),
        "contact_coverage_branch_counts": {key: int(value.sum()) for key, value in quality.items() if value.dtype == bool},
        "same_action_repeat_noise_floor": {key: max(v["maximum_absolute_repeat_errors"][key] for v in qa) for key in ("state", "qpos", "qvel", "metrics")},
        "quality_failures_retained": True, "branch_failures": [e["failure"] for e in episodes],
        "wall_seconds": time.monotonic() - start, "verified_teacher": False,
        "limits": ["Separate warm-state intervention branches, not complete independent Ollie episodes.",
                   "Post-forward force samples are not exact execution impulses.",
                   "Coverage of contact events must be checked by heldout group; no pop response reliability follows from standing response alone.",
                   "No counterfactual model, action resampling, training, or threshold tuning was used."],
    }
    assert summary["wall_seconds"] <= 120
    write_json(args.out_dir / "anchor_snapshots.json", anchor_records)
    write_json(args.out_dir / "repeat_determinism.json", qa)
    write_json(args.out_dir / "paired_gain_sign.json", paired)
    write_json(args.out_dir / "result.json", summary)
    write_json(args.out_dir / "dataset_manifest.json", {**manifest, "files": {
        name: {"sha256": digest(args.out_dir / name)} for name in ("paired_responses.npz", "transitions.npz", "quality.npz")}})
    write_json(args.out_dir / "output_hashes.json", {f.name: digest(f) for f in args.out_dir.iterdir() if f.is_file()})
    print(json.dumps(summary, indent=2, default=json_default))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "run"), required=True)
    parser.add_argument("--source", type=Path, default=Path("logs/ollie_contact_sequence_01/actual_rollout.npz"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.source) != SOURCE_SHA:
        parser.error("source SHA does not match the frozen 65 actual trajectory")
    source = load_archive(args.source)
    manifest = make_manifest(args, source)
    # JSON round-trip makes tuple/list representations identical for binding.
    manifest = json.loads(json.dumps(manifest))
    if args.mode == "inspect":
        args.out_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.out_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
    else:
        if not (args.out_dir / "manifest.json").exists() or json.loads((args.out_dir / "manifest.json").read_text()) != manifest:
            parser.error("run needs the unchanged manifest from an earlier inspect")
        if (args.out_dir / "result.json").exists():
            parser.error("do not rerun or overwrite the fixed response batch")
        run(args, source, manifest)


if __name__ == "__main__":
    main()
