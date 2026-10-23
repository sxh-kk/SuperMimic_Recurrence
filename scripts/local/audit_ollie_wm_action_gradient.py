"""Compare learned action Jacobians with warm native-MuJoCo local branches.

Only copied checkpoints and new audit outputs are written. Native branch data
are complete mjData copies after uninterrupted replay, including solver history.
Contact finite differences are conditional secants, not certified derivatives.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.learning import JointLearningConfig, JointResidualPolicy, JointWorldModel
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices

OUTPUT_NAMES = ("robot_x", "robot_y", "robot_z", "robot_world_vz",
                "board_x", "board_y", "board_z", "board_world_vz")
OUTPUT_INDICES = np.array([0, 1, 2, 38, 71, 72, 73, 84])
CONTROL_DT = 0.02
PHYSICS_DT = 0.0025


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def history_fingerprint(data):
    fields = ("qpos", "qvel", "ctrl", "qacc_warmstart", "efc_force", "qfrc_applied", "xfrc_applied")
    return {name: sha256(np.asarray(getattr(data, name)).tobytes()) for name in fields}


def freeze_checkpoint(source: Path, destination: Path, *, expected_schema="supermimic.ollie.actor.v1"):
    # The writer atomically renames its checkpoint. Reading one open inode gets
    # a complete old or new version, even if training is still progressing.
    blob = source.read_bytes()
    destination.write_bytes(blob)
    checkpoint = torch.load(BytesIO(blob), map_location="cpu", weights_only=True)
    if checkpoint["schema"] != expected_schema:
        raise ValueError("unsupported checkpoint schema")
    return checkpoint, {"source": str(source.resolve()), "copy": str(destination.resolve()),
                        "schema": checkpoint["schema"], "updates": checkpoint.get("updates"),
                        "sha256": sha256(blob), "iteration": checkpoint.get("iteration"),
                        "policy_updates": checkpoint.get("policy_updates_completed"),
                        "reference_sha256": checkpoint.get("reference_sha256")}


def contact_signature(model, data):
    pairs = set()
    for cid, contact in enumerate(data.contact[:data.ncon]):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, cid, force)
        if force[0] > 0.1:
            pairs.add(tuple(sorted(map(int, contact.geom))))
    return tuple(sorted(pairs))


def native_branch(model, index, indices, observer, snapshot, action):
    data = copy.copy(snapshot)
    if np.shares_memory(data.qpos, snapshot.qpos) or np.shares_memory(data.qacc_warmstart, snapshot.qacc_warmstart):
        raise RuntimeError("mjData copy unexpectedly aliases its source")
    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "efc_force", "qfrc_applied", "xfrc_applied"):
        if not np.array_equal(getattr(data, name), getattr(snapshot, name)):
            raise RuntimeError(f"mjData copy failed to preserve {name}")
    data.ctrl[index.robot_actuators] = action
    signatures = []
    for _ in range(8):
        before = float(data.time)
        mujoco.mj_step(model, data)
        if not np.isclose(data.time, before + PHYSICS_DT, rtol=0, atol=1e-10):
            raise RuntimeError("native branch reset its physical timeline")
        mujoco.mj_forward(model, data)
        signatures.append(contact_signature(model, data))
    state = indices.pack(data.qpos.copy(), data.qvel.copy()).numpy()
    if not np.isfinite(state).all():
        raise RuntimeError("nonfinite native branch")
    return state, signatures, observer.observe(data)


def model_jacobian(model, state, action):
    s = torch.tensor(state, dtype=torch.float32)[None]
    a = torch.tensor(action, dtype=torch.float32, requires_grad=True)
    predicted = model.step(s, a[None])[0]
    gradients = [torch.autograd.grad(predicted[int(i)], a, retain_graph=True)[0]
                 for i in OUTPUT_INDICES]
    return predicted.detach().numpy(), torch.stack(gradients).detach().numpy()


def comparison(native, learned, valid, tolerance):
    n, w = native[valid], learned[valid]
    nn, wn = float(np.linalg.norm(n)), float(np.linalg.norm(w))
    meaningful = np.abs(n) > tolerance
    return {"native_gradient_norm": nn, "wm_gradient_norm": wn,
            "wm_to_native_norm_ratio": wn / nn if nn > 1e-12 else None,
            "cosine": float(n @ w / (nn * wn)) if nn * wn > 1e-18 else None,
            "sign_agreement_on_nontrivial_native": float(np.mean(np.sign(n[meaningful]) == np.sign(w[meaningful]))) if meaningful.any() else None,
            "nontrivial_joint_count": int(meaningful.sum()), "unclipped_joint_count": int(valid.sum())}


def run(args):
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("audit requires no global control callback")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    checkpoints, manifest = {}, {}
    for label, path in (("latest", args.checkpoint), ("initial", args.initial_model)):
        if path is not None:
            checkpoints[label], manifest[label] = freeze_checkpoint(path, args.out_dir / f"{label}.pt")
    if args.offline_model is not None:
        checkpoint, provenance = freeze_checkpoint(
            args.offline_model, args.out_dir / "air.pt",
            expected_schema="supermimic.ollie.offline_wm.v1",
        )
        source_manifest = args.offline_model.parent / "manifest.json"
        manifest_blob = source_manifest.read_bytes()
        if sha256(manifest_blob) != checkpoint["manifest_sha256"]:
            raise ValueError("offline checkpoint/data manifest hash mismatch")
        if tuple(checkpoint["robot_joint_names"]) != G1_JOINT_NAMES:
            raise ValueError("offline checkpoint robot joint order mismatch")
        (args.out_dir / "air_data_manifest.json").write_bytes(manifest_blob)
        provenance["data_manifest_sha256"] = sha256(manifest_blob)
        provenance["reference_contract"] = "reference-agnostic actual-transition dynamics; no reference hash"
        checkpoints["air"], manifest["air"] = checkpoint, provenance
    native_summary = json.loads((args.native_dir / "result.json").read_text())
    if "initial" in manifest and manifest["initial"]["sha256"] != native_summary["checkpoint_sha256"]:
        raise ValueError("initial checkpoint is not the native trajectory's recorded source")
    reference_path = Path(native_summary["reference"])
    reference_blob = reference_path.read_bytes()
    reference_hash = sha256(reference_blob)
    if reference_hash != native_summary["reference_sha256"]:
        raise ValueError("native trajectory and current reference hash differ")
    reference = JointReference.load(reference_path)
    for label, checkpoint in checkpoints.items():
        if label != "air" and checkpoint.get("reference_sha256") != reference_hash:
            raise ValueError("checkpoint/reference hash mismatch")
        if tuple(checkpoint["board_joint_names"]) != BOARD_JOINT_NAMES:
            raise ValueError("board joint order mismatch")
    model, data, index = compile_scene(timestep=PHYSICS_DT)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    observer = ContactObserver(model, index, 0.1)
    torch.set_num_threads(args.torch_threads)
    networks = {}
    for label, checkpoint in checkpoints.items():
        cfg = JointLearningConfig(**checkpoint["learning_config"])
        network = JointWorldModel(reference.layout, cfg, "cpu")
        network.load_state_dict(checkpoint["model"], strict=True)
        networks[label] = network.eval().requires_grad_(False)
    cfg = JointLearningConfig(**checkpoints["latest"]["learning_config"])
    policy = JointResidualPolicy(reference.layout, cfg, "cpu")
    policy.load_state_dict(checkpoints["latest"]["policy"], strict=True)
    policy.eval().requires_grad_(False)
    archive_path = args.native_dir / "rollout.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        qposes, qvels = archive["qpos"].copy(), archive["qvel"].copy()
        actions, times = archive["control"].copy(), archive["time_physics"].copy()
    selected = [round(t / CONTROL_DT) for t in args.times]
    if any(not np.isclose(step * CONTROL_DT, t) for step, t in zip(selected, args.times, strict=True)):
        raise ValueError("snapshot times must lie on 20ms boundaries")
    if min(selected) < 0 or max(selected) >= len(actions):
        raise ValueError("requested snapshots fall outside the native action trace")
    data.qpos[:], data.qvel[:] = qposes[0], qvels[0]
    data.ctrl[index.robot_actuators] = reference.robot_state[0, 7:36]
    mujoco.mj_forward(model, data)
    snapshots, replay_error = {}, {"qpos": 0.0, "qvel": 0.0}
    if 0 in selected:
        snapshots[0] = copy.copy(data)
    for step in range(max(selected)):
        data.ctrl[index.robot_actuators] = actions[step]
        for _ in range(8):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        replay_error["qpos"] = max(replay_error["qpos"], float(np.max(np.abs(data.qpos - qposes[step + 1]))))
        replay_error["qvel"] = max(replay_error["qvel"], float(np.max(np.abs(data.qvel - qvels[step + 1]))))
        if step + 1 in selected:
            snapshots[step + 1] = copy.copy(data)
    if max(replay_error.values()) > 1e-10:
        raise RuntimeError(f"continuous native prefix did not reproduce archive: {replay_error}")
    limits = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]]
    low, high = limits[:, 0] + 0.005, limits[:, 1] - 0.005
    refs = torch.tensor(reference.state, dtype=torch.float32)
    rows, cases, arrays, actor_changes = [], [], {}, []
    for step in selected:
        snapshot = snapshots[step]
        state = indices.pack(snapshot.qpos.copy(), snapshot.qvel.copy()).numpy()
        goals = np.minimum(step + 1 + np.asarray(cfg.reference_offsets), len(refs) - 1)
        with torch.no_grad():
            actor_raw = policy(torch.tensor(state, dtype=torch.float32)[None], refs[goals][None])[0].numpy().astype(float)
        baselines = {"native_initial_actor": actions[step], "latest_actor": np.clip(actor_raw, low, high)}
        ends, predictions = {}, {}
        for base_label, action in baselines.items():
            prefix = f"t{step:04d}_{base_label}"
            endpoint, base_signature, contact = native_branch(model, index, indices, observer, snapshot, action)
            repeated, _, _ = native_branch(model, index, indices, observer, snapshot, action)
            if not np.array_equal(endpoint, repeated):
                raise RuntimeError("identical full-history native branches are nondeterministic")
            if base_label == "native_initial_actor":
                expected = indices.pack(qposes[step + 1], qvels[step + 1]).numpy()
                if np.max(np.abs(expected - endpoint)) > 1e-10:
                    raise RuntimeError("native unperturbed fork does not match recorded continuation")
            ends[base_label] = endpoint
            case = {"time_s": float(times[step]), "baseline": base_label,
                    "snapshot_history_sha256": history_fingerprint(snapshot),
                    "snapshot_contacts": observer.observe(snapshot), "endpoint_contacts": contact,
                    "baseline_repeat_error": 0.0, "action": action.tolist(),
                    "native_next_outputs": dict(zip(OUTPUT_NAMES, endpoint[OUTPUT_INDICES].tolist(), strict=True)),
                    "models": {}, "epsilons": {}}
            gradients = {}
            predictions[base_label] = {}
            for label, network in networks.items():
                predicted, jacobian = model_jacobian(network, state, action)
                gradients[label] = jacobian
                predictions[base_label][label] = predicted
                case["models"][label] = {"prediction_error": dict(zip(OUTPUT_NAMES, (predicted[OUTPUT_INDICES] - endpoint[OUTPUT_INDICES]).tolist(), strict=True))}
                arrays[f"{prefix}_{label}_jacobian"] = jacobian
            previous_native = None
            for epsilon in args.epsilons:
                plus_actions = np.clip(action + epsilon * np.eye(29), low, high)
                minus_actions = np.clip(action - epsilon * np.eye(29), low, high)
                plus_span = plus_actions[np.arange(29), np.arange(29)] - action
                minus_span = action - minus_actions[np.arange(29), np.arange(29)]
                span = plus_span + minus_span
                valid = np.isclose(plus_span, epsilon, rtol=0, atol=1e-12) & np.isclose(minus_span, epsilon, rtol=0, atol=1e-12)
                plus, minus, changes = [], [], []
                for joint in range(29):
                    p, ps, _ = native_branch(model, index, indices, observer, snapshot, plus_actions[joint])
                    n, ns, _ = native_branch(model, index, indices, observer, snapshot, minus_actions[joint])
                    plus.append(p)
                    minus.append(n)
                    changes.append((sum(a != b for a, b in zip(ps, base_signature, strict=True)),
                                    sum(a != b for a, b in zip(ns, base_signature, strict=True))))
                plus, minus = np.asarray(plus), np.asarray(minus)
                native = ((plus[:, OUTPUT_INDICES] - minus[:, OUTPUT_INDICES]) / span[:, None]).T
                epsilon_case = {"contact_signature_changed_joints": int(np.count_nonzero(np.asarray(changes).sum(axis=1))),
                                "clipped_joint_names": [G1_JOINT_NAMES[j] for j in np.flatnonzero(~valid)], "models": {}}
                for label, network in networks.items():
                    with torch.no_grad():
                        batch_s = torch.tensor(state, dtype=torch.float32).expand(29, -1)
                        wp = network.step(batch_s, torch.tensor(plus_actions, dtype=torch.float32)).numpy()
                        wn = network.step(batch_s, torch.tensor(minus_actions, dtype=torch.float32)).numpy()
                    wm_fd = ((wp[:, OUTPUT_INDICES] - wn[:, OUTPUT_INDICES]) / span[:, None]).T
                    epsilon_case["models"][label] = {}
                    for channel, name in enumerate(OUTPUT_NAMES):
                        tolerance = 1e-3 if "vz" in name else 1e-4
                        epsilon_case["models"][label][name] = comparison(native[channel], gradients[label][channel], valid, tolerance)
                        for joint in range(29):
                            rows.append({"time_s": float(times[step]), "baseline": base_label, "model": label,
                                         "epsilon_rad": epsilon, "output": name, "joint": G1_JOINT_NAMES[joint],
                                         "unclipped_central": bool(valid[joint]), "native_secant": float(native[channel, joint]),
                                         "wm_autograd": float(gradients[label][channel, joint]), "wm_secant": float(wm_fd[channel, joint]),
                                         "native_plus_change": float(plus[joint, OUTPUT_INDICES[channel]] - endpoint[OUTPUT_INDICES[channel]]),
                                         "native_minus_change": float(minus[joint, OUTPUT_INDICES[channel]] - endpoint[OUTPUT_INDICES[channel]]),
                                         "plus_contact_changed_substeps": changes[joint][0], "minus_contact_changed_substeps": changes[joint][1]})
                if previous_native is not None:
                    previous, previous_valid = previous_native
                    common = valid & previous_valid
                    epsilon_case["native_epsilon_consistency"] = {
                        name: comparison(native[ch], previous[ch], common, 1e-3 if "vz" in name else 1e-4)
                        for ch, name in enumerate(OUTPUT_NAMES)}
                previous_native = native.copy(), valid.copy()
                case["epsilons"][str(epsilon)] = epsilon_case
                arrays[f"{prefix}_eps{epsilon}_native_plus"] = plus
                arrays[f"{prefix}_eps{epsilon}_native_minus"] = minus
                arrays[f"{prefix}_eps{epsilon}_native_jacobian"] = native
                arrays[f"{prefix}_eps{epsilon}_plus_actions"] = plus_actions
                arrays[f"{prefix}_eps{epsilon}_minus_actions"] = minus_actions
                arrays[f"{prefix}_eps{epsilon}_unclipped_central"] = valid
            case["wm_ascent_direction_tests"] = []
            for label, network in networks.items():
                for channel in (2, 3, 6, 7):
                    gradient = gradients[label][channel]
                    for radius in args.direction_radii:
                        # A bounded step in the model's own upward direction,
                        # tested independently from every other branch.
                        delta = radius * gradient / max(float(np.linalg.norm(gradient)), 1e-12)
                        directed_action = np.clip(action + delta, low, high)
                        directed_end, signature, _ = native_branch(
                            model, index, indices, observer, snapshot, directed_action
                        )
                        with torch.no_grad():
                            wm_end = network.step(
                                torch.tensor(state, dtype=torch.float32)[None],
                                torch.tensor(directed_action, dtype=torch.float32)[None],
                            )[0].numpy()
                        executed_delta = directed_action - action
                        case["wm_ascent_direction_tests"].append({
                            "model": label, "objective": OUTPUT_NAMES[channel],
                            "requested_l2_radius_rad": radius,
                            "actual_l2_delta_rad": float(np.linalg.norm(executed_delta)),
                            "max_abs_delta_rad": float(np.max(np.abs(executed_delta))),
                            "actual_action_delta": executed_delta.tolist(),
                            "native_output_change": dict(zip(OUTPUT_NAMES, (directed_end[OUTPUT_INDICES] - endpoint[OUTPUT_INDICES]).tolist(), strict=True)),
                            "wm_output_change": dict(zip(OUTPUT_NAMES, (wm_end[OUTPUT_INDICES] - predictions[base_label][label][OUTPUT_INDICES]).tolist(), strict=True)),
                            "native_contact_changed_substeps": sum(a != b for a, b in zip(signature, base_signature, strict=True)),
                        })
            arrays[f"{prefix}_snapshot_state"] = state
            arrays[f"{prefix}_baseline_action"] = action
            arrays[f"{prefix}_baseline_next_state"] = endpoint
            for field in ("qpos", "qvel", "ctrl", "qacc_warmstart", "efc_force"):
                arrays[f"{prefix}_snapshot_{field}"] = np.asarray(getattr(snapshot, field)).copy()
            cases.append(case)
            print(json.dumps({"time_s": float(times[step]), "baseline": base_label,
                              "contact_wheels": case["snapshot_contacts"]["unique_wheels_ground"],
                              "completed_branches": 2 * 29 * len(args.epsilons)}), flush=True)
        actual_change = ends["latest_actor"][OUTPUT_INDICES] - ends["native_initial_actor"][OUTPUT_INDICES]
        actor_changes.append({"time_s": float(times[step]),
                              "action_delta_norm": float(np.linalg.norm(baselines["latest_actor"] - baselines["native_initial_actor"])),
                              "native_output_change": dict(zip(OUTPUT_NAMES, actual_change.tolist(), strict=True)),
                              "wm_output_change": {label: dict(zip(OUTPUT_NAMES, (predictions["latest_actor"][label][OUTPUT_INDICES] - predictions["native_initial_actor"][label][OUTPUT_INDICES]).tolist(), strict=True)) for label in networks}})
    with (args.out_dir / "gradients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out_dir / "branches.npz", **arrays)
    summary = {"checkpoints": manifest, "reference": str(reference_path), "reference_sha256": reference_hash,
               "native_archive_sha256": sha256(archive_path.read_bytes()), "prefix_max_abs_error": replay_error,
               "mujoco_version": mujoco.__version__, "native_solver_history": "full mjData copies after exact uninterrupted prefix replay; no cold qpos resets",
               "native_dtype": "float64", "wm_dtype": "float32", "control_dt_s": CONTROL_DT,
               "physics_dt_s": PHYSICS_DT, "output_names": OUTPUT_NAMES, "cases": cases,
               "robot_joint_names": G1_JOINT_NAMES, "joint_target_limits_rad": np.stack([low, high], axis=1).tolist(),
               "actor_action_changes": actor_changes,
               "scope": "20ms conditional finite-difference secants at three fixed histories; not a smooth-contact derivative certificate or a full-horizon BPTT proof"}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--offline-model", type=Path,
                        help="Optional independently fitted offline WM; retains the same actor/action baselines")
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--times", nargs="+", type=float, default=[2.60, 2.76, 2.88])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.005, 0.02])
    parser.add_argument("--direction-radii", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if (not args.times or not args.epsilons or not args.direction_radii
            or any(not np.isfinite(t) or t < 0 for t in args.times)
            or any(e <= 0 or not np.isfinite(e) for e in (*args.epsilons, *args.direction_radii))):
        parser.error("positive finite epsilon values and snapshot times required")
    run(args)


if __name__ == "__main__":
    main()
