"""Fit joint dynamics to actual executed transitions, with episode-level holdout.

Counterfactual expert labels and success/teacher flags are never training targets.
This trains a world model only; no Ollie policy or physical success is certified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.kinematics.rotations import quat_apply, quat_mul, quat_normalize, rotvec_from_quat
from supermimic.ollie.learning import JointLearningConfig, JointWorldModel
from supermimic.ollie.physics import BOARD_JOINT_NAMES
from supermimic.ollie.state import STATE_SCHEMA, JointStateLayout
from supermimic.world_model.state_spec import quat_apply_inverse_quat


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def content_hash(states, actions):
    """Hash model-precision content, canonicalizing quaternion sign and -0."""
    states = np.asarray(states, dtype="<f4").copy()
    for offset in (3, 74):
        quaternion = states[..., offset : offset + 4]
        quaternion *= np.where(quaternion[..., :1] < 0, -1.0, 1.0)
    states[states == 0] = 0
    actions = np.asarray(actions, dtype="<f4").copy()
    actions[actions == 0] = 0
    return digest(states.tobytes() + actions.tobytes())


@dataclass
class Episode:
    source: int
    episode_id: int
    states: np.ndarray
    actions: np.ndarray
    fingerprint: str
    fixed_split: str | None = None
    snapshot_fingerprint: str | None = None


PAIR_FEATURES = tuple(
    f"{entity}_{kind}_{axis}"
    for entity in ("robot", "board")
    for kind in ("position_m", "rotation_rad", "velocity_m_s", "angular_world_rad_s")
    for axis in "xyz"
)


def root_response(positive, negative):
    """24 raw SI features, with rotations/angular velocities in the world frame."""
    result = []
    for position, quaternion, velocity, angular in (
        (slice(0, 3), slice(3, 7), slice(36, 39), slice(39, 42)),
        (slice(71, 74), slice(74, 78), slice(82, 85), slice(85, 88)),
    ):
        qp, qn = (
            quat_normalize(positive[..., quaternion]),
            quat_normalize(negative[..., quaternion]),
        )
        rotation = rotvec_from_quat(quat_mul(qp, quat_apply_inverse_quat(qn)))
        result.extend(
            (
                positive[..., position] - negative[..., position],
                rotation,
                positive[..., velocity] - negative[..., velocity],
                quat_apply(qp, positive[..., angular]) - quat_apply(qn, negative[..., angular]),
            )
        )
    return torch.cat(result, dim=-1)


def pair_scales(horizons, board_vz_h1, board_vz_h8, device):
    # Fixed engineering scales; none are estimated from held-out responses.
    scale = torch.tensor(
        [0.001] * 3
        + [0.005] * 3
        + [0.05] * 3
        + [0.1] * 3
        + [0.001] * 3
        + [0.005] * 3
        + [0.01] * 3
        + [0.05] * 3,
        device=device,
    ).repeat(len(horizons), 1)
    for row, horizon in enumerate(horizons):
        scale[row, 20] = board_vz_h1 if horizon == 1 else board_vz_h8
    return scale


def read_paired_source(path, source_id):
    """Use collector-assigned snapshot splits; never randomly split sibling episodes."""
    directory = path if path.is_dir() else path.parent
    paired_path = directory / "paired_branches.npz"
    transition_path = directory / "transitions.npz"
    metadata_path = directory / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    for file in (paired_path, transition_path):
        if digest(file.read_bytes()) != metadata["artifacts_sha256"][file.name]:
            raise ValueError(f"Paired artifact hash mismatch: {file}")
    with np.load(paired_path, allow_pickle=False) as file:
        values = {key: file[key] for key in file.files}
    with np.load(transition_path, allow_pickle=False) as file:
        transitions = {key: file[key] for key in file.files}
    if not np.array_equal(transitions["action"], transitions["ctrl_float64"].astype(np.float32)):
        raise ValueError("Paired standard action differs from actual executed ctrl")
    for collection in (values, transitions):
        if (
            str(collection["schema"]) != STATE_SCHEMA
            or tuple(collection["robot_joint_names"]) != G1_JOINT_NAMES
            or tuple(collection["board_joint_names"]) != BOARD_JOINT_NAMES
            or not np.isclose(float(collection["control_dt"]), 0.02)
        ):
            raise ValueError("Paired source has an incompatible state/joint/time schema")
    states, actions = values["state"], values["action"]
    count = len(states)
    if states.shape != (count, 9, 92) or actions.shape != (count, 8, 29):
        raise ValueError("Paired source must contain complete H8 state/action branches")
    ids, groups = values["episode_id"], values["snapshot_group"]
    fingerprints, splits = values["snapshot_fingerprint_sha256"], values["split"]
    baseline_ids, joints, signs = (
        values["baseline_episode_id"],
        values["joint_index"],
        values["sign"],
    )
    valid = values["quality_valid"].astype(bool)
    if (
        any(
            x.shape != (count,)
            for x in (ids, groups, fingerprints, splits, baseline_ids, joints, signs, valid)
        )
        or len(np.unique(ids)) != count
        or not set(splits).issubset({"train", "heldout"})
        or not {"train", "heldout"}.issubset(set(splits[valid]))
    ):
        raise ValueError(
            "Paired source requires unique episode IDs and explicit train/heldout group splits"
        )
    if not np.array_equal(values["initial_state"], states[:, 0]):
        raise ValueError("Paired initial_state differs from state[:,0]")
    assignments = {row["snapshot_group"]: row for row in metadata["snapshots"]}
    by_id, episode_splits, fingerprint_splits = {}, {}, {}
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(set(splits[rows])) != 1 or len(set(fingerprints[rows])) != 1:
            raise ValueError("A snapshot group's branches crossed splits or fingerprints")
        split, fingerprint = str(splits[rows[0]]), str(fingerprints[rows[0]])
        recorded = assignments[int(group)]
        if recorded["split"] != split or recorded["fingerprint_sha256"] != fingerprint:
            raise ValueError("Paired manifest and NPZ snapshot association differ")
        if fingerprint in fingerprint_splits and fingerprint_splits[fingerprint] != split:
            raise ValueError("An identical snapshot fingerprint crossed train/heldout")
        fingerprint_splits[fingerprint] = split
        for row in rows:
            by_id[int(ids[row])] = int(row)
            episode_splits[int(ids[row])] = split
    episodes, combinations = [], {}
    for row in range(count):
        baseline = by_id[int(baseline_ids[row])]
        if (
            groups[row] != groups[baseline]
            or fingerprints[row] != fingerprints[baseline]
            or joints[baseline] != -1
            or signs[baseline] != 0
            or not np.array_equal(states[row, 0], states[baseline, 0])
        ):
            raise ValueError("Paired baseline is not the same initial snapshot")
        if not np.array_equal(
            actions[row] - actions[baseline], values["actual_delta_from_baseline"][row]
        ):
            raise ValueError("Saved actual paired intervention differs from actual controls")
        if joints[row] >= 0:
            if not 0 <= joints[row] < 29 or signs[row] not in (-1, 1):
                raise ValueError("Invalid paired joint/sign")
            if str(values["joint_name"][row]) != G1_JOINT_NAMES[joints[row]]:
                raise ValueError("Paired intervention joint name differs from its canonical index")
            other = np.arange(29) != joints[row]
            if not np.array_equal(actions[row, :, other], actions[baseline, :, other]):
                raise ValueError("A supposedly single-joint branch changed another joint")
            key = (int(groups[row]), int(joints[row]), int(signs[row]))
            if key in combinations:
                raise ValueError("Duplicate paired group/joint/sign")
            combinations[key] = row
        if not valid[row]:
            continue
        if (
            values["completed_controls"][row] != 8
            or not np.isfinite(states[row]).all()
            or not np.isfinite(actions[row]).all()
        ):
            raise ValueError("Quality-valid paired episode is incomplete or nonfinite")
        for offset in (3, 74):
            if (
                np.max(np.abs(np.linalg.norm(states[row, :, offset : offset + 4], axis=-1) - 1))
                > 1e-4
            ):
                raise ValueError("Paired source has a nonunit root quaternion")
        transition_rows = np.flatnonzero(transitions["episode_id"] == ids[row])
        if len(transition_rows) != 8:
            raise ValueError("Paired episode does not map to eight standard transitions")
        before = transitions.get("state_float64", transitions["state"])[transition_rows]
        after = transitions.get("next_state_float64", transitions["next_state"])[transition_rows]
        control = transitions.get("ctrl_float64", transitions["action"])[transition_rows]
        if (
            not np.array_equal(before, states[row, :-1])
            or not np.array_equal(after, states[row, 1:])
            or not np.array_equal(control, actions[row])
            or not np.all(transitions["snapshot_group"][transition_rows] == groups[row])
            or not np.all(
                transitions["source_snapshot_fingerprint"][transition_rows] == fingerprints[row]
            )
            or not np.all(transitions["split"][transition_rows] == splits[row])
            or not np.allclose(
                transitions["time_s"][transition_rows], np.arange(8) * 0.02, rtol=0, atol=1e-10
            )
        ):
            raise ValueError(
                "Paired/standard transition controls, state chain, time or group disagree"
            )
        converted, targets = states[row].astype(np.float32), actions[row].astype(np.float32)
        episodes.append(
            Episode(
                source_id,
                int(ids[row]),
                converted,
                targets,
                content_hash(converted, targets),
                str(splits[row]),
                str(fingerprints[row]),
            )
        )
    pairs = []
    for (group, joint, sign), positive in combinations.items():
        if sign != 1:
            continue
        negative = combinations.get((group, joint, -1))
        if negative is None:
            raise ValueError("Paired source is missing a negative counterpart")
        baseline = by_id[int(baseline_ids[positive])]
        if valid[positive] and valid[negative] and valid[baseline]:
            if np.max(np.abs(actions[positive, 0] - actions[negative, 0])) <= 1e-10:
                raise ValueError("First-step positive/negative actions coincide after clipping")
            pairs.append((positive, negative))
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or not all(
        any(splits[a] == split for a, _ in pairs) for split in ("train", "heldout")
    ):
        raise ValueError("Both splits require quality-valid action intervention pairs")
    # Compute tiny native response differences in float64 BEFORE casting model targets.
    raw_response = root_response(
        torch.from_numpy(states[pairs[:, 0], 1:]), torch.from_numpy(states[pairs[:, 1], 1:])
    ).numpy()
    paired = {
        "initial": states[pairs[:, 0], 0].astype(np.float32),
        "positive_action": actions[pairs[:, 0]].astype(np.float32),
        "negative_action": actions[pairs[:, 1]].astype(np.float32),
        "response": raw_response.astype(np.float32),
        "split": splits[pairs[:, 0]],
        "group": groups[pairs[:, 0]],
        "joint": joints[pairs[:, 0]],
        "positive_episode_id": ids[pairs[:, 0]],
        "negative_episode_id": ids[pairs[:, 1]],
    }
    return (
        episodes,
        paired,
        {
            "path": str(paired_path.resolve()),
            "sha256": digest(paired_path.read_bytes()),
            "transition_sha256": digest(transition_path.read_bytes()),
            "manifest_sha256": digest(metadata_path.read_bytes()),
            "episodes": len(episodes),
            "transitions": len(episodes) * 8,
            "valid_pairs": len(pairs),
            "snapshot_assignments": metadata["snapshots"],
            "split_unit": "Complete preserved snapshot group",
            "action_field": "actual action, never expert_label",
            "expert_label_used": False,
            "paired_target_precision": "Native float64 response differences, then float32 model targets",
            "source_metadata": metadata["source_metadata"],
        },
    )


def read_source(path, source_id, *, infer_dagger, producer):
    blob = path.read_bytes()
    with np.load(io.BytesIO(blob), allow_pickle=False) as stored:
        values = {key: stored[key] for key in stored.files}
    schema_fields = ("schema", "robot_joint_names", "board_joint_names", "control_dt")
    missing = [key for key in schema_fields if key not in values]
    inferred = None
    if missing:
        if (
            not infer_dagger
            or path.name != "dagger_actual_transitions.npz"
            or not {"expert_label", "label_used", "source"}.issubset(values)
        ):
            raise ValueError(
                f"{path}: missing schema fields {missing}; only explicitly reviewed DAgger producer inference is supported"
            )
        text = producer.read_text()
        required = (
            "action=np.stack([r[1] for r in real_data])",
            "state_now = expert.indices.pack(data.qpos, data.qvel).numpy()",
            "data.ctrl[index.robot_actuators] = executed",
            "mujoco.mj_step(native_model, data, nstep=8)",
            "step / 50",
        )
        if not all(fragment in text for fragment in required):
            raise ValueError(
                "Reviewed DAgger producer no longer matches the recorded action/state/time contract"
            )
        inferred = {
            "missing_fields": missing,
            "basis": "Inferred from the locally reviewed producer, not self-certified by this NPZ",
            "producer": str(producer.resolve()),
            "producer_sha256": digest(producer.read_bytes()),
            "robot_joint_names": list(G1_JOINT_NAMES),
            "board_joint_names": list(BOARD_JOINT_NAMES),
            "control_dt": 0.02,
            "schema": STATE_SCHEMA,
        }
        print(json.dumps({"source": str(path), "schema_inference": inferred}), flush=True)
    else:
        if (
            str(values["schema"]) != STATE_SCHEMA
            or tuple(values["robot_joint_names"]) != G1_JOINT_NAMES
            or tuple(values["board_joint_names"]) != BOARD_JOINT_NAMES
            or not np.isclose(float(values["control_dt"]), 0.02)
        ):
            raise ValueError(f"{path}: incompatible state/action/joint/time schema")
    state, nxt, action = [
        np.asarray(values[key], dtype=np.float32) for key in ("state", "next_state", "action")
    ]
    n = len(state)
    if (
        state.shape != (n, 92)
        or nxt.shape != state.shape
        or action.shape != (n, 29)
        or not all(np.isfinite(x).all() for x in (state, nxt, action))
    ):
        raise ValueError(f"{path}: nonfinite or incompatible transition arrays")
    if "ctrl" in values and not np.array_equal(
        action, np.asarray(values["ctrl"], dtype=np.float32)
    ):
        raise ValueError("Saved action differs from actual ctrl")
    ids, times = values["episode_id"], values["time_s"]
    if ids.shape != (n,) or times.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Need one integer episode_id and pre-action time_s per transition")
    for data in (state, nxt):
        for offset in (3, 74):
            if np.max(np.abs(np.linalg.norm(data[:, offset : offset + 4], axis=-1) - 1)) > 1e-4:
                raise ValueError("Nonunit robot/board wxyz quaternion")
    raw_state = np.asarray(values.get("state_float64", values["state"]))
    raw_next = np.asarray(values.get("next_state_float64", values["next_state"]))
    episodes, chain_error, raw_chain_error = [], 0.0, 0.0
    for episode_id in np.unique(ids):
        rows = np.flatnonzero(ids == episode_id)
        if (
            not np.all(np.diff(rows) == 1)
            or not np.allclose(np.diff(times[rows]), 0.02, rtol=0, atol=1e-8)
            or not np.isclose(times[rows[0]], 0)
        ):
            raise ValueError(f"{path}: episode {episode_id} is reordered, reset or not 50Hz")
        error = float(np.max(np.abs(nxt[rows[:-1]] - state[rows[1:]]))) if len(rows) > 1 else 0.0
        chain_error = max(chain_error, error)
        raw_error = (
            float(np.max(np.abs(raw_next[rows[:-1]] - raw_state[rows[1:]])))
            if len(rows) > 1
            else 0.0
        )
        raw_chain_error = max(raw_chain_error, raw_error)
        if raw_error > 1e-9:
            raise ValueError(f"{path}: source-precision next-state chain discontinuity {raw_error}")
        if error != 0:
            raise ValueError(f"{path}: next-state chain discontinuity {error}")
        sequence = np.concatenate([state[rows], nxt[rows[-1:]]])
        actions = action[rows]
        episodes.append(
            Episode(source_id, int(episode_id), sequence, actions, content_hash(sequence, actions))
        )
    return episodes, {
        "path": str(path.resolve()),
        "sha256": digest(blob),
        "transitions": n,
        "episodes": len(episodes),
        "chain_max_abs_error_float32": chain_error,
        "chain_max_abs_error_source_precision": raw_chain_error,
        "schema_inference": inferred,
        "source_description": str(values.get("source", "")),
        "action_field": "action",
        "expert_label_present": "expert_label" in values,
        "expert_label_used": False,
        "verified_teacher_required_for_dynamics": False,
    }


def split_and_window(episodes, sources, horizon, seed, fraction):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.fingerprint].append(episode)
    fixed = {}
    for key, members in groups.items():
        requested = {ep.fixed_split for ep in members if ep.fixed_split is not None}
        if len(requested) > 1:
            raise ValueError("Duplicate episode content has contradictory fixed group splits")
        if requested:
            fixed[key] = requested.pop()
    heldout_groups = {key for key, split in fixed.items() if split == "heldout"}
    for source in range(sources):
        candidates = [
            key
            for key, members in groups.items()
            if any(ep.source == source for ep in members) and fixed.get(key) != "train"
        ]
        rng.shuffle(candidates)
        wanted = max(1, math.ceil(sum(ep.source == source for ep in episodes) * fraction))
        existing = sum(ep.source == source for key in heldout_groups for ep in groups[key])
        for key in candidates:
            if existing >= wanted:
                break
            if key not in heldout_groups:
                heldout_groups.add(key)
                existing += sum(ep.source == source for ep in groups[key])
    windows = {name: [] for name in ("train", "heldout")}
    manifest = []
    for episode in episodes:
        split = "heldout" if episode.fingerprint in heldout_groups else "train"
        manifest.append(
            {
                "source": episode.source,
                "episode_id": episode.episode_id,
                "transitions": len(episode.actions),
                "fingerprint": episode.fingerprint,
                "split": split,
                **(
                    {
                        "snapshot_fingerprint": episode.snapshot_fingerprint,
                        "fixed_group_split": episode.fixed_split,
                    }
                    if episode.snapshot_fingerprint is not None
                    else {}
                ),
            }
        )
        for start in range(len(episode.actions) - horizon + 1):
            states = episode.states[start : start + horizon + 1]
            actions = episode.actions[start : start + horizon]
            windows[split].append((episode.source, states, actions, content_hash(states, actions)))
    heldout_hashes = {row[3] for row in windows["heldout"]}
    before = len(windows["train"])
    heldout_overlap = sum(row[3] in heldout_hashes for row in windows["train"])
    seen = set()
    unique = []
    for row in windows["train"]:
        if row[3] not in heldout_hashes and row[3] not in seen:
            unique.append(row)
            seen.add(row[3])
    windows["train"] = unique
    if {row[3] for row in unique} & heldout_hashes:
        raise RuntimeError("Duplicate content crossed the train/heldout split")
    if any(
        not any(row[0] == source for row in windows[split])
        for split in windows
        for source in range(sources)
    ):
        raise ValueError(
            "Each source needs nonduplicate train windows and at least one complete held-out episode"
        )
    return windows, {
        "episode_assignments": manifest,
        "duplicate_episode_groups": [key for key, members in groups.items() if len(members) > 1],
        "train_windows_before_deduplication": before,
        "train_windows_after_deduplication": len(unique),
        "removed_duplicate_or_heldout_overlapping_train_windows": before - len(unique),
        "removed_heldout_overlapping_train_windows": heldout_overlap,
        "removed_within_train_duplicate_windows": before - len(unique) - heldout_overlap,
        "identical_train_heldout_horizon_windows": 0,
        "split_unit": "Complete content-grouped episodes, with paired snapshot groups preserved when supplied; all duplicate replay episodes share one split",
    }


def paired_tensors(values, split, device):
    rows = np.flatnonzero(values["split"] == split)
    return {
        name: torch.as_tensor(values[name][rows], device=device)
        for name in ("initial", "positive_action", "negative_action", "response")
    }


def predict_pair_response(model, data, rows, horizons):
    initial = data["initial"][rows]
    count = len(initial)
    predicted = torch.cat((initial, initial), dim=0)
    positive, negative = data["positive_action"][rows], data["negative_action"][rows]
    responses = []
    for step in range(max(horizons)):
        predicted = model.step(predicted, torch.cat((positive[:, step], negative[:, step]), dim=0))
        if step + 1 in horizons:
            responses.append(root_response(predicted[:count], predicted[count:]))
    return torch.stack(responses, dim=1)


@torch.no_grad()
def evaluate_pairs(model, data, horizons, scales, batch_size=128):
    predictions = []
    for start in range(0, len(data["initial"]), batch_size):
        rows = torch.arange(
            start, min(start + batch_size, len(data["initial"])), device=scales.device
        )
        predictions.append(predict_pair_response(model, data, rows, horizons))
    predicted = torch.cat(predictions)
    target = data["response"][:, [h - 1 for h in horizons]]
    difference = predicted - target
    result = {
        "pairs": len(target),
        "scaled_response_mse": float((difference / scales).square().mean()),
        "zero_response_scaled_mse": float((target / scales).square().mean()),
        "horizons": {},
    }
    for row, horizon in enumerate(horizons):
        truth, estimate, error = target[:, row], predicted[:, row], difference[:, row]
        record = {
            "feature_rmse_si": error.square().mean(0).sqrt().tolist(),
            "zero_response_feature_rmse_si": truth.square().mean(0).sqrt().tolist(),
        }
        for entity, offset in (("robot", 0), ("board", 12)):
            for kind, section in (
                ("position_m", slice(offset, offset + 3)),
                ("rotation_rad", slice(offset + 3, offset + 6)),
                ("velocity_m_s", slice(offset + 6, offset + 9)),
                ("angular_world_rad_s", slice(offset + 9, offset + 12)),
            ):
                record[f"{entity}_{kind}_response_rmse"] = float(
                    error[:, section].square().sum(-1).mean().sqrt()
                )
        for name, column in (("robot_vz", 8), ("board_vz", 20)):
            expected, actual = truth[:, column], estimate[:, column]
            meaningful = expected.abs() > 1e-6
            energy = expected.square().sum()
            centered_truth, centered_pred = expected - expected.mean(), actual - actual.mean()
            norm = centered_truth.square().sum().sqrt() * centered_pred.square().sum().sqrt()
            record[name] = {
                "actual_response_abs_min": float(expected.abs().min()),
                "actual_response_abs_median": float(expected.abs().median()),
                "actual_response_abs_q90": float(torch.quantile(expected.abs(), 0.9)),
                "actual_response_abs_max": float(expected.abs().max()),
                "predicted_response_abs_median": float(actual.abs().median()),
                "response_rmse_m_s": float((actual - expected).square().mean().sqrt()),
                "zero_response_rmse_m_s": float(expected.square().mean().sqrt()),
                "response_gain_projection": float((actual * expected).sum() / energy)
                if energy > 0
                else None,
                "correlation": float((centered_truth * centered_pred).sum() / norm)
                if norm > 0
                else None,
                "sign_response_floor_m_s": 1e-6,
                "sign_samples": int(meaningful.sum()),
                "sign_accuracy": float(
                    (actual[meaningful].sign() == expected[meaningful].sign()).float().mean()
                )
                if meaningful.any()
                else None,
            }
        result["horizons"][str(horizon)] = record
    return result


def tensors(rows, device):
    return (
        torch.tensor(np.stack([row[1] for row in rows]), device=device).transpose(0, 1),
        torch.tensor(np.stack([row[2] for row in rows]), device=device).transpose(0, 1),
        np.array([row[0] for row in rows]),
    )


@torch.no_grad()
def evaluate(model, states, actions, source_ids, wheel_weight, batch_size=128):
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    for start in range(0, states.shape[1], batch_size):
        truth, control = (
            states[:, start : start + batch_size],
            actions[:, start : start + batch_size],
        )
        predicted = truth[0]
        rows = {}
        for step in range(len(control)):
            predicted = model.step(predicted, control[step])
            error = model.spec.error_vector(predicted, truth[step + 1])
            weight = error.new_ones(error.shape[-1])
            weight[-8:] = wheel_weight
            rows[f"weighted_nse_h{step + 1}"] = (error.square() * weight).sum(-1) / weight.sum()
            if step in (0, len(control) - 1):
                h = step + 1
                rows.update(
                    {
                        f"raw_nse_h{h}": error.square().mean(-1),
                        f"robot_root_pose_nse_h{h}": error[:, :6].square().mean(-1),
                        f"board_root_pose_nse_h{h}": error[:, 70:76].square().mean(-1),
                        f"board_root_state_nse_h{h}": error[:, 70:82].square().mean(-1),
                        f"robot_root_position_squared_m_h{h}": (error[:, :3] * 0.05)
                        .square()
                        .sum(-1),
                        f"robot_root_rotation_squared_rad_h{h}": (error[:, 3:6] * 0.1)
                        .square()
                        .sum(-1),
                        f"board_root_position_squared_m_h{h}": (error[:, 70:73] * 0.05)
                        .square()
                        .sum(-1),
                        f"board_root_rotation_squared_rad_h{h}": (error[:, 73:76] * 0.1)
                        .square()
                        .sum(-1),
                    }
                )
        rows["weighted_window_nse"] = torch.stack(
            [rows[f"weighted_nse_h{k + 1}"] for k in range(len(control))]
        ).mean(0)
        ids = source_ids[start : start + batch_size]
        for source in [-1, *np.unique(ids).tolist()]:
            mask = np.ones(len(ids), dtype=bool) if source == -1 else ids == source
            counts[source] += int(mask.sum())
            for name, values in rows.items():
                sums[source][name] += float(
                    values[torch.as_tensor(mask, device=values.device)].sum()
                )
    result = {}
    for source, values in sums.items():
        averaged = {key: value / counts[source] for key, value in values.items()}
        for key in list(averaged):
            if "_squared_" in key:
                averaged[key.replace("_squared_", "_rmse_")] = math.sqrt(averaged.pop(key))
        result[str(source)] = {"windows": counts[source], **averaged}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--paired-source",
        type=Path,
        help="Collector-38 directory (or paired_branches.npz), preserving its snapshot-group splits",
    )
    parser.add_argument("--paired-loss-weight", type=float, default=0.0)
    parser.add_argument("--paired-horizons", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--paired-board-vz-scale-h1", type=float, default=1e-4)
    parser.add_argument("--paired-board-vz-scale-h8", type=float, default=0.01)
    parser.add_argument(
        "--resume-wm",
        type=Path,
        help="Strict WM weights initialization; always use a fresh optimizer",
    )
    parser.add_argument(
        "--config-checkpoint", type=Path, default=Path("logs/ollie_bc_prefix_03/best.pt")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--infer-dagger-schema", action="store_true")
    parser.add_argument(
        "--dagger-producer", type=Path, default=Path("scripts/local/warm_start_ollie_policy.py")
    )
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--wheel-weight", type=float, default=0.05)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        args.out_dir.exists()
        or args.updates < 1
        or args.batch_size < 1
        or args.window != 8
        or not 0 < args.heldout_fraction < 0.5
        or not 0 <= args.wheel_weight <= 1
        or not math.isfinite(args.paired_loss_weight)
        or args.paired_loss_weight < 0
        or (args.paired_loss_weight > 0 and args.paired_source is None)
        or args.paired_horizons != sorted(set(args.paired_horizons))
        or not args.paired_horizons
        or min(args.paired_horizons) < 1
        or max(args.paired_horizons) > 8
        or not math.isfinite(args.paired_board_vz_scale_h1)
        or args.paired_board_vz_scale_h1 <= 0
        or not math.isfinite(args.paired_board_vz_scale_h8)
        or args.paired_board_vz_scale_h8 <= 0
    ):
        parser.error("Need a new output directory, positive budget, H8, and valid split/weight")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    episodes, sources = [], []
    for source_id, path in enumerate(args.datasets):
        loaded, record = read_source(
            path, source_id, infer_dagger=args.infer_dagger_schema, producer=args.dagger_producer
        )
        episodes.extend(loaded)
        sources.append(record)
    paired = None
    if args.paired_source is not None:
        loaded, paired, record = read_paired_source(args.paired_source, len(sources))
        if any(record["transition_sha256"] == source["sha256"] for source in sources):
            raise ValueError("Do not also include paired transitions via ordinary --datasets")
        episodes.extend(loaded)
        sources.append(record)
    windows, split = split_and_window(
        episodes, len(sources), args.window, args.seed, args.heldout_fraction
    )
    heldout_hashes = {row[3] for row in windows["heldout"]}
    if any(ep.fixed_split == "train" and ep.fingerprint in heldout_hashes for ep in episodes):
        raise ValueError("A paired training branch also appears in held-out horizon content")
    checkpoint = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False)
    config = JointLearningConfig(**checkpoint["learning_config"])
    if tuple(checkpoint["board_joint_names"]) != BOARD_JOINT_NAMES or not np.isclose(
        checkpoint["environment_config"]["control_dt"], 0.02
    ):
        raise ValueError("Base model configuration has an incompatible joint/time contract")
    manifest = {
        "schema": "supermimic.ollie.offline_wm_manifest.v1",
        "sources": sources,
        **split,
        "config_checkpoint": str(args.config_checkpoint.resolve()),
        "config_checkpoint_sha256": digest(args.config_checkpoint.read_bytes()),
        "initialization": "Strict resume-WM weights; fresh optimizer, policy weights unused"
        if args.resume_wm
        else "Load the base checkpoint's world-model weights; policy weights are unused",
        "resume_wm": str(args.resume_wm.resolve()) if args.resume_wm else None,
        "resume_wm_sha256": digest(args.resume_wm.read_bytes()) if args.resume_wm else None,
        "optimizer_reset": True,
        "learning_config": asdict(config),
        "robot_joint_names": list(G1_JOINT_NAMES),
        "board_joint_names": list(BOARD_JOINT_NAMES),
        "control_dt": 0.02,
        "window": 8,
        "wheel_weight": args.wheel_weight,
        "seed": args.seed,
        "sampling": "Uniform sources, then uniform nonduplicate training windows within the source",
        "heldout_role": "Validation episodes; diagnostics do not certify an independent task-success test",
    }
    scales = pair_scales(
        args.paired_horizons,
        args.paired_board_vz_scale_h1,
        args.paired_board_vz_scale_h8,
        args.device,
    )
    if paired is not None:
        manifest["paired_loss"] = {
            "weight": args.paired_loss_weight,
            "horizons": args.paired_horizons,
            "response_feature_names": PAIR_FEATURES,
            "engineering_scales_si_by_horizon": scales.cpu().tolist(),
            "response": "Positive minus negative world root position, world-relative rotation log, world linear/angular velocity, computed from native float64 before target conversion",
            "training": "Roll both frozen actual H8 action sequences from the same initial state, without teacher forcing or treating diverged later states as a same-state one-step pair",
            "sampling": "Uniform train-only group/joint pairs, separate RNG from ordinary source-balanced batches",
            "heldout_groups": np.unique(paired["group"][paired["split"] == "heldout"]).tolist(),
            "train_pairs": int(np.count_nonzero(paired["split"] == "train")),
            "heldout_pairs": int(np.count_nonzero(paired["split"] == "heldout")),
            "zero_response_baseline": "A model that ignores action has zero positive-minus-negative response",
        }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    train_states, train_actions, train_sources = tensors(windows["train"], args.device)
    held_states, held_actions, held_sources = tensors(windows["heldout"], args.device)
    pools = [
        torch.tensor(np.flatnonzero(train_sources == source), device=args.device)
        for source in range(len(sources))
    ]
    model = JointWorldModel(JointStateLayout(BOARD_JOINT_NAMES), config, args.device)
    initializer = checkpoint
    if args.resume_wm:
        initializer = torch.load(args.resume_wm, map_location="cpu", weights_only=False)
        if (
            tuple(initializer["board_joint_names"]) != BOARD_JOINT_NAMES
            or not np.isclose(initializer["environment_config"]["control_dt"], 0.02)
            or tuple(initializer["learning_config"]["hidden"]) != tuple(config.hidden)
        ):
            raise ValueError("Resume WM has incompatible model/joint/time configuration")
    model.load_state_dict(initializer["model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model_lr)
    initial = evaluate(model, held_states, held_actions, held_sources, args.wheel_weight)
    pair_train, pair_held = None, None
    pair_initial = None
    pair_generator = torch.Generator(device=args.device).manual_seed(args.seed + 12345)
    if paired is not None:
        pair_train = paired_tensors(paired, "train", args.device)
        pair_held = paired_tensors(paired, "heldout", args.device)
        pair_initial = {
            "train": evaluate_pairs(model, pair_train, args.paired_horizons, scales),
            "heldout": evaluate_pairs(model, pair_held, args.paired_horizons, scales),
        }
    history = []
    batch_digest = hashlib.sha256()
    for update in range(1, args.updates + 1):
        choices = torch.randint(len(sources), (args.batch_size,), device=args.device)
        batch = torch.empty(args.batch_size, dtype=torch.long, device=args.device)
        for source, pool in enumerate(pools):
            mask = choices == source
            batch[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=args.device)]
        batch_digest.update(batch.cpu().numpy().tobytes())
        absolute_loss, _ = model.window_loss(
            train_states[:, batch], train_actions[:, batch], args.wheel_weight
        )
        paired_loss = absolute_loss.new_zeros(())
        if pair_train is not None and args.paired_loss_weight:
            pair_rows = torch.randint(
                len(pair_train["initial"]),
                (args.batch_size,),
                device=args.device,
                generator=pair_generator,
            )
            predicted = predict_pair_response(model, pair_train, pair_rows, args.paired_horizons)
            target = pair_train["response"][pair_rows][:, [h - 1 for h in args.paired_horizons]]
            paired_loss = ((predicted - target) / scales).square().mean()
        loss = absolute_loss + args.paired_loss_weight * paired_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(gradient):
            raise RuntimeError("Nonfinite world-model loss or gradient")
        optimizer.step()
        if update % 100 == 0 or update == 1:
            row = {
                "update": update,
                "training_weighted_window_nse": float(absolute_loss.detach()),
                **(
                    {
                        "absolute_weighted_window_nse": float(absolute_loss.detach()),
                        "paired_response_scaled_mse": float(paired_loss.detach()),
                        "training_total_loss": float(loss.detach()),
                    }
                    if paired is not None
                    else {}
                ),
                "gradient_norm": float(gradient),
                "wall_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    final = evaluate(model, held_states, held_actions, held_sources, args.wheel_weight)
    pair_final = (
        None
        if paired is None
        else {
            "train": evaluate_pairs(model, pair_train, args.paired_horizons, scales),
            "heldout": evaluate_pairs(model, pair_held, args.paired_horizons, scales),
        }
    )
    torch.save(
        {
            "schema": "supermimic.ollie.offline_wm.v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_config": asdict(config),
            "environment_config": checkpoint["environment_config"],
            "board_joint_names": BOARD_JOINT_NAMES,
            "robot_joint_names": G1_JOINT_NAMES,
            "updates": args.updates,
            "initialization_wm_sha256": manifest["resume_wm_sha256"],
            "manifest_sha256": digest((args.out_dir / "manifest.json").read_bytes()),
            "policy_trained": False,
            "ollie_success_certified": False,
        },
        args.out_dir / "model.pt",
    )
    diagnostics = {
        "initial_heldout": initial,
        "final_heldout": final,
        **(
            {
                "initial_paired": pair_initial,
                "final_paired": pair_final,
                "paired_loss_weight": args.paired_loss_weight,
            }
            if paired is not None
            else {}
        ),
        "history": history,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "window": 8,
        "wheel_weight": args.wheel_weight,
        "device": args.device,
        "wall_seconds": time.monotonic() - started,
        "ordinary_training_batch_indices_sha256": batch_digest.hexdigest(),
        "metric_definitions": "raw_nse = unweighted normalized 90-d squared error, not unscaled state MSE; weighted NSE downweights eight passive-wheel angle/velocity terms",
        "scope": "Actual-transition dynamics fitting only; failed episodes valid; expert labels unused; no policy or Ollie success claim",
    }
    (args.out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
