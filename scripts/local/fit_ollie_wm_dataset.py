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
from supermimic.ollie.learning import JointLearningConfig, JointWorldModel
from supermimic.ollie.physics import BOARD_JOINT_NAMES
from supermimic.ollie.state import STATE_SCHEMA, JointStateLayout


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
    heldout_groups = set()
    for source in range(sources):
        candidates = [
            key for key, members in groups.items() if any(ep.source == source for ep in members)
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
        "split_unit": "Complete content-grouped episodes; all duplicate replay episodes share one split",
    }


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
    windows, split = split_and_window(
        episodes, len(sources), args.window, args.seed, args.heldout_fraction
    )
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
        "initialization": "Load the base checkpoint's untrained world-model weights; policy weights are unused",
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
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model_lr)
    initial = evaluate(model, held_states, held_actions, held_sources, args.wheel_weight)
    history = []
    for update in range(1, args.updates + 1):
        choices = torch.randint(len(sources), (args.batch_size,), device=args.device)
        batch = torch.empty(args.batch_size, dtype=torch.long, device=args.device)
        for source, pool in enumerate(pools):
            mask = choices == source
            batch[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=args.device)]
        loss, _ = model.window_loss(
            train_states[:, batch], train_actions[:, batch], args.wheel_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(gradient):
            raise RuntimeError("Nonfinite world-model loss or gradient")
        optimizer.step()
        if update % 100 == 0 or update == 1:
            row = {
                "update": update,
                "training_weighted_window_nse": float(loss.detach()),
                "gradient_norm": float(gradient),
                "wall_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    final = evaluate(model, held_states, held_actions, held_sources, args.wheel_weight)
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
            "manifest_sha256": digest((args.out_dir / "manifest.json").read_bytes()),
            "policy_trained": False,
            "ollie_success_certified": False,
        },
        args.out_dir / "model.pt",
    )
    diagnostics = {
        "initial_heldout": initial,
        "final_heldout": final,
        "history": history,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "window": 8,
        "wheel_weight": args.wheel_weight,
        "device": args.device,
        "wall_seconds": time.monotonic() - started,
        "metric_definitions": "raw_nse = unweighted normalized 90-d squared error, not unscaled state MSE; weighted NSE downweights eight passive-wheel angle/velocity terms",
        "scope": "Actual-transition dynamics fitting only; failed episodes valid; expert labels unused; no policy or Ollie success claim",
    }
    (args.out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
