"""Frozen two-arm SuperMimic WM comparison on default-model real phase data.

State-only versus state plus train-RMS-normalized, phase-balanced board-VZ
action response. Original ContactWorldModel/JointStateSpec integration stays.
No actor, BPTT policy update, simulator transition, or teacher promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from scripts.local.collect_ollie_mu09_responses import digest, load_archive, write_json
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.contact_learning import ContactWorldModel, rollout
from supermimic.ollie.learning import JointLearningConfig
from supermimic.ollie.physics import BOARD_JOINT_NAMES, BOARD_XML
from supermimic.ollie.state import JointStateLayout

SCHEMA = "supermimic.ollie.phase_response_wm.v1"
DATA_SHA = "7e638c771d017ce6b225734ef08f75e24fc31c9e5ba2dba6fed3a1e8ff6d0a34"
SOURCE_SHA = "ac6effe28909ca875149e6f94cdb17b3b59b0152a4960dfff4ab7e38beb35526"
ROLES = (
    "preparation_supported",
    "tail_impact_neighborhood",
    "air_rising",
    "air_descending",
    "first_board_landing_neighborhood",
    "late_front_support_neighborhood",
)
ARMS = ("state_only", "state_plus_response")
SEED, UPDATES, BATCH, PAIRS_PER_ROLE = 94, 500, 64, 10
TOTAL_BUDGET, FIT_CUTOFF = 300.0, 270.0


def weight_hash(weights):
    value = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def atomic_save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_dataset(directory):
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    if digest(directory / "paired_responses.npz") != DATA_SHA:
        raise ValueError("Requires the fixed93 real default-model dataset")
    data = load_archive(directory / "paired_responses.npz")
    meta = json.loads(data["metadata_json"].item())
    for key, value in meta.items():
        if manifest.get(key) != value:
            raise ValueError(f"Archive/manifest mismatch:{key}")
    if (
        meta["schema"] != "supermimic.ollie.default_phase_warm_responses.v1"
        or meta["source_sha256"] != SOURCE_SHA
        or meta["foot_deck_friction"] != 0.6
        or meta["explicit_contact_pairs"] != 0
        or meta["control_dt"] != 0.02
        or meta["physics_dt"] != 0.0025
        or tuple(meta["robot_joint_names"]) != G1_JOINT_NAMES
        or tuple(meta["board_joint_names"]) != BOARD_JOINT_NAMES
        or meta["canonical_board_xml_sha256"] != digest(BOARD_XML)
        or digest(Path(meta["source"])) != SOURCE_SHA
    ):
        raise ValueError("Source/physics/material/joint/time provenance mismatch")
    for filename, row in manifest["files"].items():
        if digest(directory / filename) != row["sha256"]:
            raise ValueError(f"Frozen dataset file mismatch:{filename}")
    for key, shape in (
        ("state", (558, 9, 92)),
        ("action", (558, 8, 29)),
        ("post_forward_metrics", (558, 8, 8, 49)),
    ):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid finite/shape contract:{key}")
    if (
        data["state"].dtype != np.float64
        or not data["valid_complete"].all()
        or not data["transition_valid"].all()
    ):
        raise ValueError("Refuse float32 native differencing or incomplete/padded data")
    if not np.array_equal(data["train_mask"], ~data["heldout_mask"]):
        raise ValueError("Split must be complementary whole groups")
    pairs, roles = [], []
    for group, row in enumerate(meta["anchor_plan"]):
        ids = np.flatnonzero(data["group_id"] == group)
        if len(ids) != 31 or len(np.unique(data["heldout_mask"][ids])) != 1:
            raise ValueError("Expected31 siblings in one split")
        if bool(data["heldout_mask"][ids[0]]) != row["heldout"]:
            raise ValueError("Frozen whole-anchor split differs")
        if not np.array_equal(
            data["state"][ids, 0], np.repeat(data["state"][ids[:1], 0], 31, axis=0)
        ):
            raise ValueError("Paired branches must share the exact hot initial state")
        base = int(ids[data["is_baseline"][ids]][0])
        if not np.array_equal(
            data["actual_delta"][ids], data["action"][ids] - data["action"][base]
        ):
            raise ValueError("Saved true perturbations differ from controls")
        for joint in np.unique(data["joint_index"][ids[~data["is_baseline"][ids]]]):
            pair = []
            for sign in (1, -1):
                selected = ids[(data["joint_index"][ids] == joint) & (data["sign"][ids] == sign)]
                if len(selected) != 1:
                    raise ValueError("Missing positive/negative sibling")
                pair.append(int(selected[0]))
            pairs.append(pair)
            roles.append(ROLES.index(row["role"]))
    pairs, roles = np.asarray(pairs), np.asarray(roles)
    if pairs.shape != (270, 2):
        raise ValueError("Expected270 real joint pairs")
    # Subtract native float64 first; only the finished difference enters float32.
    response = data["state"][pairs[:, 0]][:, [1, 8], 84] - data["state"][pairs[:, 1]][:, [1, 8], 84]
    metrics = data["post_forward_metrics"]
    names = list(data["metric_names"])
    for name in ("overflow", "external_generalized_force_norm", "external_body_force_norm"):
        if np.any(metrics[..., names.index(name)] != 0):
            raise ValueError(f"Refuse nonzero{name}")
    quality = load_archive(directory / "quality.npz")
    return data, pairs, roles, response, quality, meta


def train_scales(response, roles, train_pair_mask):
    floors = np.array([0.001, 0.005])
    return np.stack(
        [
            np.maximum(
                np.sqrt(np.mean(response[train_pair_mask & (roles == g)] ** 2, axis=0)), floors
            )
            for g in range(6)
        ]
    )


def state_loss(model, states, actions):
    predicted, _ = rollout(model, states[:, 0], actions)
    error = model.spec.error_vector(predicted, states[:, 1:])
    weights = error.new_ones(error.shape[-1])
    weights[-8:] = 0.05
    return (error.square() * weights).sum(-1).mean() / weights.sum()


def response_loss(model, initial, plus_actions, minus_actions, target, roles, scales):
    positive, _ = rollout(model, initial, plus_actions)
    negative, _ = rollout(model, initial, minus_actions)
    predicted = positive[:, [0, 7], 84] - negative[:, [0, 7], 84]
    square = ((predicted - target) / scales[roles]).square()
    # Explicit role/horizon means; not a mixed-units48-coordinate average.
    return torch.stack([square[roles == g].mean(0) for g in range(6)]).mean()


def paired_commit(models, optimizers):
    """Commit both Adam steps or restore both weights and optimizer states."""
    before = {
        name: (
            copy.deepcopy(models[name].state_dict()),
            copy.deepcopy(optimizers[name].state_dict()),
        )
        for name in ARMS
    }
    try:
        for name in ARMS:
            optimizers[name].step()
    except BaseException:
        for name in ARMS:
            models[name].load_state_dict(before[name][0], strict=True)
            optimizers[name].load_state_dict(before[name][1])
        raise


def prepare(args):
    data, pairs, roles, response, quality, meta = load_dataset(args.dataset)
    train_pair = ~data["heldout_mask"][pairs[:, 0]]
    scales = train_scales(response, roles, train_pair)
    generator = torch.Generator().manual_seed(SEED)
    train_rows = np.flatnonzero(data["train_mask"])
    state_batches, pair_batches = [], []
    for _ in range(UPDATES):
        state_batches.append(
            train_rows[torch.randint(len(train_rows), (BATCH,), generator=generator).numpy()]
        )
        per_phase = []
        for g in range(6):
            choices = np.flatnonzero(train_pair & (roles == g))
            per_phase.append(
                choices[torch.randint(len(choices), (PAIRS_PER_ROLE,), generator=generator).numpy()]
            )
        selected = np.concatenate(per_phase)
        pair_batches.append(selected[torch.randperm(len(selected), generator=generator).numpy()])
    torch.manual_seed(SEED)
    initial = ContactWorldModel(
        JointStateLayout(BOARD_JOINT_NAMES), JointLearningConfig(hidden=(512, 512, 512))
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    atomic_save(
        args.out_dir / "initial.pt",
        {"schema": SCHEMA, "model": initial.state_dict(), "seed": SEED, "actor_updates": 0},
    )
    np.savez_compressed(
        args.out_dir / "frozen_batches.npz",
        state_rows=np.array(state_batches),
        pair_rows=np.array(pair_batches),
    )
    manifest = {
        "schema": SCHEMA,
        "collector_dataset": str(args.dataset.resolve()),
        "dataset_sha256": DATA_SHA,
        "dataset_manifest_sha256": digest(args.dataset / "dataset_manifest.json"),
        "source_sha256": SOURCE_SHA,
        "physics_metadata": meta,
        "fitter_sha256": digest(__file__),
        "initial_file_sha256": digest(args.out_dir / "initial.pt"),
        "initial_weights_sha256": weight_hash(initial.state_dict()),
        "frozen_batches_sha256": digest(args.out_dir / "frozen_batches.npz"),
        "roles": ROLES,
        "train_only_board_vz_scales_m_s": scales,
        "scale_rule": "max(train-role RMS real positive-minus-negative boardVZ,[.001,.005]m/s)",
        "losses": {
            "state_only": "original H8 normalized state error, wheel weight.05",
            "state_plus_response": "same state loss +1.0 * mean6roles2horizons of normalized paired boardVZ response squared error",
        },
        "auxiliary_weight_both_arms": 0.0,
        "original_codec_and_integrator_unchanged": True,
        "seed": SEED,
        "maximum_paired_updates": UPDATES,
        "batch_state": BATCH,
        "batch_pairs": 60,
        "pairs_per_role_per_update": 10,
        "hidden": [512, 512, 512],
        "learning_rate": 2e-4,
        "clip_gradient_norm": 1.0,
        "total_budget_s": TOTAL_BUDGET,
        "stop_accepting_paired_updates_s": FIT_CUTOFF,
        "common_arms_initialization": True,
        "optimizer_commit": "both arm Adam steps transactional with rollback; one shared paired-update counter",
        "gradient_diagnostics_after_paired_updates": [0, 100, 250, 500],
        "heldout_use": "final diagnostics only; never scales, batches, hyperparameters, stopping or checkpoint selection",
        "train_branches": 372,
        "heldout_branches": 186,
        "train_pairs": 180,
        "heldout_pairs": 90,
        "failed_quality_branches_included": int((~quality["penetration_and_joint_quality"]).sum()),
        "gates": {
            "response_rmse_max_fraction_zero": 0.8,
            "gain_min": 0.5,
            "gain_max": 1.5,
            "sign_min_fraction": 0.75,
            "sign_native_threshold_m_s": 1e-6,
            "H8_velocity_no_worse_than_control_and_constant_velocity": True,
            "position_no_more_than10percent_worse_than_control": True,
        },
        "scope": "single failed trajectory, nearby overlapping anchor windows, no airborne both-top or stable-landing teacher",
        "verified_teacher": False,
        "board_bptt_allowed": False,
        "actor_updates": 0,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "initial_weights_sha256": manifest["initial_weights_sha256"],
                "train_only_scales": scales.tolist(),
                "mode": "inspect",
                "optimizer_updates": 0,
            },
            indent=2,
        )
    )


def input_tensors(data, pairs, roles, response, scales, device):
    return {
        "states": torch.tensor(data["state"], dtype=torch.float32, device=device),
        "actions": torch.tensor(data["action"], dtype=torch.float32, device=device),
        "pairs": torch.tensor(pairs, device=device),
        "roles": torch.tensor(roles, device=device),
        "response": torch.tensor(response, dtype=torch.float32, device=device),
        "scales": torch.tensor(scales, dtype=torch.float32, device=device),
    }


def losses(model, values, state_rows, pair_rows, with_response):
    state = state_loss(model, values["states"][state_rows], values["actions"][state_rows])
    if not with_response:
        return state, state.new_zeros(())
    pair = values["pairs"][pair_rows]
    effect = response_loss(
        model,
        values["states"][pair[:, 0], 0],
        values["actions"][pair[:, 0]],
        values["actions"][pair[:, 1]],
        values["response"][pair_rows],
        values["roles"][pair_rows],
        values["scales"],
    )
    return state, effect


def gradient_diagnostic(model, values, state_rows, pair_rows, update):
    state, effect = losses(model, values, state_rows, pair_rows, True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    sg = torch.autograd.grad(state, parameters, allow_unused=True)
    eg = torch.autograd.grad(effect, parameters, allow_unused=True)
    state_norm = torch.sqrt(
        sum((g.square().sum() for g in sg if g is not None), state.new_zeros(()))
    )
    effect_norm = torch.sqrt(
        sum((g.square().sum() for g in eg if g is not None), state.new_zeros(()))
    )
    dot = sum(
        (
            a.flatten().dot(b.flatten())
            for a, b in zip(sg, eg, strict=True)
            if a is not None and b is not None
        ),
        state.new_zeros(()),
    )
    return {
        "after_paired_update": update,
        "state_loss": float(state.detach()),
        "response_loss": float(effect.detach()),
        "state_gradient_norm": float(state_norm),
        "response_gradient_norm": float(effect_norm),
        "gradient_cosine": float(dot / (state_norm * effect_norm).clamp_min(1e-30)),
        "used_to_change_parameters": False,
    }


@torch.no_grad()
def predictions(model, states, actions, batch=64):
    pieces = []
    for first in range(0, len(states), batch):
        value, _ = rollout(model, states[first : first + batch, 0], actions[first : first + batch])
        pieces.append(value.cpu().numpy())
    return np.concatenate(pieces)


def state_metrics(predicted, truth, rows):
    result = {}
    for h in (1, 8):
        result[f"H{h}"] = {}
        for name, section in (("board_xyz_m", slice(71, 74)), ("board_Vxyz_m_s", slice(82, 85))):
            error = predicted[rows, h - 1, section] - truth[rows, h, section]
            result[f"H{h}"][name] = {
                "rmse_axes": np.sqrt(np.mean(error**2, axis=0)),
                "rmse_vector": float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
            }
    return result


def response_metrics(predicted, truth, pair_rows, pairs):
    selected = pairs[pair_rows]
    estimate = predicted[selected[:, 0]][:, [0, 7], 84] - predicted[selected[:, 1]][:, [0, 7], 84]
    target = truth[pair_rows]
    result = {}
    for k, h in enumerate((1, 8)):
        p, t = estimate[:, k], target[:, k]
        energy = np.sum(t * t)
        valid = abs(t) > 1e-6
        result[f"H{h}"] = {
            "pairs": len(t),
            "rmse_m_s": float(np.sqrt(np.mean((p - t) ** 2))),
            "zero_response_rmse_m_s": float(np.sqrt(np.mean(t * t))),
            "gain_projection": float(np.sum(p * t) / energy) if energy > 0 else None,
            "sign_samples": int(valid.sum()),
            "sign_agreement": float(np.mean(np.sign(p[valid]) == np.sign(t[valid])))
            if valid.any()
            else None,
        }
    return result


def diagnostics(predicted, data, pairs, roles, response):
    result = {}
    for split, mask in (("train", data["train_mask"]), ("heldout", data["heldout_mask"])):
        pmask = mask[pairs[:, 0]]
        groups = {
            "all": np.ones(len(pairs), dtype=bool),
            **{role: roles == i for i, role in enumerate(ROLES)},
            "tail": roles == 1,
            "air": np.isin(roles, [2, 3]),
            "landing": np.isin(roles, [4, 5]),
        }
        result[split] = {
            "state": state_metrics(predicted, data["state"], mask),
            "paired": {
                name: response_metrics(predicted, response, np.flatnonzero(pmask & chosen), pairs)
                for name, chosen in groups.items()
            },
        }
    return result


def gates(metrics, baseline, constant_velocity):
    paired = {}
    for group in ("all", "tail", "air", "landing"):
        paired[group] = {}
        for h in ("H1", "H8"):
            m = metrics["heldout"]["paired"][group][h]
            paired[group][h] = {
                "rmse_pass": m["rmse_m_s"] <= 0.8 * m["zero_response_rmse_m_s"],
                "gain_pass": m["gain_projection"] is not None
                and 0.5 <= m["gain_projection"] <= 1.5,
                "sign_pass": m["sign_agreement"] is not None and m["sign_agreement"] >= 0.75,
            }
    h8 = metrics["heldout"]["state"]["H8"]["board_Vxyz_m_s"]["rmse_vector"]
    result = {
        "paired": paired,
        "H8_velocity_no_worse_than_control": h8
        <= baseline["heldout"]["state"]["H8"]["board_Vxyz_m_s"]["rmse_vector"],
        "H8_velocity_no_worse_than_constant_velocity": h8
        <= constant_velocity["heldout"]["H8"]["board_Vxyz_m_s"]["rmse_vector"],
        "position_no_more_than10percent_worse_than_control": all(
            metrics["heldout"]["state"][h]["board_xyz_m"]["rmse_vector"]
            <= 1.1 * baseline["heldout"]["state"][h]["board_xyz_m"]["rmse_vector"]
            for h in ("H1", "H8")
        ),
    }
    result["all_preregistered_local_gates_pass"] = all(
        v for group in paired.values() for horizon in group.values() for v in horizon.values()
    ) and all(v for v in result.values() if isinstance(v, bool))
    result["board_bptt_allowed"] = False
    return result


def run(args):
    started = time.monotonic()
    manifest = json.loads((args.out_dir / "manifest.json").read_text())
    if (
        digest(__file__) != manifest["fitter_sha256"]
        or digest(args.out_dir / "initial.pt") != manifest["initial_file_sha256"]
        or digest(args.out_dir / "frozen_batches.npz") != manifest["frozen_batches_sha256"]
    ):
        raise ValueError("Frozen fitter/initial/batches changed")
    if (args.out_dir / "result.json").exists() or any(
        (args.out_dir / f"{name}.pt").exists() for name in ARMS
    ):
        raise ValueError("Never restart or overwrite the unique comparison")
    data, pairs, roles, response, quality, _meta = load_dataset(args.dataset)
    if str(args.dataset.resolve()) != manifest["collector_dataset"]:
        raise ValueError("Frozen dataset location changed")
    scales = train_scales(response, roles, ~data["heldout_mask"][pairs[:, 0]])
    if not np.array_equal(scales, np.array(manifest["train_only_board_vz_scales_m_s"])):
        raise ValueError("Frozen train-only scales changed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = torch.load(args.out_dir / "initial.pt", map_location="cpu", weights_only=True)
    if weight_hash(initial["model"]) != manifest["initial_weights_sha256"]:
        raise ValueError("Frozen initial weight hash changed")
    values = input_tensors(data, pairs, roles, response, scales, args.device)
    models, optimizers = {}, {}
    for name in ARMS:
        models[name] = ContactWorldModel(
            JointStateLayout(BOARD_JOINT_NAMES),
            JointLearningConfig(hidden=(512, 512, 512)),
            args.device,
        )
        models[name].load_state_dict(initial["model"], strict=True)
        optimizers[name] = torch.optim.Adam(models[name].parameters(), lr=2e-4, foreach=False)
    batches = load_archive(args.out_dir / "frozen_batches.npz")
    state_rows = torch.tensor(batches["state_rows"], device=args.device)
    pair_rows = torch.tensor(batches["pair_rows"], device=args.device)
    updates, history, gradient_records = 0, [], []
    reason = "maximum_paired_updates"
    interrupted = {"signal": None}
    handlers = {}

    def stop_on_boundary(sig, _frame):
        interrupted["signal"] = int(sig)

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, stop_on_boundary)
    batch_hash = {name: hashlib.sha256() for name in ARMS}
    try:
        gradient_records.append(
            gradient_diagnostic(models[ARMS[1]], values, state_rows[0], pair_rows[0], 0)
        )
        for update in range(UPDATES):
            if interrupted["signal"] is not None:
                reason = f"signal_at_paired_update_boundary:{interrupted['signal']}"
                break
            if time.monotonic() - started >= FIT_CUTOFF:
                reason = "wall_budget_no_new_pair_after270s"
                break
            row = {"paired_update": update + 1}
            for name in ARMS:
                model = models[name]
                optimizers[name].zero_grad(set_to_none=True)
                state, effect = losses(
                    model, values, state_rows[update], pair_rows[update], name == ARMS[1]
                )
                total = state + effect
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite loss:{name}")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                row[name] = {
                    "state_loss": float(state.detach()),
                    "response_loss": float(effect.detach()),
                    "total_loss": float(total.detach()),
                    "gradient_norm_before_clip": float(norm),
                }
            paired_commit(models, optimizers)
            updates += 1
            blob = batches["state_rows"][update].tobytes() + batches["pair_rows"][update].tobytes()
            for name in ARMS:
                batch_hash[name].update(blob)
            history.append(row)
            if updates in (100, 250, 500) and time.monotonic() - started < FIT_CUTOFF:
                gradient_records.append(
                    gradient_diagnostic(
                        models[ARMS[1]], values, state_rows[0], pair_rows[0], updates
                    )
                )
    except BaseException as error:
        reason = f"error:{type(error).__name__}:{error}"
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for name in ARMS:
            atomic_save(
                args.out_dir / f"{name}.pt",
                {
                    "schema": SCHEMA,
                    "arm": name,
                    "model": models[name].state_dict(),
                    "optimizer": optimizers[name].state_dict(),
                    "updates": updates,
                    "seed": SEED,
                    "manifest": manifest,
                    "stop_reason": reason,
                    "batch_sha256": batch_hash[name].hexdigest(),
                    "actor_updates": 0,
                    "board_bptt_allowed": False,
                },
            )
        write_json(args.out_dir / "history.json", history)
        write_json(args.out_dir / "gradient_diagnostics.json", gradient_records)
        write_json(
            args.out_dir / "training_completion.json",
            {
                "paired_updates_both_arms": updates,
                "arm_batch_sha256": {name: batch_hash[name].hexdigest() for name in ARMS},
                "stop_reason": reason,
                "wall_seconds_at_checkpoint_save": time.monotonic() - started,
            },
        )
    all_metrics, prediction_arrays = {}, {}
    if time.monotonic() - started < TOTAL_BUDGET - 5:
        for name in ARMS:
            prediction_arrays[name] = predictions(models[name], values["states"], values["actions"])
            all_metrics[name] = diagnostics(prediction_arrays[name], data, pairs, roles, response)
        np.savez_compressed(args.out_dir / "final_gpu_predictions.npz", **prediction_arrays)
    cv = np.repeat(data["state"][:, 0:1], 8, axis=1)
    cv[:, :, 71:74] = (
        data["state"][:, 0:1, 71:74]
        + np.arange(1, 9)[None, :, None] * 0.02 * data["state"][:, 0:1, 82:85]
    )
    constant_velocity = {
        split: state_metrics(cv, data["state"], mask)
        for split, mask in (("train", data["train_mask"]), ("heldout", data["heldout_mask"]))
    }
    result = {
        "schema": SCHEMA,
        "paired_updates_both_arms": updates,
        "stop_reason": reason,
        "arms": all_metrics,
        "constant_velocity": constant_velocity,
        "arm_batch_sha256": {name: batch_hash[name].hexdigest() for name in ARMS},
        "initial_weights_sha256": manifest["initial_weights_sha256"],
        "final_weights_sha256": {name: weight_hash(models[name].state_dict()) for name in ARMS},
        "train_only_scales_m_s": scales,
        "failed_quality_branches_included": int((~quality["penetration_and_joint_quality"]).sum()),
        "wall_seconds": time.monotonic() - started,
        "wall_budget_s": TOTAL_BUDGET,
        "final_gpu_diagnostics_completed": len(all_metrics) == 2,
        "verified_teacher": False,
        "actor_updates": 0,
        "board_bptt_allowed": False,
        "preregistered_gates": gates(all_metrics[ARMS[1]], all_metrics[ARMS[0]], constant_velocity)
        if len(all_metrics) == 2
        else None,
    }
    write_json(args.out_dir / "result.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("arms", "constant_velocity", "train_only_scales_m_s")
            },
            indent=2,
        )
    )


def readback(args):
    started = time.monotonic()
    manifest = json.loads((args.out_dir / "manifest.json").read_text())
    recorded = json.loads((args.out_dir / "result.json").read_text())
    data, pairs, roles, response, _, _ = load_dataset(args.dataset)
    torch.set_num_threads(2)
    scales = np.asarray(manifest["train_only_board_vz_scales_m_s"])
    values = input_tensors(data, pairs, roles, response, scales, "cpu")
    cpu, errors = {}, {}
    saved = (
        load_archive(args.out_dir / "final_gpu_predictions.npz")
        if (args.out_dir / "final_gpu_predictions.npz").exists()
        else None
    )
    for name in ARMS:
        checkpoint = torch.load(args.out_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        if (
            checkpoint["schema"] != SCHEMA
            or checkpoint["updates"] != recorded["paired_updates_both_arms"]
            or weight_hash(checkpoint["model"]) != recorded["final_weights_sha256"][name]
        ):
            raise ValueError("Checkpoint/readback hash, schema or paired-update mismatch")
        model = ContactWorldModel(
            JointStateLayout(BOARD_JOINT_NAMES), JointLearningConfig(hidden=(512, 512, 512))
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        pred = predictions(model, values["states"], values["actions"])
        cpu[name] = diagnostics(pred, data, pairs, roles, response)
        errors[name] = {
            "all_state92_prediction_max_abs_cpu_gpu": float(abs(pred - saved[name]).max())
            if saved is not None
            else None,
            "board_xyz_prediction_max_abs_cpu_gpu": float(
                abs(pred[:, :, 71:74] - saved[name][:, :, 71:74]).max()
            )
            if saved is not None
            else None,
            "board_Vxyz_prediction_max_abs_cpu_gpu": float(
                abs(pred[:, :, 82:85] - saved[name][:, :, 82:85]).max()
            )
            if saved is not None
            else None,
        }
    output = {
        "weights_only_checkpoints_read_and_hashes_exact": True,
        "same_paired_updates_and_batch_hash": len(set(recorded["arm_batch_sha256"].values())) == 1,
        "arms": cpu,
        "cpu_gpu_prediction_errors": errors,
        "cpu_preregistered_gates": gates(cpu[ARMS[1]], cpu[ARMS[0]], recorded["constant_velocity"]),
        "wall_seconds": time.monotonic() - started,
        "new_optimizer_updates": 0,
        "new_simulation_steps": 0,
        "actor_updates": 0,
        "board_bptt_allowed": False,
    }
    write_json(args.out_dir / "independent_cpu_readback.json", output)
    print(json.dumps({key: value for key, value in output.items() if key != "arms"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "run", "readback"), required=True)
    parser.add_argument("--dataset", type=Path, default=Path("logs/ollie_phase_responses_01"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"inspect": prepare, "run": run, "readback": readback}[args.mode](args)


if __name__ == "__main__":
    main()
