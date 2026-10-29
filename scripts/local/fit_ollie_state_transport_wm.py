"""Add real different-state response transport to the frozen96 TF1 objective.

Both arms reuse96 INITIAL weights and batches, never its trained weights.
Transport uses actual different states k1..7 and identical subsequent actions.
No actor, simulator, codec change or policy BPTT is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from scripts.local import fit_ollie_transition_response_wm as previous
from scripts.local.collect_ollie_mu09_responses import digest, load_archive, write_json

old = previous.old
SCHEMA = "supermimic.ollie.state_transport_wm.v1"
ARMS = ("action_baseline", "action_plus_transport")
PREVIOUS_RUN = Path("logs/ollie_transition_response_wm_01")
PREVIOUS_MANIFEST_SHA = "ff1440968e1b1af5e8733ed529c4298ee8b8117a1336e6e0cc501b88608a53df"
CORE_FILES = (
    "src/supermimic/ollie/state.py",
    "src/supermimic/ollie/learning.py",
    "src/supermimic/ollie/contact_learning.py",
    "src/supermimic/world_model/dynamics.py",
)


def transport_targets(data, pairs):
    if not np.array_equal(data["action"][pairs[:, 0], 1:], data["action"][pairs[:, 1], 1:]):
        raise ValueError("Transport k1..7 requires identical subsequent actual actions")
    return data["state"][pairs[:, 0], 2:, 84] - data["state"][pairs[:, 1], 2:, 84]


def transport_scales(target, roles, train):
    # Actual finite state differences, float64, train only; no heldout tuning.
    return np.stack(
        [
            np.maximum(np.sqrt(np.mean(target[train & (roles == g)] ** 2, axis=0)), 0.001)
            for g in range(6)
        ]
    )


def transport_loss(model, positive_current, negative_current, same_actions, target, roles, scales):
    if positive_current.shape[1:] != (7, 92) or same_actions.shape[1:] != (7, 29):
        raise ValueError("Expect seven true different-state k1..7 transitions")
    positive = model.step(positive_current.reshape(-1, 92), same_actions.reshape(-1, 29)).reshape(
        -1, 7, 92
    )
    negative = model.step(negative_current.reshape(-1, 92), same_actions.reshape(-1, 29)).reshape(
        -1, 7, 92
    )
    response = positive[:, :, 84] - negative[:, :, 84]
    square = ((response - target) / scales[roles]).square()
    # Equal weight for every role and every real input index k1..7.
    return torch.stack([square[roles == g].mean(0) for g in range(6)]).mean()


def make_values(data, pairs, roles, response, h1_scales, t_target, t_scale, device):
    result = previous.make_values(data, pairs, roles, response, h1_scales, device)
    result["transport_target"] = torch.tensor(t_target, dtype=torch.float32, device=device)
    result["transport_scale"] = torch.tensor(t_scale, dtype=torch.float32, device=device)
    return result


def losses(model, values, rows, pair_rows, with_transport):
    state, action = previous.losses(model, values, rows, pair_rows, True)
    if not with_transport:
        return state, action, state.new_zeros(())
    pairs = values["pairs"][pair_rows]
    transport = transport_loss(
        model,
        values["states"][pairs[:, 0], 1:8],
        values["states"][pairs[:, 1], 1:8],
        values["actions"][pairs[:, 0], 1:8],
        values["transport_target"][pair_rows],
        values["roles"][pair_rows],
        values["transport_scale"],
    )
    return state, action, transport


def paired_commit(models, optimizers):
    old.paired_commit(
        {a: models[b] for a, b in zip(old.ARMS, ARMS, strict=True)},
        {a: optimizers[b] for a, b in zip(old.ARMS, ARMS, strict=True)},
    )


def prepare(args):
    if digest(PREVIOUS_RUN / "manifest.json") != PREVIOUS_MANIFEST_SHA:
        raise ValueError("Requires frozen96 initial/batch provenance")
    pm = json.loads((PREVIOUS_RUN / "manifest.json").read_text())
    data, pairs, roles, response, quality, meta = old.load_dataset(args.dataset)
    train = ~data["heldout_mask"][pairs[:, 0]]
    h1 = previous.frozen_h1_scales(response, roles, train)
    target = transport_targets(data, pairs)
    scales = transport_scales(target, roles, train)
    initial = torch.load(PREVIOUS_RUN / "initial.pt", weights_only=True, map_location="cpu")
    batches = load_archive(PREVIOUS_RUN / "frozen_batches.npz")
    if (
        digest(PREVIOUS_RUN / "initial.pt") != pm["initial_file_sha256"]
        or old.weight_hash(initial["model"]) != pm["initial_weights_sha256"]
        or digest(PREVIOUS_RUN / "frozen_batches.npz") != pm["frozen_batches_sha256"]
    ):
        raise ValueError("Frozen96 initial weights or batches changed")
    if (
        not np.repeat(data["train_mask"], 8)[batches["transition_rows"]].all()
        or not data["train_mask"][pairs[batches["pair_rows"], 0]].all()
    ):
        raise ValueError("Frozen batches include heldout rows")
    if any(
        not np.array_equal(np.bincount(roles[row], minlength=6), np.full(6, 10))
        for row in batches["pair_rows"]
    ):
        raise ValueError("Each pair batch requires ten pairs per role")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    old.atomic_save(
        args.out_dir / "initial.pt",
        {"schema": SCHEMA, "model": initial["model"], "seed": 96, "actor_updates": 0},
    )
    np.savez_compressed(args.out_dir / "frozen_batches.npz", **batches)
    manifest = {
        "schema": SCHEMA,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": old.DATA_SHA,
        "dataset_manifest_sha256": digest(args.dataset / "dataset_manifest.json"),
        "physics_metadata": meta,
        "fitter_sha256": digest(__file__),
        "unchanged96_fitter_sha256": digest(previous.__file__),
        "unchanged94_fitter_sha256": digest(old.__file__),
        "core_code_sha256": {name: digest(name) for name in CORE_FILES},
        "previous96_manifest_sha256": PREVIOUS_MANIFEST_SHA,
        "previous96_initial_weights_sha256": pm["initial_weights_sha256"],
        "initial_file_sha256": digest(args.out_dir / "initial.pt"),
        "initial_weights_sha256": old.weight_hash(initial["model"]),
        "frozen_batches_sha256": digest(args.out_dir / "frozen_batches.npz"),
        "seed": 96,
        "reuse96_initial_not_trained_weights": True,
        "reuse_all96_TF1_and_H1_batches": True,
        "H1_scales_frozen94_m_s": h1,
        "transport_scales_train_role_k_m_s": scales,
        "transport_scale_rule": "train-only per-role/per-k RMS true next boardVZ positive-minus-negative finite state response, floor.001m/s",
        "losses": {
            "action_baseline": "frozen96 TF1+.25 H1 same-state action-response",
            "action_plus_transport": "same A+.25 mean6roles7k normalized actual different-state one-step boardVZ response transport",
        },
        "transport_contract": "actual state+ and state- at k1..7 with identical subsequent actions; target actual next-state boardVZ difference at k+1; never same-state action partial or infinitesimal Jacobian",
        "TF1_weight": 1.0,
        "H1_action_weight": 0.25,
        "state_transport_weight_B": 0.25,
        "free_rollout_loss_weight": 0.0,
        "maximum_paired_updates": 500,
        "transition_batch": 64,
        "pair_batch": 60,
        "pairs_per_role": 10,
        "hidden": [512, 512, 512],
        "learning_rate": 2e-4,
        "clip_grad_norm": 1.0,
        "total_budget_s": 300.0,
        "stop_new_pair_after_s": 270.0,
        "pair_commit": "both Adam steps or rollback both weights/optimizer states",
        "gradient_diagnostic_updates": [0, 100, 250, 500],
        "planned_real_input_coverage": previous.input_coverage(data, batches["transition_rows"]),
        "quality_failed_branches_retained": int((~quality["penetration_and_joint_quality"]).sum()),
        "heldout_use": "final diagnostics only; no scales, batches, hyperparameter tuning, stopping or checkpoint selection",
        "gates": "native free H1/H8 and different-state transport H2..H8: RMSE<=.8zero, gain.5..1.5, sign>=.75 for all/six roles/tail/air/landing; free H8 velocity<=A and CV; free H1/H8 position<=1.1A",
        "scope": "single failed93 trajectory, nearby overlapping anchors; no airborne both-top or stable-landing teacher",
        "verified_teacher": False,
        "actor_updates": 0,
        "board_bptt_allowed": False,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "initial_weights_sha256": manifest["initial_weights_sha256"],
                "reuse96_initial_and_batches": True,
                "transport_scales": scales.tolist(),
                "optimizer_updates": 0,
            },
            indent=2,
        )
    )


def gradient_diagnostic(model, values, rows, pair_rows, update):
    state, action, transport = losses(model, values, rows, pair_rows, True)
    params = list(model.parameters())
    gradients = [
        torch.autograd.grad(loss, params, allow_unused=True) for loss in (state, action, transport)
    ]
    vectors = [
        torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for p, g in zip(params, parts, strict=True)
            ]
        )
        for parts in gradients
    ]
    norms = [float(v.norm()) for v in vectors]
    cosine = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
    return {
        "after_paired_update": update,
        "TF1_loss": float(state.detach()),
        "H1_loss_unweighted": float(action.detach()),
        "transport_loss_unweighted": float(transport.detach()),
        "gradient_norms_unweighted": {"TF1": norms[0], "H1": norms[1], "transport": norms[2]},
        "gradient_norms_weighted": {
            "TF1": norms[0],
            "H1": 0.25 * norms[1],
            "transport": 0.25 * norms[2],
        },
        "cos_TF1_H1": cosine(vectors[0], vectors[1]),
        "cos_TF1_transport": cosine(vectors[0], vectors[2]),
        "cos_H1_transport": cosine(vectors[1], vectors[2]),
        "used_for_tuning": False,
    }


@torch.no_grad()
def true_state_predictions(model, values):
    outputs = []
    for start in range(0, len(values["current"]), 256):
        outputs.append(
            model.step(
                values["current"][start : start + 256],
                values["transition_actions"][start : start + 256],
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs).reshape(-1, 8, 92)


def transport_metrics(predicted, data, pairs, roles):
    output = {}
    groups = {
        "all": np.ones(len(pairs), dtype=bool),
        **{name: roles == g for g, name in enumerate(old.ROLES)},
        "tail": roles == 1,
        "air": np.isin(roles, [2, 3]),
        "landing": np.isin(roles, [4, 5]),
    }
    for split, mask in (("train", data["train_mask"]), ("heldout", data["heldout_mask"])):
        output[split] = {}
        for name, group in groups.items():
            chosen = pairs[mask[pairs[:, 0]] & group]
            output[split][name] = {}
            for h in range(2, 9):
                target = data["state"][chosen[:, 0], h, 84] - data["state"][chosen[:, 1], h, 84]
                estimate = predicted[chosen[:, 0], h - 1, 84] - predicted[chosen[:, 1], h - 1, 84]
                valid = abs(target) > 1e-6
                energy = np.dot(target, target)
                output[split][name][f"H{h}"] = {
                    "pairs": len(chosen),
                    "rmse_m_s": float(np.sqrt(np.mean((estimate - target) ** 2))),
                    "zero_response_rmse_m_s": float(np.sqrt(np.mean(target**2))),
                    "gain_projection": float(np.dot(estimate, target) / energy)
                    if energy > 0
                    else None,
                    "sign_samples": int(valid.sum()),
                    "sign_agreement": float(
                        np.mean(np.sign(estimate[valid]) == np.sign(target[valid]))
                    )
                    if valid.any()
                    else None,
                    "input_contract": "different real states k=h-1, same next actual action",
                }
    return output


def response_gate(record):
    return {
        "rmse_pass": record["rmse_m_s"] <= 0.8 * record["zero_response_rmse_m_s"],
        "gain_pass": record["gain_projection"] is not None
        and 0.5 <= record["gain_projection"] <= 1.5,
        "sign_pass": record["sign_agreement"] is not None and record["sign_agreement"] >= 0.75,
    }


def gates(free, transport, baseline, cv):
    result = old.gates(free, baseline, cv)
    result["free_all_six_role_gates"] = {
        role: {h: response_gate(free["heldout"]["paired"][role][h]) for h in ("H1", "H8")}
        for role in old.ROLES
    }
    result["native_state_transport_gates"] = {
        group: {h: response_gate(record) for h, record in records.items()}
        for group, records in transport["heldout"].items()
    }
    role_pass = all(
        v
        for record in result["free_all_six_role_gates"].values()
        for h in record.values()
        for v in h.values()
    )
    transport_pass = all(
        v
        for record in result["native_state_transport_gates"].values()
        for h in record.values()
        for v in h.values()
    )
    result["all_preregistered_local_gates_pass"] = (
        result["all_preregistered_local_gates_pass"] and role_pass and transport_pass
    )
    result["board_bptt_allowed"] = False
    return result


def run(args):
    started = time.monotonic()
    m = json.loads((args.out_dir / "manifest.json").read_text())
    paths = [
        (Path(__file__), "fitter_sha256"),
        (Path(previous.__file__), "unchanged96_fitter_sha256"),
        (Path(old.__file__), "unchanged94_fitter_sha256"),
        (args.out_dir / "initial.pt", "initial_file_sha256"),
        (args.out_dir / "frozen_batches.npz", "frozen_batches_sha256"),
    ]
    if any(digest(path) != m[key] for path, key in paths) or any(
        digest(name) != value for name, value in m["core_code_sha256"].items()
    ):
        raise ValueError("Frozen input/code/core model changed")
    if (args.out_dir / "result.json").exists() or any(
        (args.out_dir / f"{arm}.pt").exists() for arm in ARMS
    ):
        raise ValueError("Never restart/overwrite the unique98 fit")
    data, pairs, roles, response, quality, _ = old.load_dataset(args.dataset)
    target = transport_targets(data, pairs)
    train = ~data["heldout_mask"][pairs[:, 0]]
    h1 = previous.frozen_h1_scales(response, roles, train)
    scales = transport_scales(target, roles, train)
    if str(args.dataset.resolve()) != m["dataset"] or not np.array_equal(
        scales, np.array(m["transport_scales_train_role_k_m_s"])
    ):
        raise ValueError("Frozen dataset/transport scales changed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = torch.load(args.out_dir / "initial.pt", weights_only=True, map_location="cpu")
    if old.weight_hash(initial["model"]) != m["initial_weights_sha256"]:
        raise ValueError("Initial weight hash changed")
    v = make_values(data, pairs, roles, response, h1, target, scales, args.device)
    models = {}
    optimizers = {}
    for arm in ARMS:
        models[arm] = old.ContactWorldModel(
            old.JointStateLayout(old.BOARD_JOINT_NAMES),
            old.JointLearningConfig(hidden=(512, 512, 512)),
            args.device,
        )
        models[arm].load_state_dict(initial["model"], strict=True)
        optimizers[arm] = torch.optim.Adam(models[arm].parameters(), lr=2e-4, foreach=False)
    batches = load_archive(args.out_dir / "frozen_batches.npz")
    tb = torch.tensor(batches["transition_rows"], device=args.device)
    pb = torch.tensor(batches["pair_rows"], device=args.device)
    updates = 0
    history = []
    gradients = []
    reason = "maximum_paired_updates"
    hashes = {a: hashlib.sha256() for a in ARMS}
    stopped = {"signal": None}
    handlers = {}

    def stop(sig, _frame):
        stopped["signal"] = int(sig)

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, stop)
    try:
        gradients.append(gradient_diagnostic(models[ARMS[1]], v, tb[0], pb[0], 0))
        for update in range(500):
            if stopped["signal"] is not None:
                reason = f"signal_at_pair_boundary:{stopped['signal']}"
                break
            if time.monotonic() - started >= 270:
                reason = "wall_budget_no_new_pair_after270s"
                break
            row = {"paired_update": update + 1}
            for arm in ARMS:
                optimizers[arm].zero_grad(set_to_none=True)
                state, action, transport = losses(
                    models[arm], v, tb[update], pb[update], arm == ARMS[1]
                )
                total = state + 0.25 * action
                if arm == ARMS[1]:
                    total = total + 0.25 * transport
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite98 loss:{arm}")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    models[arm].parameters(), 1.0, error_if_nonfinite=True
                )
                row[arm] = {
                    "TF1_loss": float(state.detach()),
                    "H1_loss_unweighted": float(action.detach()),
                    "transport_loss_unweighted": float(transport.detach()),
                    "total_loss": float(total.detach()),
                    "gradient_norm_before_clip": float(norm),
                }
            paired_commit(models, optimizers)
            updates += 1
            blob = (
                batches["transition_rows"][update].tobytes()
                + batches["pair_rows"][update].tobytes()
            )
            for arm in ARMS:
                hashes[arm].update(blob)
            history.append(row)
            if updates in (100, 250, 500) and time.monotonic() - started < 270:
                gradients.append(gradient_diagnostic(models[ARMS[1]], v, tb[0], pb[0], updates))
    except BaseException as error:
        reason = f"error:{type(error).__name__}:{error}"
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for arm in ARMS:
            old.atomic_save(
                args.out_dir / f"{arm}.pt",
                {
                    "schema": SCHEMA,
                    "arm": arm,
                    "model": models[arm].state_dict(),
                    "optimizer": optimizers[arm].state_dict(),
                    "updates": updates,
                    "stop_reason": reason,
                    "batch_sha256": hashes[arm].hexdigest(),
                    "manifest": m,
                    "actor_updates": 0,
                    "board_bptt_allowed": False,
                },
            )
        write_json(args.out_dir / "history.json", history)
        write_json(args.out_dir / "gradient_diagnostics.json", gradients)
        write_json(
            args.out_dir / "training_completion.json",
            {
                "paired_updates_both_arms": updates,
                "stop_reason": reason,
                "arm_batch_sha256": {a: hashes[a].hexdigest() for a in ARMS},
                "wall_at_save_s": time.monotonic() - started,
            },
        )
    free_pred = {}
    native_pred = {}
    metrics = {}
    native_metrics = {}
    if time.monotonic() - started < 290:
        for arm in ARMS:
            free_pred[arm] = old.predictions(models[arm], v["states"], v["actions"])
            metrics[arm] = old.diagnostics(free_pred[arm], data, pairs, roles, response)
            native_pred[arm] = true_state_predictions(models[arm], v)
            native_metrics[arm] = transport_metrics(native_pred[arm], data, pairs, roles)
        np.savez_compressed(args.out_dir / "final_free_gpu_predictions.npz", **free_pred)
        np.savez_compressed(args.out_dir / "final_true_input_gpu_predictions.npz", **native_pred)
    cv = previous.constant_velocity_metrics(data)
    final_hash = {arm: old.weight_hash(models[arm].state_dict()) for arm in ARMS}
    old96 = json.loads((PREVIOUS_RUN / "result.json").read_text())
    result = {
        "schema": SCHEMA,
        "paired_updates_both_arms": updates,
        "stop_reason": reason,
        "arms_free_rollout": metrics,
        "arms_native_state_transport": native_metrics,
        "constant_velocity": cv,
        "initial_weights_sha256": m["initial_weights_sha256"],
        "arm_batch_sha256": {a: hashes[a].hexdigest() for a in ARMS},
        "final_weights_sha256": final_hash,
        "baseline_matches96_final_tensor_hash": final_hash[ARMS[0]]
        == old96["final_weights_sha256"][previous.ARMS[1]],
        "actual_true_input_coverage": previous.input_coverage(
            data, batches["transition_rows"][:updates]
        ),
        "preregistered_gates": gates(
            metrics[ARMS[1]], native_metrics[ARMS[1]], metrics[ARMS[0]], cv
        )
        if len(metrics) == 2
        else None,
        "final_diagnostics_completed": len(metrics) == 2,
        "quality_failed_branches_retained": int((~quality["penetration_and_joint_quality"]).sum()),
        "wall_seconds": time.monotonic() - started,
        "wall_budget_s": 300.0,
        "actor_updates": 0,
        "board_bptt_allowed": False,
        "verified_teacher": False,
    }
    write_json(args.out_dir / "result.json", result)
    print(
        json.dumps(
            {
                k: value
                for k, value in result.items()
                if k
                not in (
                    "arms_free_rollout",
                    "arms_native_state_transport",
                    "constant_velocity",
                    "preregistered_gates",
                )
            },
            indent=2,
        )
    )
    print(
        "all_preregistered_gates_pass",
        result["preregistered_gates"]["all_preregistered_local_gates_pass"]
        if result["preregistered_gates"]
        else None,
    )


def readback(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    m = json.loads((args.out_dir / "manifest.json").read_text())
    r = json.loads((args.out_dir / "result.json").read_text())
    data, pairs, roles, response, _, _ = old.load_dataset(args.dataset)
    target = transport_targets(data, pairs)
    v = make_values(
        data,
        pairs,
        roles,
        response,
        np.array(m["H1_scales_frozen94_m_s"]),
        target,
        np.array(m["transport_scales_train_role_k_m_s"]),
        "cpu",
    )
    saved_free = (
        load_archive(args.out_dir / "final_free_gpu_predictions.npz")
        if (args.out_dir / "final_free_gpu_predictions.npz").exists()
        else None
    )
    saved_native = (
        load_archive(args.out_dir / "final_true_input_gpu_predictions.npz")
        if (args.out_dir / "final_true_input_gpu_predictions.npz").exists()
        else None
    )
    free = {}
    native = {}
    errors = {}
    adam = {}
    for arm in ARMS:
        q = torch.load(args.out_dir / f"{arm}.pt", weights_only=True, map_location="cpu")
        if (
            q["schema"] != SCHEMA
            or q["updates"] != r["paired_updates_both_arms"]
            or old.weight_hash(q["model"]) != r["final_weights_sha256"][arm]
        ):
            raise ValueError("98 checkpoint hash/schema/update mismatch")
        steps = [float(s["step"]) for s in q["optimizer"]["state"].values()]
        if not steps or any(s != q["updates"] for s in steps):
            raise ValueError("98 Adam counters disagree")
        adam[arm] = sorted(set(steps))
        model = old.ContactWorldModel(
            old.JointStateLayout(old.BOARD_JOINT_NAMES),
            old.JointLearningConfig(hidden=(512, 512, 512)),
        )
        model.load_state_dict(q["model"], strict=True)
        fp = old.predictions(model, v["states"], v["actions"])
        tp = true_state_predictions(model, v)
        free[arm] = old.diagnostics(fp, data, pairs, roles, response)
        native[arm] = transport_metrics(tp, data, pairs, roles)
        errors[arm] = {
            "free_board_Vxyz_max_abs_m_s": float(
                abs(fp[:, :, 82:85] - saved_free[arm][:, :, 82:85]).max()
            )
            if saved_free is not None
            else None,
            "native_next_board_Vxyz_max_abs_m_s": float(
                abs(tp[:, :, 82:85] - saved_native[arm][:, :, 82:85]).max()
            )
            if saved_native is not None
            else None,
        }
    out = {
        "weights_only_and_tensor_hashes_verified": True,
        "adam_steps": adam,
        "same_actual_batch_hash": len(set(r["arm_batch_sha256"].values())) == 1,
        "arms_free_rollout": free,
        "arms_native_state_transport": native,
        "cpu_gpu_errors": errors,
        "cpu_preregistered_gates": gates(
            free[ARMS[1]], native[ARMS[1]], free[ARMS[0]], r["constant_velocity"]
        ),
        "wall_seconds": time.monotonic() - started,
        "new_optimizer_updates": 0,
        "new_simulation_steps": 0,
        "actor_updates": 0,
        "board_bptt_allowed": False,
    }
    write_json(args.out_dir / "independent_cpu_readback.json", out)
    print(
        json.dumps(
            {
                k: value
                for k, value in out.items()
                if k
                not in (
                    "arms_free_rollout",
                    "arms_native_state_transport",
                    "cpu_preregistered_gates",
                )
            },
            indent=2,
        )
    )
    print(
        "cpu_all_preregistered_gates_pass",
        out["cpu_preregistered_gates"]["all_preregistered_local_gates_pass"],
    )


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
