"""Real-intermediate-state TF1 versus TF1+.25 H1 action-response supervision.

Original SuperMimic model/codec and the frozen93 default-mu=.6 dataset stay.
H8 is evaluated freely but never differentiated in this experiment.
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

from scripts.local import fit_ollie_phase_response_wm as old
from scripts.local.collect_ollie_mu09_responses import digest, load_archive, write_json

SCHEMA = "supermimic.ollie.transition_response_wm.v1"
ARMS = ("tf1_only", "tf1_plus_response")
SEED, UPDATES, BATCH, PAIR_PER_ROLE = 96, 500, 64, 10
RESPONSE_WEIGHT = 0.25
SCALE_MANIFEST = Path("logs/ollie_phase_response_wm_01/manifest.json")
SCALE_MANIFEST_SHA = "501ea9a12f4a9ea85fadd52e99dbe2e540b70c263973ee0a3201b50c206f5ad2"


def frozen_h1_scales(response, roles, train_pairs):
    if digest(SCALE_MANIFEST) != SCALE_MANIFEST_SHA:
        raise ValueError("Requires the unchanged94 train-only scale manifest")
    previous = json.loads(SCALE_MANIFEST.read_text())
    scales = np.asarray(previous["train_only_board_vz_scales_m_s"])[:, 0]
    if not np.array_equal(scales, old.train_scales(response, roles, train_pairs)[:, 0]):
        raise ValueError("Frozen94 H1 scales do not match93 training groups")
    return scales


def transition_loss(model, current, actions, next_state):
    """Every current state here is real, independently indexed k=0..7."""
    predicted = model.step(current, actions)
    error = model.spec.error_vector(predicted, next_state)
    weights = error.new_ones(error.shape[-1])
    weights[-8:] = 0.05
    return (error.square() * weights).sum(-1).mean() / weights.sum()


def h1_response_loss(model, initial, positive_action, negative_action, target, roles, scales):
    # Only these first controls share the same true hot state. At k>=1 the
    # positive/negative real states differ; they are not same-state actions.
    positive = model.step(initial, positive_action)
    negative = model.step(initial, negative_action)
    estimate = positive[:, 84] - negative[:, 84]
    square = ((estimate - target) / scales[roles]).square()
    return torch.stack([square[roles == g].mean() for g in range(6)]).mean()


def make_values(data, pairs, roles, response, scales, device):
    return {
        "states": torch.tensor(data["state"], dtype=torch.float32, device=device),
        "actions": torch.tensor(data["action"], dtype=torch.float32, device=device),
        "current": torch.tensor(
            data["state"][:, :8].reshape(-1, 92), dtype=torch.float32, device=device
        ),
        "next": torch.tensor(
            data["state"][:, 1:].reshape(-1, 92), dtype=torch.float32, device=device
        ),
        "transition_actions": torch.tensor(
            data["action"].reshape(-1, 29), dtype=torch.float32, device=device
        ),
        "pairs": torch.tensor(pairs, device=device),
        "roles": torch.tensor(roles, device=device),
        "response_h1": torch.tensor(response[:, 0], dtype=torch.float32, device=device),
        "scales_h1": torch.tensor(scales, dtype=torch.float32, device=device),
    }


def losses(model, values, rows, pair_rows, with_response):
    state = transition_loss(
        model, values["current"][rows], values["transition_actions"][rows], values["next"][rows]
    )
    if not with_response:
        return state, state.new_zeros(())
    pairs = values["pairs"][pair_rows]
    response = h1_response_loss(
        model,
        values["states"][pairs[:, 0], 0],
        values["actions"][pairs[:, 0], 0],
        values["actions"][pairs[:, 1], 0],
        values["response_h1"][pair_rows],
        values["roles"][pair_rows],
        values["scales_h1"],
    )
    return state, response


def paired_commit(models, optimizers):
    # Reuse the tested rollback implementation without changing94 globals.
    old.paired_commit(
        {previous: models[current] for previous, current in zip(old.ARMS, ARMS, strict=True)},
        {previous: optimizers[current] for previous, current in zip(old.ARMS, ARMS, strict=True)},
    )


def input_coverage(data, rows):
    chosen = np.unique(rows)
    true_input = data["state"][:, :8].reshape(-1, 92)[chosen]
    train_rows = np.flatnonzero(np.repeat(data["train_mask"], 8))
    return {
        "sampled_transition_draws": int(np.size(rows)),
        "unique_real_transition_rows": len(chosen),
        "available_train_transition_rows": len(train_rows),
        "unique_real_state92_inputs": len(np.unique(true_input, axis=0)),
        "available_unique_real_train_state92_inputs": len(
            np.unique(data["state"][data["train_mask"], :8].reshape(-1, 92), axis=0)
        ),
        "unique_row_counts_by_step": [int((chosen % 8 == step).sum()) for step in range(8)],
        "all_sampled_rows_are_training": bool(np.repeat(data["train_mask"], 8)[chosen].all()),
    }


def prepare(args):
    data, pairs, roles, response, quality, meta = old.load_dataset(args.dataset)
    train_pair = ~data["heldout_mask"][pairs[:, 0]]
    scales = frozen_h1_scales(response, roles, train_pair)
    generator = torch.Generator().manual_seed(SEED)
    choices = np.flatnonzero(np.repeat(data["train_mask"], 8))
    tb, pb = [], []
    for _ in range(UPDATES):
        tb.append(choices[torch.randint(len(choices), (BATCH,), generator=generator).numpy()])
        selected = np.concatenate(
            [
                (group := np.flatnonzero(train_pair & (roles == g)))[
                    torch.randint(len(group), (PAIR_PER_ROLE,), generator=generator).numpy()
                ]
                for g in range(6)
            ]
        )
        pb.append(selected[torch.randperm(len(selected), generator=generator).numpy()])
    torch.manual_seed(SEED)
    initial = old.ContactWorldModel(
        old.JointStateLayout(old.BOARD_JOINT_NAMES), old.JointLearningConfig(hidden=(512, 512, 512))
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    old.atomic_save(
        args.out_dir / "initial.pt",
        {"schema": SCHEMA, "model": initial.state_dict(), "seed": SEED, "actor_updates": 0},
    )
    np.savez_compressed(
        args.out_dir / "frozen_batches.npz", transition_rows=np.array(tb), pair_rows=np.array(pb)
    )
    manifest = {
        "schema": SCHEMA,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": old.DATA_SHA,
        "dataset_manifest_sha256": digest(args.dataset / "dataset_manifest.json"),
        "physics_metadata": meta,
        "fitter_sha256": digest(__file__),
        "unchanged_shared94_fitter_sha256": digest(old.__file__),
        "scale_source_manifest_sha256": SCALE_MANIFEST_SHA,
        "frozen94_train_only_H1_scales_m_s": scales,
        "initial_file_sha256": digest(args.out_dir / "initial.pt"),
        "initial_weights_sha256": old.weight_hash(initial.state_dict()),
        "frozen_batches_sha256": digest(args.out_dir / "frozen_batches.npz"),
        "seed": SEED,
        "maximum_paired_updates": UPDATES,
        "transition_batch": BATCH,
        "pair_batch": 60,
        "pair_samples_per_role": 10,
        "hidden": [512, 512, 512],
        "learning_rate": 2e-4,
        "clip_grad_norm": 1.0,
        "losses": {
            "tf1_only": "original normalized one-step error on real state_b,k/action_b,k/next_state_b,k+1 for k0..7; wheel.05",
            "tf1_plus_response": "same TF1+.25*phase-equally-weighted H1-only boardVZ true positive-minus-negative response MSE/frozen94 H1 RMS squared",
        },
        "response_weight": RESPONSE_WEIGHT,
        "free_H8_loss_weight": 0.0,
        "H8_response_loss_weight": 0.0,
        "same_state_action_response_only_at_anchor_k0": True,
        "original_model_codec_integration_unchanged": True,
        "train_branches": 372,
        "heldout_branches": 186,
        "train_transitions": 2976,
        "heldout_transitions": 1488,
        "planned_input_coverage": input_coverage(data, np.asarray(tb)),
        "heldout_use": "final H1/H8 free native diagnostics only; no scales/batches/weight/stop/checkpoint selection",
        "total_budget_s": 300.0,
        "stop_accepting_paired_updates_s": 270.0,
        "optimizer_commit": "transactional pair with both Adam states/weights rollback on failure",
        "gradient_diagnostic_updates": [0, 100, 250, 500],
        "gates": "unchanged94 native H1/H8 response .8zero/gain.5..1.5/sign.75 for all/tail/air/landing; H8 velocity no worse than control and CV, position<=1.1control",
        "quality_failed_branches_retained": int((~quality["penetration_and_joint_quality"]).sum()),
        "scope": "same failed93 episode, overlapping nearby anchors; comparison with94 changes input sampling, response horizon/weight and seed, not a single-factor ablation",
        "verified_teacher": False,
        "actor_updates": 0,
        "board_bptt_allowed": False,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "initial_weights_sha256": manifest["initial_weights_sha256"],
                "input_coverage": manifest["planned_input_coverage"],
                "optimizer_updates": 0,
            },
            indent=2,
        )
    )


def gradient_diagnostic(model, values, rows, pair_rows, update):
    state, response = losses(model, values, rows, pair_rows, True)
    params = list(model.parameters())
    gs = torch.autograd.grad(state, params, allow_unused=True)
    gr = torch.autograd.grad(response, params, allow_unused=True)
    norm = lambda values: torch.sqrt(
        sum((g.square().sum() for g in values if g is not None), state.new_zeros(()))
    )
    ns, nr = norm(gs), norm(gr)
    dot = sum(
        (
            a.flatten().dot(b.flatten())
            for a, b in zip(gs, gr, strict=True)
            if a is not None and b is not None
        ),
        state.new_zeros(()),
    )
    return {
        "after_paired_update": update,
        "tf1_loss": float(state.detach()),
        "H1_response_loss_unweighted": float(response.detach()),
        "tf1_gradient_norm": float(ns),
        "H1_response_gradient_norm_unweighted": float(nr),
        "H1_response_gradient_norm_weighted25percent": float(RESPONSE_WEIGHT * nr),
        "cosine": float(dot / (ns * nr).clamp_min(1e-30)),
        "used_for_weight_tuning": False,
    }


def constant_velocity_metrics(data):
    cv = np.repeat(data["state"][:, 0:1], 8, axis=1)
    cv[:, :, 71:74] = (
        data["state"][:, 0:1, 71:74]
        + np.arange(1, 9)[None, :, None] * 0.02 * data["state"][:, 0:1, 82:85]
    )
    return {
        split: old.state_metrics(cv, data["state"], mask)
        for split, mask in (("train", data["train_mask"]), ("heldout", data["heldout_mask"]))
    }


def run(args):
    started = time.monotonic()
    m = json.loads((args.out_dir / "manifest.json").read_text())
    for path, key in (
        (Path(__file__), "fitter_sha256"),
        (Path(old.__file__), "unchanged_shared94_fitter_sha256"),
        (args.out_dir / "initial.pt", "initial_file_sha256"),
        (args.out_dir / "frozen_batches.npz", "frozen_batches_sha256"),
    ):
        if digest(path) != m[key]:
            raise ValueError(f"Frozen input/code changed:{key}")
    if (args.out_dir / "result.json").exists() or any(
        (args.out_dir / f"{arm}.pt").exists() for arm in ARMS
    ):
        raise ValueError("Never overwrite/restart the unique96 comparison")
    data, pairs, roles, response, quality, _ = old.load_dataset(args.dataset)
    scales = frozen_h1_scales(response, roles, ~data["heldout_mask"][pairs[:, 0]])
    if str(args.dataset.resolve()) != m["dataset"] or not np.array_equal(
        scales, np.array(m["frozen94_train_only_H1_scales_m_s"])
    ):
        raise ValueError("Frozen dataset/scales changed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = torch.load(args.out_dir / "initial.pt", weights_only=True, map_location="cpu")
    if old.weight_hash(initial["model"]) != m["initial_weights_sha256"]:
        raise ValueError("Initial tensor hash changed")
    v = make_values(data, pairs, roles, response, scales, args.device)
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
    hashes = {arm: hashlib.sha256() for arm in ARMS}
    stopped = {"signal": None}
    handlers = {}

    def stop(sig, _frame):
        stopped["signal"] = int(sig)

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, stop)
    try:
        gradients.append(gradient_diagnostic(models[ARMS[1]], v, tb[0], pb[0], 0))
        for update in range(UPDATES):
            if stopped["signal"] is not None:
                reason = f"signal_at_pair_boundary:{stopped['signal']}"
                break
            if time.monotonic() - started >= 270:
                reason = "wall_budget_no_new_pair_after270s"
                break
            row = {"paired_update": update + 1}
            for arm in ARMS:
                optimizers[arm].zero_grad(set_to_none=True)
                state, effect = losses(models[arm], v, tb[update], pb[update], arm == ARMS[1])
                total = state + RESPONSE_WEIGHT * effect
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite96 loss:{arm}")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    models[arm].parameters(), 1.0, error_if_nonfinite=True
                )
                row[arm] = {
                    "tf1_loss": float(state.detach()),
                    "H1_response_loss_unweighted": float(effect.detach()),
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
                "arm_batch_sha256": {arm: hashes[arm].hexdigest() for arm in ARMS},
                "wall_at_checkpoint_save_s": time.monotonic() - started,
            },
        )
    predictions = {}
    metrics = {}
    if time.monotonic() - started < 295:
        for arm in ARMS:
            predictions[arm] = old.predictions(models[arm], v["states"], v["actions"])
            metrics[arm] = old.diagnostics(predictions[arm], data, pairs, roles, response)
        np.savez_compressed(args.out_dir / "final_gpu_predictions.npz", **predictions)
    cv = constant_velocity_metrics(data)
    result = {
        "schema": SCHEMA,
        "paired_updates_both_arms": updates,
        "stop_reason": reason,
        "arms": metrics,
        "constant_velocity": cv,
        "initial_weights_sha256": m["initial_weights_sha256"],
        "arm_batch_sha256": {arm: hashes[arm].hexdigest() for arm in ARMS},
        "final_weights_sha256": {arm: old.weight_hash(models[arm].state_dict()) for arm in ARMS},
        "actual_input_coverage": input_coverage(data, batches["transition_rows"][:updates]),
        "preregistered_gates": old.gates(metrics[ARMS[1]], metrics[ARMS[0]], cv)
        if len(metrics) == 2
        else None,
        "final_gpu_diagnostics_completed": len(metrics) == 2,
        "quality_failed_branches_retained": int((~quality["penetration_and_joint_quality"]).sum()),
        "wall_seconds": time.monotonic() - started,
        "wall_budget_s": 300.0,
        "training_response_horizons": [1],
        "training_free_rollout_horizons": [],
        "actor_updates": 0,
        "board_bptt_allowed": False,
        "verified_teacher": False,
    }
    write_json(args.out_dir / "result.json", result)
    print(
        json.dumps(
            {k: value for k, value in result.items() if k not in ("arms", "constant_velocity")},
            indent=2,
        )
    )


def readback(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    m = json.loads((args.out_dir / "manifest.json").read_text())
    r = json.loads((args.out_dir / "result.json").read_text())
    data, pairs, roles, response, _, _ = old.load_dataset(args.dataset)
    v = make_values(
        data, pairs, roles, response, np.array(m["frozen94_train_only_H1_scales_m_s"]), "cpu"
    )
    saved = (
        load_archive(args.out_dir / "final_gpu_predictions.npz")
        if (args.out_dir / "final_gpu_predictions.npz").exists()
        else None
    )
    cpu = {}
    errors = {}
    adam_steps = {}
    for arm in ARMS:
        q = torch.load(args.out_dir / f"{arm}.pt", weights_only=True, map_location="cpu")
        if (
            q["schema"] != SCHEMA
            or q["updates"] != r["paired_updates_both_arms"]
            or old.weight_hash(q["model"]) != r["final_weights_sha256"][arm]
        ):
            raise ValueError("96 checkpoint schema, update or weight hash mismatch")
        steps = [float(s["step"]) for s in q["optimizer"]["state"].values()]
        if not steps or any(s != q["updates"] for s in steps):
            raise ValueError("96 Adam step counters differ from shared update count")
        adam_steps[arm] = sorted(set(steps))
        model = old.ContactWorldModel(
            old.JointStateLayout(old.BOARD_JOINT_NAMES),
            old.JointLearningConfig(hidden=(512, 512, 512)),
        )
        model.load_state_dict(q["model"], strict=True)
        pred = old.predictions(model, v["states"], v["actions"])
        cpu[arm] = old.diagnostics(pred, data, pairs, roles, response)
        errors[arm] = {
            "board_xyz_max_abs_m": float(abs(pred[:, :, 71:74] - saved[arm][:, :, 71:74]).max())
            if saved is not None
            else None,
            "board_Vxyz_max_abs_m_s": float(abs(pred[:, :, 82:85] - saved[arm][:, :, 82:85]).max())
            if saved is not None
            else None,
        }
    out = {
        "weights_only_and_weight_hashes_verified": True,
        "adam_steps": adam_steps,
        "same_paired_update_count_and_batch_hash": len(set(r["arm_batch_sha256"].values())) == 1,
        "arms": cpu,
        "cpu_gpu_errors": errors,
        "cpu_preregistered_gates": old.gates(cpu[ARMS[1]], cpu[ARMS[0]], r["constant_velocity"]),
        "wall_seconds": time.monotonic() - started,
        "new_optimizer_updates": 0,
        "new_simulation_steps": 0,
        "actor_updates": 0,
        "board_bptt_allowed": False,
    }
    write_json(args.out_dir / "independent_cpu_readback.json", out)
    print(json.dumps({k: value for k, value in out.items() if k != "arms"}, indent=2))


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
