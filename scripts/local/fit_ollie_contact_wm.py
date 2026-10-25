"""Bounded two-arm supervised WM comparison. No actor, PPO, or simulator rollout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.contact_learning import (
    AUX_NAMES,
    RESPONSE_NAMES,
    SCHEMA,
    ContactFitConfig,
    ContactWorldModel,
    auxiliary_targets,
    response_scales,
    rollout,
    root_response,
    training_losses,
)
from supermimic.ollie.learning import JointLearningConfig
from supermimic.ollie.physics import BOARD_JOINT_NAMES
from supermimic.ollie.state import JointStateLayout

ANCHORS = (120, 130, 140, 146, 149, 160, 240)
SOURCE_SHA = "8b602ad7f88266ba831c71df6c0f25bc84366a3af13427166e559f45d4470cb7"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def load_dataset(directory):
    path = directory / "paired_responses.npz"
    manifest_blob = (directory / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_blob)
    blob = path.read_bytes()
    expected = manifest.get("dataset_sha256")
    if expected is None:
        expected = manifest.get("files", {}).get(path.name, {}).get("sha256")
    if expected != sha(blob):
        raise ValueError("dataset hash absent from or inconsistent with manifest")
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    meta = json.loads(str(data["metadata_json"].item()))
    if (
        meta.get("foot_deck_friction") != 0.9
        or meta.get("source_sha256") != SOURCE_SHA
        or meta.get("control_dt") != 0.02
        or meta.get("physics_dt") != 0.0025
        or tuple(meta.get("robot_joint_names", ())) != G1_JOINT_NAMES
        or tuple(meta.get("board_joint_names", ())) != BOARD_JOINT_NAMES
    ):
        raise ValueError("data material/source/joint/time provenance mismatch")
    board = Path("assets/ollie/passive_skateboard.xml")
    if meta.get("board_xml_sha256") != sha(board.read_bytes()):
        raise ValueError("canonical board XML hash mismatch")
    source_path = Path(meta["source"])
    if sha(source_path.read_bytes()) != SOURCE_SHA:
        raise ValueError("actual source episode hash mismatch")
    with np.load(source_path, allow_pickle=False) as source:
        source_action_hash = sha(np.ascontiguousarray(source["action"]).tobytes())
    if meta.get("action_sha256") != source_action_hash:
        raise ValueError("source257 action array hash mismatch")
    if not data["valid_complete"].all() or not data["transition_valid"].all():
        raise ValueError(
            "incomplete native branches retained by collector; refusing padded samples"
        )
    shape = {"state": (217, 9, 92), "action": (217, 8, 29), "post_forward_metrics": (217, 8, 8, 49)}
    for key, required in shape.items():
        if data[key].shape != required or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid {key} shape/finite contract")
    if data["state"].dtype != np.float64 or data["action"].dtype != np.float64:
        raise ValueError(
            "native states and actions must remain float64 before response differencing"
        )
    names = tuple(meta["metric_names"])
    if len(names) != 49 or len(set(names)) != 49:
        raise ValueError("expected 49 unique post-forward metrics")
    auxiliary_targets(torch.zeros(1, 49), names)  # Reject missing named outputs on CPU.
    for name in ("overflow", "external_generalized_force_norm", "external_body_force_norm"):
        if np.any(data["post_forward_metrics"][..., names.index(name)] != 0):
            raise ValueError(f"nonzero {name}: refuse overflow or externally forced samples")
    if manifest.get("schema") != meta.get("schema") or any(
        manifest.get(k) != v for k, v in meta.items()
    ):
        raise ValueError("dataset manifest and archived frozen metadata differ")
    anchors = data["anchor_frame"]
    if tuple(np.unique(anchors)) != ANCHORS:
        raise ValueError("anchor group inventory differs from the frozen design")
    held = np.isin(anchors, (146, 160))
    if not np.array_equal(data["heldout_mask"].astype(bool), held):
        raise ValueError("entire anchor146/160 groups must be held out")
    if not np.array_equal(data["train_mask"].astype(bool), ~held):
        raise ValueError("train and heldout partition must be exact complements")
    pairs = []
    for anchor in ANCHORS:
        rows = np.flatnonzero(anchors == anchor)
        if len(rows) != 31:
            raise ValueError("each anchor requires exactly 31 baseline/paired branches")
        baseline = rows[data["is_baseline"][rows].astype(bool)]
        if len(baseline) != 1:
            raise ValueError("each anchor requires one baseline")
        base = int(baseline[0])
        if not np.array_equal(data["paired_baseline_id"][rows], np.full(31, base)):
            raise ValueError(
                "paired baseline must be an index within the same actual snapshot group"
            )
        if not np.array_equal(
            data["state"][rows, 0], np.repeat(data["state"][base : base + 1, 0], 31, 0)
        ):
            raise ValueError("same-anchor branches do not start at the same state")
        if not np.allclose(
            data["actual_delta"][rows],
            data["action"][rows] - data["action"][base],
            atol=1e-14,
            rtol=0,
        ):
            raise ValueError("actual_delta does not match saved clipped controls")
        joints = np.unique(data["joint_index"][rows[rows != base]])
        if len(joints) != 15:
            raise ValueError("expected 15 perturbed leg/waist joints per anchor")
        for joint in joints:
            pair = []
            for sign in (1, -1):
                found = rows[(data["joint_index"][rows] == joint) & (data["sign"][rows] == sign)]
                if len(found) != 1:
                    raise ValueError("incomplete positive/negative pair")
                pair.append(int(found[0]))
            pairs.append(pair)
    for q in (slice(3, 7), slice(74, 78)):
        if np.max(np.abs(np.linalg.norm(data["state"][..., q], axis=-1) - 1)) > 1e-8:
            raise ValueError("root quaternion norm mismatch")
    pairs = np.asarray(pairs)
    # Compute small native differences in float64, not after model float32 conversion.
    response = root_response(
        torch.from_numpy(data["state"][pairs[:, 0], 1:]),
        torch.from_numpy(data["state"][pairs[:, 1], 1:]),
    ).numpy()
    record = {
        "dataset": str(path.resolve()),
        "dataset_sha256": sha(blob),
        "intervened_action_array_sha256": sha(np.ascontiguousarray(data["action"]).tobytes()),
        "source_manifest_sha256": sha(manifest_blob),
        "metadata": meta,
        "train_branches": int((~held).sum()),
        "heldout_branches": int(held.sum()),
        "train_pairs": int((~held[pairs[:, 0]]).sum()),
        "heldout_pairs": int(held[pairs[:, 0]].sum()),
        "heldout_unit": "entire actual anchor groups146/160, all sibling branches",
        "heldout_limitation": "Same source motion and nearby warm states; anchor146 H8 overlaps source time windows of train140/149. This is local interpolation validation, not unseen-skill or independent-episode generalization.",
    }
    return data, pairs, response, held, record


def tensors(data, pairs, response, device):
    return {
        "states": torch.tensor(data["state"], dtype=torch.float32, device=device),
        "actions": torch.tensor(data["action"], dtype=torch.float32, device=device),
        "aux": auxiliary_targets(
            torch.tensor(data["post_forward_metrics"], dtype=torch.float32, device=device),
            json.loads(str(data["metadata_json"].item()))["metric_names"],
        ),
        "pairs": torch.tensor(pairs, device=device),
        "response": torch.tensor(response[:, [0, 7]], dtype=torch.float32, device=device),
    }


def coverage(data, mask, names):
    values = data["post_forward_metrics"][mask]

    def column(name):
        return values[..., names.index(name)]

    return {
        "branches": int(mask.sum()),
        "post_forward_samples": int(np.prod(values.shape[:-1])),
        "tail_force_ge5_samples": int((column("tail_ground_normal_force_n") >= 5).sum()),
        "positive_front_nose_force_ge1_samples": int((column("front_nose_board_fz_n") >= 1).sum()),
        "maximum_tail_force_n": float(column("tail_ground_normal_force_n").max()),
        "maximum_front_nose_force_n": float(column("front_nose_board_fz_n").max()),
        "scope": "coverage of post-forward forces; not execution impulses or Ollie success",
    }


@torch.no_grad()
def diagnostics(model, values, rows, pair_rows):
    predicted, aux = rollout(
        model, values["states"][rows, 0], values["actions"][rows], auxiliary=True
    )
    target = values["states"][rows, 1:]
    result = {"state": {}, "auxiliary_mse": float((aux - values["aux"][rows]).square().mean())}
    for h in (1, 8):
        result["state"][f"H{h}"] = {}
        for name, section in (("board_xyz_m", slice(71, 74)), ("board_Vxyz_m_s", slice(82, 85))):
            error = predicted[:, h - 1, section] - target[:, h - 1, section]
            result["state"][f"H{h}"][name] = {
                "rmse_axes": error.square().mean(0).sqrt().tolist(),
                "rmse_vector": float(error.square().sum(-1).mean().sqrt()),
            }
    pair = values["pairs"][pair_rows]
    plus, _ = rollout(model, values["states"][pair[:, 0], 0], values["actions"][pair[:, 0]])
    minus, _ = rollout(model, values["states"][pair[:, 1], 0], values["actions"][pair[:, 1]])
    estimate = root_response(plus[:, [0, 7]], minus[:, [0, 7]])
    truth = values["response"][pair_rows]
    result["paired"] = {}
    for k, h in enumerate((1, 8)):
        records = []
        for i in range(24):
            t, p = truth[:, k, i], estimate[:, k, i]
            energy, valid = t.square().sum(), t.abs() > 1e-6
            records.append(
                {
                    "feature": RESPONSE_NAMES[i],
                    "rmse_si": float((p - t).square().mean().sqrt()),
                    "zero_effect_rmse_si": float(t.square().mean().sqrt()),
                    "gain_projection": float((p * t).sum() / energy) if energy > 0 else None,
                    "sign_samples": int(valid.sum()),
                    "sign_agreement": float((p[valid].sign() == t[valid].sign()).float().mean())
                    if valid.any()
                    else None,
                }
            )
        result["paired"][f"H{h}"] = records
    return result


def atomic_save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def synthetic_check(output):
    torch.set_num_threads(2)
    torch.manual_seed(73)
    config = JointLearningConfig(hidden=(32, 32))
    model = ContactWorldModel(JointStateLayout(BOARD_JOINT_NAMES), config)
    state = torch.zeros(3, 92)
    state[:, 3] = state[:, 74] = 1
    action = torch.randn(3, 29)
    primary = model.step(state, action)
    combined, aux = model.step_with_aux(state, action)
    if not torch.equal(primary, combined) or aux.shape != (3, 8, len(AUX_NAMES)):
        raise AssertionError("shared trunk/state head contract failed")
    aux.square().mean().backward()
    gradient = model.dynamics.mlp.net[0].weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
        raise AssertionError("auxiliary gradient failed to reach the shared trunk")
    model.zero_grad(set_to_none=True)
    states = state[:, None].repeat(1, 9, 1)
    actions = action[:, None].repeat(1, 8, 1)
    losses = training_losses(
        model,
        states,
        actions,
        torch.zeros(3, 8, 8, len(AUX_NAMES)),
        (state, actions + 0.02, actions - 0.02),
        torch.full((3, 2, 24), 0.001),
        ContactFitConfig(),
    )
    (losses[0] + losses[1] + 0.1 * losses[2]).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError("full H8 state+pair+aux gradient is nonfinite")
    report = {
        "passed": True,
        "full_loss_values": [float(x.detach()) for x in losses],
        "full_h8_backward_finite": True,
        "device": "cpu",
        "optimizer_updates": 0,
        "gpu_initialized": torch.cuda.is_initialized(),
        "state_shape": list(primary.shape),
        "aux_shape": list(aux.shape),
        "auxiliary_names": AUX_NAMES,
        "state_step_exact_equal": True,
        "shared_trunk_gradient_norm": float(gradient.norm()),
    }
    output.mkdir(parents=True, exist_ok=False)
    temporary_model = output / "synthetic_weights.pt"
    torch.save(
        {"schema": SCHEMA, "model": model.state_dict(), "synthetic_test_only": True},
        temporary_model,
    )
    recovered = torch.load(temporary_model, map_location="cpu", weights_only=True)
    if recovered["schema"] != SCHEMA or not all(
        torch.equal(t, recovered["model"][k]) for k, t in model.state_dict().items()
    ):
        raise AssertionError("weights_only roundtrip failed")
    report["weights_only_roundtrip_exact"] = True
    (output / "synthetic_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--synthetic-check", action="store_true")
    parser.add_argument("--execute-approved-fit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.synthetic_check:
        synthetic_check(args.out_dir)
        return
    if args.dataset is None or args.out_dir.exists():
        parser.error("provide a dataset and a fresh output directory")
    started = time.monotonic()
    config = ContactFitConfig()
    data, pairs, response, held, source = load_dataset(args.dataset)
    manifest = {
        "schema": SCHEMA,
        "config": asdict(config),
        "source": source,
        "main_initialization": "fresh shared DynamicsMLP, no old WM/BC weights",
        "auxiliary_names": AUX_NAMES,
        "auxiliary_transforms": "forces asinh(N/100), nine clearances / .05m, vz / 1m/s, raw flags",
        "paired_response_contract": "42: positive-minus-negative 24 SI root features, H1/H8; actual float64 differences",
        "paired_scales": response_scales().tolist(),
        "paired_response_names": RESPONSE_NAMES,
        "arms": {"control": 0.0, "auxiliary": config.auxiliary_weight},
        "coverage": {
            "train": coverage(data, ~held, source["metadata"]["metric_names"]),
            "heldout": coverage(data, held, source["metadata"]["metric_names"]),
        },
        "actor_training": False,
        "simulation_force_feedback": False,
        "board_bptt_allowed": False,
        "ollie_gradient_certified": False,
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.inspect or not args.execute_approved_fit:
        print(json.dumps(manifest, indent=2))
        return
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    values = tensors(data, pairs, response, args.device)
    model_cfg = JointLearningConfig(hidden=config.hidden)
    initial = ContactWorldModel(JointStateLayout(BOARD_JOINT_NAMES), model_cfg, args.device)
    initial_weights = copy.deepcopy(initial.state_dict())
    models = {name: copy.deepcopy(initial) for name in ("control", "auxiliary")}
    optimizers = {
        name: torch.optim.Adam(m.parameters(), lr=config.learning_rate)
        for name, m in models.items()
    }
    train = torch.tensor(np.flatnonzero(~held), device=args.device)
    train_pairs = torch.tensor(np.flatnonzero(~held[pairs[:, 0]]), device=args.device)
    splits = {
        label: (
            torch.tensor(np.flatnonzero(mask), device=args.device),
            torch.tensor(np.flatnonzero(mask[pairs[:, 0]]), device=args.device),
        )
        for label, mask in (("train", ~held), ("heldout", held))
    }
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history, completed, reason = [], {name: 0 for name in models}, "maximum_updates"
    batch_hash = hashlib.sha256()
    arm_batch_hash = {name: hashlib.sha256() for name in models}
    initial_hash = hashlib.sha256()
    for name, tensor in initial_weights.items():
        initial_hash.update(name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes())
    for model in models.values():
        if not all(
            torch.equal(tensor, model.state_dict()[key]) for key, tensor in initial_weights.items()
        ):
            raise AssertionError("comparison arms have different initial main/head weights")
    manifest_hash = sha((args.out_dir / "manifest.json").read_bytes())
    atomic_save(
        args.out_dir / "initial.pt",
        {
            "schema": SCHEMA,
            "model": initial_weights,
            "learning_config": asdict(model_cfg),
            "manifest_sha256": manifest_hash,
            "metadata": manifest,
            "updates": 0,
            "policy": None,
        },
    )
    try:
        for update in range(config.updates):
            if time.monotonic() - started >= config.wall_seconds - 20:
                reason = "wall_budget_fit_reserve20s"
                break
            index = torch.randint(len(train), (config.batch_size,), generator=generator)
            pair_index = torch.randint(len(train_pairs), (config.batch_size,), generator=generator)
            batch_hash.update(index.numpy().tobytes() + pair_index.numpy().tobytes())
            rows, pr = train[index.to(args.device)], train_pairs[pair_index.to(args.device)]
            pair = values["pairs"][pr]
            row = {"update": update + 1}
            for name, model in models.items():
                state_loss, paired_loss, aux_loss = training_losses(
                    model,
                    values["states"][rows],
                    values["actions"][rows],
                    values["aux"][rows],
                    (
                        values["states"][pair[:, 0], 0],
                        values["actions"][pair[:, 0]],
                        values["actions"][pair[:, 1]],
                    ),
                    values["response"][pr],
                    config,
                )
                loss = state_loss + config.paired_weight * paired_loss
                if name == "auxiliary":
                    loss = loss + config.auxiliary_weight * aux_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite {name} loss")
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizers[name].step()
                completed[name] += 1
                arm_batch_hash[name].update(index.numpy().tobytes() + pair_index.numpy().tobytes())
                row[name] = {
                    "state_loss": float(state_loss.detach()),
                    "paired_loss": float(paired_loss.detach()),
                    "auxiliary_mse": float(aux_loss.detach()),
                    "total": float(loss.detach()),
                }
            history.append(row)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception as error:
        reason = f"error:{type(error).__name__}:{error}"
        raise
    finally:
        for name, model in models.items():
            atomic_save(
                args.out_dir / f"{name}.pt",
                {
                    "schema": SCHEMA,
                    "arm": name,
                    "model": model.state_dict(),
                    "optimizer": optimizers[name].state_dict(),
                    "learning_config": asdict(model_cfg),
                    "updates": completed[name],
                    "manifest_sha256": manifest_hash,
                    "metadata": manifest,
                    "policy": None,
                    "batch_indices_sha256": arm_batch_hash[name].hexdigest(),
                    "common_initial_weights_sha256": initial_hash.hexdigest(),
                    "stop_reason": reason,
                },
            )
        result = {
            "updates": completed,
            "stop_reason": reason,
            "wall_seconds_before_diagnostics": time.monotonic() - started,
            "batch_indices_sha256": batch_hash.hexdigest(),
            "arm_batch_indices_sha256": {k: v.hexdigest() for k, v in arm_batch_hash.items()},
            "common_initial_weights_sha256": initial_hash.hexdigest(),
            "actor_updates": 0,
            "ollie_success_certified": False,
            "board_bptt_allowed": False,
        }
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    # Evaluation is bounded by the same deadline; partial diagnostics are labeled.
    reports = {}
    for name, model in models.items():
        reports[name] = {}
        for label, (rows, pr) in splits.items():
            if time.monotonic() - started >= config.wall_seconds:
                reports[name][label] = {"skipped": "wall_budget"}
            else:
                reports[name][label] = diagnostics(model, values, rows, pr)
    result["diagnostics"] = reports
    result["wall_seconds"] = time.monotonic() - started
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
