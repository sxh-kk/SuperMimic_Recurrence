"""One fixed, offline two-codec WM comparison; no actor or physics execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from scripts.local.fit_ollie_contact_wm import (
    atomic_save,
    coverage,
    diagnostics,
    load_dataset,
    tensors,
)
from supermimic.ollie.contact_learning import (
    RESPONSE_NAMES,
    ContactFitConfig,
    response_scales,
    rollout,
    training_losses,
)
from supermimic.ollie.learning import JointLearningConfig
from supermimic.ollie.physics import BOARD_JOINT_NAMES
from supermimic.ollie.position_codec import (
    BUFFER_NAME,
    CODECS,
    SCHEMA,
    PositionCodecContactWorldModel,
    load_codec_checkpoint,
)
from supermimic.ollie.state import JointStateLayout

DATA_SHA = "e46105385a51b5dc151064bbdff477e854b4ade155efb53963d737e7f6437e8b"
OLD_INITIAL_SHA = "b60eef34918c3a3675fd86ef17d9d43a62ead020e377bac365f81bd2f78e13b8"
OLD_BATCH_SHA = "ae98ebdb4fa5c43816595163fb09d13c57833cd16d54d3b50a81ad6d0b0703bb"
FROZEN_FILES = (
    "src/supermimic/ollie/state.py",
    "src/supermimic/ollie/learning.py",
    "src/supermimic/ollie/contact_learning.py",
    "scripts/local/fit_ollie_contact_wm.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_hash(weights):
    result = hashlib.sha256()
    for name, tensor in weights.items():
        if name != BUFFER_NAME:
            result.update(name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def vector_stats(error):
    return {
        "rmse_vector": float(error.square().sum(-1).mean().sqrt()),
        "rmse_axes": error.square().mean(tuple(range(error.ndim - 1))).sqrt().tolist(),
        "maximum_vector_norm": float(error.norm(dim=-1).max()),
        "maximum_absolute_axis": float(error.abs().max()),
    }


def closure(states, predictions):
    before = torch.cat((states[:, :1], predictions[:, :-1]), 1)
    position_step = predictions[..., 71:74] - before[..., 71:74]
    velocity, velocity_next = before[..., 82:85], predictions[..., 82:85]
    trap = position_step - 0.01 * (velocity + velocity_next)
    semi = position_step - 0.02 * velocity_next
    return {
        name: {
            "all_H8_steps": vector_stats(value),
            "H1": vector_stats(value[:, 0]),
            "H8": vector_stats(value[:, 7]),
        }
        for name, value in (("trapezoid_residual_m", trap), ("semiimplicit_residual_m", semi))
    }


@torch.no_grad()
def evaluate(model, values, data, held):
    result = {}
    predicted, _ = rollout(model, values["states"][:, 0], values["actions"])
    masks = {"train": ~held, "heldout": held}
    masks.update({f"anchor{a}": data["anchor_frame"] == a for a in np.unique(data["anchor_frame"])})
    for label, mask in masks.items():
        rows = torch.tensor(np.flatnonzero(mask), device=predicted.device)
        pair_rows = torch.tensor(
            np.flatnonzero(mask[values["pairs"][:, 0].cpu().numpy()]), device=predicted.device
        )
        record = diagnostics(model, values, rows, pair_rows)
        # Both heads are unsupervised and fixed; auxiliary MSE is not an outcome.
        record.pop("auxiliary_mse")
        truth = values["states"][rows, 1:]
        p = predicted[rows]
        for h in (1, 8):
            error = model.spec.error_vector(p[:, h - 1], truth[:, h - 1])
            record["state"][f"H{h}"]["normalized_state90"] = {
                "vector_rmse": float(error.square().sum(-1).mean().sqrt()),
                "per_component_rmse": float(error.square().mean().sqrt()),
                "units": "dimensionless original90 error scales, not raw mixed SI92",
            }
            for name, section in (
                ("robot_xyz_m", slice(0, 3)),
                ("robot_Vxyz_m_s", slice(36, 39)),
            ):
                record["state"][f"H{h}"][name] = vector_stats(
                    p[:, h - 1, section] - truth[:, h - 1, section]
                )
        record["predicted_position_velocity_closure"] = closure(values["states"][rows], p)
        record["native_position_velocity_closure"] = closure(values["states"][rows], truth)
        result[label] = record
    return result, predicted.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("logs/ollie_mu09_responses_01"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--execute-approved-fit", action="store_true")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("use a fresh output directory; preserving every prior result")
    started = time.monotonic()
    config = replace(ContactFitConfig(), auxiliary_weight=0.0, wall_seconds=600.0)
    data, pairs, response, held, source = load_dataset(args.dataset)
    if source["dataset_sha256"] != DATA_SHA:
        raise ValueError("Only the frozen72 mu.9 dataset is registered for this experiment")
    frozen = {p: digest(p) for p in FROZEN_FILES}
    manifest = {
        "schema": SCHEMA,
        "config": asdict(config),
        "source": source,
        "arms": CODECS,
        "only_intervention": "board p_next=p+.5*dt*(v+predicted_v_next); ignore board dp3",
        "unchanged": "state92/delta90,robot,board quaternion/wheels,main/head,loss/scales/batches",
        "truth_velocity_used_by_decoder": False,
        "auxiliary_weight_both_arms": 0.0,
        "paired_scales": response_scales().tolist(),
        "paired_response_names": RESPONSE_NAMES,
        "common_initial_weights_expected73": OLD_INITIAL_SHA,
        "full500_batch_indices_expected73": OLD_BATCH_SHA,
        "coverage": {
            "train": coverage(data, ~held, source["metadata"]["metric_names"]),
            "heldout": coverage(data, held, source["metadata"]["metric_names"]),
        },
        "frozen_legacy_files_sha256": frozen,
        "new_source_sha256": {
            str(Path(__file__).resolve()): digest(__file__),
            "src/supermimic/ollie/position_codec.py": digest(
                "src/supermimic/ollie/position_codec.py"
            ),
        },
        "actor_updates": 0,
        "board_bptt_allowed": False,
        "ollie_success_certified": False,
        "physical_profile": "72 explicit foot-deck mu.9, not current canonical mu.6 trials",
    }
    args.out_dir.mkdir(parents=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.inspect or not args.execute_approved_fit:
        print(json.dumps(manifest, indent=2))
        return
    torch.set_num_threads(2)
    values = tensors(data, pairs, response, args.device)
    model_cfg = JointLearningConfig(hidden=config.hidden)
    models = {}
    for mode in CODECS:
        torch.manual_seed(config.seed)
        models[mode] = PositionCodecContactWorldModel(
            JointStateLayout(BOARD_JOINT_NAMES), model_cfg, args.device, mode=mode
        )
    common_hashes = {mode: weight_hash(m.state_dict()) for mode, m in models.items()}
    if set(common_hashes.values()) != {OLD_INITIAL_SHA}:
        raise AssertionError("Main/head initialization differs from frozen73")
    initial_heads = {
        mode: {k: v.detach().cpu().clone() for k, v in m.contact_head.state_dict().items()}
        for mode, m in models.items()
    }
    manifest_hash = digest(manifest_path)

    def checkpoint(mode, updates, **extra):
        codec = models[mode].codec_metadata()
        return {
            "schema": SCHEMA,
            "arm": mode,
            "position_codec": codec,
            "model": models[mode].state_dict(),
            "learning_config": asdict(model_cfg),
            "metadata": {**manifest, "position_codec": codec},
            "manifest_sha256": manifest_hash,
            "common_initial_weights_sha256": common_hashes[mode],
            "updates": updates,
            "policy": None,
            **extra,
        }

    reports = {"initial": {}, "final": {}}
    for mode, model in models.items():
        atomic_save(args.out_dir / f"initial_{mode}.pt", checkpoint(mode, 0))
        reports["initial"][mode], prediction = evaluate(model, values, data, held)
        np.savez_compressed(args.out_dir / f"initial_{mode}_predictions.npz", predicted=prediction)
    optimizers = {
        mode: torch.optim.Adam(m.parameters(), lr=config.learning_rate)
        for mode, m in models.items()
    }
    train = torch.tensor(np.flatnonzero(~held), device=args.device)
    train_pairs = torch.tensor(np.flatnonzero(~held[pairs[:, 0]]), device=args.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history, completed, reason = [], dict.fromkeys(CODECS, 0), "maximum_updates"
    batch_hashes = {mode: hashlib.sha256() for mode in models}
    try:
        for update in range(config.updates):
            if time.monotonic() - started >= config.wall_seconds - 20:
                reason = "wall_budget_fit_reserve20s"
                break
            index = torch.randint(len(train), (config.batch_size,), generator=generator)
            pair_index = torch.randint(len(train_pairs), (config.batch_size,), generator=generator)
            rows, pr = train[index.to(args.device)], train_pairs[pair_index.to(args.device)]
            pair = values["pairs"][pr]
            entry = {"update": update + 1}
            for mode, model in models.items():
                state_loss, pair_loss, unused_aux_loss = training_losses(
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
                del unused_aux_loss
                loss = state_loss + config.paired_weight * pair_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss: {mode}")
                optimizers[mode].zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizers[mode].step()
                completed[mode] += 1
                batch_hashes[mode].update(index.numpy().tobytes() + pair_index.numpy().tobytes())
                entry[mode] = {
                    "state_loss": float(state_loss.detach()),
                    "paired_loss": float(pair_loss.detach()),
                    "total_loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm),
                }
            history.append(entry)
            if (update + 1) % 100 == 0:
                print(
                    json.dumps({"updates": completed, "wall_s": time.monotonic() - started}),
                    flush=True,
                )
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception as error:
        reason = f"error:{type(error).__name__}:{error}"
        raise
    finally:
        for mode in models:
            atomic_save(
                args.out_dir / f"{mode}.pt",
                checkpoint(
                    mode,
                    completed[mode],
                    optimizer=optimizers[mode].state_dict(),
                    batch_indices_sha256=batch_hashes[mode].hexdigest(),
                    stop_reason=reason,
                ),
            )
        result = {
            "updates": completed,
            "stop_reason": reason,
            "common_initial_weights_sha256": common_hashes,
            "arm_batch_indices_sha256": {k: v.hexdigest() for k, v in batch_hashes.items()},
            "actor_updates": 0,
            "board_bptt_allowed": False,
            "ollie_success_certified": False,
            "wall_seconds_before_diagnostics": time.monotonic() - started,
        }
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    readback = {}
    for mode, model in models.items():
        saved = torch.load(args.out_dir / f"{mode}.pt", weights_only=True, map_location="cpu")
        if not all(torch.isfinite(t).all() for t in saved["model"].values()):
            raise AssertionError("nonfinite saved checkpoint")
        load_codec_checkpoint(model, saved)
        readback[mode] = {
            "weights_only_readback_passed": True,
            "codec": model.codec_metadata(),
            "contact_head_unchanged": all(
                torch.equal(v.detach().cpu(), initial_heads[mode][k])
                for k, v in model.contact_head.state_dict().items()
            ),
            "batch_matches73_if500": completed[mode] != 500
            or batch_hashes[mode].hexdigest() == OLD_BATCH_SHA,
        }
        if not all(readback[mode][k] for k in ("contact_head_unchanged", "batch_matches73_if500")):
            raise AssertionError("head update or batch contract changed")
        if time.monotonic() - started < config.wall_seconds:
            reports["final"][mode], prediction = evaluate(model, values, data, held)
            np.savez_compressed(args.out_dir / f"{mode}_predictions.npz", predicted=prediction)
        else:
            reports["final"][mode] = {"skipped": "wall_budget"}
    legacy_unchanged = {p: digest(p) == expected for p, expected in frozen.items()}
    if not all(legacy_unchanged.values()):
        raise AssertionError("frozen legacy file changed during comparison")
    result.update(
        diagnostics=reports,
        checkpoint_readback=readback,
        frozen_legacy_files_unchanged=legacy_unchanged,
        wall_seconds=time.monotonic() - started,
    )
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    hashes = {p.name: digest(p) for p in args.out_dir.iterdir() if p.is_file()}
    (args.out_dir / "output_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
