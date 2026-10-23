"""Fit the joint policy to a verified real-control prefix, then screen in physics.

This is behavior cloning for initialization. Neither a low fitting error nor a
successful preparation prefix certifies an Ollie or a model-based policy update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.local.evaluate_ollie_policy import evaluate
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.gpu_env import JointEnvConfig
from supermimic.ollie.learning import JointLearningConfig, JointResidualPolicy, JointWorldModel
from supermimic.ollie.physics import BOARD_JOINT_NAMES
from supermimic.ollie.reference import JointReference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--base-residual-limits", type=Path, required=True)
    parser.add_argument("--policy-init", type=Path, help="Weights-only initialization; refitted to this teacher/goals")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-updates", type=int, default=500)
    parser.add_argument("--dagger-noise", type=float, default=0.002)
    args = parser.parse_args()
    if args.updates < 1 or args.eval_every < 1 or not 0 < args.lr < math.inf:
        parser.error("positive finite budgets and learning rate required")
    if (args.dagger_rounds < 0 or args.dagger_updates < 1
            or not math.isfinite(args.dagger_noise) or args.dagger_noise < 0):
        parser.error("nonnegative DAgger rounds/noise and positive updates required")
    if args.out_dir.exists():
        parser.error("choose a fresh output directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    reference = JointReference.load(args.reference)
    reference_hash = digest(args.reference)
    with np.load(args.teacher, allow_pickle=False) as teacher:
        if not bool(teacher["verified_teacher"].item()):
            raise ValueError("only a physically verified teacher may initialize this policy")
        if str(teacher["reference_sha256"].item()) != reference_hash:
            raise ValueError("teacher reference hash differs")
        if tuple(teacher["joint_names"].tolist()) != G1_JOINT_NAMES:
            raise ValueError("teacher robot joint order differs")
        if tuple(teacher["board_joint_names"].tolist()) != BOARD_JOINT_NAMES:
            raise ValueError("teacher board joint order differs")
        offsets = tuple(int(value) for value in teacher["reference_offsets"])
        state = teacher["state"].copy()
        action = teacher["action"].copy()
        goals = teacher["future_goals"].copy()
        indices = teacher["reference_goal_indices"].copy()
        times = teacher["time_s"].copy()
        teacher_metrics = json.loads(str(teacher["metadata_json"].item()))
    if (not all(teacher_metrics.get("gates", {}).values())
            or not teacher_metrics.get("gates")
            or teacher_metrics.get("initializations") != 1
            or teacher_metrics.get("intermediate_resets") != 0
            or teacher_metrics.get("board_actuators") != 0
            or teacher_metrics.get("forces_applied") is not False):
        raise ValueError("teacher must pass its declared physical gates without hidden controls")
    count = len(action)
    if (state.shape != (count, 92) or action.shape != (count, 29)
            or goals.shape != (count, len(offsets), 92)
            or indices.shape != (count, len(offsets))
            or not np.isfinite(np.concatenate([state.ravel(), action.ravel(), goals.ravel()])).all()):
        raise ValueError("invalid teacher tensor shapes or nonfinite values")
    expected = np.minimum(np.arange(1, count + 1)[:, None] + offsets, len(reference.state) - 1)
    if (not np.array_equal(indices, expected)
            or not np.array_equal(goals, reference.state[indices].astype(np.float32))
            or not np.allclose(times, np.arange(count) / 50, rtol=0, atol=1e-8)):
        raise ValueError("teacher time/next-goal alignment differs from native evaluation")
    base_doc = json.loads(args.base_residual_limits.read_text())
    if tuple(base_doc["joint_names"]) != G1_JOINT_NAMES:
        raise ValueError("base residual-limit joint order differs")
    base_limits = np.asarray(base_doc["limits_rad"])
    needed = np.abs(action - goals[:, 0, 7:36]).max(axis=0)
    # Explicitly record changes; do not silently clip away teacher feedforward.
    limits = np.maximum(base_limits, np.ceil(1.10 * needed / 0.05) * 0.05)
    limits_doc = {
        "joint_names": list(G1_JOINT_NAMES), "limits_rad": limits.tolist(),
        "teacher_max_abs_residual_rad": needed.tolist(),
        "base_limits_rad": base_limits.tolist(),
        "rule": "max(base, ceil(1.10 * actual_teacher_peak / 0.05) * 0.05)",
        "teacher_sha256": digest(args.teacher), "reference_sha256": reference_hash,
        "original_mechanical_joint_and_torque_limits_unchanged": True,
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "residual_limits.json").write_text(json.dumps(limits_doc, indent=2) + "\n")
    config = JointLearningConfig(residual_limits_rad=tuple(map(float, limits)), reference_offsets=offsets)
    policy = JointResidualPolicy(reference.layout, config, args.device)
    model = JointWorldModel(reference.layout, config, args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.model_lr)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_lr)
    s, a, g = [torch.as_tensor(v, dtype=torch.float32, device=args.device) for v in (state, action, goals)]
    if args.policy_init:
        previous = torch.load(args.policy_init, map_location=args.device, weights_only=True)
        if (previous.get("schema") != "supermimic.ollie.actor.v1"
                or tuple(previous["board_joint_names"]) != BOARD_JOINT_NAMES
                or asdict(JointLearningConfig(**previous["learning_config"])) != asdict(config)):
            raise ValueError("initial policy network/limit/goal contract differs")
        policy.load_state_dict(previous["policy"], strict=True)
    else:
        with torch.no_grad():
            average = ((a - g[:, 0, 7:36]) / policy.residual_limits).mean(0).clamp(-0.999, 0.999)
            policy.decoder.net[-1].bias.copy_(torch.atanh(average))
    history = []
    best_native_cost = float("inf")

    def save(path, update):
        temporary = path.with_suffix(".tmp")
        torch.save({
            "schema": "supermimic.ollie.actor.v1", "policy": policy.state_dict(),
            "model": model.state_dict(), "model_optimizer": model_optimizer.state_dict(),
            "policy_optimizer": actor_optimizer.state_dict(), "learning_config": asdict(config),
            "environment_config": asdict(JointEnvConfig()),
            "board_joint_names": BOARD_JOINT_NAMES, "reference": str(args.reference.resolve()),
            "reference_sha256": reference_hash, "iteration": 0,
            "policy_updates_completed": 0, "behavior_cloning_updates_completed": update,
            "policy_initialization": "behavior_cloning", "world_model_trained": False,
            "teacher": str(args.teacher.resolve()), "teacher_sha256": digest(args.teacher),
            "policy_initialization_source": str(args.policy_init.resolve()) if args.policy_init else None,
            "policy_initialization_source_sha256": digest(args.policy_init) if args.policy_init else None,
            "teacher_verified_seconds": count / 50, "teacher_metrics": teacher_metrics,
            "seed": args.seed, "ollie_success_certified": False,
            "observations": "full simulator state engineering prototype",
        }, temporary)
        temporary.replace(path)

    def screen(checkpoint, output):
        native = evaluate(checkpoint, args.reference, output, seconds=count / 50)
        with np.load(output / "rollout.npz", allow_pickle=False) as rollout:
            maximum_xy = float(np.linalg.norm(rollout["state"][:, :2] - rollout["state"][:, 71:73], axis=-1).max())
        minimum_upright = native["minimum_robot_upright_cosine"]
        cost = (native["ranking_cost"] + 10 * max(0, 0.85 - minimum_upright)
                + 10 * max(0, maximum_xy - 0.2))
        # A prefix is moving, so the full Ollie end-standing criterion is not
        # its selection rule. Keep finite-window screening distinct from it.
        return native, {"cost": cost, "minimum_upright": minimum_upright,
                        "maximum_root_board_xy_m": maximum_xy,
                        "full_ollie_success": False,
                        "rule": "native ranking + 10*max(.85-minupright,0) + 10*max(maxXY-.20,0)"}

    for update in range(1, args.updates + 1):
        policy.train()
        prediction = policy(s, g)
        loss = (prediction - a).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(gradient):
            raise RuntimeError("nonfinite behavior-cloning loss or gradient")
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * update / args.updates)))
        if update % args.eval_every == 0 or update == args.updates:
            policy.eval()
            with torch.no_grad():
                error = policy(s, g) - a
                row = {"update": update, "training_action_rmse_rad": float(error.square().mean().sqrt()),
                       "training_action_max_abs_rad": float(error.abs().max()),
                       "training_fit_only_no_heldout_generalization_claim": True,
                       "seconds": time.monotonic() - started}
            checkpoint = args.out_dir / f"bc_{update:05d}.pt"
            save(checkpoint, update)
            native, screening = screen(checkpoint, args.out_dir / f"native_{update:05d}")
            row["native_preparation_prefix"] = native
            row["prefix_screening"] = screening
            if screening["cost"] < best_native_cost:
                best_native_cost = screening["cost"]
                save(args.out_dir / "best.pt", update)
            history.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    completed_updates = args.updates
    if args.dagger_rounds:
        import mujoco

        from scripts.local.track_ollie_quasistatic import TrackingExperiment

        surface_patches = teacher_metrics.get("contact_surface_patches", False)
        if not isinstance(surface_patches, bool):
            raise TypeError("Teacher surface-patch mode must be an explicit boolean")
        expert = TrackingExperiment(reference, seconds=count / 50, surface_patches=surface_patches)
        limits_low = expert.solver.target_limits[:, 0] + 0.005
        limits_high = expert.solver.target_limits[:, 1] - 0.005
        rng = np.random.default_rng(args.seed)
        real_data = []
        for round_index in range(args.dagger_rounds):
            beta = 1 - round_index / max(args.dagger_rounds - 1, 1)
            data, native_model, index = expert.data, expert.model, expert.index
            mujoco.mj_resetData(native_model, data)
            data.qpos[:] = expert.qpos[0]
            data.qvel[:] = 0
            data.ctrl[index.robot_actuators] = reference.robot_state[0, 7:36]
            mujoco.mj_forward(native_model, data)
            added_state, added_goals, added_labels = [], [], []
            policy.eval()
            for step in range(count):
                frame = step + 1
                state_now = expert.indices.pack(data.qpos, data.qvel).numpy()
                future = reference.state[np.minimum(frame + np.array(offsets), len(reference.state) - 1)]
                _, raw = expert.solver.pd_target(reference.robot_state[frame, 7:36],
                                               reference.robot_state[frame, 42:71], expert.solutions[frame])
                correction, _ = expert.feedback(frame, 0.45)
                expert_action = np.clip(raw + correction, limits_low, limits_high)
                with torch.inference_mode():
                    proposed = policy(torch.as_tensor(state_now, dtype=torch.float32, device=args.device)[None],
                                      torch.as_tensor(future, dtype=torch.float32, device=args.device)[None])[0].cpu().numpy()
                executed = np.clip(beta * expert_action + (1 - beta) * proposed
                                   + rng.normal(0, args.dagger_noise, 29), limits_low, limits_high)
                upright = float(data.xmat[expert.root_body, 8])
                representable = bool(np.all(np.abs(expert_action - future[0, 7:36]) < limits))
                keep = upright > 0.7 and representable
                if keep:
                    added_state.append(state_now)
                    added_goals.append(future)
                    added_labels.append(expert_action)
                data.ctrl[index.robot_actuators] = executed
                mujoco.mj_step(native_model, data, nstep=8)
                mujoco.mj_forward(native_model, data)
                next_state = expert.indices.pack(data.qpos, data.qvel).numpy()
                if not np.isfinite(next_state).all() or abs(data.time - frame * 0.02) > 1e-8:
                    raise RuntimeError("nonfinite or reset DAgger physics")
                real_data.append((state_now, executed, expert_action, next_state, future, round_index, step / 50, keep))
            if added_labels:
                s = torch.cat([s, torch.as_tensor(np.asarray(added_state), dtype=torch.float32, device=args.device)])
                g = torch.cat([g, torch.as_tensor(np.asarray(added_goals), dtype=torch.float32, device=args.device)])
                a = torch.cat([a, torch.as_tensor(np.asarray(added_labels), dtype=torch.float32, device=args.device)])
            policy.train()
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.1
            for _ in range(args.dagger_updates):
                batch = torch.randint(len(s), (min(len(s), 256),), device=args.device)
                error = policy(s[batch], g[batch]) - a[batch]
                loss = error.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                if not torch.isfinite(loss) or not torch.isfinite(gradient):
                    raise RuntimeError("nonfinite DAgger supervised update")
                optimizer.step()
            completed_updates += args.dagger_updates
            checkpoint = args.out_dir / f"dagger_{round_index:02d}.pt"
            save(checkpoint, completed_updates)
            native, screening = screen(checkpoint, args.out_dir / f"native_dagger_{round_index:02d}")
            row = {"dagger_round": round_index, "expert_mixing_beta": beta,
                   "added_expert_labels": len(added_labels), "total_labels": len(s),
                   "actual_transitions": len(real_data), "native_preparation_prefix": native,
                   "prefix_screening": screening}
            if screening["cost"] < best_native_cost:
                best_native_cost = screening["cost"]
                save(args.out_dir / "best.pt", completed_updates)
            history.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
        np.savez_compressed(args.out_dir / "dagger_actual_transitions.npz",
                            state=np.stack([r[0] for r in real_data]),
                            action=np.stack([r[1] for r in real_data]),
                            expert_label=np.stack([r[2] for r in real_data]),
                            next_state=np.stack([r[3] for r in real_data]),
                            future_goals=np.stack([r[4] for r in real_data]),
                            episode_id=np.array([r[5] for r in real_data]),
                            time_s=np.array([r[6] for r in real_data]),
                            label_used=np.array([r[7] for r in real_data]),
                            schema=np.array("supermimic.ollie.entity_state.v1"),
                            robot_joint_names=np.array(G1_JOINT_NAMES),
                            board_joint_names=np.array(BOARD_JOINT_NAMES),
                            control_dt=np.array(0.02),
                            reference_sha256=np.array(reference_hash),
                            verified_teacher=np.array(False),
                            source=np.array("Actual mixed-policy physics with counterfactual expert labels; not a verified demonstration"))
    save(args.out_dir / "latest.pt", completed_updates)
    (args.out_dir / "result.json").write_text(json.dumps({
        "method": "verified preparation-prefix behavior cloning; not WM/BPTT training",
        "reference_sha256": reference_hash, "teacher_sha256": digest(args.teacher),
        "samples": count, "teacher_seconds": count / 50, "history": history,
        "best_native_prefix_cost": best_native_cost, "ollie_success_certified": False,
        "seconds": time.monotonic() - started,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
