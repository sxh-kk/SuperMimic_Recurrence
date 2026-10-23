"""Alternate real joint physics, learned dynamics, and frozen-model actor BPTT.

Phase resets are training data augmentation only. Selection uses an actual
continuous ground-start rollout with no intermediate state assignment. This is
an engineering full-state joint extension, not paper-exact SuperMimic sensing.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from supermimic.ollie.gpu_env import JointEnvConfig, JointPhysicsEnv
from supermimic.ollie.learning import (
    JointLearningConfig,
    JointResidualPolicy,
    JointWorldModel,
    tracking_loss,
)
from supermimic.ollie.reference import JointReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wm-init", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--collect-steps", type=int, default=48)
    parser.add_argument("--model-updates", type=int, default=100)
    parser.add_argument("--policy-updates", type=int, default=20)
    parser.add_argument("--model-window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-chunks", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--exploration", type=float, default=0.025)
    parser.add_argument("--residual-limit", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.num_envs < 8 or args.iterations < 1 or args.model_window < 1
            or args.collect_steps < args.model_window or args.horizon < 1
            or args.batch_size < 1 or args.eval_every < 1 or args.replay_chunks < 1):
        parser.error("positive budgets and at least eight environments required")
    if (args.out_dir / "latest.pt").exists():
        parser.error("choose a fresh output directory; --resume reads a separate checkpoint")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    reference = JointReference.load(args.reference)
    env_cfg = JointEnvConfig(num_envs=args.num_envs)
    if reference.metadata["physics_fps"] != 1 / env_cfg.control_dt:
        raise ValueError("reference and simulator control rates differ")
    env = JointPhysicsEnv(env_cfg, args.device)
    if reference.layout != env.layout:
        raise ValueError("reference and simulator joint schemas differ")
    refs = torch.tensor(reference.state, dtype=torch.float32, device=args.device)
    cfg = JointLearningConfig(residual_limit_rad=args.residual_limit)
    model = JointWorldModel(env.layout, cfg, args.device)
    policy = JointResidualPolicy(env.layout, cfg, args.device)
    wm_opt = torch.optim.Adam(model.parameters(), lr=cfg.model_lr)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)
    if args.wm_init:
        initial = torch.load(args.wm_init, map_location=args.device, weights_only=False)
        if tuple(initial["board_joint_names"]) != env.layout.board_joint_names:
            raise ValueError("initial world model joint schema differs")
        model.load_state_dict(initial["model"])
    if args.resume:
        initial = torch.load(args.resume, map_location=args.device, weights_only=False)
        if initial["schema"] != "supermimic.ollie.actor.v1":
            raise ValueError("unsupported actor checkpoint")
        if initial["learning_config"] != asdict(cfg):
            raise ValueError("resume learning configuration differs")
        model.load_state_dict(initial["model"])
        policy.load_state_dict(initial["policy"])
        wm_opt.load_state_dict(initial["model_optimizer"])
        actor_opt.load_state_dict(initial["policy_optimizer"])
    offsets = torch.tensor(cfg.reference_offsets, device=args.device)
    lower = env.joint_limits[:, 0] + 0.005
    upper = env.joint_limits[:, 1] - 0.005
    replay = []
    train_count = args.num_envs - max(args.num_envs // 8, 1)
    best_score = -float("inf")
    history = []

    def goals(frame):
        return refs[(frame[..., None] + offsets).clamp(0, len(refs) - 1)]

    def sample_windows(training, count, window):
        chunks = torch.randint(len(replay), (count,), device=args.device)
        ids = torch.randint(0 if training else train_count,
                            train_count if training else args.num_envs,
                            (count,), device=args.device)
        starts = torch.randint(args.collect_steps - window + 1, (count,), device=args.device)
        all_states = torch.stack([chunk[0] for chunk in replay])
        all_actions = torch.stack([chunk[1] for chunk in replay])
        all_frames = torch.stack([chunk[2] for chunk in replay])
        times = starts[None] + torch.arange(window + 1, device=args.device)[:, None]
        return (all_states[chunks[None], times, ids[None]],
                all_actions[chunks[None], times[:-1], ids[None]],
                all_frames[chunks, ids] + starts)

    @torch.no_grad()
    def real_evaluate(iteration):
        initial = refs[0].expand(args.num_envs, -1).clone()
        initial[:, 36:71] = 0
        initial[:, 82:] = 0
        state = env.reset(state=initial).clone()
        states, actions, diagnostics, losses = [state[0].cpu()], [], [], []
        for frame in range(1, len(refs)):
            index = torch.full((args.num_envs,), frame, device=args.device)
            action = policy(state, goals(index)).clamp(lower, upper)
            state = env.step(action).clone()
            if not torch.isfinite(state).all():
                raise RuntimeError("nonfinite real evaluation")
            states.append(state[0].cpu())
            actions.append(action[0].cpu())
            losses.append(float(tracking_loss(model.spec, state[:1], refs[frame:frame + 1])[0]))
            diagnostic = env.diagnostics()
            diagnostics.append({key: float(value[0]) for key, value in diagnostic.items()})
        upright = min(row["root_upright"] for row in diagnostics)
        height = min(row["root_height"] for row in diagnostics)
        # Ranking is diagnostic only. Contact-based success is independently
        # evaluated at physics substeps; this score cannot certify an Ollie.
        mean_loss = sum(losses) / len(losses)
        score = -mean_loss - 10 * max(0, 0.6 - upright) - 10 * max(0, 0.35 - height)
        result = {"iteration": iteration, "tracking_loss": mean_loss,
                  "minimum_upright": upright, "minimum_root_height": height,
                  "selection_score": score, "ollie_success_certified": False,
                  "ground_start_zero_velocity": True, "intermediate_resets": 0}
        torch.save({"states": torch.stack(states), "actions": torch.stack(actions),
                    "diagnostics": diagnostics, "metrics": result,
                    "environment_config": asdict(env_cfg)},
                   args.out_dir / f"real_rollout_{iteration:04d}.pt")
        return result

    def save(path, iteration):
        torch.save({"schema": "supermimic.ollie.actor.v1", "policy": policy.state_dict(),
                    "model": model.state_dict(), "policy_optimizer": actor_opt.state_dict(),
                    "model_optimizer": wm_opt.state_dict(), "learning_config": asdict(cfg),
                    "environment_config": asdict(env_cfg),
                    "board_joint_names": env.layout.board_joint_names,
                    "reference": str(args.reference.resolve()), "iteration": iteration,
                    "seed": args.seed, "observations": "full simulator state engineering prototype",
                    "phase_resets": "training only; evaluation has one ground-start reset",
                    "ollie_success_certified": False}, path)

    initial_metrics = real_evaluate(0)
    print(json.dumps({"initial_real_evaluation": initial_metrics}), flush=True)
    save(args.out_dir / "initial.pt", 0)
    for iteration in range(1, args.iterations + 1):
        # Independent worlds are whole episodes; held-out world IDs never enter
        # model fitting or actor BPTT. Random phases increase physical coverage.
        starts = torch.randint(max(1, len(refs) - args.collect_steps),
                               (args.num_envs,), device=args.device)
        starts[::4] = 0
        initial = refs[starts].clone()
        at_start = starts == 0
        initial[at_start, 36:71] = 0
        initial[at_start, 82:] = 0
        state = env.reset(state=initial).clone()
        states, actions = [state], []
        noise = torch.randn(args.num_envs, 29, device=args.device) * args.exploration
        with torch.no_grad():
            for step in range(args.collect_steps):
                if step % 8 == 0:
                    noise = torch.randn_like(noise) * args.exploration
                action = (policy(state, goals(starts + step + 1)) + noise).clamp(lower, upper)
                state = env.step(action).clone()
                if not torch.isfinite(state).all():
                    raise RuntimeError("nonfinite real training collection")
                states.append(state)
                actions.append(action)
        replay.append((torch.stack(states), torch.stack(actions), starts))
        replay = replay[-args.replay_chunks:]
        model.requires_grad_(True)
        model.train()
        for _ in range(args.model_updates):
            batch_states, batch_actions, _ = sample_windows(True, args.batch_size, args.model_window)
            wm_loss, _ = model.window_loss(batch_states, batch_actions)
            wm_opt.zero_grad(set_to_none=True)
            if not torch.isfinite(wm_loss):
                raise RuntimeError("nonfinite world model loss")
            wm_loss.backward()
            wm_gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(wm_gradient):
                raise RuntimeError("nonfinite world model gradient")
            wm_opt.step()
        with torch.no_grad():
            held_states, held_actions, _ = sample_windows(False, args.batch_size, args.model_window)
            validation, validation_details = model.window_loss(held_states, held_actions)
        # Model weights stay fixed; derivatives with respect to policy actions
        # remain live through the complete imagined trajectory.
        model.eval().requires_grad_(False)
        actor_loss = refs.new_zeros(())
        actor_gradient = refs.new_zeros(())
        for _ in range(args.policy_updates):
            batch_states, batch_actions, frame = sample_windows(True, args.batch_size, 1)
            predicted = batch_states[0].detach()
            previous = batch_actions[0].detach()
            actor_loss = predicted.new_zeros(())
            for step in range(args.horizon):
                target_frame = (frame + step + 1).clamp(max=len(refs) - 1)
                action = policy(predicted, goals(target_frame)).clamp(lower, upper)
                predicted = model.step(predicted, action)
                loss = tracking_loss(model.spec, predicted, refs[target_frame])
                residual = action - refs[target_frame, 7:36]
                rate = action - previous
                actor_loss = actor_loss + (0.97 ** step) * (
                    loss.mean() + 0.1 * residual.square().mean() + 0.02 * rate.square().mean())
                previous = action
            actor_loss = actor_loss / args.horizon
            actor_opt.zero_grad(set_to_none=True)
            if not torch.isfinite(actor_loss):
                raise RuntimeError("nonfinite actor loss")
            actor_loss.backward()
            actor_gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            if not torch.isfinite(actor_gradient):
                raise RuntimeError("nonfinite actor gradient")
            actor_opt.step()
        row = {"iteration": iteration, "seconds": time.monotonic() - started,
               "physical_transitions": iteration * args.num_envs * args.collect_steps,
               "world_model_loss": float(wm_loss.detach()), "heldout_model_loss": float(validation),
               **{key: float(value) for key, value in validation_details.items()},
               "actor_loss": float(actor_loss.detach()), "actor_gradient": float(actor_gradient)}
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            real = real_evaluate(iteration)
            row["real_evaluation"] = real
            if real["selection_score"] > best_score:
                best_score = real["selection_score"]
                save(args.out_dir / "best.pt", iteration)
            save(args.out_dir / "latest.pt", iteration)
        history.append(row)
        with (args.out_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    torch.save({"chunks": [(s.cpu(), a.cpu(), f.cpu()) for s, a, f in replay],
                "train_environment_count": train_count, "source": "real MuJoCo-Warp physics",
                "phase_resets_training_only": True}, args.out_dir / "physical_replay.pt")
    result = {"arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "initial_real_evaluation": initial_metrics, "history": history,
              "seconds": time.monotonic() - started, "ollie_success_certified": False}
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
