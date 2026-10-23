"""Fit a first joint world model on real GPU stance/crouch transitions.

This is a fixed-scene model smoke/pilot, not an Ollie policy success claim.
Held-out environments are entire separate episodes, not overlapping windows.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from supermimic.ollie.gpu_env import JointEnvConfig, JointPhysicsEnv
from supermimic.ollie.learning import JointLearningConfig, JointWorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--collect-steps", type=int, default=48)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.num_envs < 8 or args.collect_steps < args.window or args.window < 1
            or args.updates < 1 or args.batch_size < 1):
        parser.error("need at least 8 environments and collect_steps >= window >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "model.pt").exists():
        parser.error("choose a new output directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    cfg = JointEnvConfig(num_envs=args.num_envs)
    env = JointPhysicsEnv(cfg, args.device)
    initial = env.state().clone()
    nominal = initial[:, 7:36].clone()
    amplitude = torch.rand(args.num_envs, 1, device=args.device) * 0.45
    frequency = 0.5 + torch.rand(args.num_envs, 1, device=args.device)
    random_delta = torch.randn_like(nominal) * 0.025
    states, actions = [initial], []
    for step in range(args.collect_steps):
        target = nominal + random_delta
        crouch = amplitude[:, 0] * torch.sin(torch.pi * frequency[:, 0] * step * cfg.control_dt).square()
        for offset in (0, 6):
            target[:, offset] -= crouch * 0.5
            target[:, offset + 3] += crouch
            target[:, offset + 4] -= crouch * 0.5
        target = target.clamp(env.joint_limits[:, 0] + 0.005, env.joint_limits[:, 1] - 0.005)
        actions.append(target.clone())
        states.append(env.step(target).clone())
    states, actions = torch.stack(states), torch.stack(actions)
    if not torch.isfinite(states).all():
        raise RuntimeError("physical collection produced nonfinite states")
    collection_seconds = time.monotonic() - started
    # Environment IDs define independent full-rollout episodes; hold out the last eighth.
    train_count = args.num_envs - max(args.num_envs // 8, 1)
    learning_cfg = JointLearningConfig()
    model = JointWorldModel(env.layout, learning_cfg, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_cfg.model_lr)

    def windows(train: bool, batch_size: int):
        ids = torch.randint(0 if train else train_count, train_count if train else args.num_envs,
                            (batch_size,), device=args.device)
        start = torch.randint(0, args.collect_steps - args.window + 1,
                              (batch_size,), device=args.device)
        state_index = start[None] + torch.arange(args.window + 1, device=args.device)[:, None]
        action_index = state_index[:-1]
        return states[state_index, ids], actions[action_index, ids]

    holdout = windows(False, 256)
    with torch.no_grad():
        initial_loss, initial_metrics = model.window_loss(*holdout)
    metrics = []
    for update in range(args.updates):
        loss, _ = model.window_loss(*windows(True, args.batch_size))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite model loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise RuntimeError("nonfinite model gradient")
        optimizer.step()
        if update % 100 == 0 or update == args.updates - 1:
            with torch.no_grad():
                validation, validation_metrics = model.window_loss(*holdout)
            row = {"update": update + 1, "training_loss": float(loss.detach()),
                   "validation_loss": float(validation), "gradient_norm": float(gradient),
                   **{key: float(value) for key, value in validation_metrics.items()}}
            metrics.append(row)
            print(json.dumps(row), flush=True)
    torch.save({
        "schema": "supermimic.ollie.wm_pilot.v1", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "learning_config": asdict(learning_cfg),
        "environment_config": asdict(cfg), "board_joint_names": env.layout.board_joint_names,
        "seed": args.seed, "updates": args.updates,
        "phase_coverage": "stance/crouch attempts; no verified target pop/flight/landing coverage",
    }, args.out_dir / "model.pt")
    torch.save({"states": states.cpu(), "actions": actions.cpu(), "train_environment_count": train_count},
               args.out_dir / "physical_transitions.pt")
    result = {
        "kind": "joint world-model pilot; not a trained Ollie policy",
        "state_dim": env.layout.state_dim, "delta_dim": env.layout.delta_dim,
        "feature_dim": env.layout.feature_dim,
        "physical_transitions": args.num_envs * args.collect_steps,
        "collection_seconds_including_setup": collection_seconds,
        "initial_validation_loss": float(initial_loss),
        "initial_validation_metrics": {key: float(value) for key, value in initial_metrics.items()},
        "metrics": metrics, "seconds": time.monotonic() - started,
        "environment": asdict(cfg), "config": asdict(learning_cfg),
        "phase_coverage": "stance/crouch attempts; no verified target events", "ollie_policy_trained": False,
        "no_board_actuators": True, "step_never_writes_qpos_or_applies_external_forces": True,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
