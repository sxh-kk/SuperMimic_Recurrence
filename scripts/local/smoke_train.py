"""Validate the training pipeline on temporary synthetic motion, never benchmark quality."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from supermimic.config import TrainConfig
from supermimic.data.motion_convert import convert_csv_to_npz, load_g1_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="also validate Torch compilation")
    parser.add_argument("--full-size", action="store_true", help="use the standard networks, 4096 environments/rollouts and H=32")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--anchor-obs", action="store_true")
    args = parser.parse_args()
    if args.steps < 24:
        parser.error("--steps must be at least 24 to fill replay windows and update the models")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; a skipped CPU run is not a successful smoke test.")

    from supermimic.training.loop import Trainer

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="supermimic-smoke-") as temporary:
        root = Path(temporary)
        model = load_g1_model()
        raw = np.zeros((160, 36), dtype=np.float64)
        raw[:, :3] = model.qpos0[:3]
        raw[:, 2] = 0.8
        raw[:, 6] = 1.0  # CSV quaternion: xyzw. Converter writes wxyz to NPZ.
        raw[:, 7:] = model.qpos0[7:]
        csv = root / "synthetic.csv"
        np.savetxt(csv, raw, delimiter=",")
        convert_csv_to_npz(csv, root / "npz/synthetic.npz", input_fps=50, model=model)

        cfg = TrainConfig(run_name="synthetic-smoke", log_root=root / "logs", seed=0)
        cfg.data.npz_dir = root / "npz"
        cfg.env.num_envs = 4096 if args.full_size else 16
        cfg.env.replay_capacity = 1000 if args.full_size else 64
        cfg.env.obs_noise = False
        cfg.env.reset_noise = False
        cfg.env.domain_randomization = False
        cfg.env.sampling_mode = "start"
        cfg.env.anchor_obs = args.anchor_obs
        cfg.policy_trainer.obs_noise = False
        if not args.full_size:
            cfg.world_model.hidden = (128, 128)
            cfg.policy.encoder_hidden = (128, 64)
            cfg.policy.decoder_hidden = (128, 64)
            cfg.critic.hidden = (64, 32)
            cfg.wm_trainer.batch_size = 32
            cfg.wm_trainer.batches_per_update = 1
            cfg.policy_trainer.batch_size = 16
            cfg.policy_trainer.horizon = 8
            cfg.policy_trainer.critic_epochs = 1
            cfg.policy_trainer.critic_minibatches = 2
        mode = "default" if args.compile else None
        cfg.wm_trainer.compile = mode
        cfg.policy_trainer.compile = mode
        cfg.schedule.warmup_iterations = 16
        cfg.schedule.total_iterations = args.steps
        cfg.schedule.log_interval = args.steps
        cfg.schedule.checkpoint_interval = args.steps
        trainer = None
        try:
            trainer = Trainer(cfg)
            models = {"world_model": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
            before = {
                name: [parameter.detach().clone() for parameter in module.parameters()]
                for name, module in models.items()
            }
            trainer.run()
            if trainer.num_updates == 0:
                raise RuntimeError("No model updates were executed.")
            for name, module in models.items():
                parameters = list(module.parameters())
                if not all(torch.isfinite(parameter).all() for parameter in parameters):
                    raise RuntimeError(f"{name} contains non-finite parameters.")
                if all(torch.equal(old, new) for old, new in zip(before[name], parameters)):
                    raise RuntimeError(f"{name} did not change during training.")
            state = trainer.recorder.state_io.read()
            if not torch.isfinite(state).all():
                raise RuntimeError("Simulator state contains non-finite values.")
            with torch.no_grad():
                errors = trainer.wm_trainer.online_error(trainer.buffer, n=32, k=8)
            if not torch.isfinite(errors).all():
                raise RuntimeError("World-model open-loop predictions contain non-finite errors.")
            checkpoint = trainer.log_dir / "ckpt_latest.pt"
            if not checkpoint.is_file():
                raise RuntimeError("Checkpoint was not saved.")
            expected = next(trainer.policy.parameters()).detach().clone()
            with torch.no_grad():
                next(trainer.policy.parameters()).zero_()
            trainer.load(checkpoint)
            torch.testing.assert_close(next(trainer.policy.parameters()), expected)
            print(
                f"PASS: simulator + replay + world model + actor/critic + checkpoint reload; "
                f"steps={trainer.iteration}, updates={trainer.num_updates}, "
                f"compile={args.compile}, anchor_obs={args.anchor_obs}, "
                f"full_size={args.full_size}, nse8={errors[-1].item():.6f}, "
                f"elapsed={time.perf_counter() - started:.1f}s, "
                f"torch_peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                flush=True,
            )
            print("Synthetic motion smoke only; this is not a motion-tracking score.")
        finally:
            if trainer is not None:
                trainer.env.close()
                if trainer.writer is not None:
                    trainer.writer.close()


if __name__ == "__main__":
    main()
