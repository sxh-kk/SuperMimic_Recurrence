"""Export one deterministic first-episode rollout with an aligned reference ghost.

Rollout states are saved on CPU first, then rendered offscreen and piped to ffmpeg.
No episode auto-reset is allowed; terminal states remain visible in the video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args) -> None:
    # This must precede imports of MuJoCo, including transitive project imports.
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    files = discover_npz(args.npz_dir, names=[args.clip])
    if len(files) != 1:
        raise ValueError(f"expected exactly one NPZ for clip {args.clip!r}")
    configure_torch_backends()
    torch.set_num_threads(2)
    policy, checkpoint_cfg = load_policy_from_checkpoint(args.ckpt, args.device)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(files[0])], num_envs=1, future_frames=checkpoint_cfg["env"]["future_frames"],
        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
        seed=0, anchor_obs=checkpoint_cfg["env"].get("anchor_obs", False), debug_vis=True,
    ))
    cfg.auto_reset = False
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
    cfg.viewer.max_extra_envs = 0
    env = renderer = encoder = None
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(), "status": "starting",
        "checkpoint": str(args.ckpt), "checkpoint_sha256": sha256(args.ckpt),
        "clip": args.clip, "npz": str(files[0]), "npz_sha256": sha256(files[0]),
        "requested_seconds": args.seconds, "fps": args.fps,
        "width": args.width, "height": args.height, "device": args.device,
        "render_backend": os.environ["MUJOCO_GL"], "auto_reset": False,
        "reference_alignment": "initial reference frame, then each action's pre-step target frame",
    }
    temporary = args.out.with_name(f".{args.out.stem}.{os.getpid()}.part.mp4")
    try:
        env = ManagerBasedRlEnv(cfg, device=args.device)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        clip_start, clip_frames = int(lib.clip_start[0]), int(lib.clip_len[0])
        if clip_frames < 2:
            raise ValueError("clip needs at least two reference frames")
        dt = env.step_dt
        if args.fps > 1.0 / dt + 1e-6:
            raise ValueError("video fps must not exceed the control frequency")
        cmd.forced_frame[:] = clip_start
        obs, _ = env.reset()
        samples = []

        def capture(step: int, reference_frame: int) -> None:
            samples.append((
                step, reference_frame,
                env.sim.data.qpos[0].detach().cpu().clone(),
                env.sim.data.qvel[0].detach().cpu().clone(),
            ))

        capture(0, clip_start)
        # Stop at T-1 transitions ourselves; do not depend on a delayed clip_end flag.
        maximum_steps = min(math.ceil(args.seconds / dt), clip_frames - 1)
        next_capture = 1.0 / args.fps
        terminated = truncated = False
        steps = 0
        with torch.inference_mode():
            for steps in range(1, maximum_steps + 1):
                target_frame = int(cmd.frame[0])
                obs, _, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
                terminated, truncated = bool(term[0]), bool(tout[0])
                if steps * dt + 1e-9 >= next_capture or terminated or truncated or steps == maximum_steps:
                    capture(steps, target_frame)
                    while next_capture <= steps * dt + 1e-9:
                        next_capture += 1.0 / args.fps
                if terminated or truncated:
                    break
        end_reason = (
            "terminated" if terminated else "truncated" if truncated
            else "clip_complete" if steps == clip_frames - 1 else "requested_duration"
        )
        metadata.update(
            steps=steps, simulation_duration_seconds=steps * dt,
            terminated=terminated, truncated=truncated, end_reason=end_reason,
            reference_frames=clip_frames, reference_end_reached=steps == clip_frames - 1,
            terminal_frame_included=terminated or truncated,
            terminal_frame_policy="auto_reset=False; final simulated state is included",
            rendered_frames=len(samples), encoded_duration_seconds=len(samples) / args.fps,
            sampled_steps=[sample[0] for sample in samples],
            sampled_reference_frames=[sample[1] - clip_start for sample in samples],
            timing_note="CFR sampling rounds to control steps; the last image is held one video frame",
        )
        outcome = f"Outcome - {end_reason} at {steps * dt:.2f} s"
        labels = (
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            "text='Blue ghost - Reference | Solid - Policy':x=16:y=16:fontsize=23:"
            "fontcolor=white:box=1:boxcolor=black@0.65,"
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"text='{outcome}':x=16:y=h-45:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65"
        )
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{args.width}x{args.height}",
            "-framerate", f"{args.fps:g}", "-i", "pipe:0", "-an", "-vf", labels,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        renderer = OffscreenRenderer(
            env.sim.mj_model, cfg.viewer, env.scene,
            sim_model=env.sim.model, expanded_fields=env.sim.expanded_fields,
        )
        renderer.initialize()
        if env.sim.mj_model.nmocap:
            raise ValueError("this exporter currently supports the G1 scene without mocap bodies")
        for _, reference_frame, qpos, qvel in samples:
            cpu_data = SimpleNamespace(nworld=1, qpos=qpos[None], qvel=qvel[None])
            # Commands advance at step end. Change only the ghost's cached reference while drawing;
            # policy goals, history, physics states, and the command frame are never modified.
            saved_ref = cmd._ref
            try:
                cmd._ref = lib.ref(torch.tensor([reference_frame], device=args.device))
                renderer.update(cpu_data, debug_vis_callback=cmd.debug_vis)
            finally:
                cmd._ref = saved_ref
            image = renderer.render()
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        _, error = encoder.communicate(timeout=120)
        if encoder.returncode:
            raise RuntimeError(f"ffmpeg failed: {error.decode(errors='replace')}")
        temporary.replace(args.out)
        metadata.update(status="completed", output=str(args.out), bytes=args.out.stat().st_size)
    except Exception as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        if encoder is not None and encoder.poll() is None:
            # Only this exporter-owned encoder is terminated; no training process is touched.
            encoder.kill()
            encoder.communicate()
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        temporary.unlink(missing_ok=True)
        args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/lafan1")
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.seconds, args.fps)):
        parser.error("seconds and fps must be positive and finite")
    if min(args.width, args.height) < 64 or args.width % 2 or args.height % 2:
        parser.error("width and height must be even numbers >=64 for yuv420p")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is not installed")
    args.ckpt = args.ckpt.resolve(strict=True)
    args.out = args.out.resolve()
    if args.out.suffix.lower() != ".mp4":
        parser.error("out must have the .mp4 extension")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("output MP4/JSON already exists; choose a new filename")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(args)


if __name__ == "__main__":
    main()
