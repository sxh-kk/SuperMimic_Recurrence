"""Render saved native physics and compare it with the entire source video.

No simulation is run: saved qpos are used only in a separate rendering MjData.
The displayed verdict is recomputed from saved 400 Hz physical observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local.evaluate_ollie_policy import outcome, render_actual
from supermimic.ollie.physics import compile_scene


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
        "-of", "json", str(path),
    ]))["streams"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video", type=Path, default=Path("video/ollie.mp4"))
    parser.add_argument("--time-scale", type=float, default=1.5)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a new output directory")
    if not math.isfinite(args.time_scale) or args.time_scale <= 0:
        parser.error("time scale must be finite and positive")
    result = json.loads((args.evaluation / "result.json").read_text())
    contacts = json.loads((args.evaluation / "contacts.json").read_text())
    verdict = outcome(
        contacts,
        full_reference=result["criteria"]["full_reference_episode"],
        numerical_failure=result["numerical_failure"],
    )
    if any(result[key] != value for key, value in verdict.items()):
        raise ValueError("saved observations and result disagree")
    if not result["criteria"]["full_reference_episode"] or result.get("partial_control"):
        raise ValueError("this exporter requires a complete physical episode")
    archive = args.evaluation / "actual_rollout.npz"
    original_hash = digest(archive)
    with np.load(archive, allow_pickle=False) as saved:
        qpos, times = saved["qpos"], saved["time_physics"]
        if (
            qpos.shape != (258, 47) or times.shape != (258,)
            or not np.isfinite(qpos).all()
            or not np.allclose(times, np.arange(258) * 0.02, rtol=0, atol=1e-10)
            or len(contacts) != 2057
            or abs(contacts[-1]["time_physics"] - times[-1]) > 1e-10
        ):
            raise ValueError("expected complete 5.14 s / 50 Hz states and 400 Hz contacts")
    source = probe(args.video)
    duration = float(source["duration"])
    # Round up to include the entire source. At most one output frame is held.
    frames, fps = math.ceil(duration * 24), 24
    if abs(duration - times[-1] * args.time_scale) > 0.1:
        raise ValueError("source duration and declared physical time scale differ by >100ms")
    args.out_dir.mkdir(parents=True)
    model, _, _ = compile_scene(timestep=0.0025)
    rendered = render_actual(
        model, qpos, times,
        {"mode": "Native MPC - continuous 5.14 s", "outcome": verdict["outcome"],
         "render_truth_label": "saved actual physics"},
        SimpleNamespace(out_dir=args.out_dir, video_fps=50),
    )
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters = (
        "[0:v]setpts=PTS-STARTPTS,scale=640:480:force_original_aspect_ratio=decrease,"
        "pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24,"
        f"drawtext=fontfile={font}:text='Source video':"
        "x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7,"
        f"drawtext=fontfile={font}:text='Slowdown {args.time_scale:g}x - user estimate':"
        "x=12:y=h-38:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7,"
        f"tpad=stop_mode=clone:stop_duration=0.15,trim=end_frame={frames}[src];"
        f"[1:v]setpts={args.time_scale:.12g}*(PTS-STARTPTS),scale=640:480,"
        f"setsar=1,fps=24,tpad=stop_mode=clone:stop_duration=0.15,trim=end_frame={frames}[sim];"
        "[src][sim]hstack=inputs=2[out]"
    )
    comparison = args.out_dir / "comparison_full.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(args.video), "-i", rendered["path"], "-filter_complex", filters,
        "-map", "[out]", "-frames:v", str(frames), "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(comparison),
    ], check=True)
    media = {}
    for path, expected_frames in ((Path(rendered["path"]), 258), (comparison, frames)):
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-",
        ], check=True)
        info = probe(path)
        if int(info["nb_read_frames"]) != expected_frames:
            raise RuntimeError("encoded frame count mismatch")
        media[path.name] = {"sha256": digest(path), **info}
    if digest(archive) != original_hash:
        raise RuntimeError("saved physical trajectory changed during rendering")
    report = {
        "schema": "supermimic.ollie.actual_video_export.v1",
        "evaluation": str(args.evaluation.resolve()),
        "physical_outcome": verdict,
        "trajectory_sha256": original_hash,
        "source_video_sha256": digest(args.video),
        "source_video_probe": source,
        "media": media,
        "panels": ["entire original source", "saved actual continuous physics only"],
        "physical_time_s": [0, float(times[-1])],
        "time_scale_user_estimate": args.time_scale,
        "display_fps": fps,
        "display_frames": frames,
        "last_actual_state_display_time_s": float(times[-1] * args.time_scale),
        "maximum_terminal_hold_s": max(0, (frames - 1) / fps - times[-1] * args.time_scale),
        "rendering": rendered,
        "state_archive_unchanged": True,
        "independent_physics_replay": False,
    }
    (args.out_dir / "video_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"video": str(comparison.resolve()), "outcome": verdict["outcome"],
                      "frames": frames, "decode_verified": True}, indent=2))


if __name__ == "__main__":
    main()
