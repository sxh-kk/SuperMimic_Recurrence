"""Compare source video, explicitly kinematic target, and recorded real physics.

Rendering reference poses is visualization only. It never creates a physics
success claim; the third panel's outcome comes from its saved evaluation JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from evaluate_ollie_policy import render_actual

from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=Path("video/ollie.mp4"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a new output directory")
    evaluation = json.loads((args.evaluation / "result.json").read_text())
    reference = JointReference.load(args.reference)
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if evaluation["reference_sha256"] != reference_hash:
        raise ValueError("evaluation and kinematic reference hashes differ")
    if evaluation.get("video") is None:
        raise ValueError("render the actual evaluation video first")
    actual_video = Path(evaluation["video"]["path"])
    if not actual_video.is_file():
        raise FileNotFoundError(actual_video)
    args.out_dir.mkdir(parents=True)
    model, _, _ = compile_scene()
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    qposes, _ = indices.unpack(torch.tensor(reference.state, dtype=torch.float64))
    target_dir = args.out_dir / "target"
    target_dir.mkdir()
    # Same camera convention as the real evaluation, separate rendering data.
    target_video = render_actual(
        model, qposes.numpy(), reference.time_physics,
        {"mode": "Kinematic target ONLY", "outcome": "REFERENCE",
         "render_truth_label": "kinematic pose visualization"},
        SimpleNamespace(reference_pd=False, out_dir=target_dir, video_fps=50),
    )
    time_scale = reference.metadata["time_scale"]
    origin = reference.metadata["video_time_origin_s"]
    duration = float(reference.time_physics[-1] * time_scale)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    source_label = (
        f"drawtext=fontfile={font}:text='Source video':"
        "x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7"
    )
    filters = (
        f"[0:v]trim=start={origin}:duration={duration},setpts=PTS-STARTPTS,"
        f"scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps=24,{source_label}[source];"
        f"[1:v]setpts={time_scale}*(PTS-STARTPTS),scale=640:480,setsar=1,fps=24[target];"
        f"[2:v]setpts={time_scale}*(PTS-STARTPTS),scale=640:480,setsar=1,fps=24[actual];"
        "[source][target][actual]hstack=inputs=3[out]"
    )
    destination = args.out_dir / "comparison.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(args.video), "-i", target_video["path"], "-i", str(actual_video),
        "-filter_complex", filters, "-map", "[out]", "-an", "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ], check=True)
    report = {
        "video": str(destination.resolve()), "panels_left_to_right": [
            "original video", "kinematic reference only", "recorded continuous physical evaluation"],
        "reference_sha256": reference_hash, "source_video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "evaluation": str(args.evaluation.resolve()), "physical_success": evaluation["success"],
        "failure_reasons": evaluation["failure_reasons"],
        "time_scale_unverified": time_scale,
        "display_time": "source-video time; both simulation panels slowed by the declared scale",
        "duration_seconds": duration, "kinematic_target_is_not_physics": True,
        "actual_video_modified_only_for_layout_and_display_timing": True,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
