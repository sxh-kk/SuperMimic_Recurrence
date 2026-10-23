"""Recover a real video with the pinned GVHMR clone in its separate environment.

Run using references/GVHMR/.venv/bin/python. The default skips the upstream
ground-locking postprocess because feet can contact a moving skateboard.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GVHMR_REVISION = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def probe(path: Path, frames: bool = False) -> dict:
    options = (
        ["-show_frames", "-show_entries", "frame=best_effort_timestamp_time"]
        if frames
        else ["-show_format", "-show_streams"]
    )
    return json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *options, "-of", "json", str(path)]
        )
    )


def prepare_video(source: Path, output: Path, force: bool) -> dict:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_probe = probe(source)
    source_pts = [float(row["best_effort_timestamp_time"]) for row in probe(source, True)["frames"]]
    manifest_path = output.parent / "input_manifest.json"
    if output.exists() and not force:
        if not manifest_path.exists():
            raise RuntimeError(
                "Existing normalized video has no manifest; use --force-input to replace it"
            )
        previous = json.loads(manifest_path.read_text())
        if previous["source_sha256"] != source_hash:
            raise RuntimeError("Source differs from cached video; use a new output directory")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-threads",
                "4",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-vf",
                "setpts=PTS-STARTPTS,fps=30",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
    normalized_pts = [
        float(row["best_effort_timestamp_time"]) for row in probe(output, True)["frames"]
    ]
    source_pts = [value - source_pts[0] for value in source_pts]
    mapping = []
    for value in normalized_pts:
        index = bisect.bisect_left(source_pts, value)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(source_pts)]
        mapping.append(min(candidates, key=lambda i: abs(source_pts[i] - value)))
    result = {
        "source": str(source),
        "source_sha256": source_hash,
        "source_probe": source_probe,
        "source_pts_seconds": source_pts,
        "normalized_video": str(output),
        "normalized_fps": 30.0,
        "normalized_pts_seconds": normalized_pts,
        "nearest_source_frame": mapping,
        "timing": "Preserves source presentation speed; capture speed and slow-motion ratio remain unknown.",
    }
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def skeleton_overlay(video: Path, output: Path, joints_camera, intrinsics, parents) -> None:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for index in range(len(joints_camera)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {index}")
            xyz = joints_camera[index]
            camera = intrinsics[index]
            projected = xyz @ camera.T
            valid = np.isfinite(projected).all(axis=1) & (xyz[:, 2] > 1e-5)
            xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-5)
            for child, parent in enumerate(parents):
                if parent >= 0 and valid[child] and valid[parent]:
                    a, b = (
                        tuple(np.round(xy[child]).astype(int)),
                        tuple(np.round(xy[parent]).astype(int)),
                    )
                    cv2.line(frame, a, b, (0, 255, 80), 4, cv2.LINE_AA)
            cv2.putText(
                frame,
                "GVHMR: recovered 3D body (not robot simulation)",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 80),
                2,
                cv2.LINE_AA,
            )
            process.stdin.write(frame.tobytes())
    finally:
        capture.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Overlay encoding failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("references/GVHMR"))
    parser.add_argument("--output-root", type=Path, default=Path("logs/ollie_gvhmr"))
    parser.add_argument("--camera", choices=("static", "background"), default="static")
    parser.add_argument("--postprocess", choices=("none", "ground"), default="none")
    parser.add_argument("--focal-mm", type=int)
    parser.add_argument(
        "--stage", choices=("prepare", "preprocess", "recover", "all"), default="all"
    )
    parser.add_argument("--force-input", action="store_true")
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    source, repo, output_root = (
        args.video.resolve(),
        args.repo.resolve(),
        args.output_root.resolve(),
    )
    output_dir = output_root / source.stem
    video = output_dir / "0_input_video.mp4"
    manifest = prepare_video(source, video, args.force_input)
    if args.stage == "prepare":
        print(
            json.dumps(
                {"normalized_video": str(video), "frames": len(manifest["normalized_pts_seconds"])}
            )
        )
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if revision != GVHMR_REVISION:
        raise RuntimeError(f"Unexpected GVHMR revision: {revision}")
    required = [
        "gvhmr/gvhmr_siga24_release.ckpt",
        "hmr2/epoch=10-step=25000.ckpt",
        "vitpose/vitpose-h-multi-coco.pth",
        "yolo/yolov8x.pt",
    ]
    if args.stage != "preprocess":
        required.append("body_models/smplx/SMPLX_NEUTRAL.npz")
    missing = [
        str(repo / "inputs/checkpoints" / path)
        for path in required
        if not (repo / "inputs/checkpoints" / path).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing inference assets:\n" + "\n".join(missing))
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.net_utils import detach_to_cpu
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from tools.demo.demo import load_data_dict, run_preprocess

    if not torch.cuda.is_available():
        raise RuntimeError("The upstream inference pipeline requires a CUDA GPU")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        overrides = [
            f"video_name={source.stem}",
            f"output_root={output_root}",
            f"static_cam={args.camera == 'static'}",
            "use_dpvo=False",
        ]
        if args.focal_mm is not None:
            overrides.append(f"f_mm={args.focal_mm}")
        cfg = compose(config_name="demo", overrides=overrides)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    if args.stage in ("preprocess", "all"):
        run_preprocess(cfg)
    if args.stage == "preprocess":
        return
    data = load_data_dict(cfg)
    batch = {
        "length": data["length"][None],
        "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
        "bbx_xys": data["bbx_xys"][None],
        "K_fullimg": data["K_fullimg"][None],
        "cam_angvel": data["cam_angvel"][None],
        "f_imgseq": data["f_imgseq"][None],
    }
    batch = {key: value.cuda() for key, value in batch.items()}
    model = instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    with torch.no_grad():
        outputs = model.pipeline(
            batch, train=False, postproc=args.postprocess == "ground", static_cam=cfg.static_cam
        )
        world = outputs["pred_smpl_params_global"]
        camera = outputs["pred_smpl_params_incam"]
        joints_world, _, transforms_world = model.pipeline.endecoder.fk_v2(
            **world, get_intermediate=True
        )
        joints_camera, _, transforms_camera = model.pipeline.endecoder.fk_v2(
            **camera, get_intermediate=True
        )
    pred = detach_to_cpu(
        {
            "smpl_params_global": {key: value[0] for key, value in world.items()},
            "smpl_params_incam": {key: value[0] for key, value in camera.items()},
            "K_fullimg": data["K_fullimg"],
            "static_conf_logits": outputs["static_conf_logits"][0],
        }
    )
    for group in (pred["smpl_params_global"], pred["smpl_params_incam"]):
        if not all(torch.isfinite(value).all() for value in group.values()):
            raise RuntimeError("Recovered body parameters are nonfinite")
    arrays = {
        "joints_world": joints_world[0].cpu().numpy(),
        "joints_camera": joints_camera[0].cpu().numpy(),
        "joint_rotmat_world": transforms_world[0, ..., :3, :3].cpu().numpy(),
        "joint_rotmat_camera": transforms_camera[0, ..., :3, :3].cpu().numpy(),
        "joint_names": np.asarray(JOINT_NAMES),
        "parents": model.pipeline.endecoder.parents_tensor.cpu().numpy(),
        "K_fullimg": data["K_fullimg"].cpu().numpy(),
        "keypoints2d_coco17": data["kp2d"].cpu().numpy(),
        "bbox_center_size": data["bbx_xys"].cpu().numpy(),
        "predicted_static_joint_confidence": outputs["static_conf_logits"][0]
        .sigmoid()
        .cpu()
        .numpy(),
        "time_seconds": np.asarray(manifest["normalized_pts_seconds"]),
        "nearest_source_frame": np.asarray(manifest["nearest_source_frame"]),
        "mocap_framerate": np.asarray(30.0),
        "up_axis": np.asarray("Y"),
        "body_model": np.asarray("SMPL-X neutral"),
        **{key: value.numpy() for key, value in pred["smpl_params_global"].items()},
    }
    rotation_camera_to_world = arrays["joint_rotmat_world"][:, 0] @ arrays["joint_rotmat_camera"][
        :, 0
    ].transpose(0, 2, 1)
    translation_camera_to_world = arrays["joints_world"][:, 0] - np.einsum(
        "fij,fj->fi", rotation_camera_to_world, arrays["joints_camera"][:, 0]
    )
    arrays["camera_to_world_rotation"] = rotation_camera_to_world
    arrays["camera_to_world_translation"] = translation_camera_to_world
    aligned_camera_joints = (
        np.einsum("fij,fkj->fki", rotation_camera_to_world, arrays["joints_camera"])
        + translation_camera_to_world[:, None]
    )
    camera_alignment_error = float(np.max(np.abs(aligned_camera_joints - arrays["joints_world"])))
    if not np.isfinite(arrays["joints_world"]).all():
        raise RuntimeError("Recovered body joints are nonfinite")
    if len(arrays["joints_world"]) != len(arrays["time_seconds"]):
        raise RuntimeError("Motion and normalized video lengths differ")
    suffix = "raw" if args.postprocess == "none" else "ground"
    torch.save(pred, output_dir / f"hmr4d_{suffix}.pt")
    np.savez_compressed(output_dir / f"body_world_{suffix}.npz", **arrays)
    if args.overlay:
        skeleton_overlay(
            video,
            output_dir / f"body_overlay_{suffix}.mp4",
            arrays["joints_camera"],
            arrays["K_fullimg"],
            arrays["parents"],
        )
    report = {
        "source": str(source),
        "gvhmr_revision": revision,
        "camera": args.camera,
        "postprocess": args.postprocess,
        "body_model": "SMPL-X neutral",
        "coordinate_frame": "GVHMR world, gravity-aligned Y up; camera poses use upstream camera axes",
        "camera_assumption": "Approximately fixed camera; small shake is not estimated"
        if args.camera == "static"
        else "Background SimpleVO estimates camera angular motion",
        "camera_to_world": "Per-frame rigid transforms derived from matching human root poses; not independently calibrated SLAM extrinsics.",
        "camera_alignment_max_abs_error_m": camera_alignment_error,
        "camera_intrinsics": arrays["K_fullimg"][0].tolist(),
        "frames": len(arrays["time_seconds"]),
        "fps": 30.0,
        "seconds": time.monotonic() - started,
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "parameters": str(output_dir / f"hmr4d_{suffix}.pt"),
        "body_joints": str(output_dir / f"body_world_{suffix}.npz"),
        "limitations": [
            "Monocular estimates are not ground truth.",
            "Presentation timing retained; slow-motion ratio unknown.",
            "No moving skateboard is estimated by this frontend.",
            "Camera intrinsics are approximate unless explicitly configured.",
        ],
    }
    (output_dir / f"report_{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
