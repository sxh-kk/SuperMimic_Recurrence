"""Headless, CPU GMR initialization from recovered SMPL-X global joint poses.

This is inverse kinematics, not a physics rollout or a trained SuperMimic policy.
No per-frame floor alignment, foot locking, or board contact correction is applied.
Run with references/GMR/.venv/bin/python; the training/frontend environments stay separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    quat_rotate_inverse,
    resample_motion,
)

GMR_REVISION = "25f56db79ac411258b377b6d4ccba0c8ff3372ac"
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def load_gmr_class(repo: Path):
    """Load unmodified IK modules without the upstream GUI/Torch import side effects."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != GMR_REVISION:
        raise RuntimeError(f"Expected GMR {GMR_REVISION}, got {revision}")
    package = types.ModuleType("_supermimic_headless_gmr")
    package.__path__ = [str(repo / "general_motion_retargeting")]
    sys.modules[package.__name__] = package
    module = importlib.import_module(f"{package.__name__}.motion_retarget")
    return module.GeneralMotionRetargeting


def load_transform(path: Path | None, joints_field: str, rotations_field: str):
    if path is None:
        if (joints_field, rotations_field) != ("joints_world", "joint_rotmat_world"):
            raise ValueError("Camera/custom fields require an explicit fixed --camera-world-transform")
        return Y_UP_TO_Z_UP, np.zeros(3), {
            "kind": "official_gmr_gvhmr_basis_change",
            "warning": "GVHMR global translation/orientation can drift; this is initialization only.",
        }
    metadata = json.loads(path.read_text())
    rotation = np.asarray(metadata["rotation_camera_to_world"], dtype=np.float64)
    translation = np.asarray(metadata.get("translation_camera_to_world", [0, 0, 0]), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Transform must contain one fixed 3x3 rotation and one 3-vector translation")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Transform contains nonfinite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("Transform rotation must be a proper orthonormal matrix")
    return rotation, translation, {
        "kind": "fixed_camera_to_common_z_up_world", "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "contents": metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT / "references/GMR")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-scale", type=float, default=1.5,
                        help="Hypothesis: presentation duration / physical duration")
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--human-height", type=float,
                        help="Metres; otherwise use upstream heuristic 1.66+0.1*beta[0,0]")
    parser.add_argument("--joints-field", default="joints_world")
    parser.add_argument("--rotations-field", default="joint_rotmat_world")
    parser.add_argument("--camera-world-transform", type=Path)
    parser.add_argument("--source-last-time", type=float,
                        help="Last original sampled presentation timestamp, excluding container hold")
    return parser.parse_args()


def main():
    args = parse_args()
    if not np.isfinite([args.time_scale, args.output_fps]).all() or min(
        args.time_scale, args.output_fps
    ) <= 0:
        raise ValueError("Time scale and output fps must be positive finite numbers")
    body, repo, out = args.body.resolve(), args.repo.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(body, allow_pickle=False) as source:
        names = [str(value) for value in source["joint_names"]]
        joints = source[args.joints_field].astype(np.float64)
        rotations = source[args.rotations_field].astype(np.float64)
        time_video = source["time_seconds"].astype(np.float64)
        betas = source["betas"].astype(np.float64)
        nearest_source_frame = source["nearest_source_frame"].copy()
        up_axis = str(source["up_axis"])
        input_fps = float(source["mocap_framerate"])
    frames = len(time_video)
    if frames < 3 or len(set(names)) != len(names):
        raise ValueError("Need at least three frames and unique joint names")
    if joints.shape != (frames, len(names), 3) or rotations.shape != (frames, len(names), 3, 3):
        raise ValueError("Joint/rotation arrays have inconsistent dimensions")
    if not all(np.isfinite(value).all() for value in (joints, rotations, time_video, betas)):
        raise ValueError("Input contains nonfinite values")
    if abs(time_video[0]) > 1e-8 or (np.diff(time_video) <= 0).any():
        raise ValueError("Input presentation time must start at zero and increase strictly")
    if not np.isfinite(input_fps) or input_fps <= 0:
        raise ValueError("Input mocap_framerate must be positive and finite")
    if not np.allclose(time_video, np.arange(frames) / input_fps, atol=2e-6):
        raise ValueError("Input must have a uniform presentation time grid")
    if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(
        np.linalg.det(rotations), 1.0, atol=1e-4
    ):
        raise ValueError("Input global rotations are not proper rotation matrices")
    if args.camera_world_transform is None and up_axis != "Y":
        raise ValueError("Default official GMR basis change requires GVHMR Y-up input")
    basis, translation, transform_report = load_transform(
        args.camera_world_transform, args.joints_field, args.rotations_field
    )
    positions_z_up = joints @ basis.T + translation
    quaternions = Rotation.from_matrix((basis @ rotations).reshape(-1, 3, 3)).as_quat(
        scalar_first=True
    ).reshape(frames, len(names), 4)
    height = args.human_height if args.human_height is not None else 1.66 + 0.1 * float(betas[0, 0])
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Human height must be positive and finite")

    gmr = load_gmr_class(repo)(
        src_human="smplx", tgt_robot="unitree_g1", actual_human_height=height,
        solver="daqp", verbose=False, use_velocity_limit=False,
    )
    gmr.set_ground_offset(0.0)
    model = gmr.model
    native_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0]
    if model.nq != 36 or model.nv != 35 or set(native_names) != set(G1_JOINT_NAMES):
        raise ValueError("Pinned GMR model is not the expected free-root + 29-scalar-joint G1")
    joint_ids = [model.joint(name).id for name in G1_JOINT_NAMES]
    qpos_indices = [int(model.jnt_qposadr[index]) for index in joint_ids]
    required = set(gmr.human_scale_table)
    if not required.issubset(names):
        raise ValueError(f"Missing GMR target bodies: {sorted(required - set(names))}")
    poses = np.empty((frames, model.nq), dtype=np.float64)
    errors = np.empty((frames, 2), dtype=np.float64)
    started = time.monotonic()
    for frame in range(frames):
        human = {name: (positions_z_up[frame, i].copy(), quaternions[frame, i].copy())
                 for i, name in enumerate(names)}
        poses[frame] = gmr.retarget(human, offset_to_ground=False)
        errors[frame] = gmr.error1(), gmr.error2()
        if not np.isfinite(poses[frame]).all():
            raise RuntimeError(f"IK produced nonfinite qpos at frame {frame}")
        if frame % 30 == 0 or frame == frames - 1:
            print(f"GMR {frame + 1}/{frames} frames", flush=True)

    source_last_time = args.source_last_time
    manifest_path = body.parent / "input_manifest.json"
    if source_last_time is None and manifest_path.exists():
        source_last_time = json.loads(manifest_path.read_text())["source_pts_seconds"][-1]
    sampled_end = min(float(time_video[-1]), source_last_time) if source_last_time is not None else float(
        time_video[-1]
    )
    if not np.isfinite(sampled_end) or sampled_end <= 0:
        raise ValueError("Source last timestamp must be positive and finite")
    raw_csv = np.concatenate([poses[:, :3], poses[:, [4, 5, 6, 3]], poses[:, qpos_indices]], axis=1)
    motion = resample_motion(raw_csv, input_fps * args.time_scale, args.output_fps)
    count = min(motion.num_frames, int(np.floor(sampled_end / args.time_scale * args.output_fps + 1e-7)) + 1)
    if count < 3:
        raise ValueError("Output time span must contain at least three poses")
    time_physics = np.arange(count, dtype=np.float64) / args.output_fps
    root_angular_body = quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w)
    robot_state = np.concatenate([
        motion.root_pos, motion.root_quat, motion.joint_pos, motion.root_lin_vel_w,
        root_angular_body, motion.joint_vel,
    ], axis=1)[:count]
    if not np.isfinite(robot_state).all():
        raise RuntimeError("Resampled robot state contains nonfinite values")
    ordered_qpos = np.concatenate([motion.root_pos, motion.root_quat, motion.joint_pos], axis=1)[:count]
    ordered_csv = np.concatenate([ordered_qpos[:, :3], ordered_qpos[:, [4, 5, 6, 3]], ordered_qpos[:, 7:]], axis=1)
    ranges = model.jnt_range[joint_ids]
    violations = np.maximum(ranges[:, 0] - poses[:, qpos_indices], 0) + np.maximum(
        poses[:, qpos_indices] - ranges[:, 1], 0
    )
    artifacts = {
        "native": f"gmr_native_{input_fps:g}hz.npz",
        "csv": f"g1_{args.output_fps:g}hz.csv",
        "robot_state": f"g1_robot_{args.output_fps:g}hz.npz",
    }
    report = {
        "method": "Unmodified GMR CPU IK; robot-only kinematic initialization, not a physics rollout",
        "body": str(body), "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "gmr_revision": GMR_REVISION, "gmr_xml": str(gmr.xml_file),
        "joints_field": args.joints_field, "rotations_field": args.rotations_field,
        "transform": transform_report, "rotation_input_to_z_up": basis.tolist(),
        "translation_input_to_z_up_m": translation.tolist(),
        "actual_human_height_m": height,
        "height_source": "explicit argument" if args.human_height else "GMR heuristic 1.66 + 0.1*beta[0,0]; not measured",
        "beta_0": float(betas[0, 0]), "human_height_assumption_m": 1.8,
        "height_ratio": height / 1.8, "human_scale_table": gmr.human_scale_table,
        "pelvis_scale": gmr.human_scale_table["pelvis"],
        "scale_caveat": "GMR scales root and local limbs differently; board geometry must share calibrated robot-world metres.",
        "offset_to_ground": False, "ground_offset_m": gmr.ground_offset,
        "velocity_limits": False, "velocity_limit_note": "Upstream default IK; temporal/dynamic feasibility is not guaranteed",
        "native_frames": frames, "presentation_fps": input_fps, "time_scale_hypothesis": args.time_scale,
        "last_original_sample_video_s": source_last_time,
        "output_frames": count, "physics_fps": args.output_fps, "last_physics_sample_s": float(time_physics[-1]),
        "raw_qpos_layout": "root xyz m, root quaternion wxyz, native named 29 joint angles rad",
        "csv_layout": "no header: root xyz m, root quaternion xyzw, G1_JOINT_NAMES angles rad",
        "robot_state_layout": "root xyz, root wxyz, joints29, root linear world3, root angular body3, joint velocities29",
        "max_joint_limit_violation_rad": float(violations.max()),
        "gmr_weighted_error_mean": errors.mean(axis=0).tolist(),
        "elapsed_cpu_s": time.monotonic() - started,
        "artifacts": artifacts,
        "limitations": ["No board/contact or force feasibility solved", "No policy/WM training performed",
                        "Monocular input includes scale and world-frame uncertainty", "No frame-by-frame floor correction"],
    }
    np.savez_compressed(
        out / artifacts["native"], qpos=poses, joint_names=np.asarray(native_names),
        quaternion_order=np.asarray("wxyz"), time_video=time_video,
        nearest_source_frame=nearest_source_frame, weighted_ik_errors=errors,
        metadata_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    np.savetxt(out / artifacts["csv"], ordered_csv, delimiter=",", fmt="%.10g")
    np.savez_compressed(
        out / artifacts["robot_state"], robot_state=robot_state, qpos=ordered_qpos,
        joint_names=np.asarray(G1_JOINT_NAMES), time_physics=time_physics,
        time_video=time_physics * args.time_scale, fps=np.asarray(args.output_fps),
        quaternion_order=np.asarray("wxyz"), metadata_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "frames": count, "elapsed_s": report["elapsed_cpu_s"]}))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
