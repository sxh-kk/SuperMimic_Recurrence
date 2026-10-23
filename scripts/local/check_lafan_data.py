"""Validate the complete LAFAN1 conversion on CPU and emit a JSON summary.

No simulation or resampling is run. Collection SHA256 hashes cover sorted UTF-8
lines ``relative_filename<TAB>file_sha256<NEWLINE>`` (one line per file).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def collection_hash(files: list[Path], directory: Path) -> str:
    aggregate = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(directory).as_posix()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        name = path.relative_to(directory).as_posix()
        aggregate.update(f"{name}\t{digest.hexdigest()}\n".encode())
    return aggregate.hexdigest()


def model_names(xml_path: Path | None) -> tuple[list[str], list[str]]:
    """Read the pinned G1 model with CPU MuJoCo; never initialize Warp or CUDA."""
    import mujoco

    if xml_path is None:
        spec = importlib.util.find_spec("mjlab")
        if spec is None or spec.origin is None:
            raise RuntimeError("mjlab is not installed; provide --model-xml")
        xml_path = Path(spec.origin).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if (model.nq, model.nv, model.njnt, model.nbody) != (36, 35, 30, 31):
        raise ValueError("expected G1 model with 29 joints and 30 robot bodies")
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    if "pelvis" not in bodies:
        raise ValueError("G1 model has no pelvis body")
    return joints, bodies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/lafan1/g1")
    parser.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/lafan1")
    parser.add_argument("--expected-clips", type=int, default=40)
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--quat-atol", type=float, default=1e-5)
    parser.add_argument("--first-frame-atol", type=float, default=1e-5)
    parser.add_argument("--model-xml", type=Path)
    args = parser.parse_args()
    if (
        args.expected_clips < 1
        or not math.isfinite(args.input_fps)
        or not math.isfinite(args.output_fps)
        or args.input_fps <= 0
        or args.output_fps <= 0
        or not math.isfinite(args.quat_atol)
        or not math.isfinite(args.first_frame_atol)
        or args.quat_atol <= 0
        or args.first_frame_atol <= 0
    ):
        parser.error("clip count, frame rates, and tolerances must be positive and finite")

    errors: list[str] = []
    warnings: list[str] = []
    raw_dir, npz_dir = args.raw_dir.resolve(), args.npz_dir.resolve()
    raw_files = sorted(raw_dir.rglob("*.csv")) if raw_dir.is_dir() else []
    npz_files = sorted(npz_dir.glob("*.npz")) if npz_dir.is_dir() else []
    raw_by_name = {p.stem: p for p in raw_files}
    npz_by_name = {p.stem: p for p in npz_files}
    if len(raw_by_name) != len(raw_files):
        errors.append("CSV filenames have duplicate stems; conversion would overwrite clips")
    for label, files in (("CSV", raw_files), ("NPZ", npz_files)):
        if len(files) != args.expected_clips:
            errors.append(f"expected {args.expected_clips} {label} files, found {len(files)}")
    missing = sorted(raw_by_name.keys() - npz_by_name.keys())
    extra = sorted(npz_by_name.keys() - raw_by_name.keys())
    if missing:
        errors.append(f"missing NPZ clips: {', '.join(missing)}")
    if extra:
        errors.append(f"NPZ clips without source CSV: {', '.join(extra)}")

    expected_joints = expected_bodies = None
    try:
        expected_joints, expected_bodies = model_names(args.model_xml)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"cannot validate model names: {exc}")

    metrics = {
        "raw_quaternion_max_norm_deviation": 0.0,
        "npz_quaternion_max_norm_deviation": 0.0,
        "first_pelvis_position_max_abs_error_m": 0.0,
        "first_pelvis_quaternion_max_abs_error": 0.0,
        "raw_joint_max_abs_rad": 0.0,
    }
    raw_z_min, raw_z_max = math.inf, -math.inf
    raw_frames = output_frames = paired = 0
    raw_duration_s = output_duration_s = 0.0
    for name in sorted(raw_by_name.keys() & npz_by_name.keys()):
        prefix = f"{name}: "
        before = len(errors)
        try:
            raw = np.loadtxt(raw_by_name[name], delimiter=",", dtype=np.float64, ndmin=2)
            if raw.ndim != 2 or raw.shape[1] != 36 or raw.shape[0] < 2:
                errors.append(prefix + f"CSV must have shape (N>=2, 36), got {raw.shape}")
                continue
            if not np.isfinite(raw).all():
                errors.append(prefix + "CSV contains non-finite values")
                continue
            norms = np.linalg.norm(raw[:, 3:7], axis=1)
            if not np.isfinite(norms).all() or (norms < 1e-12).any():
                errors.append(prefix + "CSV contains zero or invalid quaternion norms")
                continue
            metrics["raw_quaternion_max_norm_deviation"] = max(
                metrics["raw_quaternion_max_norm_deviation"], float(np.abs(norms - 1).max())
            )
            metrics["raw_joint_max_abs_rad"] = max(
                metrics["raw_joint_max_abs_rad"], float(np.abs(raw[:, 7:]).max())
            )
            raw_z_min = min(raw_z_min, float(raw[:, 2].min()))
            raw_z_max = max(raw_z_max, float(raw[:, 2].max()))
            n_in = raw.shape[0]
            # Independent grid-count calculation; do not call resample_motion.
            n_out = math.floor((n_in - 1) * args.output_fps / args.input_fps + 1e-6) + 1
            with np.load(npz_by_name[name], allow_pickle=False) as data:
                shapes = {
                    "fps": (1,),
                    "input_fps": (1,),
                    "joint_pos": (n_out, 29),
                    "joint_vel": (n_out, 29),
                    "body_pos_w": (n_out, 30, 3),
                    "body_quat_w": (n_out, 30, 4),
                    "body_lin_vel_w": (n_out, 30, 3),
                    "body_ang_vel_w": (n_out, 30, 3),
                    "joint_names": (29,),
                    "body_names": (30,),
                }
                arrays = {}
                for key, shape in shapes.items():
                    if key not in data:
                        errors.append(prefix + f"NPZ missing {key}")
                        continue
                    array = data[key]
                    arrays[key] = array
                    if array.shape != shape:
                        errors.append(prefix + f"{key} shape {array.shape}, expected {shape}")
                    if key not in ("joint_names", "body_names") and (
                        not np.issubdtype(array.dtype, np.number)
                        or np.iscomplexobj(array)
                        or not np.isfinite(array).all()
                    ):
                        errors.append(prefix + f"{key} contains non-numeric/non-finite values")
                if len(errors) != before:
                    continue
                for key, expected in (("fps", args.output_fps), ("input_fps", args.input_fps)):
                    if not np.isclose(float(arrays[key][0]), expected, rtol=0, atol=1e-6):
                        errors.append(prefix + f"{key}={arrays[key][0]}, expected {expected}")
                if (
                    expected_joints is not None
                    and arrays["joint_names"].tolist() != expected_joints
                ):
                    errors.append(prefix + "joint_names order differs from G1 model")
                if expected_bodies is not None and arrays["body_names"].tolist() != expected_bodies:
                    errors.append(prefix + "body_names order differs from G1 model")
                qdev = float(
                    np.abs(
                        np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1) - 1
                    ).max()
                )
                if math.isfinite(qdev):
                    metrics["npz_quaternion_max_norm_deviation"] = max(
                        metrics["npz_quaternion_max_norm_deviation"], qdev
                    )
                if not math.isfinite(qdev) or qdev > args.quat_atol:
                    errors.append(
                        prefix + f"NPZ quaternion norm deviation {qdev:.6g} exceeds tolerance"
                    )
                bodies = arrays["body_names"].tolist()
                if bodies.count("pelvis") != 1:
                    errors.append(prefix + "body_names must contain exactly one pelvis")
                else:
                    pelvis = bodies.index("pelvis")
                    p_error = float(np.abs(arrays["body_pos_w"][0, pelvis] - raw[0, :3]).max())
                    # CSV xyzw -> normalized MuJoCo wxyz, accepting equivalent +/-q.
                    q_expected = raw[0, [6, 3, 4, 5]] / norms[0]
                    q_actual = arrays["body_quat_w"][0, pelvis]
                    q_error = float(
                        min(
                            np.abs(q_actual - q_expected).max(), np.abs(q_actual + q_expected).max()
                        )
                    )
                    metrics["first_pelvis_position_max_abs_error_m"] = max(
                        metrics["first_pelvis_position_max_abs_error_m"], p_error
                    )
                    metrics["first_pelvis_quaternion_max_abs_error"] = max(
                        metrics["first_pelvis_quaternion_max_abs_error"], q_error
                    )
                    if p_error > args.first_frame_atol or q_error > args.first_frame_atol:
                        errors.append(
                            prefix
                            + f"first pelvis mapping error: position={p_error:.6g}, quaternion={q_error:.6g}"
                        )
                if not np.allclose(
                    arrays["joint_pos"][0], raw[0, 7:], rtol=0, atol=args.first_frame_atol
                ):
                    errors.append(prefix + "first-frame joints differ from source CSV radians")
                raw_frames += n_in
                output_frames += n_out
                raw_duration_s += (n_in - 1) / args.input_fps
                output_duration_s += (n_out - 1) / args.output_fps
                if len(errors) == before:
                    paired += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(prefix + f"cannot validate files: {exc}")

    if metrics["raw_quaternion_max_norm_deviation"] > args.quat_atol:
        warnings.append(
            "raw CSV quaternions are not exactly unit length; conversion normalizes them"
        )
    hashes = {}
    for label, files, directory in (("csv", raw_files, raw_dir), ("npz", npz_files, npz_dir)):
        try:
            hashes[label + "_collection_sha256"] = collection_hash(files, directory)
        except OSError as exc:
            errors.append(f"cannot hash {label} collection: {exc}")
    summary = {
        "status": "passed" if not errors else "failed",
        "paths": {"csv": str(raw_dir), "npz": str(npz_dir)},
        "clips": {"csv": len(raw_files), "npz": len(npz_files), "validated_pairs": paired},
        "frames": {"csv": raw_frames, "npz": output_frames},
        "duration_hours": {"csv": raw_duration_s / 3600, "npz": output_duration_s / 3600},
        "fps": {"csv": args.input_fps, "npz": args.output_fps},
        "metrics": metrics,
        "raw_pelvis_height_range_m": [raw_z_min, raw_z_max] if math.isfinite(raw_z_min) else None,
        "hashes": hashes,
        "hash_format": "SHA256 of sorted relative_filename<TAB>file_SHA256<NEWLINE> lines (UTF-8)",
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
