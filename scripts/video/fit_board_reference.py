"""Fit an explicitly uncertain board reference from image evidence and priors.

This CPU tool preserves both IPPE solutions and missing observations. Its final
trajectory is a regularized kinematic reference, not an observed physical rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp

PHASE_NAMES = ("prepare", "crouch", "pop", "flight", "landing", "recovery")


def graphic_points(length: float, width: float) -> np.ndarray:
    # OpenCV 4.12 IPPE must receive z=0, then apply the top/root offset separately.
    return np.array(
        [
            [-length / 2, width / 2, 0],
            [length / 2, width / 2, 0],
            [length / 2, -width / 2, 0],
            [-length / 2, -width / 2, 0],
        ],
        dtype=np.float64,
    )


def project_world(
    points: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    camera = (points - camera_translation) @ camera_rotation
    image = camera @ intrinsics.T
    return image[..., :2] / np.maximum(image[..., 2:3], 1e-5)


def candidates(
    observations: dict,
    intrinsics: np.ndarray,
    world_rotation: np.ndarray,
    world_translation: np.ndarray,
    args,
) -> dict[str, np.ndarray]:
    rows = observations["frames"]
    count = len(rows)
    positions_camera = np.full((count, 2, 3), np.nan)
    rotations_camera = np.full((count, 2, 3, 3), np.nan)
    positions_world = np.full((count, 2, 3), np.nan)
    rotations_world = np.full((count, 2, 3, 3), np.nan)
    errors = np.full((count, 2), np.nan)
    quads = np.full((count, 4, 2), np.nan)
    observed = np.array([bool(row["observed"]) for row in rows])
    confidence = np.array([row["confidence"] for row in rows])
    points = graphic_points(args.graphic_length, args.graphic_width)
    for index, row in enumerate(rows):
        if not row["observed"]:
            continue
        quads[index] = np.asarray(row["quad"], dtype=np.float64)
        success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            points, quads[index], intrinsics, None, flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            continue
        for branch, (rvec, tvec) in enumerate(zip(rvecs[:2], tvecs[:2], strict=True)):
            rotation = cv2.Rodrigues(rvec)[0]
            projected = cv2.projectPoints(points, rvec, tvec, intrinsics, None)[0][:, 0]
            positions_camera[index, branch] = tvec.ravel() - rotation[:, 2] * args.top_offset
            rotations_camera[index, branch] = rotation
            positions_world[index, branch] = (
                world_rotation @ positions_camera[index, branch] + world_translation
            )
            rotations_world[index, branch] = world_rotation @ rotation
            errors[index, branch] = np.sqrt(np.mean((projected - quads[index]) ** 2))
    valid_indices = np.flatnonzero(np.isfinite(errors).all(axis=1))
    choices = np.full(count, -1, dtype=np.int8)
    if len(valid_indices):
        costs = np.zeros((len(valid_indices), 2))
        back = np.zeros((len(valid_indices), 2), dtype=np.int8)
        for order, index in enumerate(valid_indices):
            normal_z = rotations_world[index, :, 2, 2]
            node = (errors[index] / 3) ** 2 + 40 * np.maximum(0, 0.15 - normal_z) ** 2
            if index <= 105 or index >= 130:
                node += np.minimum(
                    100, ((positions_world[index, :, 2] - args.rest_height) / 0.15) ** 2
                )
            if not order:
                costs[order] = node
                continue
            previous = valid_indices[order - 1]
            dt = rows[index]["t_video"] - rows[previous]["t_video"]
            for branch in range(2):
                transitions = []
                for old in range(2):
                    distance = np.linalg.norm(
                        positions_world[index, branch] - positions_world[previous, old]
                    )
                    relative = rotations_world[index, branch] @ rotations_world[previous, old].T
                    angle = Rotation.from_matrix(relative).magnitude()
                    transitions.append(
                        0.2 * min(100, (distance / (0.05 + dt * 0.5)) ** 2)
                        + 0.2 * (angle / (0.2 + dt * 3)) ** 2
                    )
                total = costs[order - 1] + transitions
                back[order, branch] = np.argmin(total)
                costs[order, branch] = node[branch] + total[back[order, branch]]
        branch = int(np.argmin(costs[-1]))
        for order in range(len(valid_indices) - 1, -1, -1):
            choices[valid_indices[order]] = branch
            branch = int(back[order, branch])
    selected_position = np.full((count, 3), np.nan)
    selected_rotation = np.full((count, 3, 3), np.nan)
    selected_error = np.full(count, np.nan)
    for index in valid_indices:
        branch = choices[index]
        selected_position[index] = positions_world[index, branch]
        selected_rotation[index] = rotations_world[index, branch]
        selected_error[index] = errors[index, branch]
    return {
        "time_video": np.array([row["t_video"] for row in rows]),
        "time_physics": np.array([row["t_video"] / args.slowdown for row in rows]),
        "observed_2d": observed,
        "image_confidence": confidence,
        "image_quad": quads,
        "position_camera_candidates": positions_camera,
        "rotation_camera_candidates": rotations_camera,
        "position_world_candidates": positions_world,
        "rotation_world_candidates": rotations_world,
        "reprojection_rmse_candidates_px": errors,
        "selected_branch": choices,
        "selected_position_world": selected_position,
        "selected_rotation_world": selected_rotation,
        "selected_reprojection_rmse_px": selected_error,
        "metric_pose_supported": observed
        & (confidence >= 0.15)
        & (selected_error <= 3)
        & (np.arange(count) >= 112)
        & (np.arange(count) <= 122),
        "K": intrinsics,
        "rotation_camera_to_world": world_rotation,
        "translation_camera_to_world": world_translation,
    }


def support_points(args) -> np.ndarray:
    center = np.array(
        [[x, y, z] for x in (-0.28, 0.28) for y in (-0.105, 0.105) for z in (-0.006, 0.006)]
    )
    ends = []
    for sign in (-1, 1):
        tilt = Rotation.from_euler("y", -sign * np.deg2rad(12)).as_matrix()
        ends.extend(
            np.array([sign * 0.34, 0, 0.01275]) + np.array([x, y, z]) @ tilt.T
            for x in (-0.062, 0.062)
            for y in (-0.102, 0.102)
            for z in (-0.006, 0.006)
        )
    return np.vstack([center, ends])


def minimum_local_support(rotations: np.ndarray, args) -> np.ndarray:
    """Lowest transformed collision point relative to board root, in world Z.

    Box corners are exact. Wheels use the support function of the XML cylinder
    (axis local Y), rather than rotating a preselected local bottom point.
    """
    lowest_box = np.min(np.einsum("fij,kj->fki", rotations, support_points(args))[..., 2], axis=1)
    wheel_centers = np.array([[x, y, -0.0465] for x in (-0.21, 0.21) for y in (-0.086, 0.086)])
    wheel_center_z = np.einsum("fij,kj->fki", rotations, wheel_centers)[..., 2]
    local_vertical = rotations[:, 2]
    wheel_support = 0.012 * np.abs(local_vertical[:, 1]) + 0.0275 * np.linalg.norm(
        local_vertical[:, [0, 2]], axis=1
    )
    lowest_wheel = np.min(wheel_center_z, axis=1) - wheel_support
    return np.minimum(lowest_box, lowest_wheel)


def initial_trajectory(
    raw: dict, rotation: np.ndarray, translation: np.ndarray, args
) -> np.ndarray:
    count = len(raw["time_video"])
    result = np.zeros((count, 6))
    known = np.flatnonzero(raw["observed_2d"])
    centers = np.stack(
        [
            np.nanmean(quad, axis=0) if np.isfinite(quad).all() else [np.nan, np.nan]
            for quad in raw["image_quad"]
        ]
    )
    centers = np.stack(
        [np.interp(np.arange(count), known, centers[known, axis]) for axis in range(2)], axis=1
    )
    rays = np.column_stack([centers, np.ones(count)]) @ np.linalg.inv(raw["K"]).T @ rotation.T
    depth = (args.rest_height + args.top_offset - translation[2]) / rays[:, 2]
    result[:, :3] = translation + rays * depth[:, None]
    result[:, 2] -= args.top_offset
    anchors = np.flatnonzero(raw["metric_pose_supported"])
    if len(anchors) < 3:
        raise RuntimeError("Too few supported airborne image poses to construct a reference")
    for index in anchors:
        result[index, :3] = raw["selected_position_world"][index]
        result[index, 3:] = Rotation.from_matrix(raw["selected_rotation_world"][index]).as_euler(
            "xyz"
        )
    interp_indices = np.unique(np.concatenate([np.arange(0, 103), anchors, np.arange(130, count)]))
    for axis in range(6):
        result[:, axis] = PchipInterpolator(interp_indices, result[interp_indices, axis])(
            np.arange(count)
        )
    result[0, :3] = [0, 0, args.rest_height]
    return result


def fit_trajectory(raw: dict, landmarks: list, args) -> tuple[np.ndarray, dict]:
    count = len(raw["time_video"])
    camera_rotation = raw["rotation_camera_to_world"]
    camera_translation = raw["translation_camera_to_world"]
    start = initial_trajectory(raw, camera_rotation, camera_translation, args)
    local_graphic = graphic_points(args.graphic_length, args.graphic_width)
    local_graphic[:, 2] = args.top_offset
    observed = np.flatnonzero(raw["observed_2d"])
    full = raw["metric_pose_supported"]
    ground = np.flatnonzero((np.arange(count) <= 102) | (np.arange(count) >= 130))
    constraints = []
    for index in observed:
        constraints.append(("image_full" if full[index] else "image_center", int(index)))
    for index in ground:
        constraints.append(("ground", int(index)))
    for index in range(count - 2):
        constraints.append(("acceleration", index))
    for index in range(count):
        constraints.append(("support", index))
    constraints.append(("origin", 0))
    for landmark in landmarks:
        constraints.append(("landmark", int(landmark["frame"]), landmark))

    def residual_blocks(values):
        poses = values.reshape(count, 6)
        rotations = Rotation.from_euler("xyz", poses[:, 3:]).as_matrix()
        points_world = np.einsum("fij,kj->fki", rotations, local_graphic) + poses[:, None, :3]
        projected = project_world(points_world, camera_rotation, camera_translation, raw["K"])
        lowest = minimum_local_support(rotations, args) + poses[:, 2]
        output = []
        for constraint in constraints:
            kind, index = constraint[:2]
            if kind.startswith("image"):
                # Similarity-propagated thin graphics constrain their centroid only.
                weight = np.sqrt(max(0.02, raw["image_confidence"][index]))
                if kind == "image_full":
                    block = (projected[index] - raw["image_quad"][index]).ravel() * weight / 2
                else:
                    block = (
                        (projected[index].mean(axis=0) - raw["image_quad"][index].mean(axis=0))
                        * weight
                        / 3
                    )
            elif kind == "ground":
                block = np.array(
                    [
                        (poses[index, 2] - args.rest_height) / 0.008,
                        poses[index, 3] / 0.035,
                        poses[index, 4] / 0.035,
                    ]
                )
            elif kind == "acceleration":
                second = poses[index + 2] - 2 * poses[index + 1] + poses[index]
                block = second / np.array([0.006, 0.012, 0.006, 0.035, 0.035, 0.035])
                if 113 <= index <= 119:
                    dt = float(raw["time_physics"][index + 1] - raw["time_physics"][index])
                    block[2] = (second[2] + 9.81 * dt**2) / 0.006
            elif kind == "support":
                block = np.array(
                    [min(0, lowest[index]) / 0.008, max(0, 0.1 - rotations[index, 2, 2]) / 0.08]
                )
            elif kind == "origin":
                block = (poses[0] - np.array([0, 0, args.rest_height, 0, 0, 0])) / np.array(
                    [0.015, 0.04, 0.005, 0.02, 0.02, 0.03]
                )
            else:
                item = constraint[2]
                local = np.asarray(item["points_local"], dtype=float)
                world = local @ rotations[index].T + poses[index, :3]
                image = project_world(world, camera_rotation, camera_translation, raw["K"])
                block = ((image - item["image_points"]) / item.get("sigma_px", 8)).ravel()
            output.append(block)
        return output

    blocks = residual_blocks(start.ravel())
    sparsity = lil_matrix((sum(len(block) for block in blocks), count * 6), dtype=np.int8)
    cursor = 0
    for constraint, block in zip(constraints, blocks, strict=True):
        index = constraint[1]
        frames = range(index, index + 3) if constraint[0] == "acceleration" else (index,)
        for frame in frames:
            sparsity[cursor : cursor + len(block), frame * 6 : (frame + 1) * 6] = 1
        cursor += len(block)
    result = least_squares(
        lambda x: np.concatenate(residual_blocks(x)),
        start.ravel(),
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=2,
        max_nfev=180,
        ftol=2e-5,
        xtol=2e-5,
        gtol=2e-5,
    )
    poses = result.x.reshape(count, 6)
    rotations = Rotation.from_euler("xyz", poses[:, 3:]).as_matrix()
    needed_height = -minimum_local_support(rotations, args)
    corrected = poses[:, 2] < needed_height
    correction = np.maximum(0, needed_height - poses[:, 2])
    # Preserve the unconstrained fit, and label this final geometric feasibility adjustment.
    unconstrained = poses.copy()
    poses[:, 2] += correction
    return poses, {
        "success": bool(result.success),
        "message": result.message,
        "evaluations": result.nfev,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "support_adjusted_frames": np.flatnonzero(corrected).tolist(),
        "max_support_adjustment_m": float(correction.max()),
        "unconstrained_pose": unconstrained,
        "assumptions": [
            "Approximately fixed camera; common_world.json is used as one fixed extrinsic transform.",
            "Graphic and board dimensions are engineering assumptions, not measured dimensions.",
            "Preparation/settled landing has a ground/near-flat pose prior, not a depth observation.",
            "Missing transitions are filled by temporal regularization, not observed 3D measurements.",
            "Central flight vertical acceleration is weakly regularized toward gravity using an unverified slowdown.",
            "Geometric nonpenetration of a simplified board envelope is not dynamical feasibility.",
        ],
    }


def phases(source_frame: np.ndarray) -> np.ndarray:
    # Coarse visual phase prior; these indices are not measured contact events.
    return np.select(
        [
            source_frame < 75,
            source_frame < 103,
            source_frame < 112,
            source_frame < 123,
            source_frame < 136,
        ],
        [0, 1, 2, 3, 4],
        default=5,
    ).astype(np.int8)


def save_reference(raw: dict, poses: np.ndarray, args) -> dict:
    source_time = raw["time_physics"]
    times = np.arange(int(np.floor(source_time[-1] * args.fps)) + 1) / args.fps
    position = PchipInterpolator(source_time, poses[:, :3], axis=0)(times)
    rotations = Slerp(source_time, Rotation.from_euler("xyz", poses[:, 3:]))(times)
    # Interpolation can reintroduce tiny penetration; retain an explicit audit.
    interpolated_position = position.copy()
    correction = np.maximum(0, -minimum_local_support(rotations.as_matrix(), args) - position[:, 2])
    position[:, 2] += correction
    quat = rotations.as_quat()[:, [3, 0, 1, 2]]
    for index in range(1, len(quat)):
        if np.dot(quat[index], quat[index - 1]) < 0:
            quat[index] *= -1
    frame_index = np.clip(
        np.searchsorted(source_time, times, side="right") - 1, 0, len(source_time) - 1
    )
    state = np.zeros((len(times), 21))
    state[:, :3], state[:, 3:7] = position, quat
    state[:, 11:14] = np.gradient(position, times, axis=0)
    delta = rotations[:-1].inv() * rotations[1:]
    angular = delta.as_rotvec() / np.diff(times)[:, None]
    state[:-1, 14:17] = angular
    state[-1, 14:17] = angular[-1]
    source_quality = np.where(
        raw["metric_pose_supported"], np.clip(raw["image_confidence"], 0, 1), 0.35
    )
    source_quality[~raw["observed_2d"]] = 0.1
    quality = np.interp(times, source_time, source_quality)
    prior = np.full((len(times), 2), -1, dtype=np.int8)
    # Explicit engineering priors only; source labels stay separately marked.
    trusted_standing = (frame_index <= 70) | (frame_index >= 145)
    prior[trusted_standing] = 1
    provenance = np.where(
        raw["observed_2d"][frame_index], "image_plus_prior", "temporal_physics_completion"
    )
    data = {
        "time_physics": times,
        "time_video": times * args.slowdown,
        "board_qpos": state[:, :7],
        "board_state": state,
        "position_before_resample_support_correction": interpolated_position,
        "resample_support_correction_m": correction,
        "board_pose_confidence": quality,
        "observed_2d": raw["observed_2d"][frame_index],
        "metric_pose_supported": raw["metric_pose_supported"][frame_index],
        "source_frame": frame_index,
        "pose_provenance": provenance,
        "foot_board_prior": prior,
        "contact_prior_is_observation": np.asarray(False),
        "phase": phases(frame_index),
        "phase_names": np.asarray(PHASE_NAMES),
        "wheel_motion_observed": np.asarray(False),
        "wheel_angles_placeholder": np.asarray(True),
        "slowdown_estimate": np.asarray(args.slowdown),
        "slowdown_verified": np.asarray(False),
        "quaternion_order": np.asarray("wxyz"),
        "up_axis": np.asarray("Z"),
        "units": np.asarray("SI"),
    }
    np.savez_compressed(args.out_dir / "board_reference_50hz.npz", **data)
    return data


def render_review(raw: dict, poses: np.ndarray, args) -> dict:
    capture = cv2.VideoCapture(str(args.video))
    width, height = int(capture.get(3)), int(capture.get(4))
    fps = capture.get(cv2.CAP_PROP_FPS)
    output = args.out_dir / "board_reprojection_review.mp4"
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
            str(fps),
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
    pattern = graphic_points(args.graphic_length, args.graphic_width)
    pattern[:, 2] = args.top_offset
    outline = np.array(
        [
            [-args.board_length / 2, 0.102, 0.02564],
            [-0.28, 0.105, 0.006],
            [0.28, 0.105, 0.006],
            [args.board_length / 2, 0.102, 0.02564],
            [args.board_length / 2, -0.102, 0.02564],
            [0.28, -0.105, 0.006],
            [-0.28, -0.105, 0.006],
            [-args.board_length / 2, -0.102, 0.02564],
        ]
    )
    errors = []
    all_projected = []
    try:
        for index, pose in enumerate(poses):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before source frame {index}")
            rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
            projected_pattern = project_world(
                pattern @ rotation.T + pose[:3],
                raw["rotation_camera_to_world"],
                raw["translation_camera_to_world"],
                raw["K"],
            )
            projected_outline = project_world(
                outline @ rotation.T + pose[:3],
                raw["rotation_camera_to_world"],
                raw["translation_camera_to_world"],
                raw["K"],
            )
            all_projected.append(projected_pattern)
            cv2.polylines(
                image, [np.round(projected_outline).astype(np.int32)], True, (0, 255, 60), 3
            )
            cv2.polylines(
                image, [np.round(projected_pattern).astype(np.int32)], True, (255, 255, 0), 3
            )
            if raw["observed_2d"][index]:
                cv2.polylines(
                    image,
                    [np.round(raw["image_quad"][index]).astype(np.int32)],
                    True,
                    (0, 255, 255),
                    2,
                )
                errors.append(
                    float(np.sqrt(np.mean((projected_pattern - raw["image_quad"][index]) ** 2)))
                )
            else:
                errors.append(None)
            label = f"frame {index}: {'image + prior' if raw['observed_2d'][index] else 'MISSING: temporal/physics prior'}"
            cv2.putText(image, label, (30, 45), 0, 0.85, (0, 255, 255), 2)
            cv2.putText(
                image,
                "Green board / cyan assumed graphic / yellow image candidate; kinematic fit only",
                (30, 80),
                0,
                0.7,
                (255, 255, 255),
                2,
            )
            process.stdin.write(image.tobytes())
    finally:
        capture.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Review video encoding failed")
    return {"frame_graphic_rmse_px": errors, "projected_graphic": np.asarray(all_projected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--common-world", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extra-landmarks", type=Path)
    parser.add_argument("--graphic-length", type=float, default=0.21)
    parser.add_argument("--graphic-width", type=float, default=0.18)
    parser.add_argument("--board-length", type=float, default=0.8013)
    parser.add_argument("--top-offset", type=float, default=0.006)
    parser.add_argument("--rest-height", type=float, default=0.074)
    parser.add_argument("--slowdown", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=50)
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.graphic_length, args.graphic_width, args.board_length, args.slowdown, args.fps)
        <= 0
    ):
        parser.error("dimensions, slowdown and fps must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    observations = json.loads(args.observations.read_text())
    calibration = json.loads(args.common_world.read_text())
    if hashlib.sha256(args.video.read_bytes()).hexdigest() != observations["source_sha256"]:
        raise ValueError("Image observations belong to a different source video")
    with np.load(args.body, allow_pickle=False) as body:
        intrinsics = body["K_fullimg"][0].astype(np.float64)
        if np.ptp(body["K_fullimg"], axis=0).max() > 1e-3:
            raise ValueError("This fixed-camera fit requires constant intrinsics")
    rotation = np.asarray(calibration["rotation_camera_to_world"], dtype=float)
    translation = np.asarray(calibration["translation_camera_to_world"], dtype=float)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
    ):
        raise ValueError("Invalid fixed common-world camera transform")
    raw = candidates(observations, intrinsics, rotation, translation, args)
    np.savez_compressed(args.out_dir / "raw_board_candidates.npz", **raw)
    source_report = {
        "schema": "board_pose_fit_v1",
        "source_sha256": observations["source_sha256"],
        "input_observations": str(args.observations.resolve()),
        "common_world_sha256": hashlib.sha256(args.common_world.read_bytes()).hexdigest(),
        "frames": len(raw["time_video"]),
        "observed_2d": int(raw["observed_2d"].sum()),
        "missing_source_frames": np.flatnonzero(~raw["observed_2d"]).tolist(),
        "supported_airborne_pose_frames": np.flatnonzero(raw["metric_pose_supported"]).tolist(),
        "graphic_dimensions_m": [args.graphic_length, args.graphic_width],
        "board_dimensions_m": [args.board_length, 0.21],
        "planar_pnp": "Object points z=0; two IPPE branches retained, root translation applies -R*[0,0,top_offset].",
        "camera": "One fixed common_world camera transform; never uses per-frame human-inferred extrinsics.",
        "selected_frame_119_rmse_px": float(raw["selected_reprojection_rmse_px"][119]),
        "slowdown_estimate": args.slowdown,
        "slowdown_verified": False,
    }
    (args.out_dir / "raw_diagnostics.json").write_text(json.dumps(source_report, indent=2) + "\n")
    print(json.dumps(source_report, indent=2), flush=True)
    if args.candidates_only:
        return
    tip_points = [[-args.board_length / 2, 0, 0.02564], [args.board_length / 2, 0, 0.02564]]
    landmarks = [
        {
            "frame": 0,
            "points_local": tip_points,
            "image_points": calibration["image_tip_annotations"],
            "sigma_px": 6,
            "source": "approximate_visual_common_world_calibration",
        }
    ]
    if args.extra_landmarks:
        landmarks.extend(json.loads(args.extra_landmarks.read_text())["observations"])
    poses, fit = fit_trajectory(raw, landmarks, args)
    unconstrained = fit.pop("unconstrained_pose")
    np.savez_compressed(
        args.out_dir / "fitted_source_trajectory.npz",
        time_video=raw["time_video"],
        time_physics=raw["time_physics"],
        position_world=poses[:, :3],
        rotation_world=Rotation.from_euler("xyz", poses[:, 3:]).as_matrix(),
        quat_world_wxyz=Rotation.from_euler("xyz", poses[:, 3:]).as_quat()[:, [3, 0, 1, 2]],
        unconstrained_pose_xyz_euler=unconstrained,
        observed_2d=raw["observed_2d"],
    )
    reference = save_reference(raw, poses, args)
    review = render_review(raw, poses, args)
    np.savez_compressed(
        args.out_dir / "reprojection_check.npz",
        image_quad=raw["image_quad"],
        projected_graphic=review["projected_graphic"],
        observed_2d=raw["observed_2d"],
    )
    source_report.update(
        {
            "fit": fit,
            "reference_frames": len(reference["time_physics"]),
            "reference_last_time_physics": float(reference["time_physics"][-1]),
            "frame_graphic_rmse_px": review["frame_graphic_rmse_px"],
            "extra_landmarks": landmarks,
        }
    )
    (args.out_dir / "fit_diagnostics.json").write_text(json.dumps(source_report, indent=2) + "\n")
    print(
        json.dumps(
            {"reference": str(args.out_dir / "board_reference_50hz.npz"), "fit": fit}, indent=2
        )
    )


if __name__ == "__main__":
    main()
