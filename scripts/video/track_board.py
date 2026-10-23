"""Propagate explicitly annotated grip-graphic corners; preserve failed observations.

Run with the separate requirements-board-cv.txt environment. This produces 2D
measurements and confidence, never a claim of metric 3D pose or contact forces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_video(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
    ]))
    times = np.array([float(row["best_effort_timestamp_time"]) for row in result["frames"]])
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != len(times) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("decoded frames and strictly increasing PTS do not match")
    return frames, times


def propagate(gray: list[np.ndarray], seed: dict, direction: int) -> dict[int, dict]:
    initial = int(seed["frame"])
    quad = np.asarray(seed["quad"], dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("each seed needs four finite pixel coordinates")
    mask = np.zeros_like(gray[initial])
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    points = cv2.goodFeaturesToTrack(gray[initial], 120, 0.002, 3, mask=mask)
    records = {
        initial: {"quad": quad.tolist(), "features": 0, "confidence": seed["confidence"],
                  "source": "visual_annotation", "seed_frame": initial, "observed": True}
    }
    if points is None or len(points) < 6:
        return records
    count = len(points)
    previous = initial
    lk = {"winSize": (31, 31), "maxLevel": 3}
    endpoint = len(gray) if direction > 0 else -1
    for current in range(initial + direction, endpoint, direction):
        if not seed.get("min_frame", 0) <= current <= seed.get("max_frame", len(gray) - 1):
            break
        new, status, _ = cv2.calcOpticalFlowPyrLK(gray[previous], gray[current], points, None, **lk)
        if new is None:
            break
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray[current], gray[previous], new, None, **lk)
        if back is None:
            break
        fb = np.linalg.norm(back - points, axis=2)[:, 0]
        good = (status[:, 0] > 0) & (status_back[:, 0] > 0) & (fb < 1.5)
        if good.sum() < 6:
            break
        if seed.get("tracking_model", "homography") == "similarity":
            affine, inliers = cv2.estimateAffinePartial2D(
                points[good], new[good], method=cv2.RANSAC, ransacReprojThreshold=2.0
            )
            homography = None if affine is None else np.vstack([affine, [0, 0, 1]])
        else:
            homography, inliers = cv2.findHomography(points[good], new[good], cv2.RANSAC, 2.0)
        if homography is None or inliers is None or inliers.sum() < 6:
            break
        transformed = cv2.perspectiveTransform(quad[None], homography)[0]
        if not np.isfinite(transformed).all():
            break
        if np.max(np.linalg.norm(transformed - quad, axis=1)) > 100:
            break
        if not cv2.isContourConvex(transformed.astype(np.int32)):
            break
        area = abs(cv2.contourArea(transformed))
        if not 30 < area < 30000:
            break
        points = new[good][inliers[:, 0] > 0]
        retained = len(points) / count
        confidence = float(seed["confidence"] * retained * np.exp(-abs(current - initial) / 100))
        records[current] = {
            "quad": transformed.tolist(), "features": len(points),
            "confidence": confidence, "source": "optical_flow_ransac",
            "seed_frame": initial, "observed": True,
            "tracking_model": seed.get("tracking_model", "homography"),
            "forward_backward_error_px": float(np.median(fb[good])),
        }
        quad, previous = transformed, current
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--slowdown", type=float, default=1.5)
    args = parser.parse_args()
    if not np.isfinite(args.slowdown) or args.slowdown <= 0:
        parser.error("slowdown must be finite and positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_data = json.loads(args.seeds.read_text())
    digest = hashlib.sha256(args.video.read_bytes()).hexdigest()
    if digest != seed_data["source_sha256"]:
        raise ValueError("seed coordinates belong to a different source video")
    frames, times = read_video(args.video)
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    best = {}
    for seed in seed_data["seeds"]:
        for direction in (-1, 1):
            for index, record in propagate(gray, seed, direction).items():
                if index not in best or record["confidence"] > best[index]["confidence"]:
                    best[index] = record
    rows = []
    for index, time in enumerate(times):
        rows.append({"frame": index, "t_video": float(time),
                     "t_physics": float(time / args.slowdown),
                     **best.get(index, {"observed": False, "confidence": 0.0,
                                       "quad": None, "source": "missing"})})
    output = {
        "schema": "board_image_observations_v1", "source": str(args.video.resolve()),
        "source_sha256": digest, "seed_source": str(args.seeds.resolve()),
        "seed_sha256": hashlib.sha256(args.seeds.read_bytes()).hexdigest(),
        "width": frames[0].shape[1], "height": frames[0].shape[0],
        "slowdown_estimate": args.slowdown, "slowdown_verified": False,
        "point_order": seed_data["point_order"],
        "observed_frames": len(best), "total_frames": len(frames), "frames": rows,
        "limitations": [
            "Seed corners are approximate visual annotations, not measured ground truth.",
            "Near-edge-on graphic and occlusion can terminate tracking; missing frames stay missing.",
            "Tracking confidence does not prove correct correspondence or metric 3D pose.",
            "Coordinates retain source camera motion; camera/ground calibration is still required.",
        ],
    }
    destination = args.out_dir / "board_image_observations.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    fps = 1 / float(np.median(np.diff(times)))
    writer = cv2.VideoWriter(str(args.out_dir / "tracking_review.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    if not writer.isOpened():
        raise RuntimeError("could not initialize review video writer")
    try:
        for frame, row in zip(frames, rows, strict=True):
            canvas = frame.copy()
            if row["quad"] is not None:
                cv2.polylines(canvas, [np.asarray(row["quad"], np.int32)], True, (0, 255, 255), 3)
            label = f"frame {row['frame']} | confidence {row['confidence']:.2f} | {row['source']}"
            cv2.putText(canvas, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            writer.write(cv2.resize(canvas, (960, 540)))
    finally:
        writer.release()
    print(json.dumps({"output": str(destination), "observed": len(best), "total": len(frames)}))


if __name__ == "__main__":
    main()
