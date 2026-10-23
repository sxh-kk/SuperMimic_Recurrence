"""Build a contact-aware robot + passive-board kinematic reference, not execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.contact_ik import (
    ContactIKConfig,
    correct_contact_reference,
    entity_state_from_pose,
)
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import CONTACT_NAMES, PHASE_NAMES, JointReference
from supermimic.ollie.state import JointStateLayout


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stance-video-window", type=float, nargs=2, action="append", default=[],
                        metavar=("START", "END"), help="Explicit engineering prior for both feet; seconds of video")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--max-joint-velocity", type=float, default=12.0)
    parser.add_argument("--ik-iterations", type=int, default=18)
    parser.add_argument("--smoothing-sigma", type=float, default=1.5)
    args = parser.parse_args()
    args.robot, args.board, args.out = args.robot.resolve(), args.board.resolve(), args.out.resolve()
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("Confidence threshold must be between zero and one")
    with np.load(args.robot, allow_pickle=False) as robot:
        qpos = robot["qpos"].copy()
        physical_time = robot["time_physics"].copy()
        video_time = robot["time_video"].copy()
        if tuple(robot["joint_names"]) != G1_JOINT_NAMES:
            raise ValueError("Robot joint order differs from the project G1 contract")
        robot_metadata = json.loads(str(robot["metadata_json"]))
    with np.load(args.board, allow_pickle=False) as board:
        board_qpos = board["board_qpos"].copy()
        board_time = board["time_physics"].copy()
        board_video_time = board["time_video"].copy()
        confidence = board["board_pose_confidence"].copy()
        phase = board["phase"].copy()
        priors = board["foot_board_prior"].copy()
        board_phase_names = tuple(board["phase_names"].tolist()) if "phase_names" in board else PHASE_NAMES
    frames = len(physical_time)
    if board_qpos.shape != (frames, 7) or priors.shape != (frames, 2) or confidence.shape != (frames,):
        raise ValueError("Board/robot dimensions do not match")
    if not np.allclose(board_time, physical_time, atol=1e-8) or not np.allclose(board_video_time, video_time, atol=1e-7):
        raise ValueError("Board and robot must share exactly aligned physical/video sample times")
    if not np.isin(priors, [-1, 0, 1]).all() or not np.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise ValueError("Invalid board contact priors or confidence")
    if not np.allclose(np.linalg.norm(board_qpos[:, 3:7], axis=1), 1.0, atol=1e-4):
        raise ValueError("Board quaternion must be normalized wxyz")
    if phase.shape != (frames,) or not np.isin(phase, np.arange(-1, len(board_phase_names))).all():
        raise ValueError("Invalid board phase labels")
    phase_map = {i: PHASE_NAMES.index(name) for i, name in enumerate(board_phase_names)}
    phase = np.array([phase_map[int(value)] if value >= 0 else -1 for value in phase], dtype=np.int32)
    contact_prior = priors.copy()
    if args.stance_video_window:
        # Explicit conservative windows replace upstream positive engineering
        # priors; unknown gaps must not silently become full-phase constraints.
        contact_prior[contact_prior == 1] = -1
    for start, end in args.stance_video_window:
        if not 0 <= start < end:
            raise ValueError("Stance windows must satisfy 0 <= start < end")
        marked = (video_time >= start) & (video_time <= end)
        if (phase[marked] == PHASE_NAMES.index("flight")).any():
            raise ValueError("An explicit stance window overlaps labelled flight; narrow the window")
        contact_prior[marked] = 1
    mask = (contact_prior == 1) & (confidence[:, None] >= args.min_confidence)
    mask[phase == PHASE_NAMES.index("flight")] = False
    fps = 1 / float(physical_time[1] - physical_time[0])
    config = ContactIKConfig(max_joint_velocity=args.max_joint_velocity,
                             iterations=args.ik_iterations,
                             smoothing_sigma_frames=args.smoothing_sigma)
    model, _, index = compile_scene(include_robot=True)
    corrected, diagnostics = correct_contact_reference(
        model, index, qpos, board_qpos, mask, physical_time, config
    )
    robot_state = entity_state_from_pose(corrected, fps)
    board_state = entity_state_from_pose(np.c_[board_qpos, np.zeros((frames, 4))], fps)
    contacts = np.full((frames, len(CONTACT_NAMES)), -1, dtype=np.int8)
    contacts[:, :2] = np.where(mask, 1, np.where(contact_prior == 0, 0, -1))
    active_gap = np.abs(diagnostics["foot_support_gap_m"][mask])
    active_slip = diagnostics["foot_anchor_error_xy_m"][mask]
    joint_velocities = np.abs(robot_state[:, 42:])
    joint_ranges = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]]
    violation = np.maximum(joint_ranges[:, 0] - corrected[:, 7:], 0) + np.maximum(corrected[:, 7:] - joint_ranges[:, 1], 0)
    qa = {
        "contact_samples_per_foot": mask.sum(axis=0).tolist(),
        "active_support_gap_max_m": float(active_gap.max()),
        "active_support_gap_mean_m": float(active_gap.mean()),
        "active_anchor_error_xy_max_m": float(active_slip.max()),
        "foot_board_penetration_max_m": float(diagnostics["foot_board_penetration_m"].max()),
        "joint_limit_violation_max_rad": float(violation.max()),
        "joint_velocity_max_rad_s": float(joint_velocities.max()),
        "joint_velocity_bound_rad_s": config.max_joint_velocity,
        "passed_geometric_thresholds": bool(active_gap.max() <= 0.025 and active_slip.max() <= 0.04 and diagnostics["foot_board_penetration_m"].max() <= 0.015 and violation.max() <= 1e-6),
        "meaning": "Kinematic geometry diagnostics only; contacts are priors, no force/dynamic success certification",
    }
    metadata = {
        "time_scale": robot_metadata["time_scale_hypothesis"],
        "physics_fps": fps, "video_time_origin_s": 0.0,
        "provenance": {
            "robot": str(args.robot), "robot_sha256": sha256(args.robot),
            "board": str(args.board), "board_sha256": sha256(args.board),
            "method": "Soft contact-aware CPU IK on real G1 geometry; not a physical rollout",
            "board_pose": "Monocular engineering estimate with uncertain/temporally completed segments",
        },
        "config": asdict(config), "qa": qa,
        "human_scale": {key: robot_metadata[key] for key in ("actual_human_height_m", "pelvis_scale", "human_scale_table")},
        "board_scale": 1.0, "board_geometry": "Original physical board dimensions retained; not multiplied by human pelvis scale",
        "root_calibration": "Translation fitted from marked stance foot supports, interpolated through gaps; engineering prior, not measured camera/world motion",
        "stance_video_windows": args.stance_video_window,
        "stance_prior_mode": "Explicit windows replace positive upstream priors" if args.stance_video_window else "Upstream foot_board_prior",
        "confidence_threshold": args.min_confidence,
        "contact_label_meaning": "Explicit stance priors passing confidence threshold, not measured forces",
        "unknown_contact_policy": "Unknown remains -1; labelled flight has no enforced foot-board constraints",
        "wheel_motion_observed": False, "wheel_initialization": "All four passive wheel angles/velocities set to zero; not observed",
        "unresolved": ["No contact-force/torque feasibility solved", "No learned Ollie policy or world model", "Board reference may itself penetrate/violate dynamics", "Foot overhang is allowed; body-board and self collisions require separate QA"],
    }
    result = JointReference(
        layout=JointStateLayout(BOARD_JOINT_NAMES), time_video=video_time,
        time_physics=physical_time, robot_state=robot_state, board_state=board_state,
        phase=phase, contacts=contacts, confidence=np.minimum(confidence, 0.7), metadata=metadata,
    )
    result.save(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(args.out.with_name(args.out.stem + "_diagnostics.npz"),
                        **diagnostics, corrected_qpos=corrected, initial_qpos=qpos,
                        time_video=video_time, time_physics=physical_time)
    print(json.dumps({"out": str(args.out), "frames": frames, "state_dim": result.layout.state_dim, "qa": qa}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
