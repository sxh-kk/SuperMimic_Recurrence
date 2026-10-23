"""Audit physical timing from mass-weighted G1 + board COM; read-only inputs.

Fits are conditional diagnostics of a reconstructed kinematic robot trajectory.
They do not identify the original video's capture/slow-motion rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic.ollie.physics import compile_scene
from supermimic.ollie.reference import JointReference


def quadratic_fit(time, height, start, end, gravity, scale):
    chosen = (time >= start - 1e-9) & (time <= end + 1e-9)
    if chosen.sum() < 6:
        raise ValueError("Need at least six samples per timing fit")
    t = time[chosen]
    origin = float(t.mean())
    a, b, c = np.polyfit(t - origin, height[chosen], 2)
    residual = height[chosen] - np.polyval([a, b, c], t - origin)
    acceleration = float(2 * a)
    equivalent_scale = float(np.sqrt(gravity / -acceleration)) if acceleration < 0 else None
    # For a fixed assumed gravity, only initial height and velocity remain free.
    gravity_term = -0.5 * gravity / scale**2 * (t - origin)**2
    intercept, slope = np.linalg.lstsq(
        np.column_stack([np.ones(len(t)), t - origin]), height[chosen] - gravity_term, rcond=None
    )[0]
    fixed_gravity_residual = height[chosen] - (gravity_term + intercept + slope * (t - origin))
    return {
        "requested_video_window_s": [start, end], "sampled_video_window_s": [float(t[0]), float(t[-1])],
        "samples": len(t), "origin_video_s": origin,
        "polynomial_abc": [float(a), float(b), float(c)],
        "video_vertical_acceleration_m_s2": acceleration,
        "physical_acceleration_at_assumed_scale_m_s2": acceleration * scale**2,
        "gravity_equivalent_scale_conditional": equivalent_scale,
        "quadratic_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "max_abs_quadratic_residual_m": float(np.abs(residual).max()),
        "fixed_gravity_at_assumed_scale_rmse_m": float(np.sqrt(np.mean(fixed_gravity_residual**2))),
    }


def geometry_minimum_z(model, data, ids):
    """Exact support along world Z for the primitive collision geometry used here."""
    values = []
    for gid in ids:
        kind = int(model.geom_type[gid])
        local_vertical = data.geom_xmat[gid].reshape(3, 3)[2]
        size = model.geom_size[gid]
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            support = size[0]
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            support = size[0] + size[1] * abs(local_vertical[2])
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            support = size[1] * abs(local_vertical[2]) + size[0] * np.linalg.norm(local_vertical[:2])
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            support = np.dot(size, np.abs(local_vertical))
        else:
            raise ValueError(f"Unimplemented active collision geometry support: {model.geom(gid).name}, {kind}")
        values.append(data.geom_xpos[gid, 2] - support)
    return float(min(values))


def segments(mask):
    indices = np.flatnonzero(mask)
    return [part for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(part)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/ollie/joint_reference_50hz.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/ollie/timing_audit"))
    parser.add_argument("--video-window", type=float, nargs=2, default=[4.71, 5.26])
    parser.add_argument("--support-clearance", type=float, default=0.01,
                        help="Metres; geometrically clear candidates, not measured force-free flight")
    args = parser.parse_args()
    reference = JointReference.load(args.reference)
    model, data, index = compile_scene(include_robot=True)
    mass = model.body_mass.copy()
    board_mask = model.body_rootid == model.body_rootid[index.board_body]
    robot_mask = (mass > 0) & ~board_mask
    board_mass, robot_mass = float(mass[board_mask].sum()), float(mass[robot_mask].sum())
    total_mass = float(mass.sum())
    gravity = abs(float(model.opt.gravity[2]))
    scale = reference.metadata["time_scale"]
    time = reference.time_video
    frames = len(time)
    ground = model.geom("ground").id
    active = [gid for gid in range(model.ngeom) if model.geom_bodyid[gid] != 0 and (
        (model.geom_contype[gid] & model.geom_conaffinity[ground]) or
        (model.geom_contype[ground] & model.geom_conaffinity[gid])
    )]
    board_geoms = [gid for gid in active if board_mask[model.geom_bodyid[gid]]]
    com, robot_com, board_com = [np.empty((frames, 3)) for _ in range(3)]
    clearance = np.empty(frames)
    board_clearance = np.empty(frames)
    ground_contact = np.zeros(frames, dtype=bool)
    consistency = 0.0

    def assign(robot_pose, board):
        data.qpos[index.robot_qpos:index.robot_qpos + 7] = robot_pose[:7]
        data.qpos[index.robot_joint_qpos] = robot_pose[7:36]
        data.qpos[index.board_qpos:index.board_qpos + 7] = board[:7]
        data.qpos[index.wheel_qpos] = board[7:11]
        mujoco.mj_forward(model, data)

    for frame in range(frames):
        assign(reference.robot_state[frame], reference.board_state[frame])
        com[frame] = (mass[:, None] * data.xipos).sum(axis=0) / total_mass
        robot_com[frame] = (mass[robot_mask, None] * data.xipos[robot_mask]).sum(axis=0) / robot_mass
        board_com[frame] = (mass[board_mask, None] * data.xipos[board_mask]).sum(axis=0) / board_mass
        consistency = max(consistency, float(np.max(np.abs(com[frame] - data.subtree_com[0]))))
        clearance[frame] = geometry_minimum_z(model, data, active)
        board_clearance[frame] = geometry_minimum_z(model, data, board_geoms)
        ground_contact[frame] = any(ground in item.geom and item.dist <= 0.002 for item in data.contact[:data.ncon])

    start, end = args.video_window
    chosen = (time >= start) & (time <= end)
    primary = quadratic_fit(time, com[:, 2], start, end, gravity, scale)
    jitter = [quadratic_fit(time, com[:, 2], start + left, end + right, gravity, scale)
              for left in np.arange(-0.09, 0.091, 0.03)
              for right in np.arange(-0.09, 0.091, 0.03)
              if end + right - start - left >= 0.3]
    scales = [row["gravity_equivalent_scale_conditional"] for row in jitter
              if row["gravity_equivalent_scale_conditional"] is not None]
    near_flight = [part for part in segments(clearance > args.support_clearance)
                   if time[part[0]] <= (start + end) / 2 <= time[part[-1]]]
    flight_summary = None
    if near_flight:
        part = near_flight[0]
        lo, hi = int(part[0]), int(part[-1])
        flight_summary = {
            "clearance_threshold_m": args.support_clearance,
            "sampled_video_window_s": [float(time[lo]), float(time[hi])],
            "physical_duration_at_assumed_scale_s": float((time[hi] - time[lo]) / scale),
            "bracketed_takeoff_video_s": [float(time[max(0, lo - 1)]), float(time[lo])],
            "bracketed_landing_video_s": [float(time[hi]), float(time[min(frames - 1, hi + 1)])],
            "conditional_quadratic_fit": quadratic_fit(time, com[:, 2], time[lo], time[hi], gravity, scale),
            "warning": "Geometry along a prescribed pose sequence, not force-verified flight or original video contact timing",
        }
    initial_com = None
    pre_ik_fit = None
    source = Path(reference.metadata["provenance"]["robot"])
    if source.exists():
        with np.load(source, allow_pickle=False) as original:
            poses = original["qpos"]
            if poses.shape != (frames, 36) or not np.allclose(original["time_video"], time):
                raise ValueError("GMR source no longer matches the reference time axis")
        initial_com = np.empty_like(com)
        for frame in range(frames):
            assign(poses[frame], reference.board_state[frame])
            initial_com[frame] = (mass[:, None] * data.xipos).sum(axis=0) / total_mass
        pre_ik_fit = quadratic_fit(time, initial_com[:, 2], start, end, gravity, scale)

    candidates = []
    for candidate in (1.25, 1.5, 1.75, 1.875, 2.0, 2.2):
        fit = quadratic_fit(time, com[:, 2], start, end, gravity, candidate)
        candidates.append({"scale": candidate,
                           "inferred_acceleration_m_s2": fit["physical_acceleration_at_assumed_scale_m_s2"],
                           "fixed_gravity_rmse_m": fit["fixed_gravity_at_assumed_scale_rmse_m"]})
    board_peak_index = int(reference.board_state[:, 2].argmax())
    board_rest = float(np.median(reference.board_state[time <= 2, 2]))
    board_rise = float(reference.board_state[board_peak_index, 2] - board_rest)
    central_physical_duration = (end - start) / scale
    summary = {
        "reference": str(args.reference.resolve()),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "mass_total_kg": total_mass, "robot_mass_kg": robot_mass, "board_mass_kg": board_mass,
        "com_method": "sum(all body_mass * world inertial xipos) / total body_mass; includes every robot and board body",
        "subtree_com_consistency_max_m": consistency,
        "gravity_m_s2": gravity, "time_scale_assumption": scale,
        "primary_total_com_fit": primary, "pre_contact_ik_total_com_fit": pre_ik_fit,
        "window_geometry": {
            "minimum_ground_clearance_m": float(clearance[chosen].min()),
            "ground_contact_samples": int(ground_contact[chosen].sum()),
            "scope": "Geometric absence of ground support only; no applied forces were inferred from kinematic poses",
        },
        "boundary_jitter_sensitivity": {
            "start_end_jitter_video_s": [-0.09, 0.09], "fits": len(jitter),
            "conditional_scale_min_median_max": [float(np.min(scales)), float(np.median(scales)), float(np.max(scales))],
            "not_a_confidence_interval": True,
        },
        "geometry_clear_interval_containing_apex": flight_summary,
        "board_height_check": {
            "peak_root_height_m": float(reference.board_state[board_peak_index, 2]),
            "peak_video_time_s": float(time[board_peak_index]), "preparation_root_height_m": board_rest,
            "root_rise_m": board_rise,
            "if_passive_point_same_height_total_flight_s": float(np.sqrt(8 * board_rise / gravity)),
            "central_window_physical_duration_s": central_physical_duration,
            "same_height_passive_point_rise_from_central_duration_m": gravity * central_physical_duration**2 / 8,
            "warning": "Board root is not total system COM; feet can act on the board in the air; central clear window is not the full flight",
        },
        "total_com_peak": {"z_m": float(com[:, 2].max()), "video_time_s": float(time[com[:, 2].argmax()])},
        "candidate_scale_diagnostics": candidates,
        "caveats": [
            "Retargeted G1 has different mass distribution/proportions from the real person; its COM is not the person's measured COM.",
            "Monocular scale, root calibration, board fitting and contact IK alter the reconstructed trajectory.",
            "Board fit already used a weak gravity prior with scale 1.5; this audit is not independent of that assumption.",
            "Fitted sample count includes temporal interpolation; residuals/window sensitivity do not bound reconstruction bias.",
            "One constant time scale cannot repair a visibly non-parabolic reference or unknown external support.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(args.out_dir / "com_trajectory.npz", time_video=time,
                        total_com=com, robot_com=robot_com, board_com=board_com,
                        ground_clearance=clearance, board_clearance=board_clearance,
                        ground_contact=ground_contact)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    display = (time >= start - 0.35) & (time <= end + 0.35)
    axes[0].plot(time[display], com[display, 2], label="G1 + board total COM")
    if initial_com is not None:
        axes[0].plot(time[display], initial_com[display, 2], "--", label="Before contact IK")
    fit_time = time[chosen]
    axes[0].plot(fit_time, np.polyval(primary["polynomial_abc"], fit_time - primary["origin_video_s"]), label="Quadratic fit")
    axes[0].set(ylabel="COM height (m)", title="Timing audit of a kinematic estimate; no flight success claim")
    axes[0].legend()
    axes[1].plot(time[display], reference.board_state[display, 2], label="Board root height")
    axes[1].plot(time[display], clearance[display], label="Lowest collision clearance")
    for axis in axes:
        axis.axvspan(start, end, color="gray", alpha=0.12)
        axis.grid(alpha=0.25)
    axes[1].set(xlabel="Presentation/video time (s)", ylabel="Height / clearance (m)")
    axes[1].legend()
    fig.savefig(args.out_dir / "timing_audit.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
