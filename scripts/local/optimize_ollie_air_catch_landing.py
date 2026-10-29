"""Plan preparation and airborne catch together, using only original29 PD targets.

This is physical teacher exploration for SuperMimic, not policy training.
Original geometry, passive board, continuous initialization and strict v3 stay.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import ollie_air_phase_sensors as air_sensor
from scripts.local import optimize_ollie_prelaunch_capture_landing as previous

survival = previous.survival
landing, protocol = survival.landing, survival.protocol
SCHEMA = "supermimic.ollie.air_catch_landing.v1"
WARM86_SHA = "b7866c9421eaaefd727e0774756a000e18013b65dcc8061e75e35f3ca674fc8f"
TASK = {
    "air_catch_window_samples": 4,
    "air_catch_window_s": 0.01,
    "air_catch_requires": "same native10ms window, qualified flight start through first real board-ground Fn>=.1N, descending COM, original safety",
    "catch_top_target_each_n": 1.0,
    "catch_is_soft_objective_not_new_acceptance": True,
    "no_continuous_top_force_reward_over_entire_air": True,
    "preparation_capture_window": "entire optimization start through first qualified clearance window start",
    "selection": "q/(1+q) + .25*preCP/(600+preCP) + .25*lost_fraction + .05*raw/(50000+raw)",
    "catch_quality_components": "mean both-foot finite witness gap/.04m, XY excess/.10,.035m, relative XYZ speed/.75m/s, body posture, deck tilt and1N top deficit in same window",
    "foot_xy_allowed_half_m": [0.10, 0.035],
    "relative_foot_velocity_scale_m_s": 0.75,
    "retained_raw_terms": "all83 terms except delayed top mean, same50ms, finite gap, footXY and capture point",
    "ranking_change_is_not_isolated_timing_ablation": True,
    "only_event_grades_0_1_2": True,
    "original_strict_v3_unchanged": True,
}


@dataclass(frozen=True)
class Config(survival.Config):
    prefix_controls: int = 130
    horizon: int = 127
    knot_times_s: tuple = (
        0,
        0.04,
        0.10,
        0.18,
        0.26,
        0.34,
        0.42,
        0.50,
        0.56,
        0.60,
        0.64,
        0.68,
        0.74,
        0.86,
        1.10,
        1.50,
        2.52,
    )
    seed: int = 90
    wall_budget_s: float = 1200.0
    selection: str = (
        "whole_geometry_qualified_flight_grade_then_air_catch_preparation_survival_cost"
    )


def phase_intervals(qualified, board_ground, descending, safe):
    """Keep the40ms certificate separate from when catch guidance starts.

    qualified[:,i] describes samples i..i+15, not a timestamp at its end.
    First actual ground contact after that start closes the catch interval.
    Ground/support from a later bounce cannot reopen this first-flight phase.
    """
    n, samples = board_ground.shape
    if qualified.shape != (n, samples - 15):
        raise ValueError("Expected full16-sample qualified-window array")
    index = np.arange(samples)[None]
    has = qualified.any(1)
    start = np.where(has, qualified.argmax(1), samples)
    after = board_ground & (index >= start[:, None])
    end = np.where(after.any(1), after.argmax(1), samples)
    airborne = has[:, None] & (index >= start[:, None]) & (index < end[:, None])
    opportunity = airborne & descending & safe
    return start, end, airborne, opportunity


def minimum_same_window(value, eligible, length):
    """Worst point in one contiguous fully eligible native window; never zero-fill."""
    if value.shape != eligible.shape or value.shape[1] < length:
        raise ValueError("Invalid synchronized catch arrays")
    maxima = landing.base.max_windows(value, length)
    allowed = landing.base.min_windows(eligible.astype(np.int8), length).astype(bool)
    costs = np.where(allowed, maxima, np.inf)
    index = costs.argmin(1)
    best = costs[np.arange(len(value)), index]
    return best, index, allowed.any(1)


def preparation_capture(controller, sensors, start):
    sl = controller.slices
    bq, bp = sensors[..., sl["boardquat"]], sensors[..., sl["boardpos"]]
    rc, bc = sensors[..., sl["root_com"]], sensors[..., sl["board_com"]]
    rv, bv = sensors[..., sl["root_com_v"]], sensors[..., sl["board_com_v"]]
    heading = landing.base.rotate(bq, np.array([1.0, 0, 0]))[..., :2]
    heading /= np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-8)
    lateral = np.stack((-heading[..., 1], heading[..., 0]), -1)
    omega = np.sqrt(9.81 / np.maximum(rc[..., 2] - bp[..., 2], 0.2))
    cp = rc[..., :2] - bc[..., :2] + (rv[..., :2] - bv[..., :2]) / omega[..., None]
    local = np.stack((np.sum(cp * heading, -1), np.sum(cp * lateral, -1)), -1)
    excess = np.maximum(
        abs(local - landing.TASK["capture_point_target_heading_xy_m"])
        - landing.TASK["capture_point_allowed_half_xy_m"],
        0,
    )
    penalty = np.sum((excess / [0.20, 0.06]) ** 2, -1)
    active = np.arange(sensors.shape[1])[None] < start[:, None]
    return 600 * (penalty * active).sum(1) / np.maximum(active.sum(1), 1)


def air_catch_cost(controller, states, sensors, targets, config):
    grades, old_raw, old_terms, metrics = landing.landing_cost(
        controller, states, sensors, targets, config
    )
    observed = landing.top.top_support(controller, sensors)
    real_ground = air_sensor.board_ground(controller, sensors)
    original_grade, qualified, safe, _, _, _ = landing.flight_data(
        controller, sensors, config, observed["robot_ground_above_threshold"]
    )
    # Add per-contact Fn evidence to the old conservative net-world-Fz gate.
    qualified &= landing.base.min_windows((~real_ground["active"]).astype(np.int8), 16).astype(bool)
    grade = np.where(qualified.any(1), 2, np.minimum(original_grade, 1))
    sl = controller.slices
    board_vz = sensors[..., sl["board_com_v"]][..., 2]
    start, end, airborne, opportunity = phase_intervals(
        qualified, real_ground["active"], board_vz <= 0, safe
    )
    n, samples = airborne.shape
    # Evaluate exact authored capsules only at actual airborne opportunities.
    # This covers the full horizon at400Hz, with no elapsed.60s truncation.
    rows, cols = np.nonzero(opportunity)
    selected = sensors[rows, cols]
    preceding = sensors[rows, np.maximum(cols - 1, 0)]
    quality = np.full((n, samples), np.inf)
    gap_term = np.zeros(len(rows))
    xy_term = np.zeros(len(rows))
    velocity_term = np.zeros(len(rows))
    for side in ("left", "right"):
        low, high = landing.recovery.capsule_bounds(controller, selected, side)
        center = (low + high) / 2
        old_low, old_high = landing.recovery.capsule_bounds(controller, preceding, side)
        relative_v = (center - (old_low + old_high) / 2) / config.physics_dt
        excess = np.maximum(
            abs(center[:, :2] - controller.catch_initial_centroid[side])
            - TASK["foot_xy_allowed_half_m"],
            0,
        )
        xy_term += np.sum((excess / [0.10, 0.035]) ** 2, -1) / 2
        velocity_term += (
            np.sum((relative_v / TASK["relative_foot_velocity_scale_m_s"]) ** 2, -1) / 2
        )
        gap = np.min(
            np.stack(
                [
                    landing.recovery.finite_capsule_witness_gap(
                        controller, selected, side, "board_" + box
                    )
                    for box in ("center", "tail", "nose")
                ],
                -1,
            ),
            -1,
        )
        gap_term += (gap / 0.04) ** 2 / 2
    rq = selected[..., sl["rootquat"]]
    bq = selected[..., sl["boardquat"]]
    up = 1 - 2 * (rq[..., 1] ** 2 + rq[..., 2] ** 2)
    bu = 1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)
    height = selected[..., sl["rootpos"]][..., 2] - selected[..., sl["boardpos"]][..., 2]
    pose = (np.maximum(0.9 - up, 0) / 0.2) ** 2 + (np.maximum(0.6 - height, 0) / 0.2) ** 2
    top_min = np.minimum(observed["left_top_n"][rows, cols], observed["right_top_n"][rows, cols])
    top_deficit = (1 - np.clip(top_min / TASK["catch_top_target_each_n"], 0, 1)) ** 2
    # 1N is a short-window contact hint, not continuous compression of the board.
    quality[rows, cols] = (
        gap_term
        + xy_term
        + 0.10 * velocity_term
        + pose
        + (np.maximum(0.98 - bu, 0) / 0.10) ** 2
        + top_deficit
    )
    best, at, has_window = minimum_same_window(
        quality, opportunity, TASK["air_catch_window_samples"]
    )
    best = np.where(has_window, best, 100.0)
    prep = preparation_capture(controller, sensors, start)
    geometry, depth, terminal_depth, board_up = survival.whole_episode_geometry(
        controller, states, sensors
    )
    grade = np.where((grade == 2) & ~geometry, 1, grade)
    removed = (
        "both_top_mean_after_air",
        "same50ms_both_top_and_body",
        "finite_gap_after_air",
        "foot_xy_recovery",
        "horizontal_capture_point_soft",
    )
    raw = old_raw - sum(old_terms[key] for key in removed)
    _, lost, _ = survival.priority_cost(raw, metrics["first_robot_ground_s"], config)
    lost_fraction = lost / ((config.prefix_controls + config.horizon) * 8 + 1)
    terms = {
        "air_catch_quality_bounded": np.where(has_window, best / (1 + best), 1.0),
        "preparation_capture_bounded": 0.25 * prep / (600 + prep),
        "ground_free_survival_bounded": 0.25 * lost_fraction,
        "remaining83_cost_bounded": 0.05 * raw / (50000 + raw),
    }
    score = sum(terms.values())
    valid = (grades >= 0) & np.isfinite(score) & (raw >= 0) & real_ground["valid"].all(1)
    true_both = opportunity & (observed["left_top_n"] > 1) & (observed["right_top_n"] > 1)
    rq_all = sensors[..., sl["rootquat"]]
    up_all = 1 - 2 * (rq_all[..., 1] ** 2 + rq_all[..., 2] ** 2)
    height_all = sensors[..., sl["rootpos"]][..., 2] - sensors[..., sl["boardpos"]][..., 2]
    stable_both = true_both & (up_all >= 0.9) & (height_all >= 0.6)
    origin = config.prefix_controls * config.control_dt
    metrics.update(
        original83_raw_cost=old_raw,
        remaining83_raw_cost=raw,
        air_catch_quality=best,
        air_catch_has10ms_opportunity=has_window.astype(int),
        air_catch_best_window_start_s=np.where(has_window, origin + at * config.physics_dt, -1),
        first_qualified_clearance_start_s=np.where(
            start < samples, origin + start * config.physics_dt, -1
        ),
        first_true_board_recontact_s=np.where(end < samples, origin + end * config.physics_dt, -1),
        longest_both_true_top_during_first_air_s=landing.longest_samples(true_both)
        * config.physics_dt,
        longest_both_true_top_and_pose_during_first_air_s=landing.longest_samples(stable_both)
        * config.physics_dt,
        whole_episode_geometry_pass=geometry.astype(int),
        whole_episode_maximum_penetration_m=depth,
        terminal_geometry_penetration_m=terminal_depth,
        whole_episode_minimum_board_upright=board_up,
        preparation_capture_point_cost=prep,
        board_ground_sensor_overflow=real_ground["overflow"].any(1).astype(int),
    )
    return np.where(valid, grade, -1), np.where(valid, score, np.inf), terms, metrics


def make_cost(out_dir):
    first = True

    def cost(controller, states, sensors, targets, config):
        nonlocal first
        result = air_catch_cost(controller, states, sensors, targets, config)
        grades, scores, _, _ = result
        if first:
            ids = np.flatnonzero((grades >= 0) & np.isfinite(scores))
            if not len(ids):
                raise RuntimeError("No finite air-catch candidate")
            j = int(ids[np.lexsort((scores[ids], -grades[ids]))[0]])
            protocol.write_json(
                out_dir / "first_top_stage_proof.json",
                survival.selected_top_proof(controller, states, sensors, targets, j, config),
            )
            first = False
        return result

    return cost


def prior_factory(full, warm86, warm85, warm_paths):
    factory = survival.early.prior_factory(full)

    def prepare(controller, source, extra, config):
        nominal, _, metadata = factory(controller, source, extra, config)
        metadata["air_phase_observation"] = air_sensor.add_board_ground_sensor(controller, config)
        priors = {}
        for name, warm, path, sha in (
            ("actual86_low_hop_not_teacher", warm86, warm_paths[0], WARM86_SHA),
            ("actual85_low_hop_not_teacher", warm85, warm_paths[1], previous.WARM_SHA),
        ):
            if not np.array_equal(
                warm["action"][: config.prefix_controls], full["action"][: config.prefix_controls]
            ):
                raise ValueError("Warm targets lack the identical continuous prefix")
            target = warm["action"][config.prefix_controls :].copy()
            if target.shape != nominal.shape or not np.array_equal(
                target, np.clip(target, *controller.target_limits.T)
            ):
                raise ValueError("Invalid warm target prior")
            priors[name] = target
            metadata[name] = {
                "path": str(path.resolve()),
                "sha256": sha,
                "only_pd_targets": True,
                "is_teacher": False,
            }
        priors["actual82_low_hop_not_teacher"] = nominal.copy()
        return nominal, priors, metadata

    return prepare


def main():
    parser = protocol.build_argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        source=Path("logs/ollie_recoverable_verified_candidate_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--prefix-view-dir", type=Path, default=Path("logs/ollie_top_landing_input_01")
    )
    parser.add_argument(
        "--warm86",
        type=Path,
        default=Path("logs/ollie_prelaunch_capture_landing_01/actual_rollout.npz"),
    )
    parser.add_argument(
        "--warm85",
        type=Path,
        default=Path("logs/ollie_survival_priority_landing_01/actual_rollout.npz"),
    )
    args, config = parser.parse_args(), Config()
    if any(
        (
            args.extra_prior,
            args.extra_prior_sha256,
            args.extra_prior_result,
            args.extra_prior_result_sha256,
        )
    ):
        parser.error("This experiment uses only documented actual PD-target priors")
    warm = []
    for path, sha in ((args.warm86, WARM86_SHA), (args.warm85, previous.WARM_SHA)):
        if protocol.digest(path) != sha:
            raise ValueError("Frozen warm target archive changed")
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in archive.files}
        if values["action"].shape != (257, 29) or bool(values["verified_teacher"]):
            raise ValueError("Expected complete exploration archive")
        warm.append(values)
    original = args.source
    args.source, full, metadata = landing.prepare_source(original, args.prefix_view_dir)
    protocol.run_experiment(
        args,
        config,
        cost_function=make_cost(args.out_dir),
        prior_factory=prior_factory(full, *warm, (args.warm86, args.warm85)),
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=landing.previous.highest_grade_elites,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "air_sensor_script_sha256": protocol.digest(air_sensor.__file__),
            "reference_survival_script_sha256": protocol.digest(survival.__file__),
            "reference_boundary_script_sha256": protocol.digest(survival.early.__file__),
            "reference_landing_script_sha256": protocol.digest(landing.__file__),
            "top_sensor_script_sha256": protocol.digest(landing.top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "original_full_actual_source": metadata,
            "original_source_sha256": landing.SOURCE_SHA,
            "warm_sources": [
                {"path": str(path.resolve()), "sha256": sha}
                for path, sha in ((args.warm86, WARM86_SHA), (args.warm85, previous.WARM_SHA))
            ],
            "task": TASK,
            "config": asdict(config),
            "search_is_teacher_exploration": True,
        },
    )
    if protocol.digest(original) != landing.SOURCE_SHA:
        raise RuntimeError("Original source changed")
    for path, sha in ((args.warm86, WARM86_SHA), (args.warm85, previous.WARM_SHA)):
        if protocol.digest(path) != sha:
            raise RuntimeError("Warm input changed")


if __name__ == "__main__":
    main()
