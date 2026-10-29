"""Explore directed foot recovery and early landing, without altering physics.

Strict v3 remains the only success certificate. Airborne and post-ground catch
are reported separately; airborne double support is not an extra hard rule.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import ollie_directed_top_gap as directed
from scripts.local import optimize_ollie_air_catch_landing as previous

landing, protocol, survival = previous.landing, previous.protocol, previous.survival
SCHEMA = "supermimic.ollie.directed_early_landing.v1"
ACTUAL90_SHA = "ac6effe28909ca875149e6f94cdb17b3b59b0152a4960dfff4ab7e38beb35526"
TASK = {
    "landing_grace_samples": 24,
    "landing_grace_s": 0.06,
    "quality_window_samples": 4,
    "both_support_progress_samples": 8,
    "left_support_progress_samples": 4,
    "directed_normal_cosine": 0.5,
    "gap_scale_m": 0.02,
    "upright_soft_target": 0.9,
    "root_board_height_soft_target_m": 0.55,
    "air_and_ground_landing_reported_separately": True,
    "no_requirement_for_airborne_double_support_in_original_strict_v3": True,
    "progress_tiers": "0=predicted final-standing proxy without robot-ground;1=same20ms both top+pose by first ground+60ms;2=same10ms left top+pose;3=directed approach",
    "tier_separation": 4.0,
    "selection_unrecovered": "4*tier + quality/(1+quality) + .001*survival_fraction + .001*final/(1+final)",
    "selection_recovered": "4*tier + 2*survival_fraction + .5*final/(1+final) + .25*quality/(1+quality)",
    "preparation_cp_not_in_selection": True,
    "actual_joint_violation_penalty": ".05*e/(1+e), e=(maximum_violation/.05rad)^2; target limits still original",
    "only_event_grades_0_1_2": True,
    "same_original29PD_passive_board_and_strict_v3": True,
}


@dataclass(frozen=True)
class Config(previous.Config):
    seed: int = 92
    selection: str = "whole_geometry_flight_grade_then_directed_recovery_landing_tier"


def landing_interval(start, end, descending, safe, grace_samples):
    index = np.arange(descending.shape[1])[None]
    has = start < descending.shape[1]
    # Descent is required before impact, not afterwards: contact may reverse Vz.
    eligible = has[:, None] & (index >= start[:, None])
    eligible &= index < np.minimum(end + grace_samples, descending.shape[1])[:, None]
    eligible &= descending | (index >= end[:, None])
    return eligible & safe


def selection_cost(tier, quality, has_window, lost_fraction, final_cost, joint_violation=0):
    """Disjoint tier ranges; no preparation score can buy worse foot recovery."""
    q = np.where(has_window, quality / (1 + quality), 1.0)
    terminal = final_cost / (1 + final_cost)
    approach = q + 0.001 * lost_fraction + 0.001 * terminal
    recovered = 2 * lost_fraction + 0.5 * terminal + 0.25 * q
    joint = (np.asarray(joint_violation) / 0.05) ** 2
    return 4 * tier + np.where(tier <= 1, recovered, approach) + 0.05 * joint / (1 + joint)


def directed_landing_cost(controller, states, sensors, targets, config):
    sl, ix, model = controller.slices, controller.index, controller.model
    observed = landing.top.top_support(controller, sensors)
    board_ground = previous.air_sensor.board_ground(controller, sensors)
    ground = observed["robot_ground_above_threshold"]
    grade, qualified, safe, lower, floor, bu = landing.flight_data(
        controller, sensors, config, ground
    )
    qualified &= landing.base.min_windows((~board_ground["active"]).astype(np.int8), 16).astype(
        bool
    )
    grade = np.where(qualified.any(1), 2, np.minimum(grade, 1))
    start, end, airborne, _ = previous.phase_intervals(
        qualified, board_ground["active"], sensors[..., sl["board_com_v"]][..., 2] <= 0, safe
    )
    eligible = landing_interval(
        start,
        end,
        sensors[..., sl["board_com_v"]][..., 2] <= 0,
        safe,
        TASK["landing_grace_samples"],
    )
    finite_sensors = np.isfinite(sensors).all((1, 2))
    eligible &= finite_sensors[:, None]
    n, samples = eligible.shape
    rows, cols = np.nonzero(eligible)
    selected = sensors[rows, cols]
    old_selected = sensors[rows, np.maximum(cols - 1, 0)]
    gap_cost, velocity_cost = np.zeros(len(rows)), np.zeros(len(rows))
    for side in ("left", "right"):
        geometry = directed.directed_top_gap(controller, selected, side)
        gap_cost += (geometry["gap_m"] / TASK["gap_scale_m"]) ** 2 / 2
        # Relative body origin velocity is a soft approach-speed hint, not a
        # claimed contact-point velocity or a force/contact certificate.
        local = landing.base.inverse_rotate(
            selected[..., sl["boardquat"]],
            selected[..., sl[side + "_body_pos"]] - selected[..., sl["boardpos"]],
        )
        old_local = landing.base.inverse_rotate(
            old_selected[..., sl["boardquat"]],
            old_selected[..., sl[side + "_body_pos"]] - old_selected[..., sl["boardpos"]],
        )
        velocity_cost += np.sum(((local - old_local) / config.physics_dt / 0.75) ** 2, -1) / 2
    rq = sensors[..., sl["rootquat"]]
    ru = 1 - 2 * (rq[..., 1] ** 2 + rq[..., 2] ** 2)
    height = sensors[..., sl["rootpos"]][..., 2] - sensors[..., sl["boardpos"]][..., 2]
    posture = (np.maximum(0.9 - ru, 0) / 0.2) ** 2 + (np.maximum(0.55 - height, 0) / 0.15) ** 2
    left, right = observed["left_top_n"], observed["right_top_n"]
    top_deficit = (1 - np.clip(np.minimum(left, right), 0, 1)) ** 2
    quality = np.full((n, samples), np.inf)
    # All terms refer to the same physical samples, including the balance hint.
    rc, bc = selected[..., sl["root_com"]], selected[..., sl["board_com"]]
    rv, bv = selected[..., sl["root_com_v"]], selected[..., sl["board_com_v"]]
    heading = landing.base.rotate(selected[..., sl["boardquat"]], np.array([1.0, 0, 0]))[..., :2]
    heading /= np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-8)
    lateral = np.stack((-heading[..., 1], heading[..., 0]), -1)
    omega = np.sqrt(9.81 / np.maximum(rc[..., 2] - selected[..., sl["boardpos"]][..., 2], 0.2))
    cp = rc[..., :2] - bc[..., :2] + (rv[..., :2] - bv[..., :2]) / omega[..., None]
    local_cp = np.stack((np.sum(cp * heading, -1), np.sum(cp * lateral, -1)), -1)
    excess = np.maximum(abs(local_cp - [-0.055, 0]) - [0.23, 0.065], 0)
    cp_cost = np.sum((excess / [0.20, 0.06]) ** 2, -1)
    quality[rows, cols] = (
        gap_cost
        + 0.1 * velocity_cost
        + posture[rows, cols]
        + 0.25 * cp_cost
        + (np.maximum(0.98 - bu[rows, cols], 0) / 0.10) ** 2
        + 0.25 * top_deficit[rows, cols]
    )
    q, at, has_window = previous.minimum_same_window(
        quality, eligible, TASK["quality_window_samples"]
    )
    q = np.where(has_window, q, 1e6)
    good_pose = (ru >= 0.9) & (height >= 0.55)
    both = eligible & (left > 1) & (right > 1) & good_pose
    front = eligible & (left > 1) & good_pose
    both_count, front_count = landing.longest_samples(both), landing.longest_samples(front)

    previous_state = np.concatenate(
        (np.broadcast_to(controller.state(), (n, 1, states.shape[-1])), states[:, :-1]), 1
    )
    actual_q = previous_state[..., 1 + ix.robot_joint_qpos]
    final_q = states[:, -1, 1 + ix.robot_joint_qpos]
    violation = np.maximum(
        np.maximum(controller.limits[:, 0] - actual_q, actual_q - controller.limits[:, 1]), 0
    ).max((1, 2))
    violation = np.maximum(
        violation,
        np.maximum(
            np.maximum(controller.limits[:, 0] - final_q, final_q - controller.limits[:, 1]), 0
        ).max(1),
    )
    velocity = previous_state[..., 1 + model.nq :]
    robot_v = np.linalg.norm(velocity[..., ix.robot_qvel : ix.robot_qvel + 3], axis=-1)
    robot_w = np.linalg.norm(velocity[..., ix.robot_qvel + 3 : ix.robot_qvel + 6], axis=-1)
    board_v = np.linalg.norm(velocity[..., ix.board_qvel : ix.board_qvel + 3], axis=-1)
    board_w = np.linalg.norm(velocity[..., ix.board_qvel + 3 : ix.board_qvel + 6], axis=-1)
    final = np.arange(samples) * config.physics_dt >= config.horizon * config.control_dt - 0.30
    wheel_ids = [i for i, name in enumerate(controller.board_geoms) if name.startswith("wheel_")]
    four_wheels = (floor[..., wheel_ids] >= 0.1).all(-1)
    terminal = posture + top_deficit + (~four_wheels)
    terminal += (np.maximum(0.95 - bu, 0) / 0.20) ** 2
    for value, bound in ((robot_v, 0.3), (robot_w, 0.8), (board_v, 0.35), (board_w, 0.8)):
        terminal += (np.maximum(value - bound, 0) / bound) ** 2
    final_cost = terminal[:, final].mean(1)
    final_proxy = (
        ((left[:, final] > 1).mean(1) >= 0.8)
        & ((right[:, final] > 1).mean(1) >= 0.8)
        & (four_wheels[:, final].mean(1) >= 0.8)
        & (ru[:, final].min(1) >= 0.8)
        & (height[:, final].min(1) >= 0.55)
        & (bu[:, final].min(1) >= 0.95)
        & (robot_v[:, final].max(1) <= 0.3)
        & (robot_w[:, final].max(1) <= 0.8)
        & (board_v[:, final].max(1) <= 0.35)
        & (board_w[:, final].max(1) <= 0.8)
        & ~ground.any(1)
    )
    tier = np.where(final_proxy, 0, np.where(both_count >= 8, 1, np.where(front_count >= 4, 2, 3)))
    geometry, depth, terminal_depth, min_up = survival.whole_episode_geometry(
        controller, states, sensors
    )
    grade = np.where((grade == 2) & ~geometry, 1, grade)
    first_ground = np.where(
        ground.any(1),
        config.prefix_controls * config.control_dt + ground.argmax(1) * config.physics_dt,
        -1,
    )
    _, lost, _ = survival.priority_cost(final_cost, first_ground, config)
    lost_fraction = lost / ((config.prefix_controls + config.horizon) * 8 + 1)
    score = selection_cost(tier, q, has_window, lost_fraction, final_cost, violation)
    valid = (
        np.isfinite(states).all((1, 2))
        & finite_sensors
        & observed["valid"].all(1)
        & board_ground["valid"].all(1)
        & np.isfinite(score)
        & np.isclose(states[:, -1, 0], 5.14, rtol=0, atol=1e-9)
    )
    air_both = airborne & safe & (left > 1) & (right > 1)
    post_ground_only = eligible & (np.arange(samples)[None] >= end[:, None])
    end_seconds = config.prefix_controls * config.control_dt + end * config.physics_dt
    metrics = {
        "progress_tier": tier,
        "directed_landing_quality": q,
        "has10ms_opportunity": has_window.astype(int),
        "best_window_start_s": np.where(
            has_window, config.prefix_controls * config.control_dt + at * config.physics_dt, -1
        ),
        "first_qualified_clearance_start_s": np.where(
            start < samples,
            config.prefix_controls * config.control_dt + start * config.physics_dt,
            -1,
        ),
        "first_true_board_recontact_s": np.where(end < samples, end_seconds, -1),
        "first_robot_ground_s": first_ground,
        "maximum_whole_board_clearance_m": lower.max(1),
        "whole_episode_geometry_pass": geometry.astype(int),
        "whole_episode_maximum_penetration_m": depth,
        "terminal_geometry_penetration_m": terminal_depth,
        "whole_episode_minimum_board_upright": min_up,
        "longest_air_both_true_top_s": landing.longest_samples(air_both) * config.physics_dt,
        "longest_air_both_top_and_pose_s": landing.longest_samples(air_both & good_pose)
        * config.physics_dt,
        "longest_early_landing_both_top_and_pose_s": both_count * config.physics_dt,
        "longest_early_landing_left_top_and_pose_s": front_count * config.physics_dt,
        "longest_post_ground_only_both_top_and_pose_s": landing.longest_samples(
            post_ground_only & (left > 1) & (right > 1) & good_pose
        )
        * config.physics_dt,
        "longest_post_ground_only_left_top_and_pose_s": landing.longest_samples(
            post_ground_only & (left > 1) & good_pose
        )
        * config.physics_dt,
        "maximum_actual_joint_limit_violation_rad": violation,
        "final_left_top_fraction": (left[:, final] > 1).mean(1),
        "final_right_top_fraction": (right[:, final] > 1).mean(1),
        "final_four_wheel_proxy_fraction": four_wheels[:, final].mean(1),
        "final_standing_proxy_not_certificate": final_proxy.astype(int),
        "final_standing_cost": final_cost,
        "top_sensor_overflow": observed["overflow"].any(1).astype(int),
        "board_ground_sensor_overflow": board_ground["overflow"].any(1).astype(int),
    }
    recovered = tier <= 1
    return (
        np.where(valid, grade, -1),
        np.where(valid, score, np.inf),
        {
            "progress_tier": 4 * tier,
            "quality_bounded": np.where(recovered, 0.25, 1) * np.where(has_window, q / (1 + q), 1),
            "lost_ground_free_fraction": np.where(recovered, 2, 0.001) * lost_fraction,
            "final_standing_bounded": np.where(recovered, 0.5, 0.001)
            * final_cost
            / (1 + final_cost),
            "actual_joint_limit_violation_bounded": 0.05
            * (violation / 0.05) ** 2
            / (1 + (violation / 0.05) ** 2),
        },
        metrics,
    )


def make_cost(out_dir):
    first = True

    def cost(controller, states, sensors, targets, config):
        nonlocal first
        result = directed_landing_cost(controller, states, sensors, targets, config)
        grade, score, _, _ = result
        if first:
            ids = np.flatnonzero((grade >= 0) & np.isfinite(score))
            if not len(ids):
                raise RuntimeError("No finite directed landing candidate")
            j = int(ids[np.lexsort((score[ids], -grade[ids]))[0]])
            protocol.write_json(
                out_dir / "first_top_stage_proof.json",
                survival.selected_top_proof(controller, states, sensors, targets, j, config),
            )
            first = False
        return result

    return cost


def main():
    parser = protocol.build_argument_parser()
    parser.set_defaults(
        source=Path("logs/ollie_recoverable_verified_candidate_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--prefix-view-dir", type=Path, default=Path("logs/ollie_top_landing_input_01")
    )
    parser.add_argument("--actual-prior", type=Path, action="append", default=[])
    args, config = parser.parse_args(), Config()
    if any(
        (
            args.extra_prior,
            args.extra_prior_sha256,
            args.extra_prior_result,
            args.extra_prior_result_sha256,
        )
    ):
        parser.error("Only documented actual target priors are supported")
    original = args.source
    args.source, full, metadata = landing.prepare_source(original, args.prefix_view_dir)
    paths = args.actual_prior + [
        Path("logs/ollie_air_catch_landing_01/actual_rollout.npz"),
        Path("logs/ollie_prelaunch_capture_landing_01/actual_rollout.npz"),
        Path("logs/ollie_survival_priority_landing_01/actual_rollout.npz"),
    ]
    warm, records = [], []
    for i, path in enumerate(paths):
        sha = protocol.digest(path)
        if (
            path == Path("logs/ollie_air_catch_landing_01/actual_rollout.npz")
            and sha != ACTUAL90_SHA
        ):
            raise ValueError("Frozen actual90 changed")
        with np.load(path, allow_pickle=False) as archive:
            value = {k: archive[k].copy() for k in archive.files}
        if value["action"].shape != (257, 29) or bool(value["verified_teacher"]):
            raise ValueError("Expected complete non-teacher physical archive")
        for key in ("qpos", "qvel", "state"):
            if not np.array_equal(value[key][:131], full[key][:131]):
                raise ValueError("Actual prior does not share original continuous prefix")
        if not np.array_equal(value["action"][:130], full["action"][:130]):
            raise ValueError("Actual target prefix differs")
        warm.append(value)
        records.append(
            {
                "name": f"actual_prior_{i}_not_teacher",
                "path": str(path.resolve()),
                "sha256": sha,
                "only_pd_targets": True,
            }
        )
    base_factory = survival.early.prior_factory(full)

    def prepare(controller, source, extra, cfg):
        nominal, _, info = base_factory(controller, source, extra, cfg)
        info["air_phase_observation"] = previous.air_sensor.add_board_ground_sensor(controller, cfg)
        priors = {r["name"]: v["action"][130:].copy() for r, v in zip(records, warm, strict=True)}
        priors["actual82_low_hop_not_teacher"] = nominal.copy()
        info["actual_target_priors"] = records
        return nominal, priors, info

    protocol.run_experiment(
        args,
        config,
        cost_function=make_cost(args.out_dir),
        prior_factory=prepare,
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=landing.previous.highest_grade_elites,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "directed_geometry_script_sha256": protocol.digest(directed.__file__),
            "air_phase_caller_sha256": protocol.digest(previous.__file__),
            "air_sensor_script_sha256": protocol.digest(previous.air_sensor.__file__),
            "reference_survival_script_sha256": protocol.digest(survival.__file__),
            "reference_boundary_script_sha256": protocol.digest(survival.early.__file__),
            "reference_landing_script_sha256": protocol.digest(landing.__file__),
            "top_sensor_script_sha256": protocol.digest(landing.top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "original_full_actual_source": metadata,
            "original_source_sha256": landing.SOURCE_SHA,
            "actual_target_priors": records,
            "task": TASK,
            "config": asdict(config),
            "search_is_teacher_exploration": True,
        },
    )
    if protocol.digest(original) != landing.SOURCE_SHA or any(
        protocol.digest(r["path"]) != r["sha256"] for r in records
    ):
        raise RuntimeError("Frozen inputs changed")


if __name__ == "__main__":
    main()
