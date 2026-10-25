"""Explore simultaneous real deck-top support with the original29 PD targets.

Fixed actual82 prefix, native per-contact observations and whole-episode v3.
This collects physical teacher candidates; it is not a learned policy.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local import ollie_top_contact_sensors as top
from scripts.local import optimize_ollie_hop_preserving_catch as previous

protocol, base, recovery = previous.protocol, previous.protocol.base, previous.recovery
SCHEMA = "supermimic.ollie.simultaneous_top_landing.v1"
SOURCE_SHA = "5cf90dd936724fe920e15578b054724811c0368129fea253dfc6b6514e5c29f3"


@dataclass(frozen=True)
class Config(previous.Config):
    prefix_controls: int = 158
    horizon: int = 99
    dimensions: int = 29
    knot_times_s: tuple = (
        0,
        0.02,
        0.04,
        0.08,
        0.12,
        0.18,
        0.26,
        0.38,
        0.54,
        0.76,
        1.04,
        1.42,
        1.96,
    )
    seed: int = 83
    wall_budget_s: float = 600.0
    selection: str = (
        "original_real_flight_grade_then_simultaneous_top_body_cost; highest_grade_elites_only"
    )


TASK = {
    "top_target_each_n": 40.0,
    "simultaneous_window_samples": 20,
    "catch_search_after_first40ms_end_s": 0.30,
    "top_mean_window_after_end_s": 0.40,
    "top_ramp_after_end_s": 0.04,
    "upright_target": 0.90,
    "root_board_height_target_m": 0.60,
    "capture_point": "world horizontal, board heading; LIP soft heuristic after flight, not a certificate",
    "capture_point_target_heading_xy_m": [-0.055, 0.0],
    "capture_point_allowed_half_xy_m": [0.23, 0.065],
    "weights": {
        "robot_ground": 12000,
        "penetration": 1200,
        "joint_limit": 800,
        "top_mean": 1800,
        "top_same50ms_window": 1800,
        "side_or_underside": 500,
        "body_posture": 1200,
        "capture_point": 600,
        "foot_xy": 400,
        "finite_gap": 200,
        "final_support": 1800,
        "final_stability": 800,
    },
    "all29_targets_may_change_immediately": True,
    "only_top_reward_delayed_until_40ms_flight_completes": True,
    "top_is_canonical_upward_support_not_raw_fn": True,
    "direction_prior": {
        "waist_pitch_offset_rad": -0.45,
        "both_shoulder_pitch_offset_rad": 0.60,
        "rise_s": 0.12,
        "hold_until_s": 0.50,
        "return_s": 1.0,
        "meaning": "source static COM Jacobian direction only; native branch must verify",
    },
}


def prepare_source(path, directory):
    """Exact170-control input view for the unchanged76 replay protocol.

    qpos/qvel/state/action are literal slices of the real82 archive. The lookup
    time grid is ideal20ms as required by76; original accumulated times are
    preserved separately and the contacts retain their original native times.
    No endpoint state is used to initialize a simulation.
    """
    if protocol.digest(path) != SOURCE_SHA:
        raise ValueError("Frozen full actual82 source differs")
    with np.load(path, allow_pickle=False) as archive:
        full = {k: archive[k].copy() for k in archive.files}
    if (
        full["action"].shape != (257, 29)
        or full["qpos"].shape != (258, 47)
        or full["qvel"].shape != (258, 45)
        or full["state"].shape != (258, 92)
        or bool(full["verified_teacher"])
    ):
        raise ValueError("Expected complete actual82, not a teacher/state-only reference")
    contact_path = path.parent / "contacts.json"
    contacts = json.loads(contact_path.read_text())[:1361]
    if len(contacts) != 1361:
        raise ValueError("Expected every actual contact through3.40s")
    metadata = {
        "schema": SCHEMA + ".actual_prefix_view",
        "source": str(path.resolve()),
        "source_sha256": SOURCE_SHA,
        "source_contacts_sha256": protocol.digest(contact_path),
        "state_and_action_are_exact_slices": True,
        "controls": 170,
        "ideal_lookup_grid_only": True,
        "native_boundary_times_preserved": True,
        "simulation_steps": 0,
        "endpoint_initialization_used": False,
    }
    expected = {k: full[k][:171] for k in ("qpos", "qvel", "state")}
    expected.update(
        action=full["action"][:170],
        time_physics=np.arange(171) * 0.02,
        time_native=full["time_physics"][:171],
        robot_joint_names=full["robot_joint_names"],
        board_joint_names=full["board_joint_names"],
        verified_teacher=np.asarray(False),
    )
    destination = directory / "actual_rollout.npz"
    if directory.exists():
        if json.loads((directory / "provenance.json").read_text()) != metadata:
            raise ValueError("Existing prefix view provenance differs")
        with np.load(destination, allow_pickle=False) as archive:
            if any(not np.array_equal(archive[k], v) for k, v in expected.items()):
                raise ValueError("Existing prefix view changed")
        if json.loads((directory / "contacts.json").read_text()) != contacts:
            raise ValueError("Existing prefix contacts changed")
    else:
        directory.mkdir(parents=True)
        np.savez_compressed(destination, **expected)
        protocol.write_json(directory / "contacts.json", contacts)
        protocol.write_json(directory / "provenance.json", metadata)
    return destination, full, metadata


def longest_samples(mask):
    running, longest = np.zeros(mask.shape[0], int), np.zeros(mask.shape[0], int)
    for column in mask.T:
        running = np.where(column, running + 1, 0)
        longest = np.maximum(longest, running)
    return longest


def flight_data(controller, sensors, config, robot_ground):
    sl, m = controller.slices, controller.model
    n, samples = sensors.shape[:2]
    lower = np.stack(
        [
            base.envelope_z(
                m,
                m.geom(name).id,
                sensors[..., sl[name + "_pos"]],
                sensors[..., sl[name + "_quat"]],
            )
            for name in controller.board_geoms
        ],
        -1,
    ).min(-1)
    floor = np.stack(
        [
            np.maximum(sensors[..., sl["ground_" + name]][..., 3], 0)
            for name in controller.board_geoms
        ],
        -1,
    )
    bq = sensors[..., sl["boardquat"]]
    bu = 1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)
    bad = robot_ground | (-sensors[..., sl["depth"]][..., 1] > 0.01) | (bu <= 0)
    prefix_safe = not any(
        r["unique_robot_geoms_ground"]
        or r["maximum_penetration_m"] > 0.01
        or r["board_upright_cosine"] <= 0
        for r in controller.prefix_records
    )
    safe = ~np.maximum.accumulate(bad, axis=1) & prefix_safe
    tail = (
        (sensors[..., sl["tail_strongest"]][..., 1] >= 0.1)
        & (sensors[..., sl["board_com_v"]][..., 2] > 0)
        & safe
    )
    prefix_tail = any(r["tail_ground"] for r in controller.prefix_records)
    before = np.concatenate(
        (np.full((n, 1), prefix_tail, bool), np.maximum.accumulate(tail, 1)[:, :-1] | prefix_tail),
        1,
    )
    flight = (lower > 0.005) & (floor.max(-1) < 0.1)
    qualified = (
        base.min_windows(flight.astype(np.int8), 16).astype(bool)
        & before[:, : samples - 15]
        & safe[:, 15:]
        & (np.maximum.accumulate(lower, 1)[:, 15:] >= 0.015)
    )
    grade = np.where(qualified.any(1), 2, np.where(tail.any(1) | prefix_tail, 1, 0))
    return grade, qualified, safe, lower, floor, bu


def landing_cost(controller, states, sensors, targets, config):
    sl, m, ix = controller.slices, controller.model, controller.index
    n, samples = sensors.shape[:2]
    times = np.arange(samples) * config.physics_dt
    observed = top.top_support(controller, sensors)
    ground = observed["robot_ground_above_threshold"]
    grade, qualified, safe, lower, floor, bu = flight_data(controller, sensors, config, ground)
    left, right = observed["left_top_n"], observed["right_top_n"]
    has_air = qualified.any(1)
    first_end = (np.argmax(qualified, 1) + 16) * config.physics_dt
    delay = times[None] - first_end[:, None]
    active = base.smoothstep(delay / TASK["top_ramp_after_end_s"])
    active *= (delay <= TASK["top_mean_window_after_end_s"]) & has_air[:, None]
    denominator = np.maximum(active.sum(1), 1)

    def weighted(value):
        return np.sum(value * active, 1) / denominator

    bp, bq = sensors[..., sl["boardpos"]], sensors[..., sl["boardquat"]]
    rp, rq = sensors[..., sl["rootpos"]], sensors[..., sl["rootquat"]]
    ru = 1 - 2 * (rq[..., 1] ** 2 + rq[..., 2] ** 2)
    height = rp[..., 2] - bp[..., 2]
    top_deficit = (
        1
        - np.minimum(
            np.clip(left / TASK["top_target_each_n"], 0, 1),
            np.clip(right / TASK["top_target_each_n"], 0, 1),
        )
    ) ** 2
    pose_deficit = (np.maximum(0.90 - ru, 0) / 0.20) ** 2
    pose_deficit += (np.maximum(0.60 - height, 0) / 0.20) ** 2
    # Same20-sample window, all samples, both feet and posture synchronized.
    window = base.max_windows(top_deficit + pose_deficit, 20)
    eligible = (
        (times[None, : samples - 19] >= first_end[:, None])
        & (times[None, : samples - 19] + 0.05 <= first_end[:, None] + 0.30)
        & safe[:, 19:]
        & has_air[:, None]
    )
    simultaneous = np.min(np.where(eligible, window, np.inf), 1)
    simultaneous = np.where(np.isfinite(simultaneous), simultaneous, 4)
    previous_state = np.concatenate(
        (np.broadcast_to(controller.state(), (n, 1, states.shape[-1])), states[:, :-1]), 1
    )
    q = previous_state[..., 1 + ix.robot_joint_qpos]
    finalq = states[:, -1, 1 + ix.robot_joint_qpos]
    limit = np.maximum(np.maximum(controller.limits[:, 0] - q, q - controller.limits[:, 1]), 0).max(
        (1, 2)
    )
    limit = np.maximum(
        limit,
        np.maximum(
            np.maximum(controller.limits[:, 0] - finalq, finalq - controller.limits[:, 1]), 0
        ).max(1),
    )
    depth = np.maximum(-sensors[..., sl["depth"]][..., 1], 0)
    weights = TASK["weights"]
    terms = {
        "no_robot_ground": weights["robot_ground"] * ground.mean(1),
        "penetration": weights["penetration"] * (depth.max(1) / 0.01) ** 2,
        "actual_joint_limits": weights["joint_limit"] * (limit / 0.05) ** 2,
        "both_top_mean_after_air": weights["top_mean"] * weighted(top_deficit),
        "same50ms_both_top_and_body": weights["top_same50ms_window"] * simultaneous,
        "side_or_under_after_air": weights["side_or_underside"]
        * weighted(
            (
                np.minimum(
                    (observed["left_side_or_under_n"] + observed["right_side_or_under_n"]) / 80, 5
                )
            )
            ** 2
        ),
        "body_height_upright": weights["body_posture"] * np.mean(pose_deficit[:, times >= 0.10], 1),
        "board_not_inverted": 1000 * np.mean(np.maximum(-bu, 0) ** 2, 1),
    }
    rc, bc = sensors[..., sl["root_com"]], sensors[..., sl["board_com"]]
    rv, bv = sensors[..., sl["root_com_v"]], sensors[..., sl["board_com_v"]]
    # Project onto the WORLD horizontal plane, then board heading. Rotating the
    # full3D COM into a pitched deck would mistake height for lateral offset.
    heading = base.rotate(bq, np.array([1.0, 0, 0]))[..., :2]
    heading /= np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-8)
    lateral = np.stack((-heading[..., 1], heading[..., 0]), -1)
    omega = np.sqrt(9.81 / np.maximum(rc[..., 2] - bp[..., 2], 0.2))
    cp = rc[..., :2] - bc[..., :2] + (rv[..., :2] - bv[..., :2]) / omega[..., None]
    cp_heading = np.stack((np.sum(cp * heading, -1), np.sum(cp * lateral, -1)), -1)
    cp_error = cp_heading - TASK["capture_point_target_heading_xy_m"]
    cp_excess = np.maximum(abs(cp_error) - TASK["capture_point_allowed_half_xy_m"], 0)
    terms["horizontal_capture_point_soft"] = weights["capture_point"] * weighted(
        np.sum((cp_excess / [0.20, 0.06]) ** 2, -1)
    )
    stride = 8
    selected = sensors[:, : min(samples, 241) : stride]
    planar, gaps = 0, 0
    for side in ("left", "right"):
        low, high = recovery.capsule_bounds(controller, selected, side)
        center = (low + high) / 2
        planar += np.mean(
            np.sum(
                ((center[..., :2] - controller.catch_initial_centroid[side]) / [0.20, 0.06]) ** 2,
                -1,
            ),
            1,
        )
        gap = np.min(
            np.stack(
                [
                    recovery.finite_capsule_witness_gap(controller, selected, side, "board_" + box)
                    for box in ("center", "tail", "nose")
                ],
                -1,
            ),
            -1,
        )
        # Only late geometric proximity, still not a contact certificate.
        ramp = base.smoothstep(
            (times[: min(samples, 241) : stride][None] - first_end[:, None]) / 0.08
        )
        ramp *= has_air[:, None]
        gaps += np.sum((gap / 0.04) ** 2 * ramp, 1) / np.maximum(ramp.sum(1), 1)
    terms["foot_xy_recovery"] = weights["foot_xy"] * planar
    terms["finite_gap_after_air"] = weights["finite_gap"] * gaps
    final = times >= config.horizon * config.control_dt - 0.30
    terms["final_true_top_support"] = weights["final_support"] * np.mean(top_deficit[:, final], 1)
    wheel_ids = [i for i, name in enumerate(controller.board_geoms) if name.startswith("wheel_")]
    terms["final_four_wheels"] = 500 * np.mean(
        (1 - np.clip(floor[:, final][:, :, wheel_ids], 0, 1)) ** 2, (1, 2)
    )
    velocity = previous_state[..., 1 + m.nq :]
    robot_v = np.linalg.norm(velocity[..., ix.robot_qvel : ix.robot_qvel + 3], axis=-1)
    robot_w = np.linalg.norm(velocity[..., ix.robot_qvel + 3 : ix.robot_qvel + 6], axis=-1)
    board_v = np.linalg.norm(velocity[..., ix.board_qvel : ix.board_qvel + 3], axis=-1)
    board_w = np.linalg.norm(velocity[..., ix.board_qvel + 3 : ix.board_qvel + 6], axis=-1)
    terminal = pose_deficit + (np.maximum(robot_v - 0.3, 0) / 0.3) ** 2
    terminal += (np.maximum(robot_w - 0.8, 0) / 0.8) ** 2
    terminal += (np.maximum(board_v - 0.35, 0) / 0.35) ** 2 + (
        np.maximum(board_w - 0.8, 0) / 0.8
    ) ** 2
    terminal += (np.maximum(0.95 - bu, 0) / 0.2) ** 2
    terms["final_body_stability"] = weights["final_stability"] * terminal[:, final].mean(1)
    delta = np.diff(
        np.concatenate(
            (np.broadcast_to(controller.data.ctrl[ix.robot_actuators], (n, 1, 29)), targets), 1
        ),
        axis=1,
    )
    terms["target_change"] = 0.1 * np.mean((delta / 0.25) ** 2, (1, 2))
    cost = sum(terms.values())
    valid = (
        np.isfinite(states).all((1, 2))
        & np.isfinite(sensors).all((1, 2))
        & observed["valid"].all(1)
        & np.isfinite(cost)
        & np.isclose(states[:, -1, 0], 5.14, rtol=0, atol=1e-9)
    )
    joint_support = (
        (left > 1) & (right > 1) & safe & has_air[:, None] & (delay >= 0) & (delay <= 0.4)
    )
    stable_joint = joint_support & (ru >= 0.9) & (height >= 0.6)
    metrics = {
        "raw_real_flight_grade": grade,
        "maximum_whole_board_clearance_m": lower.max(1),
        "first40ms_flight_end_s": np.where(has_air, config.prefix_controls * 0.02 + first_end, -1),
        "longest_both_true_top_after_air_s": longest_samples(joint_support) * config.physics_dt,
        "longest_both_true_top_and_posture_after_air_s": longest_samples(stable_joint)
        * config.physics_dt,
        "first_robot_ground_s": np.where(
            ground.any(1),
            config.prefix_controls * 0.02 + np.argmax(ground, 1) * config.physics_dt,
            -1,
        ),
        "final_left_top_fraction": (left[:, final] > 1).mean(1),
        "final_right_top_fraction": (right[:, final] > 1).mean(1),
        "minimum_robot_upright": ru.min(1),
        "maximum_penetration_m": depth.max(1),
        "maximum_actual_joint_limit_violation_rad": limit,
        "sensor_overflow": observed["overflow"].any(1).astype(int),
        "qualified_flight_window_count": qualified.sum(1),
    }
    return np.where(valid, grade, -1), np.where(valid, cost, np.inf), terms, metrics


def prior_factory(full):
    def factory(controller, source, extra, config):
        extended_proof = top.add_top_sensors(controller, config)
        nominal = full["action"][config.prefix_controls :].copy()
        if nominal.shape != (config.horizon, 29):
            raise ValueError("Expected exact99 actual82 controls")
        scratch = mujoco.MjData(controller.branch_model)
        scratch.qpos[:] = full["qpos"][0]
        mujoco.mj_forward(controller.branch_model, scratch)
        controller.catch_initial_centroid = {
            side: ((low + high) / 2)[0, 0, :2]
            for side in ("left", "right")
            for low, high in [
                recovery.capsule_bounds(controller, scratch.sensordata[None, None], side)
            ]
        }
        kp = controller.model.actuator_gainprm[controller.index.robot_actuators, 0]
        stand = controller.reference.robot_state[0, 7:36] + controller.ff_tau[0] / kp
        stand = np.clip(stand, *controller.target_limits.T)
        grid = np.arange(config.horizon) * 0.02
        blend = base.smoothstep((grid - 0.12) / (0.60 - 0.12))[:, None]
        stand_prior = nominal * (1 - blend) + stand * blend
        direction_prior = nominal.copy()
        setting = TASK["direction_prior"]
        ramp = base.smoothstep(grid / setting["rise_s"])
        ramp *= 1 - base.smoothstep(
            (grid - setting["hold_until_s"]) / (setting["return_s"] - setting["hold_until_s"])
        )
        direction_prior[:, 14] += ramp * setting["waist_pitch_offset_rad"]
        direction_prior[:, [15, 22]] += ramp[:, None] * setting["both_shoulder_pitch_offset_rad"]
        direction_prior = np.clip(direction_prior, *controller.target_limits.T)
        return (
            nominal,
            {
                "exact82_hop_nominal_not_teacher": nominal.copy(),
                "original_standing_targets_after_air_not_teacher": stand_prior,
                "waist_shoulders_com_direction_not_teacher": direction_prior,
            },
            {
                "nominal_exact_source82_99targets": True,
                "root_board_state_applied": False,
                "all29_robot_targets_optimized": True,
                "extended_observation_model": extended_proof,
                "top_observation_schema": controller.top_sensor_metadata,
                "is_teacher": False,
            },
        )

    return factory


def make_cost(out_dir):
    first = True
    round_index = 0
    best_synchronized_cost = np.inf

    def cost(controller, states, sensors, targets, config):
        nonlocal first, round_index, best_synchronized_cost
        result = landing_cost(controller, states, sensors, targets, config)
        grades, costs, terms, metrics = result
        synchronized = np.flatnonzero(
            (grades == 2)
            & (metrics["longest_both_true_top_and_posture_after_air_s"] >= 0.05 - 1e-10)
            & np.isfinite(costs)
        )
        if len(synchronized):
            j = int(synchronized[np.argmin(costs[synchronized])])
            if costs[j] < best_synchronized_cost:
                best_synchronized_cost = float(costs[j])
                np.savez_compressed(
                    out_dir / "best_synchronized_catch_evaluated_branch.npz",
                    action=targets[j],
                    fullphysics=states[j],
                    sensors=sensors[j],
                )
                protocol.write_json(
                    out_dir / "best_synchronized_catch.json",
                    {
                        "round": round_index,
                        "candidate": j,
                        "cost": best_synchronized_cost,
                        "metrics": {k: float(v[j]) for k, v in metrics.items()},
                        "terms": {k: float(v[j]) for k, v in terms.items()},
                        "actual_execution_performed": False,
                        "is_teacher": False,
                        "reason": "save same50ms real top+body candidate; does not change selection or certify v3",
                    },
                )
        if first:
            grades, costs, _, _ = result
            ids = np.flatnonzero((grades >= 0) & np.isfinite(costs))
            if not len(ids):
                raise RuntimeError("No finite observed native candidate")
            j = ids[np.lexsort((costs[ids], -grades[ids]))[0]]
            scratch = copy.copy(controller.data)
            errors = {"left": 0.0, "right": 0.0}
            sensor_error = 0.0
            ground_mismatches = 0
            for frame, target in enumerate(targets[j]):
                scratch.ctrl[controller.index.robot_actuators] = target
                for step in range(8):
                    mujoco.mj_step(controller.model, scratch)
                    row = controller.observer.observe(scratch)
                    cursor = frame * 8 + step
                    measured = top.top_support(controller, sensors[j, cursor])
                    ground_mismatches += bool(measured["robot_ground_above_threshold"]) != bool(
                        row["unique_robot_geoms_ground"]
                    )
                    for side in ("left", "right"):
                        errors[side] = max(
                            errors[side],
                            abs(
                                float(measured[side + "_top_n"])
                                - row[side + "_foot_deck_upward_support_n"]
                            ),
                        )
                    sensor_error = max(
                        sensor_error,
                        float(abs(controller.state(scratch) - states[j, cursor]).max()),
                    )
                    mujoco.mj_forward(controller.model, scratch)
            if max(errors.values()) > 1e-7 or sensor_error > 1e-10 or ground_mismatches:
                raise RuntimeError(
                    "New native top sensor differs from same-stage canonical contacts"
                )
            protocol.write_json(
                out_dir / "first_top_stage_proof.json",
                {
                    "candidate": int(j),
                    "substeps": config.horizon * 8,
                    "top_support_max_errors_n": errors,
                    "fullphysics_max_error": sensor_error,
                    "canonical_robot_ground_mismatches": ground_mismatches,
                    "stage": "after mj_step, execution solve and interval-start geometry; before extra forward",
                    "main_unchanged": True,
                },
            )
            first = False
        round_index += 1
        return result

    return cost


def main():
    parser = protocol.build_argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        source=Path("logs/ollie_recoverable_verified_candidate_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--prefix-view-dir", type=Path, default=Path("logs/ollie_top_landing_input_01")
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
        parser.error("Legacy right6 prior is not this experiment")
    original = args.source
    args.source, full, metadata = prepare_source(original, args.prefix_view_dir)
    protocol.run_experiment(
        args,
        config,
        cost_function=make_cost(args.out_dir),
        prior_factory=prior_factory(full),
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=previous.highest_grade_elites,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "top_sensor_script_sha256": protocol.digest(top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "original_full_actual_source": metadata,
            "task": TASK,
            "config": asdict(config),
            "original_source_sha256": SOURCE_SHA,
            "search_is_teacher_exploration": True,
        },
    )
    if protocol.digest(original) != SOURCE_SHA:
        raise RuntimeError("Frozen actual82 source changed")


if __name__ == "__main__":
    main()
