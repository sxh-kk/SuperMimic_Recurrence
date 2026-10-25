"""Bounded native-physics recoverable unload/catch exploration, not a policy.

Only the original 29 PD targets are optimized. The passive board, initial
state, force limits and strict whole-episode success evaluator stay unchanged.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import optimize_ollie_momentum_release as protocol
from scripts.local.continue_ollie_sampling_mpc import finite_capsule_witness_gap, multiply
from supermimic.data.motion_convert import G1_JOINT_NAMES

base = protocol.base
SCHEMA = "supermimic.ollie.recoverable_catch.v1"
PRIOR_SHA = "492c034d0f7806efec403fa65941dc0c94fccbbe26893d34046775a2c0b4a5f1"
PRIOR_RESULT_SHA = "d2c05a6d235552a8333da779351b52af0544ac4f79bb50b0c8f90a08e15972a2"
PREVIOUS_SHA = "c8754cad0ec9f77dc816790c9919d918f2c1e1fd9dfb5460b13ad2581ba0f0a2"


@dataclass(frozen=True)
class Config(protocol.Config):
    prefix_controls: int = 150
    horizon: int = 107
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
        2.12,
    )
    seed: int = 80
    selection: str = "recoverable_flight_grade_then_fixed_cost_all_64_rounds"


TASK = {
    "phase_s": {
        "left_begin": 0.02,
        "A": 0.08,
        "rear_release": 0.12,
        "C": 0.22,
        "descent_begin": 0.28,
        "B": 0.40,
        "standing": 0.90,
    },
    "capsule_corridor_xy_m": [0.45, 0.15],
    "corridor_soft_excess_scale_m": 0.05,
    "corridor_cost_weight": 800.0,
    "centroid_xy_scale_m": [0.18, 0.08],
    "centroid_cost_weight": 200.0,
    "early_geometry_window_s": 0.60,
    "centroid_ramp_s": [0.0, 0.08],
    "catch_weight": 250.0,
    "catch_gap_scale_m": 0.04,
    "catch_ramp_after_first_qualified_window_end_s": 0.08,
    "catch_window_after_first_qualified_window_end_s": 0.60,
    "geometry_sample_stride": 8,
    "selection": "raw grade2 retained only if one qualified16-sample flight window also has both whole-foot capsule AABBs inside corridor and projected deck overlap at every sample",
    "geometry_is_support_certificate": False,
    "no_qualified_flight_catch_cost": 0.0,
}


def capsule_bounds(controller, sensors, side):
    """Exact AABB of all seven actual capsules in the authored board frame."""
    sl, foot = controller.slices, controller.foot_geometry[side]
    fp, fq = sensors[..., sl[side + "_body_pos"]], sensors[..., sl[side + "_body_quat"]]
    bp, bq = sensors[..., sl["boardpos"]], sensors[..., sl["boardquat"]]
    # Compose the two frames once, instead of rotating each of 14 capsule
    # centers/axes into world and back. This is the same rigid transformation.
    relative = multiply(bq * [1, -1, -1, -1], fq)
    origin = base.inverse_rotate(bq, fp - bp)
    center = origin[..., None, :] + base.rotate(relative[..., None, :], foot["center"])
    axis = base.rotate(relative[..., None, :], foot["axis"])
    extent = foot["size"][:, :1] + foot["size"][:, 1:2] * np.abs(axis)
    return (center - extent).min(-2), (center + extent).max(-2)


def qualified_windows(controller, sensors, config):
    """Reproduce 76's conservative raw flight grade, without new certification."""
    sl, m = controller.slices, controller.model
    count, samples = sensors.shape[:2]
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
    ground = np.stack(
        [
            np.maximum(sensors[..., sl["ground_" + name]][..., 3], 0)
            for name in controller.board_geoms
        ],
        -1,
    ).max(-1)
    bq = sensors[..., sl["boardquat"]]
    up = 1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)
    bad = (
        (sensors[..., sl["robot_ground"]][..., 0] > 0)
        | (np.maximum(-sensors[..., sl["depth"]][..., 1], 0) > 0.01)
        | (up <= 0)
    )
    safe = ~np.maximum.accumulate(bad, axis=1)
    safe &= not any(
        r["unique_robot_geoms_ground"]
        or r["maximum_penetration_m"] > 0.01
        or r["board_upright_cosine"] <= 0
        for r in controller.prefix_records
    )
    tail = (
        (sensors[..., sl["tail_strongest"]][..., 1] >= 0.1)
        & (sensors[..., sl["board_com_v"]][..., 2] > 0)
        & safe
    )
    before = np.concatenate((np.zeros((count, 1), bool), np.maximum.accumulate(tail, 1)[:, :-1]), 1)
    return (
        base.min_windows(((lower > 0.005) & (ground < 0.1)).astype(np.int8), 16).astype(bool)
        & before[:, : samples - 15]
        & safe[:, 15:]
        & (np.maximum.accumulate(lower, 1)[:, 15:] >= 0.015)
    )


def corridor_samples(bounds, controller):
    """Necessary projected reachability only; neither contact nor top support."""
    limit = np.asarray(TASK["capsule_corridor_xy_m"])
    # Exact authored deck-box projected X envelope (the board has no flex).
    deck_x = max(
        abs(controller.model.geom_pos[controller.model.geom("board_" + box).id, 0])
        + np.sum(
            abs(
                base.rotate(
                    controller.model.geom_quat[controller.model.geom("board_" + box).id], np.eye(3)
                )
            )[:, 0]
            * controller.model.geom_size[controller.model.geom("board_" + box).id]
        )
        for box in ("center", "tail", "nose")
    )
    deck_y = max(
        controller.model.geom_size[controller.model.geom("board_" + box).id, 1]
        for box in ("center", "tail", "nose")
    )
    valid = None
    for low, high in bounds.values():
        inside = (
            (low[..., :2] >= -limit).all(-1)
            & (high[..., :2] <= limit).all(-1)
            & (low[..., 0] < deck_x)
            & (high[..., 0] > -deck_x)
            & (low[..., 1] < deck_y)
            & (high[..., 1] > -deck_y)
        )
        valid = inside if valid is None else valid & inside
    return valid


def recoverable_cost(controller, states, sensors, targets, config):
    raw_grade, cost, terms, metrics = protocol.momentum_cost(
        controller, states, sensors, targets, config
    )
    count, samples = sensors.shape[:2]
    times = np.arange(samples) * config.physics_dt
    qualified = qualified_windows(controller, sensors, config)
    # Keep the unmodified raw event grade separately; selection cannot promote
    # a ground strike or a brief high board bounce into full episode success.
    valid = raw_grade >= 0
    if not np.array_equal(qualified.any(1)[valid], (raw_grade == 2)[valid]):
        raise RuntimeError("Raw76 flight grade reproduction differs")
    qualified &= valid[:, None]
    early = times <= TASK["early_geometry_window_s"]
    # Foot geometry only influences the early soft task or a qualified flight
    # window. Include every one of the 16 samples of every such window, even
    # when a flight happens later than the early task. All other poses have
    # no effect on any returned term/metric/selection grade.
    needed = np.broadcast_to(early, (count, samples)).copy()
    for offset in range(16):
        needed[:, offset : offset + qualified.shape[1]] |= qualified
    bounds = {}
    relevant = sensors[needed]
    for side in ("left", "right"):
        low, high = np.zeros((count, samples, 3)), np.zeros((count, samples, 3))
        low[needed], high[needed] = capsule_bounds(controller, relevant, side)
        bounds[side] = low, high
    corridor = corridor_samples(bounds, controller) & needed
    recoverable = qualified & base.min_windows(corridor.astype(np.int8), 16).astype(bool)
    grade = np.where((raw_grade == 2) & ~recoverable.any(1), 1, raw_grade)
    limit = np.asarray(TASK["capsule_corridor_xy_m"])
    excess = sum(
        np.sum(
            (np.maximum(-limit - low[:, early, :2], 0) + np.maximum(high[:, early, :2] - limit, 0))
            ** 2,
            -1,
        )
        for low, high in bounds.values()
    )
    terms["recoverable_capsule_corridor"] = TASK["corridor_cost_weight"] * np.mean(
        excess / TASK["corridor_soft_excess_scale_m"] ** 2, 1
    )
    ramp = base.smoothstep(times[early] / TASK["centroid_ramp_s"][1])
    planar = sum(
        np.sum(
            (
                (
                    (low[:, early, :2] + high[:, early, :2]) / 2
                    - controller.catch_initial_centroid[side]
                )
                / TASK["centroid_xy_scale_m"]
            )
            ** 2,
            -1,
        )
        for side, (low, high) in bounds.items()
    )
    terms["recoverable_centroid_xy"] = TASK["centroid_cost_weight"] * np.mean(planar * ramp, 1)
    # Remove the old selected-soft-anchor catch proxy, which may push onto the
    # board before a full 40ms flight. Replace it only after a real raw window.
    terms.pop("catch_support_proxy")
    first = np.argmax(qualified, 1)
    has_air = qualified.any(1)
    # Sensors are preintegration poses. The 16th sample spans one more dt;
    # begin strictly after the end of its physical step.
    end = (first + 16) * config.physics_dt
    stride = TASK["geometry_sample_stride"]
    selected = sensors[:, ::stride]
    gt = times[::stride]
    activation = base.smoothstep(
        (gt[None] - end[:, None]) / TASK["catch_ramp_after_first_qualified_window_end_s"]
    )
    activation *= (
        gt[None] <= end[:, None] + TASK["catch_window_after_first_qualified_window_end_s"]
    ) & has_air[:, None]
    # Exactly zero-weight samples contribute zero. Evaluate only the sparse
    # active poses, including no geometry work for candidates with no flight.
    active = activation > 0
    gap_square = np.zeros_like(activation)
    if active.any():
        relevant = selected[active]
        gap_square[active] = sum(
            np.min(
                np.stack(
                    [
                        finite_capsule_witness_gap(controller, relevant, side, "board_" + box)
                        for box in ("center", "tail", "nose")
                    ],
                    -1,
                ),
                -1,
            )
            ** 2
            for side in ("left", "right")
        )
    terms["finite_catch_after_qualified_flight"] = (
        TASK["catch_weight"]
        * np.sum(gap_square / TASK["catch_gap_scale_m"] ** 2 * activation, 1)
        / np.maximum(activation.sum(1), 1)
    )
    metrics.update(
        raw76_event_grade=raw_grade,
        recoverable_qualified_flight=recoverable.any(1).astype(np.int8),
        early_both_foot_corridor_fraction=corridor[:, early].mean(1),
        first_qualified_window_end_s=np.where(
            has_air, config.prefix_controls * config.control_dt + end, -1
        ),
        finite_catch_active_samples=(activation > 0).sum(1),
    )
    cost = sum(terms.values())
    valid &= np.isfinite(cost)
    return np.where(valid, grade, -1), np.where(valid, cost, np.inf), terms, metrics


def load_priors(args, config):
    if any(
        (
            args.extra_prior,
            args.extra_prior_sha256,
            args.extra_prior_result,
            args.extra_prior_result_sha256,
        )
    ):
        raise ValueError("Use the recoverable q12 protocol, not legacy right6 extra-prior")
    paths = {
        "recoverable_prior": (args.recoverable_prior, PRIOR_SHA),
        "recoverable_prior_result": (args.recoverable_prior_result, PRIOR_RESULT_SHA),
        "previous76": (args.previous76, PREVIOUS_SHA),
    }
    for path, expected in paths.values():
        if protocol.digest(path) != expected:
            raise ValueError(f"Frozen prior hash differs: {path}")
    result = json.loads(args.recoverable_prior_result.read_text())
    if not (
        result["schema"] == "supermimic.ollie.recoverable_foot_static_prior.v1"
        and result["source_sha256"] == protocol.SOURCE_SHA
        and result["source_frame"] == config.prefix_controls
        and result["board_xml_sha256"] == protocol.digest(protocol.BOARD_XML)
        and result["integration_steps"] == 0
        and not result["is_teacher"]
        and tuple(result["optimized_joint_names"]) == G1_JOINT_NAMES[:12]
        and all(
            n["static_geometry_pass"] and n["gates"]["all_other_qpos_exact"]
            for n in result["nodes"]
        )
    ):
        raise ValueError("Static recoverable prior provenance/geometry failed")
    with np.load(args.recoverable_prior, allow_pickle=False) as data:
        q12 = data["q12"].copy()
        if (
            q12.shape != (3, 12)
            or not np.isfinite(q12).all()
            or not np.array_equal(q12, [n["q12_exact"] for n in result["nodes"]])
            or tuple(data["joint_names12"]) != G1_JOINT_NAMES[:12]
        ):
            raise ValueError("Static q12 mismatch")
    with np.load(args.previous76, allow_pickle=False) as data:
        previous = data["action"].copy()
        previous_initial = data["qpos"][0].copy()
        if (
            previous.shape != (257, 29)
            or tuple(data["robot_joint_names"]) != G1_JOINT_NAMES
            or bool(data["verified_teacher"])
        ):
            raise ValueError("Previous actual archive contract differs")
    return (
        q12,
        previous,
        previous_initial,
        {
            name: {"path": str(path.resolve()), "sha256": expected}
            for name, (path, expected) in paths.items()
        },
    )


def make_prior_factory(q12, previous, previous_initial):
    def factory(controller, source, extra, config):
        if not np.array_equal(source["action"][:150], previous[:150]) or not np.array_equal(
            source["qpos"][0], previous_initial
        ):
            raise ValueError("Previous76 uses a different original initial state/prefix")
        nominal, legacy, old_metadata = protocol.prepare_priors(controller, source, None, config)
        standing = np.asarray(old_metadata["standing_target"])
        prior = next(iter(legacy.values())).copy()
        grid = np.arange(config.horizon) * config.control_dt
        phase = TASK["phase_s"]
        # Only active leg targets are replaced. Rear drive is exactly preserved
        # through +.12s, including its original tracking error/PD target.
        blend = base.smoothstep((grid - phase["left_begin"]) / (phase["A"] - phase["left_begin"]))[
            :, None
        ]
        prior[:, :6] = prior[:, :6] * (1 - blend) + q12[0, :6] * blend
        blend = base.smoothstep(
            (grid - phase["rear_release"]) / (phase["C"] - phase["rear_release"])
        )[:, None]
        prior[:, :12] = prior[:, :12] * (1 - blend) + q12[2] * blend
        blend = base.smoothstep(
            (grid - phase["descent_begin"]) / (phase["B"] - phase["descent_begin"])
        )[:, None]
        prior[:, :12] = prior[:, :12] * (1 - blend) + q12[1] * blend
        blend = base.smoothstep((grid - phase["B"]) / (phase["standing"] - phase["B"]))[:, None]
        prior[:, :12] = prior[:, :12] * (1 - blend) + standing[:12] * blend
        prior = np.clip(prior, *controller.target_limits.T)
        if not np.array_equal(prior[grid <= 0.12, 6:12], source["action"][150:157, 6:12]):
            raise RuntimeError("Prior changed the preserved rear drive")
        # Initial authored capsule centroids, not foot sites or inertial frames.
        import mujoco

        scratch = mujoco.MjData(controller.branch_model)
        scratch.qpos[:] = source["qpos"][0]
        mujoco.mj_forward(controller.branch_model, scratch)
        sensors0 = scratch.sensordata[None, None]
        controller.catch_initial_centroid = {
            side: ((low + high) / 2)[0, 0, :2]
            for side in ("left", "right")
            for low, high in [capsule_bounds(controller, sensors0, side)]
        }
        return (
            nominal,
            {
                "recoverable_ABC_static_legs_not_teacher": prior,
                "previous76_actual_low_hop_missed_catch": previous[150:].copy(),
                **legacy,
            },
            {
                **old_metadata,
                "q12_nodes": q12.tolist(),
                "phase": phase,
                "mapping": "absolute active leg IK targets; other targets from legacy; no root/board state applied",
                "rear_first7_controls_exact": True,
                "initial_capsule_centroid_boardlocal_xy_m": {
                    side: value.tolist()
                    for side, value in controller.catch_initial_centroid.items()
                },
            },
        )

    return factory


def main():
    parser = protocol.build_argument_parser()
    parser.description = __doc__
    parser.add_argument(
        "--recoverable-prior",
        type=Path,
        default=Path("logs/ollie_recoverable_foot_prior_02/recoverable_foot_priors.npz"),
    )
    parser.add_argument(
        "--recoverable-prior-result",
        type=Path,
        default=Path("logs/ollie_recoverable_foot_prior_02/result.json"),
    )
    parser.add_argument(
        "--previous76", type=Path, default=Path("logs/ollie_momentum_release_01/actual_rollout.npz")
    )
    args, config = parser.parse_args(), Config()
    q12, previous, initial, paths = load_priors(args, config)
    protocol.run_experiment(
        args,
        config,
        cost_function=recoverable_cost,
        prior_factory=make_prior_factory(q12, previous, initial),
        schema=SCHEMA,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "frozen_recoverable_sources": paths,
            "task": TASK,
            "config": asdict(config),
        },
    )
    for item in paths.values():
        if protocol.digest(item["path"]) != item["sha256"]:
            raise RuntimeError("Recoverable experiment input changed")


if __name__ == "__main__":
    main()
