"""Enable the existing capture-point hint before launch and reuse actual85.

Original82 nominal/prefix, original physics and full strict v3 are preserved.
Actual85 is an additional evaluated PD-target prior, never a state reset.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import optimize_ollie_survival_priority_landing as survival

landing, early, protocol = survival.landing, survival.early, survival.protocol
SCHEMA = "supermimic.ollie.prelaunch_capture_landing.v1"
WARM_SHA = "fc0a113a3003f2ff669510cbfe3ad4d6f0ff31f05ace9738724707d91167a9d6"
TASK = {
    "prelaunch_capture_window_s": 0.16,
    "capture_weight_unchanged": 600,
    "capture_active": "union of first160ms and original postflight ramp/window; one weighted mean",
    "capture_is_soft_world_horizontal_lip_hint_not_a_certificate": True,
    "top_rewards_still_wait_for_real40ms_flight": True,
    "warm_actual85_is_only_additional_pd_target_prior": True,
    "nominal_and_primary_initialization_still_original82": True,
    "survival_priority_and_whole_geometry_gate_unchanged": True,
}


@dataclass(frozen=True)
class Config(survival.Config):
    seed: int = 86


def corrected_capture(controller, sensors, config, first_end_absolute):
    sl = controller.slices
    times = np.arange(sensors.shape[1]) * config.physics_dt
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
    relative_end = first_end_absolute - config.prefix_controls * config.control_dt
    delay = times[None] - relative_end[:, None]
    post = landing.base.smoothstep(delay / landing.TASK["top_ramp_after_end_s"])
    post *= (delay <= landing.TASK["top_mean_window_after_end_s"]) & (
        first_end_absolute[:, None] >= 0
    )
    pre = times < TASK["prelaunch_capture_window_s"]
    active = np.maximum(post, pre[None])
    total = (
        landing.TASK["weights"]["capture_point"]
        * np.sum(penalty * active, 1)
        / np.maximum(active.sum(1), 1)
    )
    pre_cost = landing.TASK["weights"]["capture_point"] * penalty[:, pre].mean(1)
    return total, pre_cost


def make_cost(out_dir):
    previous_cost = survival.make_cost(out_dir)
    first = True

    def cost(controller, states, sensors, targets, config):
        nonlocal first
        grades, old_priority, _, metrics = previous_cost(
            controller, states, sensors, targets, config
        )
        cp, pre_cp = corrected_capture(
            controller, sensors, config, metrics["first40ms_flight_end_s"]
        )
        raw = (
            metrics["original83_raw_cost"] - metrics["raw_term_horizontal_capture_point_soft"] + cp
        )
        selection, lost, bounded = survival.priority_cost(
            raw, metrics["first_robot_ground_s"], config
        )
        grades = np.where(np.isfinite(selection) & (grades >= 0), grades, -1)
        metrics.update(
            previous85_priority_cost=old_priority,
            corrected86_raw_cost=raw,
            corrected_capture_point_cost=cp,
            prelaunch_capture_point_cost=pre_cp,
        )
        if first:
            ids = np.flatnonzero((grades >= 0) & np.isfinite(selection))
            if not len(ids):
                raise RuntimeError("No finite prelaunch-corrected candidate")
            j = int(ids[np.lexsort((selection[ids], -grades[ids]))[0]])
            protocol.write_json(
                out_dir / "prelaunch_first_top_stage_proof.json",
                survival.selected_top_proof(controller, states, sensors, targets, j, config),
            )
            first = False
        return (
            grades,
            selection,
            {"lost_ground_free_steps": 2 * lost, "bounded86_cost": bounded},
            metrics,
        )

    return cost


def prior_factory(full, warm, warm_path):
    original_factory = early.prior_factory(full)

    def factory(controller, source, extra, config):
        nominal, priors, metadata = original_factory(controller, source, extra, config)
        targets = warm["action"][config.prefix_controls :].copy()
        if not np.array_equal(
            warm["action"][: config.prefix_controls], full["action"][: config.prefix_controls]
        ):
            raise ValueError("Actual85 does not share exact original continuous prefix")
        if targets.shape != nominal.shape or not np.isfinite(targets).all():
            raise ValueError("Invalid actual85 target prior")
        if not np.array_equal(targets, np.clip(targets, *controller.target_limits.T)):
            raise ValueError("Actual85 targets exceed original target limits")
        metadata["warm_prior"] = {
            "path": str(warm_path.resolve()),
            "sha256": WARM_SHA,
            "only_action_read_for_targets": True,
            "root_board_state_applied": False,
            "is_teacher": False,
            "actual_outcome": "FAILED",
        }
        return nominal, {"actual85_hop_and_brief_catch_not_teacher": targets, **priors}, metadata

    return factory


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
        "--warm-source",
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
        parser.error("Legacy right6 prior is not this experiment")
    if protocol.digest(args.warm_source) != WARM_SHA:
        raise ValueError("Actual85 warm target source changed")
    with np.load(args.warm_source, allow_pickle=False) as archive:
        warm = {k: archive[k].copy() for k in archive.files}
    if warm["action"].shape != (257, 29) or bool(warm["verified_teacher"]):
        raise ValueError("Expected complete actual85 exploration, not a teacher")
    original = args.source
    args.source, full, metadata = landing.prepare_source(original, args.prefix_view_dir)
    protocol.run_experiment(
        args,
        config,
        cost_function=make_cost(args.out_dir),
        prior_factory=prior_factory(full, warm, args.warm_source),
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=landing.previous.highest_grade_elites,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "reference_survival_script_sha256": protocol.digest(survival.__file__),
            "reference_boundary_script_sha256": protocol.digest(early.__file__),
            "reference_landing_script_sha256": protocol.digest(landing.__file__),
            "top_sensor_script_sha256": protocol.digest(landing.top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "warm_source": {"path": str(args.warm_source.resolve()), "sha256": WARM_SHA},
            "original_full_actual_source": metadata,
            "original_source_sha256": landing.SOURCE_SHA,
            "original83_task": landing.TASK,
            "survival85_task": survival.TASK,
            "task": TASK,
            "config": asdict(config),
            "search_is_teacher_exploration": True,
        },
    )
    if (
        protocol.digest(original) != landing.SOURCE_SHA
        or protocol.digest(args.warm_source) != WARM_SHA
    ):
        raise RuntimeError("Frozen input changed")


if __name__ == "__main__":
    main()
