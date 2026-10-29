"""Keep verified front-foot recovery when updating native CEM, then stand.

The only changes from95 are an exact actual95 nominal, PD standing/hold seeds,
and event-plus-progress elite selection. Physics,95 loss and strict v3 stay.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import ollie_progress_elites as progress
from scripts.local import optimize_ollie_sole_recovery as recovery

protocol, landing, survival = recovery.protocol, recovery.landing, recovery.survival
SCHEMA = "supermimic.ollie.progress_preserving_landing.v1"
ACTUAL95_SHA = "f1356bb5765c03cef5a0d25364a87c7a572b5f4aa66527f6bb52f528b673db65"
AUDIT95_SHA = "6c11bc6327b7ca2d1d4f44a27bb8c32491971b97a6014a72e34582489cc187e7"
TASK = {
    "loss": "exact frozen95 sole_recovery_cost, no changed acceptance",
    "nominal": "exact complete actual95 PD suffix; zero residual reproduces verified15ms early double top",
    "elites": "highest physical event grade, then lowest explicit same-batch progress tier,1..16 only",
    "standing_seeds": "actual95 targets with initial standing29PD blend start3.38 and duration.16/.28/.40s",
    "hold_seed": "hold actual95 target at3.38s, then standing blend3.60..4.10; targets only",
    "is_not_a_single_factor_ablation": True,
    "all_original_physics_and_strict_v3_unchanged": True,
}


@dataclass(frozen=True)
class Config(recovery.Config):
    seed: int = 97
    progress_elite_schema: str = progress.SCHEMA
    progress_tier_stride: float = 4.0
    progress_tier_count: int = 4
    progress_tier_score_upper_bound: float = 3.0
    selection: str = "original_event_and95_progress_tier_preserving_elites_then95_cost"


def standing_targets(nominal, stand, limits):
    if nominal.shape != (127, 29) or stand.shape != (29,):
        raise ValueError("Expected original127-by29 target arrays")
    time = 2.60 + (np.arange(127) + 0.5) * 0.02
    priors = {"exact_actual95_front_recovery_not_teacher": nominal.copy()}
    for duration in (0.16, 0.28, 0.40):
        blend = landing.base.smoothstep((time - 3.38) / duration)[:, None]
        value = nominal * (1 - blend) + stand * blend
        priors[f"actual95_stand_in_{round(duration * 1000)}ms_not_teacher"] = np.clip(
            value, *limits.T
        )
    held = nominal.copy()
    frame = round((3.38 - 2.60) / 0.02)
    held[frame:] = nominal[frame]
    blend = landing.base.smoothstep((time - 3.60) / 0.50)[:, None]
    priors["actual95_hold_catch_then_stand_not_teacher"] = np.clip(
        held * (1 - blend) + stand * blend, *limits.T
    )
    return priors


def main():
    parser = protocol.build_argument_parser()
    parser.set_defaults(
        source=Path("logs/ollie_recoverable_verified_candidate_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--prefix-view-dir", type=Path, default=Path("logs/ollie_top_landing_input_01")
    )
    parser.add_argument(
        "--actual95", type=Path, default=Path("logs/ollie_sole_recovery_01/actual_rollout.npz")
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
        parser.error("Only the frozen physically evaluated actual95 source is used")
    path, audit = args.actual95, args.actual95.parent / "independent_air_stage_audit_01/result.json"
    if protocol.digest(path) != ACTUAL95_SHA or protocol.digest(audit) != AUDIT95_SHA:
        raise ValueError("Frozen actual95 or independent physical audit changed")
    with np.load(path, allow_pickle=False) as archive:
        warm = {k: archive[k].copy() for k in archive.files}
    if warm["action"].shape != (257, 29) or bool(warm["verified_teacher"]):
        raise ValueError("Expected complete original actual95, still not a teacher")
    original = args.source
    args.source, full, metadata = landing.prepare_source(original, args.prefix_view_dir)
    if not np.array_equal(warm["action"][:130], full["action"][:130]) or any(
        not np.array_equal(warm[k][:131], full[k][:131]) for k in ("qpos", "qvel", "state")
    ):
        raise ValueError("Actual95 and original continuous130-control prefix differ")
    base_factory = survival.early.prior_factory(full)

    def prepare(controller, source, extra, cfg):
        _, _, info = base_factory(controller, source, extra, cfg)
        info["air_phase_observation"] = recovery.previous.air_sensor.add_board_ground_sensor(
            controller, cfg
        )
        nominal = warm["action"][130:].copy()
        if not np.array_equal(nominal, np.clip(nominal, *controller.target_limits.T)):
            raise ValueError("Verified actual95 targets exceed original limits")
        kp = controller.model.actuator_gainprm[controller.index.robot_actuators, 0]
        stand = controller.reference.robot_state[0, 7:36] + controller.ff_tau[0] / kp
        stand = np.clip(stand, *controller.target_limits.T)
        priors = standing_targets(nominal, stand, controller.target_limits)
        priors["exact82_hop_not_teacher"] = full["action"][130:].copy()
        info.pop("nominal_exact_source82_control_count")
        info["nominal_exact_actual95_control_count"] = 127
        info["actual95_pd_target_source"] = {"path": str(path.resolve()), "sha256": ACTUAL95_SHA}
        info["independent_actual95_audit"] = {"path": str(audit.resolve()), "sha256": AUDIT95_SHA}
        info["nominal_zero_residual_is_exact_actual95"] = True
        info["source_states_applied_after_initialization"] = False
        return nominal, priors, info

    cost, selector = progress.bind_progress_elites(recovery.make_cost(args.out_dir))
    protocol.run_experiment(
        args,
        config,
        cost_function=cost,
        prior_factory=prepare,
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=selector,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "progress_elite_script_sha256": protocol.digest(progress.__file__),
            "reference_sole_recovery_script_sha256": protocol.digest(recovery.__file__),
            "directed_geometry_script_sha256": protocol.digest(recovery.directed.__file__),
            "air_phase_caller_sha256": protocol.digest(recovery.previous.__file__),
            "air_sensor_script_sha256": protocol.digest(recovery.previous.air_sensor.__file__),
            "reference_survival_script_sha256": protocol.digest(survival.__file__),
            "reference_boundary_script_sha256": protocol.digest(survival.early.__file__),
            "reference_landing_script_sha256": protocol.digest(landing.__file__),
            "top_sensor_script_sha256": protocol.digest(landing.top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "original_full_actual_source": metadata,
            "original_source_sha256": landing.SOURCE_SHA,
            "nominal_actual95_source": {"path": str(path.resolve()), "sha256": ACTUAL95_SHA},
            "independent_actual95_audit": {"path": str(audit.resolve()), "sha256": AUDIT95_SHA},
            "task": TASK,
            "config": asdict(config),
            "search_is_teacher_exploration": True,
        },
    )
    if (
        protocol.digest(original) != landing.SOURCE_SHA
        or protocol.digest(path) != ACTUAL95_SHA
        or protocol.digest(audit) != AUDIT95_SHA
    ):
        raise RuntimeError("Frozen source/audit changed")


if __name__ == "__main__":
    main()
