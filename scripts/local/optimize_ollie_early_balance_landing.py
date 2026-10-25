"""Move the same real-top/full29 landing objective before the COM drift.

Only the warm prefix boundary, horizon and fixed seed differ from83. The
physical model, targets, cost semantics and original full v3 remain unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.local import optimize_ollie_top_support_landing as landing

protocol = landing.protocol
SCHEMA = "supermimic.ollie.early_balance_top_landing.v1"


@dataclass(frozen=True)
class Config(landing.Config):
    prefix_controls: int = 145
    horizon: int = 112
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
        2.22,
    )
    seed: int = 84


def prior_factory(full):
    factory83 = landing.prior_factory(full)

    def factory(controller, source, extra, config):
        nominal, priors, metadata = factory83(controller, source, extra, config)
        priors = {
            (
                "elapsed_standing_blend_not_teacher"
                if name == "original_standing_targets_after_air_not_teacher"
                else name
            ): value
            for name, value in priors.items()
        }
        metadata.pop("nominal_exact_source82_99targets")
        metadata["nominal_exact_source82_control_count"] = config.horizon
        metadata["balance_correction_begins_at_physics_s"] = config.prefix_controls * 0.02
        metadata["prior_phase_is_elapsed_time_not_fixed_video_phase"] = True
        return nominal, priors, metadata

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
    args.source, full, metadata = landing.prepare_source(original, args.prefix_view_dir)
    protocol.run_experiment(
        args,
        config,
        cost_function=landing.make_cost(args.out_dir),
        prior_factory=prior_factory(full),
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=landing.previous.highest_grade_elites,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "reference_landing_script_sha256": protocol.digest(landing.__file__),
            "top_sensor_script_sha256": protocol.digest(landing.top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "original_full_actual_source": metadata,
            "original_source_sha256": landing.SOURCE_SHA,
            "task": landing.TASK,
            "config": asdict(config),
            "search_is_teacher_exploration": True,
            "changes_from83": [
                "continuous original145-control prefix to2.90s, not158 to3.16s",
                "exact112 saved82 suffix targets, zero residual retains original low hop",
                "last knot2.22s, seed84; same cost, weights, priors, physics and limits",
            ],
        },
    )
    if protocol.digest(original) != landing.SOURCE_SHA:
        raise RuntimeError("Frozen actual82 source changed")


if __name__ == "__main__":
    main()
