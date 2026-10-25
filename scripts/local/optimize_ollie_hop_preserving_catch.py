"""Preserve the known real hop while exploring recoverable feet; not a policy.

80's strict recoverability selection traded the hop for a cheaper non-jump.
Keep the original physical-flight grade and exact76 target sequence as the
nominal. Fit the distribution only to candidates of its highest event grade.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import optimize_ollie_recoverable_catch as recovery

protocol = recovery.protocol
SCHEMA = "supermimic.ollie.hop_preserving_catch.v1"


@dataclass(frozen=True)
class Config(recovery.Config):
    seed: int = 81
    sigma_rad: float = 0.10
    selection: str = "original_physical_event_grade_then_80_cost; highest_grade_elites_only"


def highest_grade_elites(order, grades, costs, config):
    """Never move a real-flight mean toward lower-grade non-jumps."""
    return order[grades[order] == grades[order[0]]][: config.elites]


def hop_preserving_cost(controller, states, sensors, targets, config):
    recoverable_grade, costs, terms, metrics = recovery.recoverable_cost(
        controller, states, sensors, targets, config
    )
    metrics["recoverable_selection_grade80"] = recoverable_grade
    raw = metrics["raw76_event_grade"]
    grades = np.where((recoverable_grade >= 0) & np.isfinite(costs), raw, -1)
    return grades, costs, terms, metrics


def make_prior_factory(q12, previous, initial):
    original_factory = recovery.make_prior_factory(q12, previous, initial)

    def factory(controller, source, extra, config):
        _, priors, metadata = original_factory(controller, source, extra, config)
        nominal = previous[config.prefix_controls :].copy()
        # Every timestep of the original evaluated hop, not its knot fit, is
        # the nominal. Thus zero residual reproduces the real hop exactly.
        ordered = {"exact76_actual_hop_nominal_not_teacher": nominal.copy()}
        ordered.update({k: v for k, v in priors.items() if not k.startswith("previous76")})
        metadata.update(
            nominal="exact saved76 original107 evaluated PD targets; residual0 is the real hop",
            previous76_action_sha256=recovery.PREVIOUS_SHA,
            mean_initial_residual_exact_zero=True,
            highest_grade_elites_only=True,
            static_prior_rear_drive_assert_applies_only_to_static_prior=True,
        )
        return nominal, ordered, metadata

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
    q12, previous, initial, paths = recovery.load_priors(args, config)
    protocol.run_experiment(
        args,
        config,
        cost_function=hop_preserving_cost,
        prior_factory=make_prior_factory(q12, previous, initial),
        elite_selector=highest_grade_elites,
        schema=SCHEMA,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "recovery_script_sha256": protocol.digest(recovery.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "frozen_recoverable_sources": paths,
            "task": recovery.TASK,
            "config": asdict(config),
            "changes_from80": [
                "original physical event grade; recoverability stays soft and separately reported",
                "nominal exact76 full107 evaluated robot targets, not a knot approximation",
                "highest event grade only for elite mean/std; fewer than16 allowed",
                "initial residual sigma .10rad, seed81; no physics or success criteria change",
            ],
        },
    )
    for item in paths.values():
        if protocol.digest(item["path"]) != item["sha256"]:
            raise RuntimeError("Hop-preserving experiment input changed")


if __name__ == "__main__":
    main()
