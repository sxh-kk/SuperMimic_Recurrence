"""Phase targets preserve the pop; recovered score prioritizes real survival."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.local import optimize_ollie_phase_basis as basis


def score(tiers, first_ground, final, quality):
    n = len(tiers)
    raw = 4 * np.asarray(tiers) + 0.5
    return basis.survival_first_score(
        np.full(n, 2),
        raw,
        {"progress_tier": 4 * np.asarray(tiers), "old_detail": np.ones(n)},
        {
            "progress_tier": np.asarray(tiers),
            "first_robot_ground_s": np.asarray(first_ground),
            "final_standing_cost": np.asarray(final),
            "directed_landing_quality": np.asarray(quality),
            "has10ms_opportunity": np.ones(n, dtype=bool),
        },
    )


def test_one_later_physics_step_beats_all_recovered_secondary_terms():
    _, costs, _, metrics = score([1, 1], [3.6, 3.6025], [0, 1e12], [0, 1e12])
    assert costs[1] < costs[0]
    assert metrics["lost_physics_steps_after_first_ground"][0] == (
        metrics["lost_physics_steps_after_first_ground"][1] + 1
    )


def test_recovery_tier_has_priority_and_approach_cost_is_unchanged():
    _, costs, terms, metrics = score(
        [0, 1, 2, 3], [2.6, -1, 3.6, 3.6], [1e12] * 4, [1e12] * 4
    )
    assert costs[0] < costs[1] < costs[2] < costs[3]
    np.testing.assert_array_equal(costs[2:], [8.5, 12.5])
    np.testing.assert_array_equal(terms["old_detail"], [0, 0, 1, 1])
    np.testing.assert_array_equal(metrics["raw95_term_old_detail"], np.ones(4))


def test_all_parameters_leave_pre324_targets_exact_and_zero_is_nominal():
    rng = np.random.default_rng(99)
    nominal = rng.uniform(-0.4, 0.4, (127, 29))
    stand = rng.uniform(-0.4, 0.4, 29)
    limits = np.tile([-0.5, 0.5], (29, 1))
    parameters = rng.uniform(-0.3, 0.3, (128, 25))
    parameters[:, 24] = rng.uniform(0, 1, 128)
    actual, _ = basis.make_targets(parameters, nominal, stand, limits)
    np.testing.assert_array_equal(actual[:, :32], np.broadcast_to(nominal[:32], (128, 32, 29)))
    exact, _ = basis.make_targets(np.zeros(25), nominal, stand, limits)
    np.testing.assert_array_equal(exact, nominal)
    assert (actual >= limits[:, 0]).all() and (actual <= limits[:, 1]).all()


def test_terminal_blend_is_causal_and_reaches_original_stand_target():
    nominal, stand = np.zeros((127, 29)), np.full(29, 0.25)
    parameters = np.zeros(25)
    parameters[24] = 1
    actual, _ = basis.make_targets(parameters, nominal, stand, np.tile([-1, 1], (29, 1)))
    times = 2.6 + (np.arange(127) + 0.5) * 0.02
    np.testing.assert_array_equal(actual[times < 3.6], nominal[times < 3.6])
    np.testing.assert_array_equal(actual[times >= 4.1], np.broadcast_to(stand, actual[times >= 4.1].shape))
