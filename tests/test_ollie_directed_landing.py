"""Event-window and ranking contracts for directed landing exploration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.local.optimize_ollie_directed_landing import landing_interval, selection_cost


def test_landing_grace_does_not_rewrite_airborne_phase():
    descending = np.zeros((1, 50), bool)
    descending[0, 10:20] = True
    mask = landing_interval(np.array([5]), np.array([20]), descending, np.ones_like(descending), 24)
    assert np.flatnonzero(mask[0]).tolist() == list(range(10, 44))


def test_landing_window_requires_qualified_flight_and_original_safety():
    descending = np.ones((2, 50), bool)
    safe = np.ones_like(descending)
    safe[0, 25:] = False
    mask = landing_interval(np.array([5, 50]), np.array([20, 50]), descending, safe, 24)
    assert np.flatnonzero(mask[0]).tolist() == list(range(5, 25))
    assert not mask[1].any()


def test_tiers_have_disjoint_score_ranges():
    # Even the worst survivability in a recovered tier outranks a cheaper
    # unrecovered tier, and a good preparation score is not an argument.
    tier = np.arange(4)
    worst = selection_cost(tier, np.full(4, 1e12), np.ones(4, bool), np.ones(4), np.full(4, 1e12))
    best = selection_cost(tier, np.zeros(4), np.ones(4, bool), np.zeros(4), np.zeros(4))
    assert np.all(worst[:-1] < best[1:])


def test_missing_opportunity_never_looks_like_zero_geometry_error():
    score = selection_cost(
        np.array([3, 3]), np.array([0.0, 0.0]), np.array([True, False]), np.zeros(2), np.zeros(2)
    )
    assert score.tolist() == [12.0, 13.0]
