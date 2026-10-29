"""Catch phase contracts: actual contact closes air; certificate delay does not."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.local.optimize_ollie_air_catch_landing import (
    minimum_same_window,
    phase_intervals,
)


def test_catch_starts_at_qualified_window_start_before_certificate_end():
    qualified = np.zeros((1, 40 - 15), dtype=bool)
    qualified[0, 5] = True  # 16-sample certificate finishes at index20.
    ground = np.zeros((1, 40), dtype=bool)
    ground[0, 26:] = True
    start, end, airborne, eligible = phase_intervals(
        qualified, ground, np.ones_like(ground), np.ones_like(ground)
    )
    assert start.tolist() == [5] and end.tolist() == [26]
    assert airborne[0, 5:26].all() and eligible[0, 5:26].all()
    assert not airborne[0, :5].any() and not airborne[0, 26:].any()


def test_first_recontact_closes_phase_even_if_board_bounces():
    qualified = np.zeros((1, 45 - 15), dtype=bool)
    qualified[0, 3] = True
    ground = np.zeros((1, 45), dtype=bool)
    ground[0, 22] = True
    _, end, airborne, _ = phase_intervals(
        qualified, ground, np.ones_like(ground), np.ones_like(ground)
    )
    assert end[0] == 22 and not airborne[0, 22:].any()


def test_absent_flight_never_opens_air_phase():
    ground = np.zeros((2, 40), dtype=bool)
    _, _, airborne, eligible = phase_intervals(
        np.zeros((2, 25), bool), ground, np.ones_like(ground), np.ones_like(ground)
    )
    assert not airborne.any() and not eligible.any()


def test_same_window_cannot_join_unsynchronized_or_late_samples():
    eligible = np.zeros((1, 20), bool)
    eligible[0, 3:8] = True
    value = np.zeros((1, 20))  # Late low values must not be selected.
    value[0, 3:8] = [4, 3, 2, 1, 4]
    best, index, has = minimum_same_window(value, eligible, 4)
    assert has.tolist() == [True] and best.tolist() == [4] and index.tolist() == [3]
    eligible[0, 5] = False
    best, _, has = minimum_same_window(value, eligible, 4)
    assert not has[0] and np.isinf(best[0])


def test_descent_and_original_safety_are_required_at_each_sample():
    qualified = np.zeros((1, 40 - 15), bool)
    qualified[0, 2] = True
    ground = np.zeros((1, 40), bool)
    descending, safe = np.ones_like(ground), np.ones_like(ground)
    descending[0, :12] = False
    safe[0, 20:] = False
    _, _, _, opportunity = phase_intervals(qualified, ground, descending, safe)
    assert np.flatnonzero(opportunity[0]).tolist() == list(range(12, 20))
