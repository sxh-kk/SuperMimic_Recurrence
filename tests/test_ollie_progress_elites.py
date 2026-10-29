"""Recovery progress cannot be averaged away while filling the CEM elite quota."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.local.ollie_progress_elites import (
    SCHEMA,
    bind_progress_elites,
    highest_event_and_progress_elites,
)


def config(**updates):
    result = {
        "progress_elite_schema": SCHEMA,
        "progress_tier_stride": 4.0,
        "progress_tier_count": 4,
        "progress_tier_score_upper_bound": 3.0,
        "elites": 16,
    }
    result.update(updates)
    return SimpleNamespace(**result)


def order(grades, costs):
    return np.lexsort((costs, -grades))


def test_single_catch_cannot_be_averaged_with_fifteen_non_catches():
    grades = np.full(16, 2)
    costs = np.r_[8.7, np.linspace(12.01, 12.9, 15)]
    result = highest_event_and_progress_elites(order(grades, costs), grades, costs, config())
    assert result.tolist() == [0]
    parameters = np.r_[1.0, np.full(15, -1.0)]
    assert parameters[result].mean() == 1
    assert parameters.mean() < 0  # The former16-candidate update loses the catch.


def test_physical_flight_always_beats_a_lower_grade_with_better_progress():
    grades, costs = np.array([1, 2, 0, 2]), np.array([0.1, 12.4, 0.0, 8.9])
    chosen = highest_event_and_progress_elites(order(grades, costs), grades, costs, config())
    assert chosen.tolist() == [3]


def test_best_progress_elites_are_unique_evaluated_and_limited_by_quota():
    grades, costs = np.full(20, 2), np.linspace(8.9, 8.1, 20)
    ids = np.arange(20)
    selected = highest_event_and_progress_elites(ids, grades, costs, config(elites=3))
    assert selected.tolist() == [19, 18, 17]
    assert len(np.unique(selected)) == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"progress_elite_schema": None},
        {"progress_elite_schema": "v2"},
        {"progress_tier_stride": 3.0},
        {"progress_tier_count": 5},
        {"progress_tier_score_upper_bound": 4.0},
        {"progress_tier_score_upper_bound": 4.1},
        {"progress_tier_score_upper_bound": float("nan")},
        {"elites": True},
        {"elites": 0},
        {"elites": 17},
    ],
)
def test_explicit_schema_and_disjoint_score_interval_are_required(changes):
    with pytest.raises(ValueError):
        highest_event_and_progress_elites(
            np.array([0]), np.array([2]), np.array([12.1]), config(**changes)
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0, 15.1, 16.0])
def test_nonfinite_negative_gap_interval_and_out_of_range_scores_rejected(score):
    with pytest.raises(ValueError):
        highest_event_and_progress_elites(np.array([0]), np.array([2]), np.array([score]), config())


@pytest.mark.parametrize(
    "ids", [np.array([], int), np.array([0, 0]), np.array([2]), np.array([0.0])]
)
def test_invalid_evaluated_candidate_order_rejected(ids):
    with pytest.raises(ValueError):
        highest_event_and_progress_elites(ids, np.array([2]), np.array([12.1]), config())


def test_invalid_branch_sentinel_cannot_enter_elites():
    grades, costs = np.array([-1, 2]), np.array([np.inf, 8.2])
    assert highest_event_and_progress_elites(np.array([1]), grades, costs, config()).tolist() == [1]
    with pytest.raises(ValueError):
        highest_event_and_progress_elites(np.array([0, 1]), grades, costs, config())
    with pytest.raises(ValueError):
        highest_event_and_progress_elites(np.array([1]), np.array([-1, 3]), costs, config())


def bound_cost(grades, costs, tiers):
    def evaluate(controller, states, sensors, targets, cfg):
        return grades, costs, {"score": costs}, {"progress_tier": tiers}

    return evaluate


def test_bound_selector_uses_explicit_progress_and_consumes_one_fresh_batch():
    grades, costs, tiers = np.array([2, 2]), np.array([8.7, 12.1]), np.array([2, 3])
    evaluate, select = bind_progress_elites(bound_cost(grades, costs, tiers))
    with pytest.raises(ValueError, match="fresh"):
        select(np.array([0, 1]), grades, costs, config())
    evaluate(None, None, None, None, config())
    assert select(np.array([0, 1]), grades.copy(), costs.copy(), config()).tolist() == [0]
    with pytest.raises(ValueError, match="fresh"):
        select(np.array([0, 1]), grades, costs, config())


def test_explicit_tier_catches_overflow_that_score_decoder_cannot_identify():
    grades, costs = np.array([2]), np.array([8.2])
    # The number8.2 alone is a valid tier2 score. An explicit tier1 declaration
    # proves that its true internal score overflowed4 and must be rejected.
    assert highest_event_and_progress_elites(np.array([0]), grades, costs, config()).tolist() == [0]
    evaluate, _ = bind_progress_elites(bound_cost(grades, costs, np.array([1])))
    with pytest.raises(ValueError, match="interval"):
        evaluate(None, None, None, None, config())


def test_mutated_batch_config_and_missing_metrics_do_not_use_stale_progress():
    grades, costs, tiers = np.array([2]), np.array([8.2]), np.array([2])
    evaluate, select = bind_progress_elites(bound_cost(grades, costs, tiers))
    evaluate(None, None, None, None, config())
    with pytest.raises(ValueError, match="different or mutated"):
        select(np.array([0]), grades, costs + 0.1, config())
    with pytest.raises(ValueError, match="different or mutated"):
        select(np.array([0]), grades, costs, config(elites=2))
    other_eval, other_select = bind_progress_elites(bound_cost(grades, costs, tiers))
    with pytest.raises(ValueError, match="fresh"):
        other_select(np.array([0]), grades, costs, config())
    assert select(np.array([0]), grades, costs, config()).tolist() == [0]
    assert callable(other_eval)
    malformed, _ = bind_progress_elites(lambda *args: (grades, costs, {}, {}))
    with pytest.raises(ValueError, match="explicit"):
        malformed(None, None, None, None, config())
