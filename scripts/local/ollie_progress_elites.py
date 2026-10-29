"""Keep physical-event and recovery progress when updating native CEM elites.

No physics or frozen rollout protocol is changed. A catch must not be averaged
with lower-progress approaches merely to fill an elite quota. Prefer the bound
cost/selector pair: its explicit per-batch tiers detect score-interval overflow
that cannot be distinguished from another tier by floor(score/4) alone.
"""

from __future__ import annotations

from functools import wraps
from numbers import Integral

import numpy as np

SCHEMA = "supermimic.ollie.progress_elites.v1"
STRIDE = 4.0
TIER_COUNT = 4
MAX_ELITES = 16


def _configuration(config):
    """Require an explicit score contract, rather than guessing from any cost."""
    if getattr(config, "progress_elite_schema", None) != SCHEMA:
        raise ValueError("Missing or incompatible progress_elite_schema")
    stride = getattr(config, "progress_tier_stride", None)
    count = getattr(config, "progress_tier_count", None)
    bound = getattr(config, "progress_tier_score_upper_bound", None)
    elites = getattr(config, "elites", None)
    if isinstance(stride, (bool, np.bool_)) or stride != STRIDE:
        raise ValueError("Progress schema v1 requires score stride4")
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, Integral)
        or count != TIER_COUNT
    ):
        raise ValueError("Progress schema v1 requires four tiers0..3")
    if (
        isinstance(bound, (bool, np.bool_))
        or not isinstance(bound, (int, float, np.integer, np.floating))
        or not np.isfinite(bound)
        or not 0 < bound < stride
    ):
        raise ValueError("Within-tier score upper bound must be finite, positive and below stride4")
    if (
        isinstance(elites, (bool, np.bool_))
        or not isinstance(elites, Integral)
        or not 1 <= elites <= MAX_ELITES
    ):
        raise ValueError("Expected an integer elite quota between1 and16")
    return float(stride), int(count), float(bound), int(elites)


def _integer_vector(value, name, *, length=None):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric integer-valued vector")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} length differs from the evaluated batch")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} contains nonfinite or fractional values")
    return array.astype(np.int64)


def _batch(grades, costs):
    grades = _integer_vector(grades, "event grades")
    costs = np.asarray(costs)
    if costs.ndim != 1 or costs.shape != grades.shape or costs.dtype.kind not in "iuf":
        raise ValueError("Expected matching one-dimensional numeric grades/costs")
    if np.isnan(costs).any() or np.isneginf(costs).any():
        raise ValueError("Costs must not contain NaN or negative infinity")
    if ((grades < -1) | (grades > 2)).any():
        raise ValueError("Only event grades-1,0,1,2 are supported by the frozen protocol")
    valid = grades >= 0
    if (~np.isfinite(costs[valid])).any() or (costs[valid] < 0).any():
        raise ValueError("Every nonnegative event grade requires a finite nonnegative score")
    return grades, costs, valid


def _tiers_and_scores(tiers, costs, valid, stride, count, bound):
    tiers = _integer_vector(tiers, "progress tiers", length=len(costs))
    if ((tiers < 0) | (tiers >= count)).any():
        raise ValueError("Progress tier is outside the declared0..3 schema")
    within = costs[valid] - stride * tiers[valid]
    if ((within < 0) | (within >= bound)).any():
        raise ValueError("Score lies outside its explicit progress-tier interval")
    return tiers


def _select(order, grades, costs, tiers, quota):
    order = np.asarray(order)
    if order.ndim != 1 or order.dtype.kind not in "iu" or not len(order):
        raise ValueError("Expected a nonempty integer vector of evaluated candidate IDs")
    if len(np.unique(order)) != len(order) or (order < 0).any() or (order >= len(costs)).any():
        raise ValueError("Evaluated candidate IDs are duplicated or outside the batch")
    if (grades[order] < 0).any() or not np.isfinite(costs[order]).all():
        raise ValueError("Candidate order includes an invalid or nonfinite branch")
    highest_grade = grades[order].max()
    candidates = order[grades[order] == highest_grade]
    lowest_tier = tiers[candidates].min()
    candidates = candidates[tiers[candidates] == lowest_tier]
    # Sorting inside the eligible group is explicit; even an unsorted caller
    # cannot trade away event/progress priority for a smaller absolute score.
    chosen = candidates[np.argsort(costs[candidates], kind="stable")[:quota]]
    return chosen.astype(np.int64, copy=True)


def highest_event_and_progress_elites(order, grades, costs, config):
    """Four-argument selector for explicitly declared fixed stride4 scores.

    Required config fields: progress_elite_schema=SCHEMA,
    progress_tier_stride=4., progress_tier_count=4,
    progress_tier_score_upper_bound=3. (or another bound strictly below4),
    and elites. Invalid branches may use grade-1/cost+inf outside ``order``.

    This decoder verifies disjoint intervals, but score alone cannot detect a
    hidden tier1 overflow by4 that aliases a legal tier2 score. Use
    ``bind_progress_elites`` when explicit metrics are available.
    """
    stride, count, bound, quota = _configuration(config)
    grades, costs, valid = _batch(grades, costs)
    if (costs[valid] >= stride * count).any():
        raise ValueError("Score is outside all declared progress-tier intervals")
    tiers = np.zeros(len(costs), np.int64)
    tiers[valid] = np.floor(costs[valid] / stride).astype(np.int64)
    tiers = _tiers_and_scores(tiers, costs, valid, stride, count, bound)
    return _select(order, grades, costs, tiers, quota)


def bind_progress_elites(cost_function, *, tier_metric="progress_tier"):
    """Return ``(wrapped_cost, four_argument_selector)`` with a private batch cache.

    The cost callable has the unchanged native five-argument cost signature and
    returns ``grades, costs, terms, metrics``. Its explicit metrics tiers must
    match each branch's declared score interval. The selector accepts only the
    latest unconsumed batch, equal grades/costs and equal configuration. It may
    be called once per batch. Independent factories never share state.
    """
    if not callable(cost_function) or not isinstance(tier_metric, str) or not tier_metric:
        raise ValueError("Expected a cost callable and a nonempty explicit tier metric")
    cached = None

    @wraps(cost_function)
    def evaluated_cost(controller, states, sensors, targets, config):
        nonlocal cached
        cached = None
        contract = _configuration(config)
        result = cost_function(controller, states, sensors, targets, config)
        if not isinstance(result, tuple) or len(result) != 4:
            raise ValueError("Cost must return grades,costs,terms,metrics")
        grades, costs, _, metrics = result
        grades, costs, valid = _batch(grades, costs)
        if not isinstance(metrics, dict) or tier_metric not in metrics:
            raise ValueError("Cost did not provide explicit per-candidate progress tiers")
        tiers = _tiers_and_scores(metrics[tier_metric], costs, valid, *contract[:3])
        cached = (grades.copy(), costs.copy(), tiers.copy(), contract)
        return result

    def selector(order, grades, costs, config):
        nonlocal cached
        if cached is None:
            raise ValueError("No fresh unconsumed evaluated batch for progress selection")
        expected_grades, expected_costs, tiers, contract = cached
        current_contract = _configuration(config)
        actual_grades, actual_costs, _ = _batch(grades, costs)
        if (
            current_contract != contract
            or not np.array_equal(actual_grades, expected_grades)
            or not np.array_equal(actual_costs, expected_costs)
        ):
            raise ValueError("Progress selector received a different or mutated evaluated batch")
        selected = _select(order, actual_grades, actual_costs, tiers, contract[3])
        cached = None
        return selected

    return evaluated_cost, selector
