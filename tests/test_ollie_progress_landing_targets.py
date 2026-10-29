"""Pure PD standing priors must keep the verified pre-catch controls exact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.local.optimize_ollie_progress_preserving_landing import standing_targets


def test_nominal_is_exact_and_standing_seeds_do_not_change_verified_launch():
    nominal = np.linspace(-0.2, 0.4, 127 * 29).reshape(127, 29)
    stand = np.zeros(29)
    limits = np.tile([-1, 1], (29, 1))
    targets = standing_targets(nominal, stand, limits)
    time = 2.60 + (np.arange(127) + 0.5) * 0.02
    assert np.array_equal(targets["exact_actual95_front_recovery_not_teacher"], nominal)
    for name, value in targets.items():
        assert np.array_equal(value[time < 3.38], nominal[time < 3.38])
        if name != "exact_actual95_front_recovery_not_teacher":
            assert np.allclose(value[time >= 4.10], stand, rtol=0, atol=1e-15)


def test_standing_prior_targets_are_bounded_without_mutating_nominal():
    nominal = np.full((127, 29), 0.1)
    limits = np.tile([-0.2, 0.2], (29, 1))
    original = nominal.copy()
    targets = standing_targets(nominal, np.full(29, 0.7), limits)
    assert np.array_equal(nominal, original)
    for value in targets.values():
        assert (value <= 0.2).all() and (value >= -0.2).all()
