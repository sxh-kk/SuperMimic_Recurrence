"""PD prior timing/bounds; body/board states are not an input or an output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.local.optimize_ollie_sole_recovery import flattening_targets


def inputs():
    baseline = np.full((127, 29), 0.2)
    stand = np.full(29, 0.1)
    limits = np.tile([-1, 1], (29, 1))
    roll = baseline.copy()
    roll[15:43, 1] += 0.04
    return baseline, stand, limits, roll


def test_flattening_seeds_preserve_prepop_targets_and_end_at_standing():
    baseline, stand, limits, roll = inputs()
    priors, _ = flattening_targets(baseline, stand, limits, roll)
    times = 2.6 + (np.arange(127) + 0.5) * 0.02
    for name, targets in priors.items():
        source = roll if name.startswith("short_hiproll") else baseline
        assert np.array_equal(targets[times < 3.24], source[times < 3.24])
        assert np.allclose(targets[times >= 3.90], stand, rtol=0, atol=1e-15)
        assert targets.shape == (127, 29)
        assert ((targets >= limits[:, 0]) & (targets <= limits[:, 1])).all()


def test_ankle_wave_changes_only_selected_pd_columns_before_standing_blend():
    baseline, stand, limits, roll = inputs()
    priors, _ = flattening_targets(baseline, stand, limits, roll)
    time = 2.6 + (np.arange(127) + 0.5) * 0.02
    mask = (time >= 3.24) & (time < 3.36)
    left = priors["left_ankle_minus06_then_stand_not_teacher"]
    assert np.array_equal(left[mask][:, np.arange(29) != 4], baseline[mask][:, np.arange(29) != 4])
    assert (left[mask, 4] < baseline[mask, 4]).all()


def test_prior_clipping_is_reported_and_does_not_mutate_sources():
    baseline, stand, limits, roll = inputs()
    limits[:, 0] = 0
    original = baseline.copy()
    priors, metadata = flattening_targets(baseline, stand, limits, roll)
    assert metadata["both_ankles_minus09_then_stand"]["target_values_clipped"] > 0
    assert (priors["both_ankles_minus09_then_stand_not_teacher"] >= 0).all()
    assert np.array_equal(baseline, original)
