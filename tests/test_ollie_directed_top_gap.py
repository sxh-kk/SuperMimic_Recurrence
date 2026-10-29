"""Independent geometry contracts for directed finite deck-top proximity."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.local.ollie_directed_top_gap import directed_top_gap, segment_rectangle_closest


def test_segment_interior_and_corner_have_analytic_finite_rectangle_distance():
    result = segment_rectangle_closest([-2, 0, 1], [2, 0, 1], [0.5, 0.5])
    assert result["distance_m"] == pytest.approx(1)
    assert result["normal_cosine"] == pytest.approx(1)
    assert abs(result["top_point"][0]) <= 0.5
    result = segment_rectangle_closest([1, 1, 1], [1, 1, 1], [0.5, 0.5])
    assert result["distance_m"] == pytest.approx(np.sqrt(1.5))
    assert np.array_equal(result["top_point"], [0.5, 0.5, 0])


def test_direction_rejects_side_and_below_even_at_zero_ordinary_surface_gap():
    points = np.array([[0.52, 0, 0], [0, 0, -0.02], [0, 0, 0.02]])
    nearest = segment_rectangle_closest(points, points, [0.5, 0.5])
    directed = segment_rectangle_closest(points, points, [0.5, 0.5], min_normal_cos=0.5)
    assert np.allclose(nearest["distance_m"], 0.02)
    assert np.array_equal(directed["feasible"], [False, False, True])
    assert np.allclose(nearest["normal_cosine"], [0, -1, 1])


def test_cone_boundary_is_solved_between_endpoints_and_stationary_points():
    result = segment_rectangle_closest([1, 0, 0.1], [1, 0, 1], [0.5, 0.5], min_normal_cos=0.5)
    expected_z = 0.5 / np.sqrt(3)
    assert result["distance_m"] == pytest.approx(1 / np.sqrt(3), abs=1e-12)
    assert result["axis_point"][2] == pytest.approx(expected_z, abs=1e-12)
    assert result["segment_fraction"] == pytest.approx((expected_z - 0.1) / 0.9, abs=1e-12)
    assert result["normal_cosine"] == pytest.approx(0.5, abs=1e-12)


def test_cone_quadratic_degenerates_to_a_linear_boundary_without_losing_root():
    start = np.array([0.8, 0, 0.1])
    difference = np.array([-0.3, 0, 0.3 * np.sqrt(3)])
    result = segment_rectangle_closest(
        start, start + difference, [0.5, 0.5], min_normal_cos=np.sqrt(3) / 2
    )
    # The path has dx**2 == dz**2/3, so the cone quadratic term vanishes.
    expected_t = (0.3 - 0.1 / np.sqrt(3)) / 0.6
    assert result["segment_fraction"] == pytest.approx(expected_t, abs=1e-12)
    assert result["normal_cosine"] == pytest.approx(np.sqrt(3) / 2, abs=1e-12)


def test_random_exact_solution_is_bounded_by_independent_dense_segment_oracle():
    generator = np.random.default_rng(91)
    start, end = generator.uniform(-1, 1, (2, 60, 3))
    half = np.array([0.4, 0.2])
    grid = np.linspace(0, 1, 20001)
    points = start[:, None] + grid[None, :, None] * (end - start)[:, None]
    top = points.copy()
    top[..., :2] = np.clip(points[..., :2], -half, half)
    top[..., 2] = 0
    delta = points - top
    distance = np.linalg.norm(delta, axis=-1)
    cosine = np.divide(delta[..., 2], distance, out=np.ones_like(distance), where=distance > 0)
    for threshold in (None, 0.5, 0.8):
        result = segment_rectangle_closest(start, end, half, min_normal_cos=threshold)
        allowed = np.ones_like(distance, bool) if threshold is None else cosine >= threshold
        oracle = np.where(allowed, distance, np.inf).min(1)
        comparable = np.isfinite(oracle)
        assert result["feasible"][comparable].all()
        # A uniform grid gives a Lipschitz upper bound, independently of the
        # piecewise polynomial solver's candidate construction.
        tolerance = np.linalg.norm(end - start, axis=-1) / 20000 + 1e-10
        assert (result["distance_m"][comparable] <= oracle[comparable] + 1e-10).all()
        assert (
            result["distance_m"][comparable] >= oracle[comparable] - tolerance[comparable]
        ).all()
        assert (result["segment_fraction"][comparable] >= 0).all()
        assert (result["segment_fraction"][comparable] <= 1).all()


def controller_and_sensors(points):
    # Seven tiny finite capsules share a known authored pose. Other deck boxes
    # are far away, so a genuine center-top contact cannot pick another box.
    names = ["board_center", "board_tail", "board_nose"]
    geometry = SimpleNamespace(
        geom=lambda name: SimpleNamespace(id=names.index(name)),
        geom_size=np.tile([0.5, 0.5, 0.05], (3, 1)),
    )
    controller = SimpleNamespace(model=geometry, slices={}, foot_geometry={})
    cursor = 0
    for prefix in ("left_body_", "right_body_", "board_center_", "board_tail_", "board_nose_"):
        for suffix, length in (("pos", 3), ("quat", 4)):
            controller.slices[prefix + suffix] = slice(cursor, cursor + length)
            cursor += length
    sensors = np.zeros((*points.shape[:-1], cursor))
    for prefix in ("left_body_", "right_body_", "board_center_", "board_tail_", "board_nose_"):
        sensors[..., controller.slices[prefix + "quat"]] = [1, 0, 0, 0]
    sensors[..., controller.slices["left_body_pos"]] = points
    sensors[..., controller.slices["board_tail_pos"]] = [20, 0, 0]
    sensors[..., controller.slices["board_nose_pos"]] = [-20, 0, 0]
    controller.foot_geometry["left"] = {
        "center": np.zeros((7, 3)),
        "axis": np.tile([1, 0, 0], (7, 1)),
        "size": np.tile([0.02, 0.01], (7, 1)),
        "geoms": np.arange(10, 17),
    }
    return controller, sensors


def test_capsule_top_side_and_under_proximity_have_directed_evidence():
    points = np.array([[0, 0, 0.07], [0.53, 0, 0.05], [0, 0, 0.03]])
    controller, sensors = controller_and_sensors(points)
    result = directed_top_gap(controller, sensors, "left", chunk_size=2)
    assert result["gap_m"][0] == pytest.approx(0, abs=1e-12)
    assert result["normal_up"][0] == pytest.approx(1)
    assert result["box_index"][0] == 0
    assert result["capsule_geom_id"][0] in range(10, 17)
    assert not result["cone_feasible"][1:].any()
    assert result["gap_m"][1] > 0.02
    assert result["gap_m"][2] == pytest.approx(0.04)
    assert np.allclose(result["undirected_gap_m"], 0, atol=1e-12)
    assert result["normal_up"][1] == pytest.approx(0)
    assert result["normal_up"][2] == pytest.approx(-1)
    witness_distance = np.linalg.norm(
        result["axis_point_world_m"] - result["target_axis_point_world_m"], axis=-1
    )
    assert np.allclose(witness_distance, result["gap_m"], atol=1e-12)


def test_rotated_finite_edge_top_contact_is_zero_without_plane_extension():
    # Sphere-axis point outside the finite XY rectangle, but a diagonal
    # upward corner normal creates a legal top-edge contact.
    radius = 0.02
    points = np.array([[0.5 + radius * 0.6 + 0.01, 0, 0.05 + radius * 0.8]])
    controller, sensors = controller_and_sensors(points)
    result = directed_top_gap(controller, sensors, "left")
    assert result["gap_m"][0] == pytest.approx(0, abs=1e-12)
    assert result["normal_up"][0] == pytest.approx(0.8, abs=1e-12)
    assert np.allclose(result["top_point_world_m"][0], [0.5, 0, 0.05])
    angle = 0.7
    quaternion = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
    from scripts.local.ollie_directed_top_gap import _rotate

    shift = np.array([1.2, -0.3, 0.4])
    for prefix in ("left_body_", "right_body_", "board_center_", "board_tail_", "board_nose_"):
        sensors[..., controller.slices[prefix + "pos"]] = (
            _rotate(quaternion, sensors[..., controller.slices[prefix + "pos"]]) + shift
        )
        sensors[..., controller.slices[prefix + "quat"]] = quaternion
    rotated = directed_top_gap(controller, sensors, "left")
    assert np.allclose(rotated["gap_m"], result["gap_m"], atol=1e-12)
    assert np.allclose(rotated["normal_up"], result["normal_up"], atol=1e-12)
    assert np.allclose(
        rotated["top_point_world_m"], _rotate(quaternion, result["top_point_world_m"]) + shift
    )


def test_scalar_batch_empty_and_invalid_geometry_contracts():
    controller, sensors = controller_and_sensors(np.zeros((2, 3, 3)))
    batch = directed_top_gap(controller, sensors, "left")
    scalar = directed_top_gap(controller, sensors[0, 0], "left")
    assert batch["gap_m"].shape == (2, 3)
    assert scalar["gap_m"].shape == ()
    empty = directed_top_gap(controller, sensors[:0], "left")
    assert empty["gap_m"].shape == (0, 3)
    assert empty["axis_point_world_m"].shape == (0, 3, 3)
    with pytest.raises(ValueError, match="finite"):
        segment_rectangle_closest([0, 0, np.nan], [0, 0, 1], [1, 1])
    with pytest.raises(ValueError, match="cosine"):
        segment_rectangle_closest([0, 0, 1], [0, 0, 1], [1, 1], min_normal_cos=1)
