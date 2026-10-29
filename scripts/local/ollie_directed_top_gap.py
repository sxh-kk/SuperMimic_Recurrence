"""Directed finite deck-top proximity for the seven original foot capsules.

No simulation or collision model is changed. In each box's authored frame a
capsule axis is a finite segment and its top is a finite rectangle. Segment to
rectangle distance is solved exactly, including an optional upward normal cone.
All reported gaps are geometry guidance, never evidence of physical support.
"""

from __future__ import annotations

import numpy as np

BOXES = ("center", "tail", "nose")
SCHEMA = "supermimic.ollie.directed_finite_top_gap.v1"


def _rotate(quaternion, vector):
    q = np.asarray(quaternion)
    xyz = q[..., 1:]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + q[..., :1] * vector)


def segment_rectangle_closest(start, end, half_xy, *, top_z=0.0, min_normal_cos=None):
    """Exact finite-segment distance to a horizontal finite rectangle.

    Inputs broadcast to (...,3), (...,3), (...,2), and (...). If specified, the
    closest-point direction from rectangle to axis must satisfy n.z>=the cosine
    and have nonnegative z. The closed cone is used for distance minimization;
    a force observer's strict angle rule still requires its own evaluation.

    Proof: x/y rectangle crossings divide the segment into at most five pieces.
    Distance squared on each piece is quadratic. A constrained minimum occurs
    at an endpoint, a stationary point, a z=0 point, or a quadratic cone-boundary
    root. All such candidates are evaluated; no projected-point sampling occurs.
    """
    start, end, half_xy = (np.asarray(value, dtype=np.float64) for value in (start, end, half_xy))
    if start.shape[-1:] != (3,) or end.shape[-1:] != (3,) or half_xy.shape[-1:] != (2,):
        raise ValueError("Expected segment XYZ endpoints and rectangle XY half sizes")
    shape = np.broadcast_shapes(
        start.shape[:-1], end.shape[:-1], half_xy.shape[:-1], np.shape(top_z)
    )
    a, b = (np.broadcast_to(value, (*shape, 3)) for value in (start, end))
    half = np.broadcast_to(half_xy, (*shape, 2))
    height = np.broadcast_to(np.asarray(top_z, dtype=np.float64), shape)
    if not all(np.isfinite(value).all() for value in (a, b, half, height)) or (half <= 0).any():
        raise ValueError("Geometry must be finite and rectangle half sizes positive")
    if min_normal_cos is not None and not 0 < min_normal_cos < 1:
        raise ValueError("The upward normal cosine must lie strictly between0 and1")
    d = b - a
    zero, one = np.zeros(shape), np.ones(shape)
    crossings = [zero, one]
    for axis in (0, 1):
        for sign in (-1, 1):
            value = np.divide(
                sign * half[..., axis] - a[..., axis],
                d[..., axis],
                out=np.zeros(shape),
                where=d[..., axis] != 0,
            )
            crossings.append(np.clip(value, 0, 1))
    boundaries = np.sort(np.stack(crossings, -1), -1)
    lower, upper = boundaries[..., :-1], boundaries[..., 1:]
    middle = (lower + upper) / 2
    middle_point = a[..., None, :] + middle[..., None] * d[..., None, :]
    active_xy = abs(middle_point[..., :2]) > half[..., None, :]
    intercept_xy = np.where(
        active_xy,
        a[..., None, :2] - np.sign(middle_point[..., :2]) * half[..., None, :],
        0,
    )
    slope_xy = np.where(active_xy, d[..., None, :2], 0)
    z_intercept = np.broadcast_to((a[..., 2] - height)[..., None], lower.shape)
    z_slope = np.broadcast_to(d[..., 2, None], lower.shape)
    quadratic = np.sum(slope_xy**2, -1) + z_slope**2
    linear = np.sum(slope_xy * intercept_xy, -1) + z_slope * z_intercept
    stationary = np.divide(-linear, quadratic, out=middle.copy(), where=quadratic > 0)
    candidates = [boundaries, np.clip(stationary, lower, upper)]
    if min_normal_cos is not None:
        scale = 1 / min_normal_cos**2 - 1
        qa = np.sum(slope_xy**2, -1) - scale * z_slope**2
        qb = np.sum(slope_xy * intercept_xy, -1) - scale * z_slope * z_intercept
        qc = np.sum(intercept_xy**2, -1) - scale * z_intercept**2
        epsilon = 32 * np.finfo(np.float64).eps
        tiny = np.finfo(np.float64).tiny
        tolerance = epsilon * (abs(qa) + abs(qb) + abs(qc) + tiny)
        is_quadratic = abs(qa) > tolerance
        discriminant = qb**2 - qa * qc
        real = discriminant >= -epsilon * (qb**2 + abs(qa * qc) + tiny)
        # The q-form avoids losing a small root when B and sqrt(D) cancel.
        qvalue = -qb - np.where(qb >= 0, 1, -1) * np.sqrt(np.maximum(discriminant, 0))
        first = np.divide(qvalue, qa, out=np.full_like(qa, np.nan), where=is_quadratic & real)
        second = np.divide(
            qc, qvalue, out=np.full_like(qc, np.nan), where=is_quadratic & real & (qvalue != 0)
        )
        linear_root = np.divide(
            -qc, 2 * qb, out=np.full_like(qc, np.nan), where=~is_quadratic & (abs(qb) > tolerance)
        )
        first = np.where(is_quadratic, first, linear_root)
        for root in (first, second):
            inside = (root >= lower - 1e-12) & (root <= upper + 1e-12)
            candidates.append(np.where(inside, np.clip(root, lower, upper), np.nan))
        zroot = np.divide(
            height - a[..., 2], d[..., 2], out=np.full(shape, np.nan), where=d[..., 2] != 0
        )
        candidates.append(np.where((zroot >= 0) & (zroot <= 1), zroot, np.nan)[..., None])
    fraction = np.concatenate(candidates, -1)
    point = a[..., None, :] + fraction[..., None] * d[..., None, :]
    target = np.concatenate(
        (
            np.clip(point[..., :2], -half[..., None, :], half[..., None, :]),
            np.broadcast_to(height[..., None, None], (*point.shape[:-1], 1)),
        ),
        -1,
    )
    difference = point - target
    distance = np.linalg.norm(difference, axis=-1)
    cosine = np.divide(
        difference[..., 2], distance, out=np.ones_like(distance), where=distance > 1e-15
    )
    allowed = np.isfinite(fraction)
    if min_normal_cos is not None:
        allowed &= (difference[..., 2] >= -1e-12) & (cosine >= min_normal_cos - 1e-12)
    objective = np.where(allowed, distance, np.inf)
    selected = np.argmin(objective, -1)
    has = allowed.any(-1)

    def take(value):
        if value.ndim == len(shape) + 1:
            return np.take_along_axis(value, selected[..., None], -1)[..., 0]
        return np.take_along_axis(value, selected[..., None, None], -2)[..., 0, :]

    axis_point, top_point = take(point), take(target)
    nearest_distance = take(objective)
    normal = np.divide(
        axis_point - top_point,
        nearest_distance[..., None],
        out=np.zeros((*shape, 3)),
        where=(has & (nearest_distance > 1e-15))[..., None],
    )
    normal = np.where((has & (nearest_distance <= 1e-15))[..., None], [0, 0, 1], normal)
    return {
        "distance_m": nearest_distance,
        "feasible": has,
        "segment_fraction": np.where(has, take(fraction), np.nan),
        "axis_point": np.where(has[..., None], axis_point, np.nan),
        "top_point": np.where(has[..., None], top_point, np.nan),
        "normal": np.where(has[..., None], normal, np.nan),
        "normal_cosine": np.where(has, normal[..., 2], np.nan),
    }


def directed_top_proximity(controller, sensors, side, *, min_normal_cos=0.5, chunk_size=2048):
    """Return finite directed gap and concrete closest-pair evidence per sample.

    ``gap_m`` is abs(axis-to-top distance minus capsule radius) when an upward
    cone point exists. Otherwise it is distance from the axis to the rectangle
    offset upward by that radius: an explicit legal vertical-contact witness
    and an upper bound on the translation needed to reach that witness. It is
    positive for side/below configurations, including ordinary distance-zero
    contacts on a box side or underside. It is never a support certificate.

    Only original seven capsule axes/radii and center/tail/nose top rectangles
    participate. Evidence points and normal are world coordinates; selected
    capsule/box IDs identify the original collision geometry.
    """
    values = np.asarray(sensors)
    if values.ndim < 1 or side not in ("left", "right") or chunk_size < 1:
        raise ValueError("Expected sensor samples, one foot side and positive chunk size")
    if not np.isfinite(values).all():
        raise ValueError("Directed geometry requires finite observations")
    shape, flat = values.shape[:-1], values.reshape(-1, values.shape[-1])
    foot, sl = controller.foot_geometry[side], controller.slices
    if len(foot["center"]) != 7:
        raise ValueError("Expected the original seven foot capsules")
    if not len(flat):
        # An empty selection is useful when no native airborne opportunity exists.
        vectors = {
            name: np.empty((*shape, 3))
            for name in (
                "axis_point_world_m",
                "top_point_world_m",
                "capsule_point_world_m",
                "target_axis_point_world_m",
                "normal_world",
            )
        }
        scalars = {
            name: np.empty(shape)
            for name in (
                "gap_m",
                "signed_gap_m",
                "normal_cosine",
                "normal_up",
                "segment_fraction",
                "undirected_gap_m",
                "undirected_normal_cosine",
            )
        }
        return {
            **vectors,
            **scalars,
            "cone_feasible": np.empty(shape, bool),
            "capsule_index": np.empty(shape, int),
            "capsule_geom_id": np.empty(shape, int),
            "box_index": np.empty(shape, int),
            "box_geom_id": np.empty(shape, int),
        }
    pieces = []
    for begin in range(0, len(flat), chunk_size):
        batch = flat[begin : begin + chunk_size]
        fq = batch[:, sl[side + "_body_quat"]]
        fp = batch[:, sl[side + "_body_pos"]]
        center = fp[:, None, :] + _rotate(fq[:, None, :], foot["center"])
        axis = _rotate(fq[:, None, :], foot["axis"])
        first = center - axis * foot["size"][:, 1, None]
        second = center + axis * foot["size"][:, 1, None]
        radius = foot["size"][:, 0]
        boxes = []
        for box_index, box in enumerate(BOXES):
            name = "board_" + box
            geom = controller.model.geom(name).id
            bp, bq = batch[:, sl[name + "_pos"]], batch[:, sl[name + "_quat"]]
            inverse = bq[:, None, :] * [1, -1, -1, -1]
            start, end = (_rotate(inverse, point - bp[:, None, :]) for point in (first, second))
            size = controller.model.geom_size[geom]
            nearest = segment_rectangle_closest(
                start, end, size[:2], top_z=size[2], min_normal_cos=min_normal_cos
            )
            ordinary = segment_rectangle_closest(start, end, size[:2], top_z=size[2])
            fallback = segment_rectangle_closest(start, end, size[:2], top_z=size[2] + radius)
            feasible = nearest["feasible"]
            signed_gap = nearest["distance_m"] - radius
            gap = np.where(feasible, abs(signed_gap), fallback["distance_m"])
            # Fallback's target is an axis-center target, not the deck surface.
            target = fallback["top_point"].copy()
            target[..., 2] -= radius
            axis_point = np.where(
                feasible[..., None], nearest["axis_point"], fallback["axis_point"]
            )
            top_point = np.where(feasible[..., None], nearest["top_point"], target)
            displacement = axis_point - top_point
            magnitude = np.linalg.norm(displacement, axis=-1)
            fallback_normal = np.divide(
                displacement,
                magnitude[..., None],
                out=np.zeros_like(displacement),
                where=magnitude[..., None] > 1e-15,
            )
            normal = np.where(feasible[..., None], nearest["normal"], fallback_normal)
            target_normal = np.where(feasible[..., None], nearest["normal"], [0, 0, 1])
            world_normal = _rotate(bq[:, None, :], normal)
            axis_world = bp[:, None, :] + _rotate(bq[:, None, :], axis_point)
            top_world = bp[:, None, :] + _rotate(bq[:, None, :], top_point)
            boxes.append(
                {
                    "gap_m": gap,
                    "signed_gap_m": np.where(feasible, signed_gap, np.nan),
                    "normal_cosine": normal[..., 2],
                    "normal_up": normal[..., 2],
                    "cone_feasible": feasible,
                    "segment_fraction": np.where(
                        feasible, nearest["segment_fraction"], fallback["segment_fraction"]
                    ),
                    "axis_point_world_m": axis_world,
                    "top_point_world_m": top_world,
                    "capsule_point_world_m": axis_world - radius[None, :, None] * world_normal,
                    "target_axis_point_world_m": top_world
                    + radius[None, :, None] * _rotate(bq[:, None, :], target_normal),
                    "normal_world": world_normal,
                    "undirected_gap_m": ordinary["distance_m"] - radius,
                    "undirected_normal_cosine": ordinary["normal_cosine"],
                    "box_index": np.full(gap.shape, box_index, int),
                    "box_geom_id": np.full(gap.shape, geom, int),
                    "capsule_index": np.broadcast_to(np.arange(7), gap.shape),
                    "capsule_geom_id": np.broadcast_to(np.asarray(foot["geoms"]), gap.shape),
                }
            )
        combined = {name: np.concatenate([box[name] for box in boxes], axis=1) for name in boxes[0]}
        index = combined["gap_m"].argmin(1)
        result = {}
        for name, value in combined.items():
            result[name] = value[np.arange(len(batch)), index]
        pieces.append(result)
    return {
        name: np.concatenate([piece[name] for piece in pieces], 0).reshape(
            (*shape, *pieces[0][name].shape[1:])
        )
        for name in pieces[0]
    }


directed_top_gap = directed_top_proximity
