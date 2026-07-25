"""Pure geometric primitives.

Every function here takes numbers and returns numbers. No configuration, no
logging, no I/O, no project types beyond `Landmark`. That makes this the one
module in the analysis stack that can be verified against values you can work
out by hand, which is exactly what `tests/test_geometry.py` does.

**Coordinate convention.** Landmarks use image coordinates: ``x`` increases to
the right, ``y`` increases *downward*. This trips people up constantly — a
point with a larger ``y`` is *lower* in the frame. Functions that care about
real-world "up" account for it internally and say so in their docstring.
"""

from __future__ import annotations

import math

from app.analysis.types import Landmark

#: Vectors shorter than this (in normalised units) are treated as degenerate.
#: Landmarks that coincide produce a zero-length vector, and the angle between
#: a zero vector and anything is undefined rather than zero.
_EPSILON = 1e-9


def midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    """Midpoint of two landmarks, as ``(x, y)``."""
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def landmark_distance(a: Landmark, b: Landmark) -> float:
    """Euclidean distance between two landmarks in the image plane."""
    return math.hypot(a.x - b.x, a.y - b.y)


def angle_between_points(
    first: tuple[float, float],
    vertex: tuple[float, float],
    second: tuple[float, float],
) -> float | None:
    """Interior angle at ``vertex``, in degrees, within ``[0, 180]``.

    Computed from the two vectors radiating out of the vertex. Returns None if
    either vector is degenerate, since the angle is genuinely undefined rather
    than zero — propagating a fake 0 degrees would look like a fully-bent joint.

    Uses ``atan2`` of the cross and dot products rather than ``acos`` of the
    normalised dot product. Both are correct in exact arithmetic, but the acos
    form loses precision badly near 0 and 180 degrees — precisely the range a
    locked-out knee sits in — and can push its argument outside [-1, 1] through
    rounding, raising a domain error.
    """
    ax, ay = first[0] - vertex[0], first[1] - vertex[1]
    bx, by = second[0] - vertex[0], second[1] - vertex[1]

    if math.hypot(ax, ay) < _EPSILON or math.hypot(bx, by) < _EPSILON:
        return None

    cross = ax * by - ay * bx
    dot = ax * bx + ay * by
    return math.degrees(math.atan2(abs(cross), dot))


def joint_angle(first: Landmark, vertex: Landmark, second: Landmark) -> float | None:
    """Interior angle at the ``vertex`` landmark, in degrees.

    For a knee, call ``joint_angle(hip, knee, ankle)``: a straight leg gives
    ~180 degrees and the value falls as the joint flexes.
    """
    return angle_between_points(
        (first.x, first.y), (vertex.x, vertex.y), (second.x, second.y)
    )


def angle_from_vertical(
    lower: tuple[float, float], upper: tuple[float, float]
) -> float | None:
    """Tilt of the ``lower -> upper`` segment away from vertical, in degrees.

    Returns 0 for a perfectly upright segment, increasing as it tilts in either
    direction, capped at 90 for horizontal. Always non-negative — this measures
    *how much* lean, not which way, because the direction depends on which side
    the camera is on and is not a form judgement.

    Args:
        lower: The lower point in the world, e.g. the hip midpoint.
        upper: The upper point in the world, e.g. the shoulder midpoint.

    Note the ``y`` inversion: because image ``y`` grows downward, an upright
    torso has ``upper.y < lower.y``, so the vertical component is computed as
    ``lower_y - upper_y`` to come out positive.
    """
    horizontal = upper[0] - lower[0]
    vertical = lower[1] - upper[1]  # positive when `upper` really is higher

    if math.hypot(horizontal, vertical) < _EPSILON:
        return None

    # atan2(horizontal, vertical) rather than the usual (y, x) ordering:
    # measuring from the vertical axis, not the horizontal one.
    angle = math.degrees(math.atan2(abs(horizontal), abs(vertical)))
    return angle


def horizontal_offset_from_line(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    """Signed horizontal gap from ``point`` to the ``start -> end`` line.

    The line is sampled at the point's *own* height, so the result answers "how
    far sideways is the knee from where the hip-to-ankle line puts it", which is
    the quantity a coach means by a knee tracking inward. Positive means the
    point lies to the right of the line in image coordinates (larger ``x``);
    callers decide what that means anatomically.

    Returns None when ``start`` and ``end`` are at the same height, since there
    is then no line to interpolate along. That is not a pathological case: it is
    exactly what a frame cropped mid-shin, or a badly tracked ankle, produces.

    Args:
        point: The point to measure, e.g. the knee.
        start: One end of the reference line, e.g. the hip.
        end: The other end, e.g. the ankle.
    """
    vertical_span = end[1] - start[1]
    if abs(vertical_span) < _EPSILON:
        return None

    # Where along start -> end the point's height falls. Deliberately not
    # clamped to [0, 1]: a knee above the hip or below the ankle is a tracking
    # failure, and extrapolating gives a large offset that the visibility gates
    # upstream will have already suppressed anyway.
    ratio = (point[1] - start[1]) / vertical_span
    expected_x = start[0] + ratio * (end[0] - start[0])
    return point[0] - expected_x


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def linear_scale(
    value: float, from_value: float, to_value: float, clamp_result: bool = True
) -> float:
    """Map ``value`` onto ``[0, 1]`` where ``from_value`` is 0 and ``to_value`` is 1.

    Works whether ``to_value`` is above or below ``from_value``, which matters
    for depth: knee angles *decrease* as the squat deepens, so the mapping runs
    from 170 degrees (0%) down to 90 degrees (100%).
    """
    span = to_value - from_value
    if abs(span) < _EPSILON:
        return 0.0
    result = (value - from_value) / span
    return clamp(result, 0.0, 1.0) if clamp_result else result
