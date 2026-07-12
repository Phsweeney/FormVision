"""Tests for the pure geometric primitives.

Every expected value here is one you can work out on paper — that is the point
of keeping this module free of configuration and I/O. If these fail, nothing
downstream can be trusted.
"""

from __future__ import annotations

import math

import pytest

from app.analysis.geometry import (
    angle_between_points,
    angle_from_vertical,
    clamp,
    joint_angle,
    landmark_distance,
    linear_scale,
    midpoint,
)
from app.analysis.types import Landmark


def lm(x: float, y: float, visibility: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, z=0.0, visibility=visibility)


class TestAngleBetweenPoints:
    @pytest.mark.parametrize(
        ("first", "vertex", "second", "expected"),
        [
            # Right angle: arms along +x and -y from the origin.
            ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), 90.0),
            # Straight line through the vertex.
            ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), 180.0),
            # Both arms in the same direction: fully closed.
            ((1.0, 0.0), (0.0, 0.0), (2.0, 0.0), 0.0),
            # 45 degrees.
            ((1.0, 0.0), (0.0, 0.0), (1.0, 1.0), 45.0),
            # 135 degrees.
            ((1.0, 0.0), (0.0, 0.0), (-1.0, 1.0), 135.0),
        ],
    )
    def test_known_angles(self, first, vertex, second, expected):
        assert angle_between_points(first, vertex, second) == pytest.approx(
            expected, abs=1e-6
        )

    def test_result_is_always_in_zero_to_one_eighty(self):
        """The interior angle is unsigned; mirrored inputs agree."""
        above = angle_between_points((1.0, 0.0), (0.0, 0.0), (1.0, 1.0))
        below = angle_between_points((1.0, 0.0), (0.0, 0.0), (1.0, -1.0))
        assert above == pytest.approx(below)
        assert 0.0 <= above <= 180.0

    def test_degenerate_vector_returns_none(self):
        """A coincident point makes the angle undefined, not zero.

        Returning 0.0 here would be read downstream as a fully-flexed joint,
        which is a wildly different claim from 'not measurable'.
        """
        assert angle_between_points((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)) is None
        assert angle_between_points((1.0, 0.0), (0.0, 0.0), (0.0, 0.0)) is None

    def test_precision_near_straight(self):
        """The acos formulation loses precision exactly here — atan2 does not.

        A locked-out knee sits at ~180 degrees, so this is the operating range
        of a standing lifter, not an edge case.
        """
        angle = angle_between_points((-1.0, 0.0), (0.0, 0.0), (1.0, 1e-7))
        assert angle is not None
        assert angle == pytest.approx(180.0, abs=1e-4)


class TestJointAngle:
    def test_straight_leg_is_one_eighty(self):
        """Hip, knee, ankle collinear: a locked-out leg."""
        angle = joint_angle(lm(0.5, 0.4), lm(0.5, 0.6), lm(0.5, 0.8))
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_right_angle_knee(self):
        """Thigh horizontal, shin vertical: a 90-degree knee."""
        angle = joint_angle(lm(0.3, 0.6), lm(0.5, 0.6), lm(0.5, 0.8))
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_angle_decreases_as_knee_flexes(self):
        """Deeper squat must give a smaller number.

        Guards the sign convention: if this ever inverts, every depth
        measurement in the app silently reverses.
        """
        knee = lm(0.5, 0.6)
        ankle = lm(0.5, 0.8)
        shallow = joint_angle(lm(0.5, 0.4), knee, ankle)
        deep = joint_angle(lm(0.35, 0.5), knee, ankle)
        assert deep < shallow


class TestAngleFromVertical:
    def test_upright_is_zero(self):
        """Note the argument order: (lower, upper) in world terms.

        Image y grows downward, so the 'upper' point has the smaller y.
        """
        assert angle_from_vertical((0.5, 0.6), (0.5, 0.4)) == pytest.approx(0.0)

    def test_horizontal_is_ninety(self):
        assert angle_from_vertical((0.5, 0.6), (0.8, 0.6)) == pytest.approx(90.0)

    def test_forty_five_degrees(self):
        assert angle_from_vertical((0.5, 0.6), (0.7, 0.4)) == pytest.approx(45.0)

    def test_lean_direction_does_not_change_magnitude(self):
        """Which way the lifter faces depends on the camera, not their form."""
        forward = angle_from_vertical((0.5, 0.6), (0.7, 0.4))
        backward = angle_from_vertical((0.5, 0.6), (0.3, 0.4))
        assert forward == pytest.approx(backward)

    def test_zero_length_returns_none(self):
        assert angle_from_vertical((0.5, 0.5), (0.5, 0.5)) is None


class TestHelpers:
    def test_midpoint(self):
        assert midpoint(lm(0.0, 0.0), lm(1.0, 1.0)) == (0.5, 0.5)

    def test_landmark_distance(self):
        assert landmark_distance(lm(0.0, 0.0), lm(3.0, 4.0)) == pytest.approx(5.0)

    def test_clamp(self):
        assert clamp(5.0, 0.0, 1.0) == 1.0
        assert clamp(-5.0, 0.0, 1.0) == 0.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_linear_scale_handles_descending_range(self):
        """Depth maps 170 degrees -> 0 and 90 degrees -> 1, i.e. backwards.

        A scaling helper that assumed an ascending range would break the single
        most important metric in the app.
        """
        assert linear_scale(170.0, 170.0, 90.0) == pytest.approx(0.0)
        assert linear_scale(90.0, 170.0, 90.0) == pytest.approx(1.0)
        assert linear_scale(130.0, 170.0, 90.0) == pytest.approx(0.5)

    def test_linear_scale_clamps_beyond_the_range(self):
        """Squatting below parallel is still 100%, not 130%."""
        assert linear_scale(70.0, 170.0, 90.0) == pytest.approx(1.0)
        assert linear_scale(180.0, 170.0, 90.0) == pytest.approx(0.0)

    def test_linear_scale_without_clamping(self):
        assert linear_scale(70.0, 170.0, 90.0, clamp_result=False) == pytest.approx(1.25)

    def test_linear_scale_zero_span(self):
        assert linear_scale(5.0, 3.0, 3.0) == 0.0


def test_radians_degrees_sanity():
    """Guards against a stray radian/degree mix-up in the module."""
    assert angle_between_points((1, 0), (0, 0), (0, 1)) == pytest.approx(
        math.degrees(math.pi / 2)
    )
