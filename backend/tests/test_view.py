"""Tests for camera view detection.

The classifier exists to stop the analysis reporting numbers a given camera
angle cannot see. These tests pin down both halves: that each orientation is
recognised, and that the decision is made on a scale-free quantity so it does
not quietly depend on how far away the camera stood.
"""

from __future__ import annotations

import pytest

from app.analysis.types import PoseSeries, VideoMetadata, ViewOrientation
from app.analysis.view import detect_view, shoulder_ratio
from app.config import Settings
from tests.synthetic import build_frame, build_squat_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestClassification:
    @pytest.mark.parametrize(
        "view",
        [ViewOrientation.SIDE, ViewOrientation.FRONT, ViewOrientation.OBLIQUE],
    )
    def test_each_orientation_is_recognised(self, view, settings):
        assert detect_view(build_squat_series(reps=2, view=view), settings) is view

    def test_untracked_footage_is_unknown(self, settings):
        """Not a guess and not a default — the question is unanswerable."""
        frames = tuple(build_frame(i, i / 30.0, 0.55, visibility=0.05) for i in range(30))
        series = PoseSeries(frames, VideoMetadata(720, 1280, 30.0, 30, 1.0), "synthetic")
        assert detect_view(series, settings) is ViewOrientation.UNKNOWN

    def test_empty_series_is_unknown(self, settings):
        series = PoseSeries((), VideoMetadata(720, 1280, 30.0, 0, 0.0), "synthetic")
        assert detect_view(series, settings) is ViewOrientation.UNKNOWN


class TestTheRatio:
    def test_side_on_collapses_the_shoulders(self, settings):
        """Turned side-on the two shoulders project onto nearly one point."""
        side = shoulder_ratio(
            build_squat_series(reps=1, view=ViewOrientation.SIDE),
            settings.landmark_visibility_threshold,
        )
        front = shoulder_ratio(
            build_squat_series(reps=1, view=ViewOrientation.FRONT),
            settings.landmark_visibility_threshold,
        )
        assert side < settings.view_side_max_shoulder_ratio
        assert front > settings.view_front_min_shoulder_ratio
        # An order of magnitude apart, which is why a plain threshold suffices.
        assert front > side * 10

    def test_is_independent_of_camera_distance(self, settings):
        """Dividing by the subject's own torso is what makes this hold.

        A raw shoulder separation in normalised units would halve when the
        lifter stood twice as far back, and the classifier would flip.
        """
        series = build_squat_series(reps=1, view=ViewOrientation.FRONT)
        shrunk = _scaled(series, 0.5)

        threshold = settings.landmark_visibility_threshold
        assert shoulder_ratio(shrunk, threshold) == pytest.approx(
            shoulder_ratio(series, threshold), rel=1e-6
        )
        assert detect_view(shrunk, settings) is ViewOrientation.FRONT

    def test_a_deep_rep_does_not_change_the_verdict(self, settings):
        """The torso foreshortens at the bottom, inflating the ratio for those
        frames. Taking the median over the clip is what absorbs it."""
        deep = build_squat_series(
            reps=3,
            depth_fraction=1.0,
            torso_lean_deg=15.0,
            bottom_lean_deg=60.0,
            view=ViewOrientation.SIDE,
        )
        assert detect_view(deep, settings) is ViewOrientation.SIDE


def _scaled(series: PoseSeries, factor: float) -> PoseSeries:
    """The same figure, filmed from further away: shrunk about the frame centre."""
    from dataclasses import replace

    return replace(
        series,
        frames=tuple(
            replace(
                frame,
                landmarks=tuple(
                    replace(
                        landmark,
                        x=0.5 + (landmark.x - 0.5) * factor,
                        y=0.5 + (landmark.y - 0.5) * factor,
                    )
                    for landmark in frame.landmarks
                ),
            )
            for frame in series.frames
        ),
    )
