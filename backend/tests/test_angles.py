"""Tests for the angle computation stage.

These run against the synthetic stick figure, so the expected relationships are
consequences of geometry rather than magic numbers: a deeper hip *must* give a
smaller knee angle, a scaled-up figure *must* give identical normalised output.
"""

from __future__ import annotations

import pytest

from app.analysis.angles import compute_angles
from app.analysis.types import PoseSeries, VideoMetadata
from app.config import Settings
from tests.synthetic import build_frame, build_squat_series, build_standing_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestScaleNormalisation:
    def test_torso_scale_is_measured(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        assert angles.torso_scale is not None
        assert angles.torso_scale > 0
        assert angles.thigh_scale is not None

    def test_camera_distance_does_not_change_normalised_output(self, settings):
        """The whole point of scale normalisation.

        The same squat filmed from further away produces smaller raw
        coordinates. Normalised signals must be unaffected, or every threshold
        in the app would mean something different per video.
        """
        near = build_squat_series(reps=2)

        # Shrink the figure about the frame centre: same movement, further away.
        def shrink(series: PoseSeries, factor: float) -> PoseSeries:
            from app.analysis.types import FramePose, Landmark

            frames = tuple(
                FramePose(
                    frame.frame_index,
                    frame.timestamp_s,
                    tuple(
                        Landmark(
                            0.5 + (point.x - 0.5) * factor,
                            0.5 + (point.y - 0.5) * factor,
                            point.z,
                            point.visibility,
                        )
                        for point in frame.landmarks
                    ),
                    frame.detected,
                )
                for frame in series.frames
            )
            return PoseSeries(frames, series.metadata, series.estimator_name)

        far = shrink(near, 0.5)

        near_angles = compute_angles(near, settings)
        far_angles = compute_angles(far, settings)

        # Raw scale genuinely differs...
        assert far_angles.torso_scale == pytest.approx(
            near_angles.torso_scale * 0.5, rel=1e-3
        )

        # ...but the normalised hip-height signal does not.
        near_hip = [v for v in near_angles.hip_height if v is not None]
        far_hip = [v for v in far_angles.hip_height if v is not None]
        assert far_hip == pytest.approx(near_hip, rel=1e-6)

        # Joint angles are scale-invariant by construction.
        near_knee = [v for v in near_angles.left_knee_deg if v is not None]
        far_knee = [v for v in far_angles.left_knee_deg if v is not None]
        assert far_knee == pytest.approx(near_knee, rel=1e-6)


class TestKneeAngles:
    def test_standing_knees_are_near_extension(self, settings):
        angles = compute_angles(build_standing_series(seconds=2.0), settings)
        knees = [v for v in angles.left_knee_deg if v is not None]
        assert knees
        assert min(knees) > 150.0

    def test_squatting_closes_the_knee(self, settings):
        angles = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)
        knees = [v for v in angles.left_knee_deg if v is not None]
        assert min(knees) < 110.0
        assert max(knees) > 150.0

    def test_deeper_squat_gives_smaller_minimum(self, settings):
        """Direction-of-effect check on the most important signal in the app."""
        shallow = compute_angles(build_squat_series(reps=1, depth_fraction=0.4), settings)
        deep = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)

        shallow_min = min(v for v in shallow.left_knee_deg if v is not None)
        deep_min = min(v for v in deep.left_knee_deg if v is not None)
        assert deep_min < shallow_min

    def test_symmetric_figure_gives_matching_knees(self, settings):
        angles = compute_angles(build_squat_series(reps=1, left_right_bias=0.0), settings)
        for left, right in zip(angles.left_knee_deg, angles.right_knee_deg, strict=True):
            if left is not None and right is not None:
                assert left == pytest.approx(right, abs=1.0)

    def test_biased_figure_gives_diverging_knees(self, settings):
        angles = compute_angles(build_squat_series(reps=1, left_right_bias=0.6), settings)
        differences = [
            abs(left - right)
            for left, right in zip(
                angles.left_knee_deg, angles.right_knee_deg, strict=True
            )
            if left is not None and right is not None
        ]
        assert max(differences) > 3.0


class TestTorsoLean:
    def test_upright_figure_reads_near_zero(self, settings):
        series = build_squat_series(reps=1, torso_lean_deg=0.0, bottom_lean_deg=0.0)
        angles = compute_angles(series, settings)
        leans = [v for v in angles.torso_lean_deg if v is not None]
        assert max(leans) < 2.0

    def test_lean_is_recovered(self, settings):
        series = build_squat_series(reps=1, torso_lean_deg=30.0, bottom_lean_deg=30.0)
        angles = compute_angles(series, settings)
        leans = [v for v in angles.torso_lean_deg if v is not None]
        assert max(leans) == pytest.approx(30.0, abs=2.0)

    def test_increasing_lean_at_the_bottom_is_detected(self, settings):
        series = build_squat_series(reps=1, torso_lean_deg=10.0, bottom_lean_deg=55.0)
        angles = compute_angles(series, settings)
        leans = [v for v in angles.torso_lean_deg if v is not None]
        assert max(leans) > 45.0


class TestHipHeight:
    def test_falls_during_the_descent(self, settings):
        angles = compute_angles(build_squat_series(reps=1), settings)
        heights = [v for v in angles.hip_height if v is not None]
        assert max(heights) > min(heights)

    def test_deeper_squat_travels_further(self, settings):
        shallow = compute_angles(build_squat_series(reps=1, depth_fraction=0.4), settings)
        deep = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)

        def travel(series):
            values = [v for v in series.hip_height if v is not None]
            return max(values) - min(values)

        assert travel(deep) > travel(shallow)

    def test_standing_still_barely_moves(self, settings):
        angles = compute_angles(build_standing_series(seconds=3.0), settings)
        heights = [v for v in angles.hip_height if v is not None]
        assert max(heights) - min(heights) < 0.05

    def test_hip_drops_below_knee_at_full_depth(self, settings):
        """hip_knee_offset >= 0 is the at-or-below-parallel signal."""
        angles = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)
        offsets = [v for v in angles.hip_knee_offset if v is not None]
        assert max(offsets) > -0.1

    def test_shallow_squat_keeps_hip_above_knee(self, settings):
        angles = compute_angles(build_squat_series(reps=1, depth_fraction=0.3), settings)
        offsets = [v for v in angles.hip_knee_offset if v is not None]
        assert max(offsets) < 0.0


class TestTrackingQuality:
    def test_clean_series_is_fully_valid(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        assert angles.valid_fraction == pytest.approx(1.0)

    def test_undetected_frames_are_marked_invalid(self, settings):
        series = build_squat_series(reps=2, undetected_frames=tuple(range(10, 30)))
        angles = compute_angles(series, settings)
        assert angles.valid_fraction < 1.0
        assert angles.valid[15] is False

    def test_low_visibility_landmarks_are_rejected(self, settings):
        """Visibility below threshold is as good as not detected.

        MediaPipe reports a position for occluded joints; using them regardless
        would produce confident-looking angles from guessed coordinates.
        """
        from app.analysis.types import VideoMetadata

        frames = tuple(build_frame(i, i / 30.0, 0.55, visibility=0.1) for i in range(30))
        series = PoseSeries(frames, VideoMetadata(720, 1280, 30.0, 30, 1.0), "synthetic")
        angles = compute_angles(series, settings)
        assert angles.valid_fraction == 0.0
        assert all(value is None for value in angles.left_knee_deg)

    def test_series_lengths_stay_aligned(self, settings):
        """Every parallel list must match the frame count, or the overlay and
        charts desynchronise from the video."""
        series = build_squat_series(reps=3, undetected_frames=(5, 6, 7))
        angles = compute_angles(series, settings)
        n = len(series.frames)
        assert len(angles.timestamps_s) == n
        assert len(angles.left_knee_deg) == n
        assert len(angles.right_knee_deg) == n
        assert len(angles.hip_deg) == n
        assert len(angles.torso_lean_deg) == n
        assert len(angles.hip_height) == n
        assert len(angles.hip_knee_offset) == n
        assert len(angles.valid) == n


class TestEmptyAndDegenerate:
    def test_empty_series_does_not_crash(self, settings):
        series = PoseSeries((), VideoMetadata(720, 1280, 30.0, 0, 0.0), "synthetic")
        angles = compute_angles(series, settings)
        assert len(angles) == 0
        assert angles.valid_fraction == 0.0

    def test_never_detected_yields_no_scale(self, settings):
        frames = tuple(build_frame(i, i / 30.0, 0.5, detected=False) for i in range(30))
        series = PoseSeries(frames, VideoMetadata(720, 1280, 30.0, 30, 1.0), "synthetic")
        angles = compute_angles(series, settings)
        assert angles.torso_scale is None
        assert all(value is None for value in angles.hip_height)

    def test_mean_knee_handles_one_missing_side(self, settings):
        angles = compute_angles(build_squat_series(reps=1), settings)
        angles.right_knee_deg[0] = None
        mean = angles.mean_knee_deg
        assert mean[0] == pytest.approx(angles.left_knee_deg[0])
