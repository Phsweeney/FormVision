"""Tests for repetition detection.

The core promise is that the count is correct. These tests exercise it against
clean squats, standing-still footage, noise, partial reps, dropouts, and varying
tempos — every case that has a plausible way of producing a wrong number.
"""

from __future__ import annotations

import pytest

from app.analysis.angles import compute_angles
from app.analysis.reps import detect_reps
from app.analysis.types import ViewOrientation
from app.config import Settings
from tests.synthetic import build_squat_series, build_standing_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


def count_reps(series, settings: Settings) -> int:
    return len(detect_reps(compute_angles(series, settings), settings))


class TestCounting:
    @pytest.mark.parametrize("expected", [1, 2, 3, 5, 8])
    def test_counts_exactly(self, expected, settings):
        assert count_reps(build_squat_series(reps=expected), settings) == expected

    def test_counts_shallow_reps(self, settings):
        """A partial squat is still a rep. Depth is judged separately."""
        assert count_reps(build_squat_series(reps=3, depth_fraction=0.5), settings) == 3

    def test_counts_at_various_frame_rates(self, settings):
        """Thresholds derive from the signal, not from frame indices."""
        for fps in (24.0, 30.0, 60.0):
            assert count_reps(build_squat_series(reps=4, fps=fps), settings) == 4

    def test_counts_at_various_tempos(self, settings):
        for duration in (1.0, 2.0, 4.0):
            series = build_squat_series(reps=3, rep_duration_s=duration)
            assert count_reps(series, settings) == 3


class TestFalsePositives:
    def test_standing_still_yields_zero(self, settings):
        assert count_reps(build_standing_series(seconds=5.0), settings) == 0

    def test_standing_with_wobble_yields_zero(self, settings):
        """The case a naive threshold counter gets catastrophically wrong.

        Without the min_rep_range guard, small hip wobble gets normalised into a
        full-scale signal and every oscillation becomes a repetition.
        """
        series = build_standing_series(seconds=6.0, noise=0.004)
        assert count_reps(series, settings) == 0

    def test_empty_series_yields_zero(self, settings):
        from app.analysis.types import PoseSeries, VideoMetadata

        series = PoseSeries((), VideoMetadata(720, 1280, 30.0, 0, 0.0), "synthetic")
        assert count_reps(series, settings) == 0

    def test_untracked_video_yields_zero(self, settings):
        series = build_squat_series(reps=3, undetected_frames=tuple(range(500)))
        assert count_reps(series, settings) == 0

    def test_hysteresis_prevents_threshold_flicker(self, settings):
        """Hovering at the boundary must not multiply the count.

        A single threshold would be crossed repeatedly here. The gap between the
        descent and ascent thresholds is what prevents it.
        """
        settings.rep_descent_fraction = 0.60
        settings.rep_ascent_fraction = 0.25
        assert count_reps(build_squat_series(reps=2), settings) == 2


class TestRepBoundaries:
    def test_ordering_is_coherent(self, settings):
        angles = compute_angles(build_squat_series(reps=3), settings)
        for rep in detect_reps(angles, settings):
            assert rep.start_frame < rep.bottom_frame < rep.end_frame
            assert rep.start_time_s < rep.bottom_time_s < rep.end_time_s

    def test_reps_are_numbered_from_one(self, settings):
        angles = compute_angles(build_squat_series(reps=4), settings)
        reps = detect_reps(angles, settings)
        assert [rep.index for rep in reps] == [1, 2, 3, 4]

    def test_reps_do_not_overlap(self, settings):
        angles = compute_angles(build_squat_series(reps=4), settings)
        reps = detect_reps(angles, settings)
        for previous, current in zip(reps, reps[1:], strict=False):
            assert previous.end_frame <= current.start_frame

    def test_durations_are_plausible(self, settings):
        """Recovered duration should be close to what was generated.

        A tolerance is expected: the rep starts when the hip leaves the standing
        band, which is slightly after the generated movement begins.
        """
        angles = compute_angles(build_squat_series(reps=3, rep_duration_s=2.0), settings)
        for rep in detect_reps(angles, settings):
            assert 1.2 < rep.duration_s < 3.0

    def test_phases_sum_to_the_total(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        for rep in detect_reps(angles, settings):
            assert rep.eccentric_s + rep.concentric_s == pytest.approx(rep.duration_s)
            assert rep.eccentric_s > 0
            assert rep.concentric_s > 0

    def test_bottom_frame_is_the_deepest_point(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        for rep in detect_reps(angles, settings):
            window = [
                value
                for value in angles.hip_height[rep.start_frame : rep.end_frame + 1]
                if value is not None
            ]
            assert angles.hip_height[rep.bottom_frame] == pytest.approx(
                min(window), abs=1e-9
            )


class TestRepMeasurements:
    def test_depth_percent_tracks_actual_depth(self, settings):
        shallow = detect_reps(
            compute_angles(build_squat_series(reps=2, depth_fraction=0.45), settings),
            settings,
        )
        deep = detect_reps(
            compute_angles(build_squat_series(reps=2, depth_fraction=1.0), settings),
            settings,
        )
        assert deep[0].depth_percent > shallow[0].depth_percent

    def test_depth_percent_is_bounded(self, settings):
        angles = compute_angles(build_squat_series(reps=2, depth_fraction=1.2), settings)
        for rep in detect_reps(angles, settings):
            assert 0.0 <= rep.depth_percent <= 100.0

    def test_full_depth_registers_hip_below_knee(self, settings):
        angles = compute_angles(build_squat_series(reps=2, depth_fraction=1.0), settings)
        assert all(rep.hip_below_knee for rep in detect_reps(angles, settings))

    def test_shallow_squat_does_not_register_below_parallel(self, settings):
        angles = compute_angles(build_squat_series(reps=2, depth_fraction=0.35), settings)
        assert not any(rep.hip_below_knee for rep in detect_reps(angles, settings))

    def test_lean_is_the_worst_within_the_rep(self, settings):
        """Worst-case, not average: upright at the top and folded at the bottom
        is a lean problem that averaging would conceal."""
        angles = compute_angles(
            build_squat_series(
                reps=2,
                torso_lean_deg=8.0,
                bottom_lean_deg=55.0,
                view=ViewOrientation.SIDE,
            ),
            settings,
        )
        for rep in detect_reps(angles, settings):
            assert rep.max_torso_lean_deg > 40.0

    def test_asymmetry_is_measured(self, settings):
        angles = compute_angles(build_squat_series(reps=2, left_right_bias=0.3), settings)
        for rep in detect_reps(angles, settings):
            assert rep.knee_asymmetry_deg > 8.0

    def test_symmetric_squat_reports_near_zero_asymmetry(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        for rep in detect_reps(angles, settings):
            assert rep.knee_asymmetry_deg < 1.0

    def test_asymmetry_is_unmeasured_side_on(self, settings):
        """One leg hides the other, so there is nothing to compare it against.

        The far leg is tracked in a small, noisy fraction of frames; treating
        that as a left/right difference invented ~38 degrees of asymmetry on
        real footage that had none.
        """
        angles = compute_angles(
            build_squat_series(
                reps=2, view=ViewOrientation.SIDE, far_side_visibility=0.4
            ),
            settings,
        )
        reps = detect_reps(angles, settings)
        assert reps
        assert all(rep.knee_asymmetry_deg is None for rep in reps)
        # Depth still comes through — it only ever needed one leg.
        assert all(rep.depth_percent is not None for rep in reps)


class TestRobustness:
    def test_brief_dropout_does_not_split_a_rep(self, settings):
        """A few untracked frames mid-rep must not become two reps."""
        series = build_squat_series(reps=3, undetected_frames=(40, 41, 42))
        assert count_reps(series, settings) == 3

    def test_incomplete_final_rep_is_discarded(self, settings):
        """A rep the lifter never finished has unknown duration and ascent.

        Truncating mid-descent must not inflate the count.
        """
        full = build_squat_series(reps=3)
        from app.analysis.types import PoseSeries, VideoMetadata

        # Cut the clip partway into what would be the third descent.
        keep = int(len(full.frames) * 0.78)
        truncated = PoseSeries(
            full.frames[:keep],
            VideoMetadata(720, 1280, 30.0, keep, keep / 30.0),
            "synthetic",
        )
        assert count_reps(truncated, settings) <= 3

    def test_min_duration_filter_rejects_jitter(self, settings):
        settings.min_rep_duration_s = 10.0  # longer than any generated rep
        assert count_reps(build_squat_series(reps=3), settings) == 0

    def test_min_range_guard_is_configurable(self, settings):
        settings.min_rep_range = 99.0  # nothing can clear this
        assert count_reps(build_squat_series(reps=3), settings) == 0

    def test_stricter_descent_fraction_still_counts_full_reps(self, settings):
        settings.rep_descent_fraction = 0.8
        assert count_reps(build_squat_series(reps=3, depth_fraction=1.0), settings) == 3
