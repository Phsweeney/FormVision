"""Tests for workout-level metric aggregation.

The recurring theme: unmeasurable must stay unmeasurable. A metric that reports
0 when it means "no data" is worse than one that reports nothing, because a
dashboard renders 0 as a real, and alarming, result.
"""

from __future__ import annotations

import pytest

from app.analysis.angles import compute_angles
from app.analysis.metrics import compute_metrics
from app.analysis.reps import detect_reps
from app.analysis.types import AngleSeries
from app.config import Settings
from tests.synthetic import build_squat_series, build_standing_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


def analyse(series, settings: Settings):
    angles = compute_angles(series, settings)
    reps = detect_reps(angles, settings)
    metrics = compute_metrics(reps, angles, series.metadata.duration_s, settings)
    return angles, reps, metrics


class TestEmptySet:
    def test_no_reps_reports_zero_count_and_no_averages(self, settings):
        """Zero reps is a count. Everything derived from reps is None."""
        _, reps, metrics = analyse(build_standing_series(seconds=4.0), settings)
        assert reps == []
        assert metrics.total_reps == 0
        assert metrics.avg_depth_percent is None
        assert metrics.max_depth_percent is None
        assert metrics.avg_rep_duration_s is None
        assert metrics.reps_per_minute is None

    def test_tracking_quality_is_still_reported(self, settings):
        """Quality describes the video, not the reps, so it survives an empty set."""
        _, _, metrics = analyse(build_standing_series(seconds=4.0), settings)
        assert metrics.tracking_quality == pytest.approx(1.0)

    def test_video_duration_is_preserved(self, settings):
        series = build_standing_series(seconds=4.0)
        _, _, metrics = analyse(series, settings)
        assert metrics.video_duration_s == pytest.approx(series.metadata.duration_s)


class TestCounts:
    @pytest.mark.parametrize("expected", [1, 3, 6])
    def test_total_reps_matches(self, expected, settings):
        _, _, metrics = analyse(build_squat_series(reps=expected), settings)
        assert metrics.total_reps == expected


class TestDepth:
    def test_max_depth_is_the_deepest_rep(self, settings):
        """'Maximum depth' means deepest, i.e. the largest depth percentage."""
        _, reps, metrics = analyse(build_squat_series(reps=4), settings)
        assert metrics.max_depth_percent == pytest.approx(
            max(rep.depth_percent for rep in reps)
        )

    def test_average_depth_is_between_the_extremes(self, settings):
        # Base depth kept below parallel-equivalent so depth_percent does not
        # saturate at 100 and the jitter is actually visible in the output.
        _, reps, metrics = analyse(
            build_squat_series(reps=4, depth_fraction=0.35, depth_jitter=0.10),
            settings,
        )
        depths = [rep.depth_percent for rep in reps]
        assert min(depths) < max(depths)
        assert min(depths) <= metrics.avg_depth_percent <= max(depths)

    def test_depth_percent_saturates_at_full_depth(self, settings):
        """Below parallel is 100%, not 130%.

        Deliberate: the metric answers "did you reach depth", so it caps. The
        consequence, worth knowing, is that depth *consistency* carries no
        information for a lifter who already hits depth on every rep - every
        value is 100 and the deviation is 0.
        """
        _, reps, _ = analyse(
            build_squat_series(reps=3, depth_fraction=1.0, depth_jitter=0.2),
            settings,
        )
        assert all(rep.depth_percent == pytest.approx(100.0) for rep in reps)

    def test_min_knee_angle_is_the_smallest_seen(self, settings):
        _, reps, metrics = analyse(build_squat_series(reps=3), settings)
        assert metrics.min_knee_angle_deg == pytest.approx(
            min(rep.min_knee_angle_deg for rep in reps)
        )

    def test_deeper_set_reports_higher_average_depth(self, settings):
        _, _, shallow = analyse(build_squat_series(reps=3, depth_fraction=0.45), settings)
        _, _, deep = analyse(build_squat_series(reps=3, depth_fraction=1.0), settings)
        assert deep.avg_depth_percent > shallow.avg_depth_percent


class TestTiming:
    def test_average_duration_is_between_fastest_and_slowest(self, settings):
        _, _, metrics = analyse(build_squat_series(reps=5), settings)
        assert metrics.fastest_rep_s <= metrics.avg_rep_duration_s
        assert metrics.avg_rep_duration_s <= metrics.slowest_rep_s

    def test_phases_sum_to_the_average_duration(self, settings):
        _, _, metrics = analyse(build_squat_series(reps=4), settings)
        assert metrics.avg_eccentric_s + metrics.avg_concentric_s == pytest.approx(
            metrics.avg_rep_duration_s
        )

    def test_workout_time_excludes_setup_and_rack_off(self, settings):
        """Working time spans first descent to last lockout, not the whole clip.

        The generator adds standing pauses at both ends, so the two must differ.
        """
        series = build_squat_series(reps=3, standing_pause_s=1.5)
        _, _, metrics = analyse(series, settings)
        assert metrics.total_workout_time_s < metrics.video_duration_s

    def test_reps_per_minute_is_consistent_with_duration(self, settings):
        _, reps, metrics = analyse(build_squat_series(reps=5), settings)
        expected = len(reps) / metrics.total_workout_time_s * 60.0
        assert metrics.reps_per_minute == pytest.approx(expected)

    def test_faster_reps_give_a_higher_rate(self, settings):
        _, _, slow = analyse(build_squat_series(reps=4, rep_duration_s=3.0), settings)
        _, _, fast = analyse(build_squat_series(reps=4, rep_duration_s=1.2), settings)
        assert fast.reps_per_minute > slow.reps_per_minute


class TestConsistency:
    def test_single_rep_has_no_consistency_figure(self, settings):
        """One rep has zero deviation by definition, which is not evidence of
        perfect consistency. None is the honest answer."""
        _, _, metrics = analyse(build_squat_series(reps=1), settings)
        assert metrics.total_reps == 1
        assert metrics.depth_consistency_percent is None
        assert metrics.duration_consistency_s is None

    def test_uniform_reps_are_highly_consistent(self, settings):
        _, _, metrics = analyse(build_squat_series(reps=5, depth_jitter=0.0), settings)
        assert metrics.depth_consistency_percent < 3.0

    def test_varying_reps_are_less_consistent(self, settings):
        _, _, uniform = analyse(
            build_squat_series(reps=6, depth_fraction=0.35, depth_jitter=0.0),
            settings,
        )
        _, _, varied = analyse(
            build_squat_series(reps=6, depth_fraction=0.35, depth_jitter=0.10),
            settings,
        )
        assert varied.depth_consistency_percent > uniform.depth_consistency_percent

    def test_std_not_range(self, settings):
        """Standard deviation reflects the whole set.

        Range would be set by the two extreme reps, making one bad rep in twenty
        indistinguishable from twenty erratic ones.
        """
        _, reps, metrics = analyse(
            build_squat_series(reps=6, depth_fraction=0.35, depth_jitter=0.10),
            settings,
        )
        depths = [rep.depth_percent for rep in reps]
        assert max(depths) > min(depths)  # the jitter is genuinely present
        assert metrics.depth_consistency_percent < (max(depths) - min(depths))


class TestFormMetrics:
    def test_lean_is_aggregated(self, settings):
        _, _, metrics = analyse(
            build_squat_series(reps=3, torso_lean_deg=10.0, bottom_lean_deg=50.0),
            settings,
        )
        assert metrics.max_torso_lean_deg > 40.0
        assert metrics.avg_torso_lean_deg is not None

    def test_max_lean_is_at_least_the_average(self, settings):
        _, _, metrics = analyse(build_squat_series(reps=4), settings)
        assert metrics.max_torso_lean_deg >= metrics.avg_torso_lean_deg

    def test_asymmetry_is_aggregated(self, settings):
        _, _, symmetric = analyse(build_squat_series(reps=3), settings)
        _, _, biased = analyse(build_squat_series(reps=3, left_right_bias=0.3), settings)
        assert biased.avg_knee_asymmetry_deg > symmetric.avg_knee_asymmetry_deg


class TestTrackingQuality:
    def test_clean_video_scores_one(self, settings):
        _, _, metrics = analyse(build_squat_series(reps=3), settings)
        assert metrics.tracking_quality == pytest.approx(1.0)

    def test_dropouts_lower_the_score(self, settings):
        series = build_squat_series(reps=3, undetected_frames=tuple(range(0, 60)))
        _, _, metrics = analyse(series, settings)
        assert metrics.tracking_quality < 1.0

    def test_quality_matches_the_valid_fraction(self, settings):
        series = build_squat_series(reps=3, undetected_frames=tuple(range(10, 40)))
        angles, _, metrics = analyse(series, settings)
        assert metrics.tracking_quality == pytest.approx(angles.valid_fraction)


def test_metrics_with_no_angle_data(settings):
    """Defensive: aggregation over nothing must not raise."""
    metrics = compute_metrics([], AngleSeries(), 0.0, settings)
    assert metrics.total_reps == 0
    assert metrics.tracking_quality == 0.0
