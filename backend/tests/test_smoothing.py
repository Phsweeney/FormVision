"""Tests for gap filling and smoothing.

The behaviours that matter most here are the ones that protect rep detection:
missing frames must not be invented, and smoothing must not shift the signal in
time.
"""

from __future__ import annotations

import math

import pytest

from app.analysis.smoothing import (
    decimate,
    decimation_indices,
    interpolate_gaps,
    moving_average,
    percentile,
    smooth_series,
    window_size_for_fps,
)


class TestWindowSize:
    def test_converts_seconds_to_frames(self):
        assert window_size_for_fps(30.0, 0.2) == 7  # 6 rounded up to odd

    def test_always_odd(self):
        """An even window is off-centre and shifts the signal in time.

        Every rep boundary would move in a consistent direction, biasing all
        timing metrics.
        """
        for fps in (24.0, 25.0, 30.0, 48.0, 60.0, 120.0):
            for seconds in (0.1, 0.15, 0.2, 0.5):
                assert window_size_for_fps(fps, seconds) % 2 == 1

    def test_respects_minimum(self):
        assert window_size_for_fps(30.0, 0.001) == 3

    def test_degenerate_fps(self):
        assert window_size_for_fps(0.0, 0.2) == 3
        assert window_size_for_fps(-5.0, 0.2) == 3


class TestInterpolateGaps:
    def test_fills_a_short_gap_linearly(self):
        result = interpolate_gaps([0.0, None, None, 3.0], max_gap=5)
        assert result == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_leaves_a_long_gap_alone(self):
        """A long dropout is missing data, not something to invent.

        Drawing a straight line across a second of unseen movement could
        fabricate a rep that never happened.
        """
        values = [0.0, None, None, None, None, 5.0]
        assert interpolate_gaps(values, max_gap=2) == values

    def test_does_not_extrapolate_at_the_edges(self):
        result = interpolate_gaps([None, 1.0, 2.0, None], max_gap=5)
        assert result[0] is None
        assert result[-1] is None

    def test_all_missing_is_unchanged(self):
        assert interpolate_gaps([None, None], max_gap=5) == [None, None]

    def test_single_known_value_is_unchanged(self):
        assert interpolate_gaps([None, 1.0, None], max_gap=5) == [None, 1.0, None]

    def test_zero_max_gap_disables_filling(self):
        values = [0.0, None, 2.0]
        assert interpolate_gaps(values, max_gap=0) == values


class TestMovingAverage:
    def test_smooths_a_spike(self):
        values = [1.0, 1.0, 10.0, 1.0, 1.0]
        result = moving_average(values, window=3)
        assert result[2] == pytest.approx(4.0)
        assert result[2] < 10.0

    def test_constant_signal_is_unchanged(self):
        values = [5.0] * 10
        assert moving_average(values, window=5) == pytest.approx(values)

    def test_preserves_length(self):
        for window in (3, 5, 9, 21):
            assert len(moving_average([1.0] * 10, window)) == 10

    def test_does_not_resurrect_missing_frames(self):
        """None must survive smoothing.

        If a smoothed value appeared where the subject was untracked, rep
        detection would run over data that was never observed.
        """
        result = moving_average([1.0, 1.0, None, 1.0, 1.0], window=3)
        assert result[2] is None

    def test_averages_only_present_values(self):
        result = moving_average([2.0, None, 4.0], window=3)
        # Index 0 sees [2.0, (None)] -> 2.0; index 2 sees [(None), 4.0] -> 4.0.
        assert result[0] == pytest.approx(2.0)
        assert result[2] == pytest.approx(4.0)

    def test_introduces_no_phase_shift(self):
        """A symmetric window must not move a peak.

        This is the property that keeps rep boundaries honest.
        """
        values = [math.sin(i * 0.2) for i in range(60)]
        smoothed = moving_average(values, window=5)
        peak_raw = max(range(len(values)), key=lambda i: values[i])
        peak_smoothed = max(
            range(len(smoothed)), key=lambda i: smoothed[i] if smoothed[i] else -9e9
        )
        assert abs(peak_raw - peak_smoothed) <= 1

    def test_window_of_one_is_identity(self):
        values = [1.0, 5.0, 2.0]
        assert moving_average(values, window=1) == values

    def test_empty_input(self):
        assert moving_average([], window=5) == []


class TestSmoothSeries:
    def test_interpolates_then_smooths(self):
        result = smooth_series(
            [0.0, None, 2.0, 3.0, 4.0], fps=30.0, seconds=0.1, max_gap=3
        )
        assert all(value is not None for value in result)

    def test_long_gap_survives_as_none(self):
        values = [0.0] + [None] * 20 + [1.0]
        result = smooth_series(values, fps=30.0, seconds=0.1, max_gap=5)
        assert result[10] is None


class TestPercentile:
    def test_ignores_missing(self):
        assert percentile([0.0, None, 10.0], 50) == pytest.approx(5.0)

    def test_all_missing_returns_none(self):
        assert percentile([None, None], 50) is None

    def test_extremes_are_resistant_to_a_single_outlier(self):
        """Why rep detection uses p10/p90 rather than min/max.

        One mis-tracked frame at an extreme would otherwise define the entire
        range that thresholds are computed from.
        """
        clean = [1.0] * 50 + [2.0] * 50
        with_outlier = clean + [100.0]
        assert percentile(with_outlier, 90) < 10.0
        assert max(with_outlier) == 100.0


class TestDecimate:
    def test_short_series_is_untouched(self):
        values = [1.0, 2.0, 3.0]
        assert decimate(values, max_points=10) == values

    def test_reduces_to_the_cap(self):
        assert len(decimate([float(i) for i in range(5000)], max_points=600)) == 600

    def test_keeps_first_and_last(self):
        result = decimate([float(i) for i in range(1000)], max_points=100)
        assert result[0] == 0.0
        assert result[-1] == 999.0

    def test_returns_only_real_samples(self):
        """Sampling, not averaging: a tooltip must never show an invented value."""
        values = [float(i) for i in range(1000)]
        assert all(value in set(values) for value in decimate(values, 50))

    def test_indices_match_decimate(self):
        """Parallel series must decimate identically or charts desynchronise."""
        values = [float(i) for i in range(1000)]
        indices = decimation_indices(len(values), 100)
        assert [values[i] for i in indices] == decimate(values, 100)
