"""Signal smoothing and gap filling for per-frame time series.

Raw landmark output jitters frame to frame. Rep detection compares a signal
against thresholds, so untreated jitter around a threshold produces phantom
repetitions — smoothing is not cosmetic here, it is what makes the rep counter
trustworthy.

All functions operate on ``list[float | None]``, where None means "not measured
in this frame". They preserve list length so every series stays index-aligned
with the video frames.
"""

from __future__ import annotations

import numpy as np


def window_size_for_fps(fps: float, seconds: float, minimum: int = 3) -> int:
    """Convert a duration into an odd frame count.

    Odd so the window is symmetric about its centre and introduces no phase
    shift — an even window would offset every rep boundary by half a frame in a
    consistent direction, which quietly biases every timing metric.
    """
    if fps <= 0 or seconds <= 0:
        return minimum
    size = int(round(fps * seconds))
    size = max(size, minimum)
    if size % 2 == 0:
        size += 1
    return size


def interpolate_gaps(values: list[float | None], max_gap: int) -> list[float | None]:
    """Linearly fill runs of None no longer than ``max_gap``.

    Short dropouts — a limb briefly occluded — are worth bridging so a rep is
    not split in two. Long dropouts are genuinely missing data and are left as
    None, because inventing a straight line across a second of unseen movement
    would fabricate a rep that may not have happened.

    Leading and trailing gaps are never filled: with data on only one side
    there is nothing to interpolate between, and extrapolating a squat's
    trajectory is guesswork.
    """
    if max_gap <= 0:
        return list(values)

    result = list(values)
    known = [i for i, value in enumerate(result) if value is not None]
    if len(known) < 2:
        return result

    for left, right in zip(known, known[1:], strict=False):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap:
            continue
        start_value = result[left]
        end_value = result[right]
        assert start_value is not None and end_value is not None  # noqa: S101
        step = (end_value - start_value) / (gap + 1)
        for offset in range(1, gap + 1):
            result[left + offset] = start_value + step * offset

    return result


def moving_average(values: list[float | None], window: int) -> list[float | None]:
    """Centred moving average that tolerates missing values.

    Each output is the mean of the present values within the window centred on
    that index. Positions that were None stay None — smoothing must not
    resurrect a frame in which the subject was not tracked, or rep detection
    would run over invented data.

    Windows are truncated at the ends rather than padded. Padding with edge
    values flattens the first and last few frames, which is where a rep often
    starts or finishes.
    """
    if window <= 1 or not values:
        return list(values)

    half = window // 2
    array = np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )
    present = ~np.isnan(array)

    # Cumulative sums give an O(n) sliding window regardless of window size.
    # nan entries contribute zero to the sum and zero to the count, so the mean
    # is automatically taken over present values only.
    filled = np.where(present, array, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0.0], np.cumsum(present.astype(float))))

    n = len(array)
    indices = np.arange(n)
    starts = np.maximum(indices - half, 0)
    ends = np.minimum(indices + half + 1, n)

    window_sums = sums[ends] - sums[starts]
    window_counts = counts[ends] - counts[starts]

    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(window_counts > 0, window_sums / window_counts, np.nan)

    # Restore the original None mask.
    averaged = np.where(present, averaged, np.nan)
    return [None if np.isnan(value) else float(value) for value in averaged]


def smooth_series(
    values: list[float | None],
    fps: float,
    seconds: float,
    max_gap: int,
) -> list[float | None]:
    """Fill short gaps, then smooth. The standard treatment for every signal.

    Order matters: interpolating first means the moving average sees a
    continuous signal across brief dropouts instead of averaging over a hole.
    """
    return moving_average(
        interpolate_gaps(values, max_gap), window_size_for_fps(fps, seconds)
    )


def percentile(values: list[float | None], q: float) -> float | None:
    """Percentile of the present values, or None if there are none.

    Rep detection uses the 10th and 90th percentiles rather than min and max to
    establish the range of hip travel, because a single mis-tracked frame at
    either extreme would otherwise define the whole scale.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(np.percentile(present, q))


def decimate(values: list[float | None], max_points: int) -> list[float | None]:
    """Reduce a series to at most ``max_points`` by uniform sampling.

    Used only when shaping API responses. Sampling rather than averaging keeps
    every returned point a real measurement, so a chart tooltip never shows a
    value that did not occur.
    """
    if max_points <= 0 or len(values) <= max_points:
        return list(values)
    indices = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    return [values[i] for i in indices]


def decimation_indices(length: int, max_points: int) -> list[int]:
    """Indices `decimate` would keep, so parallel series stay aligned."""
    if max_points <= 0 or length <= max_points:
        return list(range(length))
    return [int(i) for i in np.linspace(0, length - 1, max_points).round().astype(int)]
