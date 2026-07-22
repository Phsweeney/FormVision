"""Repetition detection.

Finds squat repetitions in the hip-height signal produced by `angles.py`.

**The approach: a hysteresis state machine over adaptive thresholds.**

The naive version — count every time the hip drops below some fixed height — has
two fatal problems. First, a fixed height is meaningless: it depends on the
lifter, the camera, and the frame. Second, a single threshold flickers. A hip
hovering right at the boundary crosses it repeatedly and registers a dozen reps
where there was one.

Both are solved here. Thresholds are derived from the range of hip travel *in
this particular video*, so they adapt to the lifter automatically. And there are
two of them — a lower one to enter the descent, a higher one to close the rep —
so the signal must genuinely traverse the gap between them to advance. That gap
is the hysteresis band, and it makes flickering impossible.
"""

from __future__ import annotations

from enum import Enum, auto

from app.analysis.geometry import linear_scale
from app.analysis.smoothing import percentile
from app.analysis.types import AngleSeries, Rep, ViewOrientation
from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class _State(Enum):
    """Where the lifter is in the movement."""

    STANDING = auto()
    DESCENDING = auto()
    ASCENDING = auto()


def detect_reps(angles: AngleSeries, settings: Settings) -> list[Rep]:
    """Find every repetition in the clip.

    Returns an empty list — rather than raising — when the video contains no
    detectable movement. "No reps" is a legitimate outcome that the coaching
    layer reports on; it is not an error.
    """
    if len(angles) == 0:
        return []

    signal = angles.hip_height
    baseline = percentile(signal, 90)
    bottom_reference = percentile(signal, 10)

    if baseline is None or bottom_reference is None:
        logger.info("No usable hip-height signal; reporting zero reps")
        return []

    travel = baseline - bottom_reference

    # Percentiles rather than min/max: one mis-tracked frame at an extreme would
    # otherwise define the entire range that every threshold derives from.
    if travel < settings.min_rep_range:
        logger.info(
            "Hip travel %.3f is below the %.3f minimum; reporting zero reps",
            travel,
            settings.min_rep_range,
        )
        return []

    descend_threshold = baseline - settings.rep_descent_fraction * travel
    ascend_threshold = baseline - settings.rep_ascent_fraction * travel

    candidates = _scan(signal, angles, descend_threshold, ascend_threshold)
    reps = _finalise(candidates, angles, settings)

    logger.info(
        "Detected %d repetitions (travel=%.3f, thresholds %.3f/%.3f)",
        len(reps),
        travel,
        descend_threshold,
        ascend_threshold,
    )
    return reps


def _scan(
    signal: list[float | None],
    angles: AngleSeries,
    descend_threshold: float,
    ascend_threshold: float,
) -> list[tuple[int, int, int]]:
    """Walk the signal and emit ``(start, bottom, end)`` frame triples.

    The state machine:

    - **STANDING** -> DESCENDING once the hip falls below ``descend_threshold``.
      The rep's *start* is backdated to when the hip last left the standing
      band, so the recorded descent includes the whole movement rather than
      only the part below the threshold.
    - **DESCENDING** -> ASCENDING once the hip begins rising again, tracked by
      remembering the lowest point seen.
    - **ASCENDING** -> STANDING once the hip rises back above
      ``ascend_threshold``, which closes the rep.

    Untracked frames (None) do not advance the machine. A rep in progress
    survives a brief dropout rather than being split in two.
    """
    triples: list[tuple[int, int, int]] = []

    state = _State.STANDING
    start_frame = 0
    bottom_frame = 0
    bottom_value = float("inf")
    # Most recent frame at which the lifter was unambiguously standing; used to
    # backdate the start of a descent.
    last_standing_frame = 0

    for index, value in enumerate(signal):
        if value is None:
            continue

        if state is _State.STANDING:
            if value >= ascend_threshold:
                last_standing_frame = index
            elif value < descend_threshold:
                state = _State.DESCENDING
                start_frame = last_standing_frame
                bottom_frame = index
                bottom_value = value

        elif state is _State.DESCENDING:
            if value < bottom_value:
                bottom_value = value
                bottom_frame = index
            elif value > bottom_value:
                # The hip has started back up: the bottom is behind us.
                state = _State.ASCENDING

        elif state is _State.ASCENDING:
            if value < bottom_value:
                # Dipped again before locking out — treat it as still the same
                # descent rather than closing a rep the lifter has not finished.
                bottom_value = value
                bottom_frame = index
                state = _State.DESCENDING
            elif value >= ascend_threshold:
                triples.append((start_frame, bottom_frame, index))
                state = _State.STANDING
                last_standing_frame = index
                bottom_value = float("inf")

    # A rep still in progress at the final frame is deliberately discarded: the
    # lifter never returned to lockout, so its duration and ascent are unknown.
    if state is not _State.STANDING:
        logger.debug("Video ended mid-repetition; incomplete rep discarded")

    return triples


def _finalise(
    triples: list[tuple[int, int, int]],
    angles: AngleSeries,
    settings: Settings,
) -> list[Rep]:
    """Turn frame triples into `Rep` objects, discarding implausible ones."""
    reps: list[Rep] = []

    for start, bottom, end in triples:
        start_time = angles.timestamps_s[start]
        end_time = angles.timestamps_s[end]

        # Anything faster than a person can physically squat is tracking
        # jitter that survived smoothing, not a repetition.
        if end_time - start_time < settings.min_rep_duration_s:
            logger.debug(
                "Discarding %.2fs candidate below the %.2fs minimum",
                end_time - start_time,
                settings.min_rep_duration_s,
            )
            continue

        reps.append(_build_rep(len(reps) + 1, start, bottom, end, angles, settings))

    return reps


def _build_rep(
    index: int,
    start: int,
    bottom: int,
    end: int,
    angles: AngleSeries,
    settings: Settings,
) -> Rep:
    """Measure one repetition.

    Depth is taken at the bottom frame. Lean and asymmetry are taken as the
    worst value across the whole rep, because a lifter who is upright at the top
    and folded at the bottom has a lean problem — averaging would hide it.
    """
    left_at_bottom = _window_min(angles.left_knee_deg, start, end)
    right_at_bottom = _window_min(angles.right_knee_deg, start, end)

    knee_values = [v for v in (left_at_bottom, right_at_bottom) if v is not None]
    min_knee = min(knee_values) if knee_values else None

    # Asymmetry is the worst instantaneous gap between the two knees across the
    # rep, not the gap between each leg's independent minimum. Each leg reaches
    # its deepest point at a slightly different frame, so differencing the two
    # minima cancels a genuine one-sided shift: a lifter visibly loading one leg
    # measured 0.2 degrees of "asymmetry" that way, because both knees passed
    # through the same minimum at different moments. Comparing frame by frame and
    # keeping the worst gap matches the same "worst across the rep" intent used
    # for lean above.
    #
    # Asymmetry needs a front-on camera. Side-on, the far leg is occluded and
    # tracked in a small, noisy fraction of frames — comparing it against the
    # near leg produced ~38 degrees of "asymmetry" on footage with none, which
    # is worse than saying nothing.
    asymmetry = (
        _window_max_gap(angles.left_knee_deg, angles.right_knee_deg, start, end)
        if angles.view is ViewOrientation.FRONT
        else None
    )

    depth_percent = None
    if min_knee is not None:
        depth_percent = (
            linear_scale(
                min_knee,
                settings.standing_knee_angle_deg,
                settings.parallel_knee_angle_deg,
            )
            * 100.0
        )

    offset = _window_max(angles.hip_knee_offset, start, end)

    return Rep(
        index=index,
        start_frame=start,
        bottom_frame=bottom,
        end_frame=end,
        start_time_s=angles.timestamps_s[start],
        bottom_time_s=angles.timestamps_s[bottom],
        end_time_s=angles.timestamps_s[end],
        min_knee_angle_deg=min_knee,
        min_left_knee_deg=left_at_bottom,
        min_right_knee_deg=right_at_bottom,
        min_hip_angle_deg=_window_min(angles.hip_deg, start, end),
        max_torso_lean_deg=_window_max(angles.torso_lean_deg, start, end),
        knee_asymmetry_deg=asymmetry,
        depth_percent=depth_percent,
        hip_below_knee=offset is not None and offset >= 0.0,
    )


def _window_min(values: list[float | None], start: int, end: int) -> float | None:
    """Smallest present value in ``[start, end]``, or None."""
    present = [v for v in values[start : end + 1] if v is not None]
    return min(present) if present else None


def _window_max(values: list[float | None], start: int, end: int) -> float | None:
    """Largest present value in ``[start, end]``, or None."""
    present = [v for v in values[start : end + 1] if v is not None]
    return max(present) if present else None


def _window_max_gap(
    left: list[float | None],
    right: list[float | None],
    start: int,
    end: int,
) -> float | None:
    """Largest ``|left - right|`` over frames in ``[start, end]``, or None.

    Only frames where both sides are present contribute; the difference is taken
    within each frame, so it reflects a same-instant left/right comparison rather
    than the gap between two values measured at different times.
    """
    gaps = [
        abs(a - b)
        for a, b in zip(left[start : end + 1], right[start : end + 1], strict=True)
        if a is not None and b is not None
    ]
    return max(gaps) if gaps else None
