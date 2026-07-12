"""Workout-level metric aggregation.

Reduces per-rep measurements to the summary numbers shown on the dashboard.

Two conventions run through this module:

**Missing stays missing.** Any metric that cannot be computed is None, never 0.
A set with no reps has no average depth; reporting 0% would read as "you squatted
terribly" rather than "there was nothing to measure".

**Consistency is a standard deviation, not a range.** Range is decided by the two
most extreme reps, so one bad rep in twenty looks identical to twenty erratic
ones. Standard deviation reflects the whole set, which is the question a lifter
is actually asking.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.analysis.types import AngleSeries, Metrics, Rep
from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _std(values: Sequence[float]) -> float | None:
    """Population standard deviation, or None when a single rep makes it
    meaningless.

    One rep has zero deviation by definition, which would be reported as perfect
    consistency. That is not a claim the data supports, so it is None instead.
    """
    return float(np.std(values)) if len(values) >= 2 else None


def _collect(reps: Sequence[Rep], attribute: str) -> list[float]:
    """Present values of ``attribute`` across reps, skipping the unmeasured."""
    return [
        value for value in (getattr(rep, attribute) for rep in reps) if value is not None
    ]


def compute_metrics(
    reps: Sequence[Rep],
    angles: AngleSeries,
    video_duration_s: float,
    settings: Settings,  # noqa: ARG001 - kept for interface stability
) -> Metrics:
    """Aggregate per-rep measurements into the dashboard summary.

    ``settings`` is unused today but kept in the signature so future
    configurable aggregation (trimmed means, per-exercise weighting) does not
    change every call site.
    """
    tracking_quality = angles.valid_fraction

    if not reps:
        logger.info("No repetitions to aggregate")
        return Metrics(
            total_reps=0,
            video_duration_s=video_duration_s,
            total_workout_time_s=0.0,
            tracking_quality=tracking_quality,
        )

    depths = _collect(reps, "depth_percent")
    durations = [rep.duration_s for rep in reps]
    eccentrics = [rep.eccentric_s for rep in reps]
    concentrics = [rep.concentric_s for rep in reps]
    leans = _collect(reps, "max_torso_lean_deg")
    asymmetries = _collect(reps, "knee_asymmetry_deg")
    knee_angles = _collect(reps, "min_knee_angle_deg")

    # Working time spans the first descent to the last lockout, deliberately
    # excluding setup and rack-off time at either end of the clip.
    total_workout_time = reps[-1].end_time_s - reps[0].start_time_s

    reps_per_minute = None
    if total_workout_time > 0:
        reps_per_minute = len(reps) / total_workout_time * 60.0

    metrics = Metrics(
        total_reps=len(reps),
        video_duration_s=video_duration_s,
        total_workout_time_s=total_workout_time,
        # "Maximum depth" is the *deepest* rep, which is the largest depth
        # percentage and therefore the smallest knee angle.
        max_depth_percent=max(depths) if depths else None,
        avg_depth_percent=_mean(depths),
        min_knee_angle_deg=min(knee_angles) if knee_angles else None,
        avg_rep_duration_s=_mean(durations),
        fastest_rep_s=min(durations),
        slowest_rep_s=max(durations),
        avg_eccentric_s=_mean(eccentrics),
        avg_concentric_s=_mean(concentrics),
        reps_per_minute=reps_per_minute,
        avg_torso_lean_deg=_mean(leans),
        max_torso_lean_deg=max(leans) if leans else None,
        avg_knee_asymmetry_deg=_mean(asymmetries),
        depth_consistency_percent=_std(depths),
        duration_consistency_s=_std(durations),
        tracking_quality=tracking_quality,
    )

    logger.info(
        "Metrics: %d reps, avg depth %s, avg duration %.2fs, quality %.0f%%",
        metrics.total_reps,
        f"{metrics.avg_depth_percent:.0f}%" if metrics.avg_depth_percent else "n/a",
        metrics.avg_rep_duration_s or 0.0,
        tracking_quality * 100,
    )
    return metrics
