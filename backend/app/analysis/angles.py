"""Convert a `PoseSeries` into per-frame joint angles and derived signals.

This is where raw landmarks become quantities a coach would recognise: knee
angle, hip angle, torso lean, and how high the hips are.

**Scale normalisation is the central idea here.** Landmark coordinates are
fractions of the frame, so a lifter filmed from three metres away produces
smaller numbers than the same lifter filmed from one metre. Any threshold in
raw units would therefore mean different things in different videos. Every
distance-based signal is divided by the subject's own torso length, measured
from the video itself, which makes the outputs comparable across camera
distances, resolutions, and body sizes.

Torso length is the reference rather than, say, leg length or overall height
because it is the one body segment whose projected length barely changes during
a squat — the legs fold, the torso does not.
"""

from __future__ import annotations

import numpy as np

from app.analysis.geometry import (
    angle_from_vertical,
    joint_angle,
    midpoint,
)
from app.analysis.smoothing import smooth_series
from app.analysis.types import (
    REQUIRED_LANDMARKS,
    AngleSeries,
    FramePose,
    Landmark,
    PoseSeries,
)
from app.analysis.types import (
    PoseLandmarkIndex as LM,
)
from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _visible(frame: FramePose, index: LM, threshold: float) -> Landmark | None:
    """Return a landmark only if it was confidently detected."""
    landmark = frame.get(index)
    if landmark is None or landmark.visibility < threshold:
        return None
    return landmark


def _frame_is_usable(frame: FramePose, threshold: float) -> bool:
    """True when every landmark squat analysis needs is confidently visible."""
    return all(
        _visible(frame, index, threshold) is not None for index in REQUIRED_LANDMARKS
    )


def _compute_scales(
    series: PoseSeries, threshold: float
) -> tuple[float | None, float | None]:
    """Measure the subject's torso and thigh length from the clip.

    Both use the median across all usable frames rather than a single reference
    frame, so one badly-tracked moment cannot set the scale for the entire
    analysis. If the scale were wrong, every normalised signal — and therefore
    every depth threshold — would be wrong with it.
    """
    torso_lengths: list[float] = []
    thigh_lengths: list[float] = []

    for frame in series.frames:
        if not _frame_is_usable(frame, threshold):
            continue

        left_shoulder = frame.get(LM.LEFT_SHOULDER)
        right_shoulder = frame.get(LM.RIGHT_SHOULDER)
        left_hip = frame.get(LM.LEFT_HIP)
        right_hip = frame.get(LM.RIGHT_HIP)
        left_knee = frame.get(LM.LEFT_KNEE)
        right_knee = frame.get(LM.RIGHT_KNEE)

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)

        torso_lengths.append(float(np.hypot(*np.subtract(shoulder_mid, hip_mid))))
        thigh_lengths.append(float(np.hypot(*np.subtract(hip_mid, knee_mid))))

    torso = float(np.median(torso_lengths)) if torso_lengths else None
    thigh = float(np.median(thigh_lengths)) if thigh_lengths else None

    # A scale of ~0 means the subject occupies almost no pixels; dividing by it
    # would produce meaningless magnitudes, so treat it as unmeasurable.
    if torso is not None and torso < 1e-6:
        torso = None
    if thigh is not None and thigh < 1e-6:
        thigh = None

    return torso, thigh


def compute_angles(series: PoseSeries, settings: Settings) -> AngleSeries:
    """Build the full `AngleSeries` for a video.

    Raw values are computed per frame, then every signal is gap-filled and
    smoothed with the same window so they remain directly comparable and
    index-aligned.
    """
    threshold = settings.landmark_visibility_threshold
    torso_scale, thigh_scale = _compute_scales(series, threshold)

    result = AngleSeries(torso_scale=torso_scale, thigh_scale=thigh_scale)

    for frame in series.frames:
        result.timestamps_s.append(frame.timestamp_s)
        usable = _frame_is_usable(frame, threshold)
        result.valid.append(usable)

        if not usable:
            result.left_knee_deg.append(None)
            result.right_knee_deg.append(None)
            result.hip_deg.append(None)
            result.torso_lean_deg.append(None)
            result.hip_height.append(None)
            result.hip_knee_offset.append(None)
            continue

        left_shoulder = frame.get(LM.LEFT_SHOULDER)
        right_shoulder = frame.get(LM.RIGHT_SHOULDER)
        left_hip = frame.get(LM.LEFT_HIP)
        right_hip = frame.get(LM.RIGHT_HIP)
        left_knee = frame.get(LM.LEFT_KNEE)
        right_knee = frame.get(LM.RIGHT_KNEE)
        left_ankle = frame.get(LM.LEFT_ANKLE)
        right_ankle = frame.get(LM.RIGHT_ANKLE)

        # Knee: angle at the knee between the thigh and the shin. ~180 when the
        # leg is locked out, falling as the lifter descends.
        result.left_knee_deg.append(joint_angle(left_hip, left_knee, left_ankle))
        result.right_knee_deg.append(joint_angle(right_hip, right_knee, right_ankle))

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)
        ankle_mid = midpoint(left_ankle, right_ankle)

        # Hip: angle at the hip between the torso and the thigh. Closes as the
        # lifter hinges forward and down.
        result.hip_deg.append(_angle_at(shoulder_mid, hip_mid, knee_mid))

        # Torso lean from vertical. 0 is upright; larger means more forward
        # inclination. Only meaningful from a side-on camera.
        result.torso_lean_deg.append(angle_from_vertical(hip_mid, shoulder_mid))

        # Hip height above the ankles, in torso lengths. Image y grows downward,
        # so `ankle_y - hip_y` is positive when the hip is above the ankles.
        if torso_scale:
            result.hip_height.append((ankle_mid[1] - hip_mid[1]) / torso_scale)
        else:
            result.hip_height.append(None)

        # Hip position relative to the knees, in thigh lengths. Positive means
        # the hip has dropped to or below knee level: at or past parallel.
        if thigh_scale:
            result.hip_knee_offset.append((hip_mid[1] - knee_mid[1]) / thigh_scale)
        else:
            result.hip_knee_offset.append(None)

    _smooth_in_place(result, series.metadata.fps, settings)

    logger.info(
        "Angles computed for %d frames (%.1f%% usable, torso_scale=%s)",
        len(result),
        result.valid_fraction * 100,
        f"{torso_scale:.4f}" if torso_scale else "unmeasurable",
    )
    return result


def _angle_at(
    first: tuple[float, float],
    vertex: tuple[float, float],
    second: tuple[float, float],
) -> float | None:
    """Joint angle helper for midpoints rather than landmarks."""
    from app.analysis.geometry import angle_between_points

    return angle_between_points(first, vertex, second)


def _smooth_in_place(series: AngleSeries, fps: float, settings: Settings) -> None:
    """Gap-fill and smooth every signal with identical parameters.

    Using one window for all signals keeps them phase-aligned. If hip height
    were smoothed more heavily than knee angle, the rep boundaries derived from
    the former would drift relative to the depth measured from the latter.
    """
    seconds = settings.smoothing_window_seconds
    max_gap = settings.max_interpolation_gap_frames

    series.left_knee_deg = smooth_series(series.left_knee_deg, fps, seconds, max_gap)
    series.right_knee_deg = smooth_series(series.right_knee_deg, fps, seconds, max_gap)
    series.hip_deg = smooth_series(series.hip_deg, fps, seconds, max_gap)
    series.torso_lean_deg = smooth_series(series.torso_lean_deg, fps, seconds, max_gap)
    series.hip_height = smooth_series(series.hip_height, fps, seconds, max_gap)
    series.hip_knee_offset = smooth_series(series.hip_knee_offset, fps, seconds, max_gap)


def unwrap_landmarks(series: PoseSeries) -> dict:
    """Serialise a `PoseSeries` for archival on disk.

    Stored so an analysis can be recomputed with improved logic without
    re-running pose estimation, which is by far the expensive step.
    """
    return {
        "estimator": series.estimator_name,
        "metadata": {
            "width": series.metadata.width,
            "height": series.metadata.height,
            "fps": series.metadata.fps,
            "frame_count": series.metadata.frame_count,
            "duration_s": series.metadata.duration_s,
        },
        "frames": [
            {
                "i": frame.frame_index,
                "t": round(frame.timestamp_s, 4),
                "d": frame.detected,
                # Rounded to 5 decimals: normalised coordinates carry no
                # meaningful precision beyond that, and it roughly halves the
                # stored size.
                "l": [
                    [
                        round(point.x, 5),
                        round(point.y, 5),
                        round(point.z, 5),
                        round(point.visibility, 3),
                    ]
                    for point in frame.landmarks
                ],
            }
            for frame in series.frames
        ],
    }
