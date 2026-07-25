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

from dataclasses import dataclass

import numpy as np

from app.analysis.geometry import (
    angle_from_vertical,
    horizontal_offset_from_line,
    joint_angle,
)
from app.analysis.smoothing import smooth_series
from app.analysis.types import (
    CORE_LANDMARKS,
    LEFT_ANKLE_ANGLE_LANDMARKS,
    LEFT_LEG_LANDMARKS,
    RIGHT_ANKLE_ANGLE_LANDMARKS,
    RIGHT_LEG_LANDMARKS,
    AngleSeries,
    FramePose,
    Landmark,
    PoseSeries,
    ViewOrientation,
)
from app.analysis.types import (
    PoseLandmarkIndex as LM,
)
from app.analysis.view import detect_view
from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _visible(frame: FramePose, index: LM, threshold: float) -> Landmark | None:
    """Return a landmark only if it was confidently detected."""
    landmark = frame.get(index)
    if landmark is None or landmark.visibility < threshold:
        return None
    return landmark


def _group_visible(frame: FramePose, group: tuple[LM, ...], threshold: float) -> bool:
    """True when every landmark in ``group`` is confidently visible."""
    return all(_visible(frame, index, threshold) is not None for index in group)


def _frame_is_usable(frame: FramePose, threshold: float) -> bool:
    """True when there is enough of the body to analyse the frame.

    The core torso points, plus **at least one** complete leg — not both.
    Requiring both is what made side-on footage unusable: the far leg is hidden
    behind the near one, so MediaPipe reports low confidence for it and ~83% of
    an otherwise clean clip was discarded. One leg is all a squat needs, and
    side-on it is the only one there is.
    """
    return _group_visible(frame, CORE_LANDMARKS, threshold) and (
        _group_visible(frame, LEFT_LEG_LANDMARKS, threshold)
        or _group_visible(frame, RIGHT_LEG_LANDMARKS, threshold)
    )


def _pair_point(
    frame: FramePose, left: LM, right: LM, threshold: float
) -> tuple[float, float] | None:
    """Midpoint of a left/right landmark pair, or the one side that is visible.

    Falling back to a single side is not an approximation forced on us — filmed
    side-on the two landmarks project to nearly the same image point, so the
    visible one *is* the midpoint to within the noise. It is what lets every
    downstream signal keep working on footage where only one side is tracked.
    """
    left_landmark = _visible(frame, left, threshold)
    right_landmark = _visible(frame, right, threshold)

    if left_landmark is not None and right_landmark is not None:
        return (
            (left_landmark.x + right_landmark.x) / 2.0,
            (left_landmark.y + right_landmark.y) / 2.0,
        )
    if left_landmark is not None:
        return (left_landmark.x, left_landmark.y)
    if right_landmark is not None:
        return (right_landmark.x, right_landmark.y)
    return None


def _medial_knee_offset(
    hip: Landmark,
    knee: Landmark,
    ankle: Landmark,
    medial_sign: float,
    torso_scale: float,
) -> float | None:
    """How far the knee has travelled inward off its own hip-to-ankle line.

    Positive is medial (valgus, the knee collapsing toward the midline) and
    negative is lateral, whichever way the lifter happens to be facing. Scaled
    by torso length like every other distance in this module, so the threshold
    that judges it means the same thing at any camera distance.
    """
    offset = horizontal_offset_from_line(
        (knee.x, knee.y), (hip.x, hip.y), (ankle.x, ankle.y)
    )
    if offset is None:
        return None
    return offset * medial_sign / torso_scale


def _medial_sign(frame: FramePose, threshold: float) -> float | None:
    """Which image direction counts as medial for the *left* leg, as +1 or -1.

    Derived per frame from where the two hips sit rather than assumed, because
    MediaPipe labels landmarks anatomically: a lifter facing away from the
    camera has their left hip on the left of the image, and one facing the
    camera has it on the right. Hard-coding either would silently invert the
    valgus sign for half of all clips.
    """
    left_hip = _visible(frame, LM.LEFT_HIP, threshold)
    right_hip = _visible(frame, LM.RIGHT_HIP, threshold)
    if left_hip is None or right_hip is None:
        return None

    separation = right_hip.x - left_hip.x
    # Filmed side-on the hips project onto each other and there is no left-right
    # axis to speak of. The caller gates this to front-on footage anyway; this
    # guard is what stops a near-zero separation from picking a sign at random.
    if abs(separation) < 1e-6:
        return None
    return 1.0 if separation > 0 else -1.0


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

        shoulder_mid = _pair_point(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, threshold)
        hip_mid = _pair_point(frame, LM.LEFT_HIP, LM.RIGHT_HIP, threshold)
        knee_mid = _pair_point(frame, LM.LEFT_KNEE, LM.RIGHT_KNEE, threshold)

        if shoulder_mid is None or hip_mid is None or knee_mid is None:
            continue

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
    view = detect_view(series, settings)

    result = AngleSeries(torso_scale=torso_scale, thigh_scale=thigh_scale, view=view)

    # Filmed front-on, the torso hinges almost directly toward the lens, so its
    # inclination barely projects into the image at all — real footage measures
    # around 1 degree however hard the lifter folds. That is not a good torso
    # position, it is the absence of a measurement, so it is recorded as one.
    lean_is_measurable = view is not ViewOrientation.FRONT

    # The two new signal families split along the same axis, for the same
    # reason. Ankle travel is a sagittal movement and shares torso lean's gate.
    # Knee valgus is a frontal-plane movement and is the mirror image: side-on,
    # the knee projects onto its own hip-to-ankle line however far it collapses.
    # Valgus takes the stricter test of the two because a wrongly-signed or
    # projection-flattened reading would accuse a lifter of a fault they do not
    # have, and silence is the cheaper error.
    ankle_is_measurable = lean_is_measurable
    valgus_is_measurable = view is ViewOrientation.FRONT

    for frame in series.frames:
        result.timestamps_s.append(frame.timestamp_s)
        usable = _frame_is_usable(frame, threshold)
        left_leg = usable and _group_visible(frame, LEFT_LEG_LANDMARKS, threshold)
        right_leg = usable and _group_visible(frame, RIGHT_LEG_LANDMARKS, threshold)

        result.valid.append(usable)
        result.left_leg_valid.append(left_leg)
        result.right_leg_valid.append(right_leg)

        if not usable:
            result.left_knee_deg.append(None)
            result.right_knee_deg.append(None)
            result.hip_deg.append(None)
            result.torso_lean_deg.append(None)
            result.hip_height.append(None)
            result.hip_knee_offset.append(None)
            result.left_hip_deg.append(None)
            result.right_hip_deg.append(None)
            result.left_ankle_deg.append(None)
            result.right_ankle_deg.append(None)
            result.left_knee_lateral.append(None)
            result.right_knee_lateral.append(None)
            continue

        # Knee: angle at the knee between the thigh and the shin. ~180 when the
        # leg is locked out, falling as the lifter descends. Measured per leg
        # from that leg's own landmarks, so an occluded side goes missing on its
        # own rather than taking the frame down with it.
        result.left_knee_deg.append(
            joint_angle(
                frame.get(LM.LEFT_HIP), frame.get(LM.LEFT_KNEE), frame.get(LM.LEFT_ANKLE)
            )
            if left_leg
            else None
        )
        result.right_knee_deg.append(
            joint_angle(
                frame.get(LM.RIGHT_HIP),
                frame.get(LM.RIGHT_KNEE),
                frame.get(LM.RIGHT_ANKLE),
            )
            if right_leg
            else None
        )

        shoulder_mid = _pair_point(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, threshold)
        hip_mid = _pair_point(frame, LM.LEFT_HIP, LM.RIGHT_HIP, threshold)
        knee_mid = _pair_point(frame, LM.LEFT_KNEE, LM.RIGHT_KNEE, threshold)
        ankle_mid = _pair_point(frame, LM.LEFT_ANKLE, LM.RIGHT_ANKLE, threshold)

        # Hip: angle at the hip between the torso and the thigh. Closes as the
        # lifter hinges forward and down.
        result.hip_deg.append(_angle_at(shoulder_mid, hip_mid, knee_mid))

        # Torso lean from vertical. 0 is upright; larger means more forward
        # inclination. Only meaningful from a side-on camera.
        result.torso_lean_deg.append(
            angle_from_vertical(hip_mid, shoulder_mid) if lean_is_measurable else None
        )

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

        _append_per_side_signals(
            result,
            frame,
            threshold=threshold,
            left_leg=left_leg,
            right_leg=right_leg,
            torso_scale=torso_scale,
            ankle_is_measurable=ankle_is_measurable,
            valgus_is_measurable=valgus_is_measurable,
        )

    _smooth_in_place(result, series.metadata.fps, settings)

    logger.info(
        "Angles computed for %d frames (%.1f%% usable, legs L/R %.0f%%/%.0f%%, "
        "view=%s, torso_scale=%s)",
        len(result),
        result.valid_fraction * 100,
        result.left_leg_coverage * 100,
        result.right_leg_coverage * 100,
        view.value,
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


@dataclass(frozen=True, slots=True)
class _Side:
    """One half of the body, so the per-side maths is written once."""

    shoulder: LM
    hip: LM
    knee: LM
    ankle: LM
    ankle_group: tuple[LM, ...]
    #: Sign relating this side to `_medial_sign`, which is defined for the left.
    medial_orientation: float


_LEFT_SIDE = _Side(
    shoulder=LM.LEFT_SHOULDER,
    hip=LM.LEFT_HIP,
    knee=LM.LEFT_KNEE,
    ankle=LM.LEFT_ANKLE,
    ankle_group=LEFT_ANKLE_ANGLE_LANDMARKS,
    medial_orientation=1.0,
)
_RIGHT_SIDE = _Side(
    shoulder=LM.RIGHT_SHOULDER,
    hip=LM.RIGHT_HIP,
    knee=LM.RIGHT_KNEE,
    ankle=LM.RIGHT_ANKLE,
    ankle_group=RIGHT_ANKLE_ANGLE_LANDMARKS,
    medial_orientation=-1.0,
)


def _side_signals(
    frame: FramePose,
    side: _Side,
    *,
    threshold: float,
    leg_visible: bool,
    torso_scale: float | None,
    medial_sign: float | None,
    ankle_is_measurable: bool,
) -> tuple[float | None, float | None, float | None]:
    """Hip angle, ankle angle, and medial knee offset for one side of one frame.

    Every one of the three is independently `None`-able. A frame can easily
    yield a hip angle but no ankle angle, because the foot left the bottom of
    the shot, and that must cost the ankle angle only.
    """
    hip = _visible(frame, side.hip, threshold)
    knee = _visible(frame, side.knee, threshold)
    ankle = _visible(frame, side.ankle, threshold)
    shoulder = _visible(frame, side.shoulder, threshold)

    hip_deg: float | None = None
    if leg_visible and shoulder is not None and hip is not None and knee is not None:
        hip_deg = joint_angle(shoulder, hip, knee)

    ankle_deg: float | None = None
    if ankle_is_measurable and _group_visible(frame, side.ankle_group, threshold):
        foot = frame.get(side.ankle_group[-1])
        if knee is not None and ankle is not None and foot is not None:
            ankle_deg = joint_angle(knee, ankle, foot)

    knee_lateral: float | None = None
    if (
        leg_visible
        and medial_sign is not None
        and torso_scale
        and hip is not None
        and knee is not None
        and ankle is not None
    ):
        knee_lateral = _medial_knee_offset(
            hip, knee, ankle, medial_sign * side.medial_orientation, torso_scale
        )

    return hip_deg, ankle_deg, knee_lateral


def _append_per_side_signals(
    result: AngleSeries,
    frame: FramePose,
    *,
    threshold: float,
    left_leg: bool,
    right_leg: bool,
    torso_scale: float | None,
    ankle_is_measurable: bool,
    valgus_is_measurable: bool,
) -> None:
    """Append this frame's per-side hip, ankle, and valgus signals."""
    medial_sign = _medial_sign(frame, threshold) if valgus_is_measurable else None

    left = _side_signals(
        frame,
        _LEFT_SIDE,
        threshold=threshold,
        leg_visible=left_leg,
        torso_scale=torso_scale,
        medial_sign=medial_sign,
        ankle_is_measurable=ankle_is_measurable,
    )
    right = _side_signals(
        frame,
        _RIGHT_SIDE,
        threshold=threshold,
        leg_visible=right_leg,
        torso_scale=torso_scale,
        medial_sign=medial_sign,
        ankle_is_measurable=ankle_is_measurable,
    )

    result.left_hip_deg.append(left[0])
    result.left_ankle_deg.append(left[1])
    result.left_knee_lateral.append(left[2])
    result.right_hip_deg.append(right[0])
    result.right_ankle_deg.append(right[1])
    result.right_knee_lateral.append(right[2])


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
    series.left_hip_deg = smooth_series(series.left_hip_deg, fps, seconds, max_gap)
    series.right_hip_deg = smooth_series(series.right_hip_deg, fps, seconds, max_gap)
    series.left_ankle_deg = smooth_series(series.left_ankle_deg, fps, seconds, max_gap)
    series.right_ankle_deg = smooth_series(series.right_ankle_deg, fps, seconds, max_gap)
    series.left_knee_lateral = smooth_series(
        series.left_knee_lateral, fps, seconds, max_gap
    )
    series.right_knee_lateral = smooth_series(
        series.right_knee_lateral, fps, seconds, max_gap
    )


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
