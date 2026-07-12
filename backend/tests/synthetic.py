"""Synthetic pose data for testing.

Builds a `PoseSeries` describing a mathematically-known squat: a stick figure,
viewed from the side, whose hips and knees follow a controlled trajectory. The
number of reps, the depth of each one, and the amount of forward lean are all
inputs, so a test can assert the analysis recovers exactly what was put in.

This is what the `PoseEstimator` abstraction buys. Every module downstream of
pose estimation is tested against these figures — no video files, no model
download, no MediaPipe, and the suite runs in about a second.

The figure is anatomically simplified but geometrically consistent: segment
lengths are fixed, and the knee is placed by solving the two-link leg so the
hip sits at the requested height. That means the knee angles the analysis
computes are real consequences of the geometry, not values written in by hand.
"""

from __future__ import annotations

import math

from app.analysis.pose.base import PoseEstimator
from app.analysis.types import (
    FramePose,
    Landmark,
    PoseSeries,
    VideoMetadata,
)
from app.analysis.types import (
    PoseLandmarkIndex as LM,
)

#: Normalised segment lengths for the synthetic figure. Chosen so a standing
#: figure occupies most of the frame, as real footage tends to.
THIGH = 0.18
SHIN = 0.18
TORSO = 0.22
SHOULDER_WIDTH = 0.09
HIP_WIDTH = 0.07

#: Where the feet sit vertically in the frame (image coordinates, y down).
ANKLE_Y = 0.92
CENTER_X = 0.5

_LANDMARK_COUNT = 33


def _leg_geometry(
    hip: tuple[float, float],
    ankle: tuple[float, float],
    knee_forward: float,
) -> tuple[float, float]:
    """Place the knee by solving the two-link leg between hip and ankle.

    Standard two-circle intersection: the knee lies at distance ``THIGH`` from
    the hip and ``SHIN`` from the ankle. Of the two solutions, the one on the
    forward side is chosen, because that is the direction a knee bends.

    When hip and ankle are further apart than the legs can span — a locked-out
    stance, possibly pushed over the limit by rounding — the intersection
    degenerates and the knee is placed on the hip-ankle line with a slight
    forward offset, which is what a real leg looks like at lockout.

    Returns the knee's ``(x, y)``.
    """
    hip_x, hip_y = hip
    ankle_x, ankle_y = ankle

    dx, dy = ankle_x - hip_x, ankle_y - hip_y
    span = math.hypot(dx, dy)

    if span < 1e-9 or span >= THIGH + SHIN:
        scale = THIGH / span if span > 1e-9 else 0.0
        return (hip_x + dx * scale + knee_forward * 0.02, hip_y + dy * scale)

    # Distance from the hip, along the hip->ankle axis, to the foot of the
    # perpendicular that reaches the knee.
    along = (span**2 + THIGH**2 - SHIN**2) / (2 * span)
    perpendicular = math.sqrt(max(THIGH**2 - along**2, 0.0))

    ux, uy = dx / span, dy / span
    base_x, base_y = hip_x + ux * along, hip_y + uy * along

    # Rotate the unit vector 90 degrees; sign chosen so positive `knee_forward`
    # pushes the knee toward +x.
    sign = 1.0 if knee_forward >= 0 else -1.0
    perp_x, perp_y = uy * sign, -ux * sign
    magnitude = perpendicular * min(abs(knee_forward), 1.0)

    return (base_x + perp_x * magnitude, base_y + perp_y * magnitude)


def build_frame(
    frame_index: int,
    timestamp_s: float,
    hip_y: float,
    torso_lean_deg: float = 10.0,
    knee_forward: float = 1.0,
    left_right_bias: float = 0.0,
    hip_setback: float = 0.0,
    visibility: float = 0.95,
    detected: bool = True,
) -> FramePose:
    """Construct one frame of the synthetic figure.

    Args:
        hip_y: Hip height in image coordinates. Smaller = standing taller.
        torso_lean_deg: Forward inclination of the torso from vertical.
        knee_forward: Direction/magnitude of knee travel; +1 is forward.
        left_right_bias: Asymmetry, as a fraction of a thigh length. Displaces
            the left foot, which changes that leg's hip-to-ankle span and so
            genuinely changes its knee angle — the geometric signature of a
            lifter shifting their weight onto one side.
        hip_setback: How far the hips have travelled backward from the ankles.
            This is what allows the hip to descend below knee level: with the
            hip stacked directly over the ankle, a symmetric two-link leg always
            places the knee at the midpoint, so depth past parallel is
            geometrically impossible. Real lifters sit back; so does this figure.
        visibility: Confidence written onto every landmark.
        detected: When False, the frame carries no landmarks at all.
    """
    if not detected:
        return FramePose(frame_index, timestamp_s, (), detected=False)

    points = [Landmark(CENTER_X, 0.5, 0.0, visibility) for _ in range(_LANDMARK_COUNT)]

    def place(index: LM, x: float, y: float) -> None:
        points[index] = Landmark(x, y, 0.0, visibility)

    hip_x = CENTER_X - hip_setback

    # Torso: shoulders sit one torso-length up, inclined forward by the
    # requested lean angle.
    lean_rad = math.radians(torso_lean_deg)
    shoulder_x = hip_x + TORSO * math.sin(lean_rad)
    shoulder_y = hip_y - TORSO * math.cos(lean_rad)

    place(LM.NOSE, shoulder_x, shoulder_y - 0.08)
    place(LM.LEFT_SHOULDER, shoulder_x - SHOULDER_WIDTH / 2, shoulder_y)
    place(LM.RIGHT_SHOULDER, shoulder_x + SHOULDER_WIDTH / 2, shoulder_y)
    place(LM.LEFT_ELBOW, shoulder_x - SHOULDER_WIDTH, shoulder_y + 0.09)
    place(LM.RIGHT_ELBOW, shoulder_x + SHOULDER_WIDTH, shoulder_y + 0.09)
    place(LM.LEFT_WRIST, shoulder_x - SHOULDER_WIDTH, shoulder_y + 0.18)
    place(LM.RIGHT_WRIST, shoulder_x + SHOULDER_WIDTH, shoulder_y + 0.18)

    place(LM.LEFT_HIP, hip_x - HIP_WIDTH / 2, hip_y)
    place(LM.RIGHT_HIP, hip_x + HIP_WIDTH / 2, hip_y)

    # Each leg is solved independently so `left_right_bias` produces a genuinely
    # different knee angle rather than a cosmetic offset.
    for side_hip, side_knee, side_ankle, bias in (
        (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, left_right_bias),
        (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, 0.0),
    ):
        offset = -HIP_WIDTH / 2 if side_hip == LM.LEFT_HIP else HIP_WIDTH / 2
        # The bias displaces the foot, altering the hip-to-ankle span for that
        # leg and therefore the angle its knee must adopt to bridge it.
        ankle_x = CENTER_X + offset + bias * THIGH
        ankle_y = ANKLE_Y
        knee_x, knee_y = _leg_geometry(
            (hip_x + offset, hip_y), (ankle_x, ankle_y), knee_forward
        )
        place(side_knee, knee_x, knee_y)
        place(side_ankle, ankle_x, ankle_y)

    place(LM.LEFT_HEEL, CENTER_X - HIP_WIDTH / 2 - 0.01, ANKLE_Y + 0.02)
    place(LM.RIGHT_HEEL, CENTER_X + HIP_WIDTH / 2 - 0.01, ANKLE_Y + 0.02)
    place(LM.LEFT_FOOT_INDEX, CENTER_X - HIP_WIDTH / 2 + 0.05, ANKLE_Y + 0.03)
    place(LM.RIGHT_FOOT_INDEX, CENTER_X + HIP_WIDTH / 2 + 0.05, ANKLE_Y + 0.03)

    return FramePose(frame_index, timestamp_s, tuple(points), detected=True)


def build_squat_series(
    reps: int = 3,
    fps: float = 30.0,
    rep_duration_s: float = 2.0,
    standing_pause_s: float = 0.5,
    depth_fraction: float = 1.0,
    torso_lean_deg: float = 12.0,
    bottom_lean_deg: float | None = None,
    left_right_bias: float = 0.0,
    depth_jitter: float = 0.0,
    undetected_frames: tuple[int, ...] = (),
    width: int = 720,
    height: int = 1280,
) -> PoseSeries:
    """Build a full synthetic squat clip.

    The hip follows a raised-cosine descent and ascent — smooth, with zero
    velocity at the top and bottom, which is how a controlled squat actually
    moves and avoids the discontinuities a triangle wave would introduce.

    Args:
        reps: How many repetitions to generate.
        depth_fraction: 1.0 descends to full depth; 0.5 is a half squat.
        bottom_lean_deg: Lean at the bottom, interpolated from
            ``torso_lean_deg`` at the top. Used to test the forward-lean rule.
        depth_jitter: Per-rep variation in depth, to test the consistency rule.
        undetected_frames: Frame indices where the subject is not found, to
            test gap handling.
    """
    standing_hip_y = ANKLE_Y - (THIGH + SHIN) * 0.97
    # Full depth places the hip a little below knee level.
    full_depth_hip_y = ANKLE_Y - THIGH * 0.85
    # How far the hips travel backward at full depth. Without this the hip
    # cannot get below the knees (see `build_frame`'s `hip_setback`).
    full_depth_setback = THIGH * 0.78

    frames: list[FramePose] = []
    frame_index = 0

    def emit(hip_y: float, lean: float, setback: float) -> None:
        nonlocal frame_index
        frames.append(
            build_frame(
                frame_index,
                frame_index / fps,
                hip_y,
                torso_lean_deg=lean,
                left_right_bias=left_right_bias,
                hip_setback=setback,
                detected=frame_index not in undetected_frames,
            )
        )
        frame_index += 1

    pause_frames = max(1, int(standing_pause_s * fps))
    rep_frames = max(2, int(rep_duration_s * fps))
    top_lean = torso_lean_deg
    low_lean = bottom_lean_deg if bottom_lean_deg is not None else torso_lean_deg

    # Lead-in standing period so rep detection has a clear baseline.
    for _ in range(pause_frames):
        emit(standing_hip_y, top_lean, 0.0)

    for rep in range(reps):
        rep_depth = depth_fraction + (depth_jitter * (1 if rep % 2 else -1))
        rep_depth = max(0.05, min(1.2, rep_depth))
        bottom_hip_y = standing_hip_y + (full_depth_hip_y - standing_hip_y) * rep_depth
        bottom_setback = full_depth_setback * rep_depth

        for step in range(rep_frames):
            # Raised cosine: 0 at the top, 1 at the bottom, 0 again at lockout.
            phase = 0.5 * (1 - math.cos(2 * math.pi * step / rep_frames))
            hip_y = standing_hip_y + (bottom_hip_y - standing_hip_y) * phase
            emit(
                hip_y,
                top_lean + (low_lean - top_lean) * phase,
                bottom_setback * phase,
            )

        for _ in range(pause_frames):
            emit(standing_hip_y, top_lean, 0.0)

    return PoseSeries(
        frames=tuple(frames),
        metadata=VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=len(frames),
            duration_s=len(frames) / fps,
        ),
        estimator_name="synthetic",
    )


def build_standing_series(
    seconds: float = 4.0, fps: float = 30.0, noise: float = 0.0
) -> PoseSeries:
    """A clip of someone standing still. Must yield zero reps.

    ``noise`` adds small random-free wobble (a deterministic sine) to the hip
    height, which is the case that naive threshold-based rep counting fails:
    without a minimum-range guard it amplifies jitter into dozens of reps.
    """
    standing_hip_y = ANKLE_Y - (THIGH + SHIN) * 0.97
    count = int(seconds * fps)
    frames = [
        build_frame(
            i,
            i / fps,
            standing_hip_y + noise * math.sin(i * 0.7),
            torso_lean_deg=8.0,
        )
        for i in range(count)
    ]
    return PoseSeries(
        frames=tuple(frames),
        metadata=VideoMetadata(720, 1280, fps, count, count / fps),
        estimator_name="synthetic",
    )


class SyntheticPoseEstimator(PoseEstimator):
    """A `PoseEstimator` that ignores the video and returns a canned series.

    Registered by the API tests so the full HTTP flow — upload, analyze, poll,
    fetch results — can be exercised without a real video or MediaPipe.
    """

    name = "synthetic"

    def __init__(self, series: PoseSeries | None = None) -> None:
        self._series = series or build_squat_series()

    def estimate(self, video_path) -> PoseSeries:  # noqa: ARG002 - path unused
        return self._series
