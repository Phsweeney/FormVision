"""Core analysis data types.

These types are the contract between every stage of the pipeline. Crucially,
nothing here imports MediaPipe, OpenCV, SQLAlchemy, or FastAPI — they are plain
dataclasses over plain floats.

That is deliberate. Because `PoseSeries` is framework-free, every module
downstream of pose estimation (angles, reps, metrics, feedback) can be tested
by constructing one by hand, with no video file, no model download, and no
CV dependency. It is also what makes the estimator swappable: a webcam stream
or an ML model in V2 only has to produce this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class PoseLandmarkIndex(IntEnum):
    """Indices into the 33-point MediaPipe Pose topology.

    Named here so analysis code reads ``LEFT_KNEE`` instead of ``25``. Only the
    points V1 actually uses are named; the full 33 are still carried in the
    data so future features do not require re-running estimation.
    """

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


#: Landmarks that must be visible for a frame to be usable for squat analysis.
REQUIRED_LANDMARKS: tuple[PoseLandmarkIndex, ...] = (
    PoseLandmarkIndex.LEFT_SHOULDER,
    PoseLandmarkIndex.RIGHT_SHOULDER,
    PoseLandmarkIndex.LEFT_HIP,
    PoseLandmarkIndex.RIGHT_HIP,
    PoseLandmarkIndex.LEFT_KNEE,
    PoseLandmarkIndex.RIGHT_KNEE,
    PoseLandmarkIndex.LEFT_ANKLE,
    PoseLandmarkIndex.RIGHT_ANKLE,
)

#: Pairs of landmark indices to connect when drawing a skeleton.
#: MediaPipe 0.10.35 removed `mp.solutions.pose.POSE_CONNECTIONS`, so the
#: topology is declared here rather than imported.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    # Arms
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Legs
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)


class Severity(StrEnum):
    """How a piece of coaching feedback should be presented."""

    GOOD = "good"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Landmark:
    """A single detected body point.

    ``x`` and ``y`` are normalised to [0, 1] against frame width and height, so
    they are resolution independent — this is what lets pose inference run on a
    downscaled frame while the overlay draws at full resolution.

    Note ``y`` increases *downward* (image convention). Analysis code that cares
    about "higher in the world" must therefore negate or subtract; the helpers
    in ``geometry.py`` and ``angles.py`` handle this so callers do not repeat it.
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


@dataclass(frozen=True, slots=True)
class FramePose:
    """All landmarks detected in one video frame.

    ``detected`` is False when the estimator found no person at all, which is
    different from finding one with low confidence — the pipeline treats the
    two differently when deciding whether a gap can be interpolated.
    """

    frame_index: int
    timestamp_s: float
    landmarks: tuple[Landmark, ...]
    detected: bool = True

    def get(self, index: PoseLandmarkIndex | int) -> Landmark | None:
        """Return a landmark by index, or None if this frame lacks it."""
        if not self.detected or index >= len(self.landmarks):
            return None
        return self.landmarks[index]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Physical properties of the source video."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class PoseSeries:
    """Per-frame pose data for an entire video.

    This is the output of the `PoseEstimator` interface and the input to every
    analysis module.
    """

    frames: tuple[FramePose, ...]
    metadata: VideoMetadata
    estimator_name: str = "unknown"

    @property
    def detected_frame_count(self) -> int:
        return sum(1 for frame in self.frames if frame.detected)

    @property
    def detection_rate(self) -> float:
        """Fraction of frames in which a person was found at all."""
        if not self.frames:
            return 0.0
        return self.detected_frame_count / len(self.frames)


@dataclass(slots=True)
class AngleSeries:
    """Per-frame joint angles and derived signals.

    Lists are parallel and all the same length as the source `PoseSeries`.
    ``None`` marks a frame where the value could not be computed (subject not
    tracked, or a required landmark below the visibility threshold), which is
    distinct from a computed value of zero.
    """

    timestamps_s: list[float] = field(default_factory=list)
    left_knee_deg: list[float | None] = field(default_factory=list)
    right_knee_deg: list[float | None] = field(default_factory=list)
    hip_deg: list[float | None] = field(default_factory=list)
    torso_lean_deg: list[float | None] = field(default_factory=list)

    #: Vertical hip position, in torso lengths above the ankles. Larger = more
    #: upright. Scale-normalised so it does not depend on camera distance.
    hip_height: list[float | None] = field(default_factory=list)

    #: Signed hip-to-knee vertical offset in thigh lengths. >= 0 means the hip
    #: has dropped to or below knee level, i.e. at or past parallel.
    hip_knee_offset: list[float | None] = field(default_factory=list)

    #: True where every landmark needed for squat analysis was visible.
    valid: list[bool] = field(default_factory=list)

    #: Median torso length in normalised units; the scale reference for the
    #: whole clip. None when the subject was never tracked well enough.
    torso_scale: float | None = None
    thigh_scale: float | None = None

    def __len__(self) -> int:
        return len(self.timestamps_s)

    @property
    def valid_fraction(self) -> float:
        """Share of frames usable for analysis — the tracking quality score."""
        if not self.valid:
            return 0.0
        return sum(self.valid) / len(self.valid)

    @property
    def mean_knee_deg(self) -> list[float | None]:
        """Average of the two knee angles per frame, ignoring missing sides."""
        result: list[float | None] = []
        for left, right in zip(self.left_knee_deg, self.right_knee_deg, strict=True):
            present = [value for value in (left, right) if value is not None]
            result.append(sum(present) / len(present) if present else None)
        return result


@dataclass(frozen=True, slots=True)
class Rep:
    """One detected repetition, from the start of the descent to lockout."""

    index: int  # 1-based, as a lifter would count

    start_frame: int
    bottom_frame: int
    end_frame: int

    start_time_s: float
    bottom_time_s: float
    end_time_s: float

    #: Knee angle at the deepest point. The primary depth measurement.
    min_knee_angle_deg: float | None = None
    min_left_knee_deg: float | None = None
    min_right_knee_deg: float | None = None
    min_hip_angle_deg: float | None = None

    #: Greatest forward torso lean from vertical during the rep.
    max_torso_lean_deg: float | None = None

    #: Largest left/right knee angle difference at the bottom.
    knee_asymmetry_deg: float | None = None

    #: 0-100, where 100 means the configured parallel standard was reached.
    depth_percent: float | None = None

    #: True when the hip dropped to or below knee level.
    hip_below_knee: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def eccentric_s(self) -> float:
        """Descent time — lockout to bottom."""
        return self.bottom_time_s - self.start_time_s

    @property
    def concentric_s(self) -> float:
        """Ascent time — bottom back to lockout."""
        return self.end_time_s - self.bottom_time_s


@dataclass(frozen=True, slots=True)
class Metrics:
    """Workout-level summary statistics."""

    total_reps: int
    video_duration_s: float
    total_workout_time_s: float

    max_depth_percent: float | None = None
    avg_depth_percent: float | None = None
    min_knee_angle_deg: float | None = None

    avg_rep_duration_s: float | None = None
    fastest_rep_s: float | None = None
    slowest_rep_s: float | None = None
    avg_eccentric_s: float | None = None
    avg_concentric_s: float | None = None

    #: Repetitions per minute across the working set.
    reps_per_minute: float | None = None

    avg_torso_lean_deg: float | None = None
    max_torso_lean_deg: float | None = None
    avg_knee_asymmetry_deg: float | None = None

    depth_consistency_percent: float | None = None
    duration_consistency_s: float | None = None

    tracking_quality: float = 0.0


@dataclass(frozen=True, slots=True)
class FeedbackItem:
    """One piece of coaching advice."""

    rule_id: str
    severity: Severity
    title: str
    message: str
    explanation: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Everything the pipeline produces for one video."""

    metadata: VideoMetadata
    angles: AngleSeries
    reps: tuple[Rep, ...]
    metrics: Metrics
    feedback: tuple[FeedbackItem, ...]
    estimator_name: str
