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


#: Landmarks required in every usable frame, whatever the camera angle. The
#: shoulders and hips are the one group that survives occlusion: filmed side-on
#: they still report ~0.99 visibility, because even the far one sits at the
#: silhouette edge rather than behind the body.
CORE_LANDMARKS: tuple[PoseLandmarkIndex, ...] = (
    PoseLandmarkIndex.LEFT_SHOULDER,
    PoseLandmarkIndex.RIGHT_SHOULDER,
    PoseLandmarkIndex.LEFT_HIP,
    PoseLandmarkIndex.RIGHT_HIP,
)

#: The legs, per side. A usable frame needs *one* of these complete, not both.
#:
#: Demanding both is what made side-on footage unusable. Filmed from the side
#: the far leg is hidden behind the near one, and MediaPipe drops its confidence
#: accordingly — measured at 0.40 for the far knee against 0.93 for the near one
#: — even though the coordinates it returns are good, agreeing with the near
#: leg's knee angle to within a few degrees. Requiring both sides therefore
#: discarded ~83% of a perfectly clear side-on clip on a confidence score, not
#: on anything actually wrong with the data.
LEFT_LEG_LANDMARKS: tuple[PoseLandmarkIndex, ...] = (
    PoseLandmarkIndex.LEFT_KNEE,
    PoseLandmarkIndex.LEFT_ANKLE,
)
RIGHT_LEG_LANDMARKS: tuple[PoseLandmarkIndex, ...] = (
    PoseLandmarkIndex.RIGHT_KNEE,
    PoseLandmarkIndex.RIGHT_ANKLE,
)

#: Everything the ankle angle needs, per side. Listed separately from the leg
#: groups because the foot index is the one landmark the rest of the analysis
#: never asks for: a frame cropped at the shins is still perfectly usable for
#: depth and lean, so a missing toe must cost the ankle angle alone rather than
#: invalidate the frame.
LEFT_ANKLE_ANGLE_LANDMARKS: tuple[PoseLandmarkIndex, ...] = (
    PoseLandmarkIndex.LEFT_KNEE,
    PoseLandmarkIndex.LEFT_ANKLE,
    PoseLandmarkIndex.LEFT_FOOT_INDEX,
)
RIGHT_ANKLE_ANGLE_LANDMARKS: tuple[PoseLandmarkIndex, ...] = (
    PoseLandmarkIndex.RIGHT_KNEE,
    PoseLandmarkIndex.RIGHT_ANKLE,
    PoseLandmarkIndex.RIGHT_FOOT_INDEX,
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


class ViewOrientation(StrEnum):
    """Where the camera stood relative to the lifter.

    This is not cosmetic: it decides which measurements mean anything. A 2D
    analysis can only see what the image plane shows, so torso lean is
    measurable side-on and left/right asymmetry front-on, and each is noise
    from the other angle. Detecting the view lets the pipeline return `None`
    for what it cannot see rather than a confident number that is wrong.
    """

    SIDE = "side"
    FRONT = "front"
    #: Somewhere between the two — neither measurement is fully trustworthy.
    OBLIQUE = "oblique"
    #: The subject was never tracked well enough to tell.
    UNKNOWN = "unknown"


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

    #: Per-side hip angle (shoulder-hip-knee on that side's own landmarks).
    #: `hip_deg` above is the midpoint version and stays the primary signal;
    #: these exist so a left/right comparison is possible at the hip as well as
    #: the knee, which is what an asymmetry judgement needs.
    left_hip_deg: list[float | None] = field(default_factory=list)
    right_hip_deg: list[float | None] = field(default_factory=list)

    #: Ankle angle (knee-ankle-foot) per side, measuring shin-over-foot travel.
    #: A heel lifting off the floor shows up here as the angle opening out.
    #: Side-on only: filmed front-on the foot points at the lens, so the whole
    #: movement projects to nothing and the number would be noise.
    left_ankle_deg: list[float | None] = field(default_factory=list)
    right_ankle_deg: list[float | None] = field(default_factory=list)

    #: Knee displacement from that leg's own hip-to-ankle line, in torso
    #: lengths, signed so that **positive means medial** — the knee travelling
    #: inward toward the midline, which is valgus. Front-on only: from the side
    #: the knee sits on that line by projection no matter what it is doing.
    left_knee_lateral: list[float | None] = field(default_factory=list)
    right_knee_lateral: list[float | None] = field(default_factory=list)

    #: Vertical hip position, in torso lengths above the ankles. Larger = more
    #: upright. Scale-normalised so it does not depend on camera distance.
    hip_height: list[float | None] = field(default_factory=list)

    #: Signed hip-to-knee vertical offset in thigh lengths. >= 0 means the hip
    #: has dropped to or below knee level, i.e. at or past parallel.
    hip_knee_offset: list[float | None] = field(default_factory=list)

    #: True where enough landmarks were visible to analyse the frame: the core
    #: torso points plus at least one complete leg.
    valid: list[bool] = field(default_factory=list)

    #: Per-side leg visibility. Tracked separately from `valid` because side-on
    #: footage is analysable on one leg, and knowing *which* legs were seen is
    #: what tells the rest of the pipeline whether a left/right comparison is
    #: meaningful.
    left_leg_valid: list[bool] = field(default_factory=list)
    right_leg_valid: list[bool] = field(default_factory=list)

    #: Median torso length in normalised units; the scale reference for the
    #: whole clip. None when the subject was never tracked well enough.
    torso_scale: float | None = None
    thigh_scale: float | None = None

    #: Where the camera stood. Signals this angle cannot measure are `None`.
    view: ViewOrientation = ViewOrientation.UNKNOWN

    def __len__(self) -> int:
        return len(self.timestamps_s)

    @property
    def valid_fraction(self) -> float:
        """Share of frames usable for analysis — the tracking quality score."""
        if not self.valid:
            return 0.0
        return sum(self.valid) / len(self.valid)

    @property
    def left_leg_coverage(self) -> float:
        """Share of frames in which the left leg was independently tracked."""
        if not self.left_leg_valid:
            return 0.0
        return sum(self.left_leg_valid) / len(self.left_leg_valid)

    @property
    def right_leg_coverage(self) -> float:
        """Share of frames in which the right leg was independently tracked."""
        if not self.right_leg_valid:
            return 0.0
        return sum(self.right_leg_valid) / len(self.right_leg_valid)

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

    #: The camera angle the clip was filmed from. Explains why the measurements
    #: that angle cannot see came back as None.
    camera_view: ViewOrientation = ViewOrientation.UNKNOWN


class FeedbackSource(StrEnum):
    """Where a piece of feedback came from.

    Carried in the type system rather than left to the wording, so the UI can
    mark model-derived advice without pattern-matching on copy. A lifter being
    told their knees cave deserves to know whether that came from a measurement
    with a threshold on it or from a model's opinion.
    """

    RULE = "rule"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class FeedbackItem:
    """One piece of coaching advice."""

    rule_id: str
    severity: Severity
    title: str
    message: str
    explanation: str

    #: Defaults to RULE so every pre-existing rule keeps its behaviour without
    #: being edited.
    source: FeedbackSource = FeedbackSource.RULE

    #: Model confidence, 0-1. None for rule-derived items, which are not
    #: probabilistic and must not be dressed up as though they were.
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class FaultPrediction:
    """One model's verdict on one repetition.

    Predictions are per-rep rather than per-set because that is the granularity
    a lifter can act on: "your knees caved on reps 4 and 5" is coachable where
    "your knees caved" is not.
    """

    fault_id: str
    rep_index: int

    #: Mean per-frame probability across the rep. Because the model is
    #: calibrated, this reads as roughly the share of the rep showing the fault.
    #: None when the rep could not be scored at all.
    probability: float | None

    #: Share of the rep's scored frames whose probability cleared `threshold`.
    affected_fraction: float

    #: The model's own operating threshold, chosen during training to hold a
    #: precision target. Travels with the artifact because it is calibrated
    #: against those particular weights and is meaningless beside any others.
    threshold: float

    #: Share of the detector's inputs that were actually measured, 0-1. A verdict
    #: resting on features the camera could not see is not a verdict.
    feature_completeness: float

    #: Whether this rises to something worth telling the lifter.
    fired: bool


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Everything the pipeline produces for one video."""

    metadata: VideoMetadata
    angles: AngleSeries
    reps: tuple[Rep, ...]
    metrics: Metrics
    feedback: tuple[FeedbackItem, ...]
    estimator_name: str

    #: Model output, empty when the ML layer is disabled or has no artifact.
    #: Defaulted so constructing a result without it stays valid, which is what
    #: keeps every existing test and code path working unchanged.
    predictions: tuple[FaultPrediction, ...] = ()
