"""API request and response models.

These are the contract between backend and frontend. `frontend/src/lib/types.ts`
mirrors this file, and the OpenAPI schema generated from it is the reference
both sides work against.

Field names are deliberately explicit about units — ``duration_s``,
``knee_angle_deg``, ``depth_percent``. A bare ``duration`` invites someone to
assume milliseconds.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.analysis.smoothing import decimation_indices
from app.analysis.types import (
    AnalysisResult,
    AngleSeries,
    FaultPrediction,
    FeedbackItem,
    Metrics,
    Rep,
    VideoMetadata,
)
from app.db.models import Analysis, AnalysisStatus


class VideoInfo(BaseModel):
    """Physical properties of the uploaded video."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata) -> VideoInfo:
        return cls(
            width=metadata.width,
            height=metadata.height,
            fps=round(metadata.fps, 3),
            frame_count=metadata.frame_count,
            duration_s=round(metadata.duration_s, 3),
        )


class UploadResponse(BaseModel):
    """Returned by ``POST /upload``."""

    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: AnalysisStatus
    created_at: datetime


class AnalyzeResponse(BaseModel):
    """Returned by ``POST /analyze``. Analysis continues in the background."""

    id: str
    status: AnalysisStatus
    message: str


class RepSchema(BaseModel):
    """One detected repetition."""

    index: int = Field(description="1-based repetition number.")
    start_time_s: float
    bottom_time_s: float
    end_time_s: float
    duration_s: float
    eccentric_s: float = Field(description="Descent duration, in seconds.")
    concentric_s: float = Field(description="Ascent duration, in seconds.")

    start_frame: int
    bottom_frame: int
    end_frame: int

    min_knee_angle_deg: float | None
    min_left_knee_deg: float | None
    min_right_knee_deg: float | None
    min_hip_angle_deg: float | None
    max_torso_lean_deg: float | None
    knee_asymmetry_deg: float | None
    depth_percent: float | None = Field(
        default=None, description="0-100, where 100 is the configured depth target."
    )
    hip_below_knee: bool

    @classmethod
    def from_rep(cls, rep: Rep) -> RepSchema:
        return cls(
            index=rep.index,
            start_time_s=round(rep.start_time_s, 3),
            bottom_time_s=round(rep.bottom_time_s, 3),
            end_time_s=round(rep.end_time_s, 3),
            duration_s=round(rep.duration_s, 3),
            eccentric_s=round(rep.eccentric_s, 3),
            concentric_s=round(rep.concentric_s, 3),
            start_frame=rep.start_frame,
            bottom_frame=rep.bottom_frame,
            end_frame=rep.end_frame,
            min_knee_angle_deg=_round(rep.min_knee_angle_deg),
            min_left_knee_deg=_round(rep.min_left_knee_deg),
            min_right_knee_deg=_round(rep.min_right_knee_deg),
            min_hip_angle_deg=_round(rep.min_hip_angle_deg),
            max_torso_lean_deg=_round(rep.max_torso_lean_deg),
            knee_asymmetry_deg=_round(rep.knee_asymmetry_deg),
            depth_percent=_round(rep.depth_percent),
            hip_below_knee=rep.hip_below_knee,
        )


class MetricsSchema(BaseModel):
    """Workout-level summary."""

    total_reps: int
    video_duration_s: float
    total_workout_time_s: float

    max_depth_percent: float | None
    avg_depth_percent: float | None
    min_knee_angle_deg: float | None

    avg_rep_duration_s: float | None
    fastest_rep_s: float | None
    slowest_rep_s: float | None
    avg_eccentric_s: float | None
    avg_concentric_s: float | None
    reps_per_minute: float | None

    avg_torso_lean_deg: float | None
    max_torso_lean_deg: float | None
    avg_knee_asymmetry_deg: float | None

    depth_consistency_percent: float | None = Field(
        default=None,
        description="Standard deviation of per-rep depth. Lower is more consistent.",
    )
    duration_consistency_s: float | None = Field(
        default=None,
        description="Standard deviation of per-rep duration. Lower is more consistent.",
    )

    tracking_quality: float = Field(
        description="Fraction of frames usable for analysis, 0-1."
    )
    camera_view: str = Field(
        default="unknown",
        description=(
            "Detected camera angle: side, front, oblique, or unknown. Explains "
            "which measurements came back null — torso lean is not measurable "
            "front-on, left/right asymmetry is not measurable side-on."
        ),
    )

    @classmethod
    def from_metrics(cls, metrics: Metrics) -> MetricsSchema:
        return cls(
            total_reps=metrics.total_reps,
            video_duration_s=round(metrics.video_duration_s, 3),
            total_workout_time_s=round(metrics.total_workout_time_s, 3),
            max_depth_percent=_round(metrics.max_depth_percent, 1),
            avg_depth_percent=_round(metrics.avg_depth_percent, 1),
            min_knee_angle_deg=_round(metrics.min_knee_angle_deg),
            avg_rep_duration_s=_round(metrics.avg_rep_duration_s),
            fastest_rep_s=_round(metrics.fastest_rep_s),
            slowest_rep_s=_round(metrics.slowest_rep_s),
            avg_eccentric_s=_round(metrics.avg_eccentric_s),
            avg_concentric_s=_round(metrics.avg_concentric_s),
            reps_per_minute=_round(metrics.reps_per_minute, 1),
            avg_torso_lean_deg=_round(metrics.avg_torso_lean_deg),
            max_torso_lean_deg=_round(metrics.max_torso_lean_deg),
            avg_knee_asymmetry_deg=_round(metrics.avg_knee_asymmetry_deg),
            depth_consistency_percent=_round(metrics.depth_consistency_percent),
            duration_consistency_s=_round(metrics.duration_consistency_s),
            tracking_quality=round(metrics.tracking_quality, 4),
            camera_view=metrics.camera_view.value,
        )


class FeedbackSchema(BaseModel):
    """One piece of coaching advice."""

    rule_id: str = Field(description="Stable identifier; safe to key UI logic on.")
    severity: str = Field(description="One of: good, info, warning, critical.")
    title: str
    message: str
    explanation: str
    source: str = Field(
        default="rule",
        description=(
            "Where the advice came from: 'rule' for a direct measurement against "
            "a threshold, 'model' for a trained classifier. Defaulted so results "
            "stored before the ML layer existed still decode."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Model confidence, 0-1. Null for rule-derived advice, which is not "
            "probabilistic and must not be presented as though it were."
        ),
    )

    @classmethod
    def from_item(cls, item: FeedbackItem) -> FeedbackSchema:
        return cls(
            rule_id=item.rule_id,
            severity=item.severity.value,
            title=item.title,
            message=item.message,
            explanation=item.explanation,
            source=item.source.value,
            confidence=_round(item.confidence, 3),
        )


class PredictionSchema(BaseModel):
    """One model verdict on one repetition.

    Exposed alongside the feedback items so a client can show which reps a fault
    was found on, rather than only that it was found somewhere in the set.
    """

    fault_id: str
    rep_index: int
    probability: float | None
    affected_fraction: float
    threshold: float
    feature_completeness: float
    fired: bool

    @classmethod
    def from_prediction(cls, prediction: FaultPrediction) -> PredictionSchema:
        return cls(
            fault_id=prediction.fault_id,
            rep_index=prediction.rep_index,
            probability=_round(prediction.probability, 3),
            affected_fraction=_round(prediction.affected_fraction, 3) or 0.0,
            threshold=_round(prediction.threshold, 4) or 0.0,
            feature_completeness=_round(prediction.feature_completeness, 3) or 0.0,
            fired=prediction.fired,
        )


class SeriesSchema(BaseModel):
    """Time series for charting.

    Every list is the same length and index-aligned with ``time_s``. Nulls mark
    frames where the value could not be measured; Recharts renders those as
    gaps, which is the honest representation of a tracking dropout.

    These are decimated (see `max_series_points`): full-resolution landmark data
    stays on disk. A 60 s clip at 60 fps would otherwise ship 3600 points per
    series, for a chart a few hundred pixels wide.
    """

    time_s: list[float]
    left_knee_deg: list[float | None]
    right_knee_deg: list[float | None]
    hip_deg: list[float | None]
    torso_lean_deg: list[float | None]
    hip_height: list[float | None]

    sample_count: int = Field(description="Points returned after decimation.")
    source_frame_count: int = Field(description="Frames before decimation.")

    @classmethod
    def from_angles(cls, angles: AngleSeries, max_points: int) -> SeriesSchema:
        # One index list drives every series, so they cannot drift apart.
        indices = decimation_indices(len(angles), max_points)

        def take(values: list[float | None]) -> list[float | None]:
            return [_round(values[i]) if values[i] is not None else None for i in indices]

        return cls(
            time_s=[round(angles.timestamps_s[i], 3) for i in indices],
            left_knee_deg=take(angles.left_knee_deg),
            right_knee_deg=take(angles.right_knee_deg),
            hip_deg=take(angles.hip_deg),
            torso_lean_deg=take(angles.torso_lean_deg),
            hip_height=[
                _round(angles.hip_height[i], 4)
                if angles.hip_height[i] is not None
                else None
                for i in indices
            ],
            sample_count=len(indices),
            source_frame_count=len(angles),
        )


class AnalysisResponse(BaseModel):
    """Returned by ``GET /analysis/{id}``.

    A single response shape covers every status. While processing, the result
    fields are null and only ``status`` is meaningful — this keeps the polling
    client simple: one endpoint, one type, check one field.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: AnalysisStatus
    filename: str
    created_at: datetime
    updated_at: datetime

    error_code: str | None = None
    error_message: str | None = None
    processing_seconds: float | None = None

    video: VideoInfo | None = None
    metrics: MetricsSchema | None = None
    reps: list[RepSchema] | None = None
    feedback: list[FeedbackSchema] | None = None
    series: SeriesSchema | None = None
    predictions: list[PredictionSchema] | None = Field(
        default=None,
        description=(
            "Per-rep model verdicts. Absent on analyses run before the ML layer "
            "existed, and empty when it is disabled or had nothing to say."
        ),
    )

    video_url: str | None = Field(
        default=None, description="Path to stream the original video."
    )
    overlay_url: str | None = Field(
        default=None, description="Path to stream the skeleton overlay, if rendered."
    )
    estimator: str | None = None

    @classmethod
    def from_record(
        cls, record: Analysis, payload: dict | None = None
    ) -> AnalysisResponse:
        """Build a response from a database row plus its decoded result blob."""
        response = cls(
            id=record.id,
            status=record.status,
            filename=record.original_filename,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error_code=record.error_code,
            error_message=record.error_message,
            processing_seconds=_round(record.processing_seconds, 2),
            video_url=f"/video/{record.id}",
            overlay_url=f"/overlay/{record.id}" if record.overlay_filename else None,
        )

        if payload:
            response.video = VideoInfo(**payload["video"])
            response.metrics = MetricsSchema(**payload["metrics"])
            response.reps = [RepSchema(**rep) for rep in payload["reps"]]
            response.feedback = [FeedbackSchema(**item) for item in payload["feedback"]]
            response.series = SeriesSchema(**payload["series"])
            response.estimator = payload.get("estimator")
            # `.get` rather than `[...]`: results stored before the ML layer
            # existed have no such key, and they must still render.
            response.predictions = [
                PredictionSchema(**item) for item in payload.get("predictions", [])
            ]

        return response


def build_result_payload(result: AnalysisResult, max_series_points: int) -> dict:
    """Serialise an `AnalysisResult` for storage in the database.

    Stored already shaped for the API so reads are a plain JSON decode rather
    than a re-run of every conversion. Analysis happens once; results are read
    many times, including on every status poll.
    """
    return {
        "video": VideoInfo.from_metadata(result.metadata).model_dump(),
        "metrics": MetricsSchema.from_metrics(result.metrics).model_dump(),
        "reps": [RepSchema.from_rep(rep).model_dump() for rep in result.reps],
        "feedback": [
            FeedbackSchema.from_item(item).model_dump() for item in result.feedback
        ],
        "series": SeriesSchema.from_angles(result.angles, max_series_points).model_dump(),
        "estimator": result.estimator_name,
        "predictions": [
            PredictionSchema.from_prediction(prediction).model_dump()
            for prediction in result.predictions
        ],
    }


def _round(value: float | None, digits: int = 2) -> float | None:
    """Round, preserving None.

    Values are rounded on the way out because normalised landmark coordinates
    carry no meaningful precision past a few decimals, and full float repr
    roughly doubles the size of every series.
    """
    return None if value is None else round(value, digits)
