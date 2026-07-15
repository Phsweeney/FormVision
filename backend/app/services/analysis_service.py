"""Coordinates uploads, background analysis, and result persistence.

Sits between the HTTP layer and the analysis pipeline. Routes call into here;
this module owns the database record's lifecycle and the storage side effects,
and the pipeline stays a pure function of a video file.
"""

from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy.orm import Session

from app.analysis.pipeline import run_pipeline
from app.analysis.pose.base import PoseEstimator
from app.config import Settings, get_settings
from app.core.exceptions import ConflictError, FormVisionError, NotFoundError
from app.db.database import session_scope
from app.db.models import Analysis, AnalysisStatus
from app.logging_config import get_logger
from app.schemas.analysis import build_result_payload
from app.services.storage import LocalStorage, validate_analysis_id

logger = get_logger(__name__)

#: Injected by tests so the API can be exercised without MediaPipe or a real
#: video. Production leaves this as None and the registry decides.
_estimator_override: PoseEstimator | None = None


def set_estimator_override(estimator: PoseEstimator | None) -> None:
    """Force a specific pose estimator for subsequent analyses."""
    global _estimator_override
    _estimator_override = estimator


def get_analysis(session: Session, analysis_id: str) -> Analysis:
    """Fetch a record, or raise 404.

    The id is validated against the strict hex pattern first: it arrives from
    the URL and is used to build filesystem paths.
    """
    validate_analysis_id(analysis_id)
    record = session.get(Analysis, analysis_id)
    if record is None:
        raise NotFoundError("Analysis not found.", detail={"id": analysis_id})
    return record


def decode_result(record: Analysis) -> dict[str, Any] | None:
    """Decode the stored result blob, tolerating corruption.

    A record whose blob will not parse is reported as having no results rather
    than raising, so one bad row cannot break the endpoint for the client.
    """
    if not record.result_json:
        return None
    try:
        return json.loads(record.result_json)
    except json.JSONDecodeError:
        logger.exception("Corrupt result payload on analysis %s", record.id)
        return None


def start_analysis(session: Session, analysis_id: str) -> Analysis:
    """Mark a record as processing, ready for the background task.

    The status is committed *before* the background task starts so a client
    polling immediately after `POST /analyze` sees `processing` rather than a
    stale `uploaded` and concluding nothing happened.
    """
    record = get_analysis(session, analysis_id)

    if record.status is AnalysisStatus.PROCESSING:
        raise ConflictError(
            "This video is already being analysed.", detail={"id": analysis_id}
        )

    record.status = AnalysisStatus.PROCESSING
    record.error_code = None
    record.error_message = None
    session.commit()
    session.refresh(record)
    return record


def run_analysis_task(analysis_id: str, settings: Settings | None = None) -> None:
    """Execute the pipeline and persist the outcome.

    Runs on a background worker thread with no request context, so it opens its
    own session and must never raise — an escaping exception in a FastAPI
    BackgroundTask is logged and lost, leaving the record stuck on `processing`
    forever and the client polling indefinitely. Every failure path therefore
    ends with the record marked `failed` and an explanation the UI can show.
    """
    settings = settings or get_settings()
    storage = LocalStorage(settings)
    started = time.perf_counter()

    try:
        with session_scope() as session:
            record = session.get(Analysis, analysis_id)
            if record is None:
                logger.error("Analysis %s vanished before processing", analysis_id)
                return
            video_path = storage.video_path(record.stored_filename)
            overlay_name = storage.build_overlay_filename(analysis_id)
            landmark_name = storage.build_landmark_filename(analysis_id)

        output = run_pipeline(
            video_path=video_path,
            overlay_path=storage.overlay_path(overlay_name),
            estimator=_estimator_override,
            settings=settings,
        )

        storage.write_landmarks(
            storage.landmark_path(landmark_name), output.landmark_payload
        )

        payload = build_result_payload(output.result, settings.max_series_points)

        with session_scope() as session:
            record = session.get(Analysis, analysis_id)
            if record is None:  # pragma: no cover - deleted mid-flight
                return
            record.status = AnalysisStatus.COMPLETED
            record.result_json = json.dumps(payload, separators=(",", ":"))
            record.overlay_filename = (
                overlay_name if output.overlay_path is not None else None
            )
            record.landmark_filename = landmark_name
            record.processing_seconds = time.perf_counter() - started
            record.error_code = None
            record.error_message = None

            metadata = output.result.metadata
            record.fps = metadata.fps
            record.frame_count = metadata.frame_count
            record.duration_seconds = metadata.duration_s
            record.width = metadata.width
            record.height = metadata.height

        logger.info(
            "Analysis %s completed in %.1fs (%d reps)",
            analysis_id,
            time.perf_counter() - started,
            output.result.metrics.total_reps,
        )

    except FormVisionError as exc:
        # A known failure mode: the message is written for a human and is safe
        # to show in the UI.
        logger.warning("Analysis %s failed: %s", analysis_id, exc.message)
        _mark_failed(analysis_id, exc.code, exc.message, started)

    except Exception:
        # Unexpected: log the trace for us, show the client something generic.
        logger.exception("Analysis %s failed unexpectedly", analysis_id)
        _mark_failed(
            analysis_id,
            "INTERNAL_ERROR",
            "An unexpected error occurred while analysing this video.",
            started,
        )


def _mark_failed(analysis_id: str, code: str, message: str, started: float) -> None:
    """Record a failure. Deliberately swallows its own errors.

    This is the last line of defence; if it raised, the record would stay on
    `processing` and the client would poll forever.
    """
    try:
        with session_scope() as session:
            record = session.get(Analysis, analysis_id)
            if record is None:
                return
            record.status = AnalysisStatus.FAILED
            record.error_code = code
            record.error_message = message
            record.processing_seconds = time.perf_counter() - started
    except Exception:  # pragma: no cover - database unavailable
        logger.exception("Could not record failure for analysis %s", analysis_id)
