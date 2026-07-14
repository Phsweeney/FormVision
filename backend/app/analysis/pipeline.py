"""The analysis pipeline.

Wires the stages together in order:

    video -> pose -> angles -> reps -> metrics -> feedback
                  \\-> skeleton overlay

This module contains no analysis logic of its own — deliberately. It is
coordination only, so each stage stays independently testable and replaceable.
Read this file to understand the shape of the system; read the individual
modules to understand the maths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.analysis.angles import compute_angles, unwrap_landmarks
from app.analysis.feedback.engine import generate_feedback
from app.analysis.metrics import compute_metrics
from app.analysis.overlay import render_overlay
from app.analysis.pose.base import PoseEstimator
from app.analysis.pose.registry import create_estimator
from app.analysis.reps import detect_reps
from app.analysis.types import AnalysisResult, PoseSeries, VideoMetadata
from app.config import Settings, get_settings
from app.core.exceptions import VideoProcessingError
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """Everything one pipeline run produces."""

    result: AnalysisResult
    #: Raw landmarks, ready to be archived so an analysis can be recomputed
    #: with improved logic without re-running the expensive pose step.
    landmark_payload: dict
    overlay_path: Path | None
    duration_s: float


def run_pipeline(
    video_path: Path,
    overlay_path: Path | None = None,
    estimator: PoseEstimator | None = None,
    settings: Settings | None = None,
) -> PipelineOutput:
    """Analyse one video end to end.

    Args:
        video_path: The uploaded video.
        overlay_path: Where to write the skeleton video. None skips rendering.
        estimator: Override the configured pose estimator. This is the seam the
            tests use to run the whole pipeline without MediaPipe or a real
            video file.
        settings: Override application settings.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    # --- 1. Pose estimation ------------------------------------------------
    estimator = estimator or create_estimator(settings=settings)
    logger.info("Analysing %s with estimator '%s'", video_path.name, estimator.name)
    pose = estimator.estimate(video_path)

    if not pose.frames:
        raise VideoProcessingError("The video contained no readable frames.")

    metadata = _corrected_metadata(pose)

    # --- 2. Joint angles and derived signals -------------------------------
    angles = compute_angles(pose, settings)

    # --- 3. Repetitions ----------------------------------------------------
    reps = detect_reps(angles, settings)

    # --- 4. Aggregate metrics ----------------------------------------------
    metrics = compute_metrics(reps, angles, metadata.duration_s, settings)

    # --- 5. Coaching feedback ----------------------------------------------
    feedback = generate_feedback(reps, metrics, angles, settings)

    # --- 6. Skeleton overlay -----------------------------------------------
    written_overlay: Path | None = None
    if overlay_path is not None:
        try:
            written_overlay = render_overlay(
                video_path, overlay_path, pose, angles, reps, settings
            )
        except Exception:
            # The overlay is a visualisation of results that already exist.
            # Losing it is a degraded experience; losing the metrics because
            # rendering failed would be a wasted upload.
            logger.exception("Overlay rendering failed; continuing without it")

    duration = time.perf_counter() - started
    logger.info(
        "Pipeline complete in %.1fs: %d reps, %d feedback items",
        duration,
        len(reps),
        len(feedback),
    )

    return PipelineOutput(
        result=AnalysisResult(
            metadata=metadata,
            angles=angles,
            reps=tuple(reps),
            metrics=metrics,
            feedback=tuple(feedback),
            estimator_name=pose.estimator_name,
        ),
        landmark_payload=unwrap_landmarks(pose),
        overlay_path=written_overlay,
        duration_s=duration,
    )


def _corrected_metadata(pose: PoseSeries) -> VideoMetadata:
    """Replace the container's declared frame count with the decoded count.

    Containers routinely report a frame count that disagrees with what actually
    decodes — a truncated file, a variable frame rate, or simply a wrong header.
    Since every timestamp in the analysis is derived from the real decoded
    frames, the duration must be too, or the charts would run past the end of
    the video.
    """
    declared = pose.metadata
    actual_frames = len(pose.frames)

    if declared.frame_count == actual_frames and declared.duration_s > 0:
        return declared

    duration = actual_frames / declared.fps if declared.fps > 0 else 0.0
    if declared.frame_count != actual_frames:
        logger.info(
            "Container declared %d frames; %d decoded. Using the decoded count.",
            declared.frame_count,
            actual_frames,
        )

    return VideoMetadata(
        width=declared.width,
        height=declared.height,
        fps=declared.fps,
        frame_count=actual_frames,
        duration_s=duration,
    )
