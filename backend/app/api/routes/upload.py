"""Video upload endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.video import probe_video, validate_upload_metadata
from app.db.database import get_db
from app.db.models import Analysis, AnalysisStatus
from app.logging_config import get_logger
from app.schemas.analysis import UploadResponse
from app.schemas.common import ErrorResponse
from app.services.storage import LocalStorage, get_storage, new_analysis_id

logger = get_logger(__name__)

router = APIRouter(tags=["upload"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a squat video",
    responses={
        400: {"model": ErrorResponse, "description": "Malformed or empty upload"},
        413: {"model": ErrorResponse, "description": "File exceeds the size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
    },
)
def upload_video(
    file: UploadFile = File(description="An MP4 or MOV video of a back squat."),
    session: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Accept a video, validate it, and create an analysis record.

    Validation runs in two passes. Headers are checked first because it is
    cheap and rejects obvious mistakes without touching the disk. The file is
    then written and actually opened, which is the pass that matters: header
    checks only inspect what the client claimed, so a text file renamed to
    `.mp4` gets through the first and is caught by the second.

    A failed probe deletes the stored file. Leaving it would accumulate junk
    that nothing ever references.
    """
    validate_upload_metadata(file.filename, file.content_type, settings)

    analysis_id = new_analysis_id()
    stored_filename = storage.build_video_filename(analysis_id, file.filename)
    destination = storage.video_path(stored_filename)

    size_bytes = storage.save_upload(file.file, destination)

    try:
        probe = probe_video(destination, settings)
    except Exception:
        storage.delete_artifacts(destination)
        raise

    record = Analysis(
        id=analysis_id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=size_bytes,
        status=AnalysisStatus.UPLOADED,
        fps=probe.fps,
        frame_count=probe.frame_count,
        duration_seconds=probe.duration_s,
        width=probe.width,
        height=probe.height,
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    logger.info(
        "Uploaded %s as %s (%dx%d, %.1fs, %.1f MB)",
        file.filename,
        analysis_id,
        probe.width,
        probe.height,
        probe.duration_s,
        size_bytes / 1024 / 1024,
    )

    return UploadResponse(
        id=record.id,
        filename=record.original_filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        status=record.status,
        created_at=record.created_at,
    )
