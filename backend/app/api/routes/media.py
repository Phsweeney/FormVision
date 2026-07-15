"""Video and overlay streaming endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.common import ErrorResponse
from app.services.storage import LocalStorage, get_storage

router = APIRouter(tags=["media"])

# Media is immutable once written - the id is a uuid tied to one file - so it
# can be cached aggressively. Without this the browser refetches the whole
# video on every dashboard visit.
_CACHE_CONTROL = "public, max-age=31536000, immutable"


@router.get(
    "/video/{analysis_id}",
    summary="Stream the original uploaded video",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse, "description": "No such video"}},
)
def get_video(
    analysis_id: str,
    session: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> FileResponse:
    """Serve the uploaded video.

    Starlette's `FileResponse` handles HTTP Range requests, which is what lets
    the browser seek within the clip rather than being forced to download it
    from the start. CORS in `main.py` exposes `Content-Range` and
    `Accept-Ranges`; without that the browser cannot read the headers and
    seeking silently stops working cross-origin.
    """
    from app.services.analysis_service import get_analysis

    record = get_analysis(session, analysis_id)
    path = storage.require_file(storage.video_path(record.stored_filename), "Video")

    return FileResponse(
        path,
        media_type=_media_type(record.stored_filename, record.content_type),
        filename=record.original_filename,
        # `inline` so the browser plays it rather than offering a download.
        content_disposition_type="inline",
        headers={"Cache-Control": _CACHE_CONTROL},
    )


@router.get(
    "/overlay/{analysis_id}",
    summary="Stream the skeleton overlay video",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse, "description": "No overlay available"}},
)
def get_overlay(
    analysis_id: str,
    session: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> FileResponse:
    """Serve the rendered skeleton overlay.

    404s when no overlay exists - the analysis may still be running, or
    rendering may have failed while the metrics succeeded. The client treats a
    missing overlay as a degraded state, not an error.
    """
    from app.core.exceptions import NotFoundError
    from app.services.analysis_service import get_analysis

    record = get_analysis(session, analysis_id)
    if not record.overlay_filename:
        raise NotFoundError(
            "No overlay has been rendered for this analysis.",
            detail={"status": record.status.value},
        )

    path = storage.require_file(storage.overlay_path(record.overlay_filename), "Overlay")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"overlay_{analysis_id}.mp4",
        content_disposition_type="inline",
        headers={"Cache-Control": _CACHE_CONTROL},
    )


def _media_type(stored_filename: str, declared: str) -> str:
    """Pick a media type the browser will accept.

    Derived from the stored extension rather than the client's declared type,
    since browsers routinely send `application/octet-stream` for .mov and a
    video element will not play a response labelled that way.
    """
    if stored_filename.lower().endswith(".mov"):
        return "video/quicktime"
    if stored_filename.lower().endswith(".mp4"):
        return "video/mp4"
    return declared or "video/mp4"
