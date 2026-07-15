"""Analysis trigger endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.logging_config import get_logger
from app.schemas.analysis import AnalyzeResponse
from app.schemas.common import ErrorResponse
from app.services.analysis_service import run_analysis_task, start_analysis

logger = get_logger(__name__)

router = APIRouter(tags=["analysis"])


class AnalyzeRequest(BaseModel):
    """Body of ``POST /analyze``."""

    analysis_id: str = Field(description="Id returned by POST /upload.")


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start analysing an uploaded video",
    responses={
        404: {"model": ErrorResponse, "description": "No such analysis"},
        409: {"model": ErrorResponse, "description": "Already being analysed"},
    },
)
def analyze(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
) -> AnalyzeResponse:
    """Queue analysis and return immediately.

    Returns **202 Accepted**, not 200: pose estimation over a 60-second clip
    takes far longer than any sensible HTTP timeout, so the work runs in a
    background task and the client polls `GET /analysis/{id}`.

    The status is committed to `processing` before the task is queued, so a
    client that polls instantly sees the correct state rather than a stale
    `uploaded` and concluding nothing happened.
    """
    record = start_analysis(session, payload.analysis_id)
    background_tasks.add_task(run_analysis_task, record.id)

    logger.info("Queued analysis %s", record.id)
    return AnalyzeResponse(
        id=record.id,
        status=record.status,
        message="Analysis started. Poll GET /analysis/{id} for progress.",
    )
