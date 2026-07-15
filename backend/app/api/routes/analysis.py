"""Analysis result retrieval."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Analysis
from app.schemas.analysis import AnalysisResponse
from app.schemas.common import ErrorResponse

router = APIRouter(tags=["analysis"])


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Fetch analysis status and results",
    responses={404: {"model": ErrorResponse, "description": "No such analysis"}},
)
def get_analysis_result(
    analysis_id: str, session: Session = Depends(get_db)
) -> AnalysisResponse:
    """Return the current state of an analysis.

    This is the polling endpoint. One response shape covers every status:
    while processing, the result fields are null and only `status` carries
    information. That keeps the client to a single endpoint and a single type,
    checking a single field.
    """
    from app.services.analysis_service import decode_result, get_analysis

    record = get_analysis(session, analysis_id)
    return AnalysisResponse.from_record(record, decode_result(record))


@router.get(
    "/analyses",
    response_model=list[AnalysisResponse],
    summary="List recent analyses",
)
def list_analyses(
    limit: int = 20, session: Session = Depends(get_db)
) -> list[AnalysisResponse]:
    """Recent analyses, newest first.

    Not required by the V1 flow, but it is the natural place for workout
    history to attach in a later version, and it makes the API browsable
    during development.

    Summaries only - the result blob is deliberately not decoded, since a list
    view needs status and filename, not every rep of every set.
    """
    limit = max(1, min(limit, 100))
    records = session.scalars(
        select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)
    ).all()
    return [AnalysisResponse.from_record(record) for record in records]
