"""Health and readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import Settings, get_settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health() -> HealthResponse:
    """Report liveness plus the details needed to diagnose a bad deployment.

    ``pose_model_available`` is included because a missing model bundle is the
    single most likely reason analysis fails on a fresh install, and it is
    invisible until someone actually uploads a video.
    """
    settings: Settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
        pose_estimator=settings.pose_estimator,
        pose_model_available=settings.pose_model_path.exists(),
    )
