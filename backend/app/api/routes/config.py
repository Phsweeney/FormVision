"""Public analysis-configuration endpoint.

Serves the analysis thresholds the live webcam client needs so it can run the
same pipeline in the browser without hard-coding a second copy of every number.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import Settings, get_settings
from app.schemas.config import ConfigResponse

router = APIRouter(tags=["system"])


@router.get("/config", response_model=ConfigResponse, summary="Analysis configuration")
def config() -> ConfigResponse:
    """Return the analysis thresholds used by both server and live client.

    The live mode fetches this once at session start. Because every value is
    read straight from the settings singleton, retuning a threshold in ``.env``
    changes the offline pipeline and the live client together, with no second
    place to edit.
    """
    settings: Settings = get_settings()
    return ConfigResponse.from_settings(settings)
