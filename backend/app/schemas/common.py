"""Shared response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Payload of ``GET /health``."""

    status: str = Field(description="Always 'ok' when the service is reachable.")
    version: str
    environment: str
    pose_estimator: str = Field(description="Active pose estimator implementation.")
    pose_model_available: bool = Field(
        description="Whether the pose model bundle is present on disk."
    )


class ErrorDetail(BaseModel):
    """Inner object of an error envelope."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    detail: dict[str, Any] = Field(
        default_factory=dict, description="Optional structured context."
    )


class ErrorResponse(BaseModel):
    """Every non-2xx response from the API uses this shape."""

    error: ErrorDetail
