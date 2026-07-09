"""Domain exceptions and the JSON error envelope.

The frontend needs to distinguish "your file was the wrong type" from "the
server fell over", and to show a useful message either way. Raising typed
exceptions from deep inside the pipeline and translating them once, at the
edge, keeps that logic out of every route handler.

Every error response has the same shape:

    {"error": {"code": "UNSUPPORTED_MEDIA", "message": "...", "detail": {...}}}
"""

from __future__ import annotations

from typing import Any


class FormVisionError(Exception):
    """Base class for all expected, domain-level failures.

    ``status_code`` and ``code`` let the exception carry its own HTTP mapping,
    so the handler in ``main.py`` stays a one-liner regardless of how many
    subclasses exist.
    """

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "detail": self.detail,
            }
        }


class ValidationError(FormVisionError):
    """The request was well-formed but its contents are unacceptable."""

    status_code = 400
    code = "VALIDATION_ERROR"


class UnsupportedMediaError(FormVisionError):
    """Wrong file extension or content type."""

    status_code = 415
    code = "UNSUPPORTED_MEDIA"


class PayloadTooLargeError(FormVisionError):
    """Upload exceeded the configured size limit."""

    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


class NotFoundError(FormVisionError):
    """No such analysis, video, or overlay."""

    status_code = 404
    code = "NOT_FOUND"


class ConflictError(FormVisionError):
    """The resource exists but is in the wrong state for this operation."""

    status_code = 409
    code = "CONFLICT"


class VideoProcessingError(FormVisionError):
    """The file could not be decoded or read as video."""

    status_code = 422
    code = "VIDEO_PROCESSING_ERROR"


class PoseEstimationError(FormVisionError):
    """The pose estimator failed to initialise or produced no usable output."""

    status_code = 422
    code = "POSE_ESTIMATION_ERROR"


class AnalysisError(FormVisionError):
    """A stage of the analysis pipeline failed."""

    status_code = 500
    code = "ANALYSIS_ERROR"
