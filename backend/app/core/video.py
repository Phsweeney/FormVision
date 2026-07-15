"""Upload validation and video probing.

Validation happens in two passes, and the split matters.

**Before writing** — extension and declared content type. Cheap, and rejects the
obvious cases without touching the disk.

**After writing** — actually open the file and read its properties. This is the
pass that matters, because the first one only inspects what the *client claimed*.
A `.txt` renamed to `.mp4` passes every header check and fails here, which is
the only place it can be caught honestly.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from app.config import Settings
from app.core.exceptions import UnsupportedMediaError, ValidationError
from app.logging_config import get_logger

logger = get_logger(__name__)

_MAX_PLAUSIBLE_FPS = 480.0
_FALLBACK_FPS = 30.0


def validate_upload_metadata(
    filename: str, content_type: str | None, settings: Settings
) -> None:
    """Check what the client told us, before anything is written to disk."""
    if not filename:
        raise ValidationError("A filename is required.")

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_video_extensions:
        raise UnsupportedMediaError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Accepted formats: {', '.join(settings.allowed_video_extensions)}.",
            detail={"allowed": settings.allowed_video_extensions},
        )

    # Browsers are inconsistent about the MIME type they attach to .mov files,
    # so a missing type is tolerated. The probe below is the real gate.
    if content_type and content_type not in settings.allowed_content_types:
        raise UnsupportedMediaError(
            f"Unsupported content type '{content_type}'.",
            detail={"allowed": settings.allowed_content_types},
        )


class ProbeResult:
    """Video properties read from a file on disk."""

    __slots__ = ("width", "height", "fps", "frame_count", "duration_s")

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
        duration_s: float,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration_s = duration_s


def probe_video(path: Path, settings: Settings) -> ProbeResult:
    """Open the file and read its real properties.

    This is where a file that is not actually a video gets rejected, regardless
    of what its name or headers claimed.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise UnsupportedMediaError(
                "This file could not be read as a video. It may be corrupt, or "
                "use a codec the server cannot decode."
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # A container can report plausible-looking headers and still contain no
        # decodable video, so require one frame to actually come out.
        ok, _ = capture.read()
        if not ok:
            raise UnsupportedMediaError(
                "This file could not be read as a video. No frames could be decoded."
            )

        if width <= 0 or height <= 0:
            raise UnsupportedMediaError("The video reports invalid dimensions.")

        if not fps or fps <= 0 or fps > _MAX_PLAUSIBLE_FPS:
            logger.warning("Video reported fps=%s; assuming %s", fps, _FALLBACK_FPS)
            fps = _FALLBACK_FPS

        duration = frame_count / fps if frame_count > 0 else 0.0

        # Only reject on a duration we actually measured. An unknown frame count
        # is common enough that refusing the upload would be wrong; the analysis
        # is bounded by the file size cap regardless.
        if duration > settings.max_video_duration_s:
            raise ValidationError(
                f"Video is {duration:.0f}s long; the limit is "
                f"{settings.max_video_duration_s:.0f}s.",
                detail={"duration_s": round(duration, 1)},
            )

        return ProbeResult(width, height, fps, max(frame_count, 0), duration)
    finally:
        capture.release()
