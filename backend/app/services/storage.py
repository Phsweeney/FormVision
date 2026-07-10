"""Local filesystem storage for videos, overlays, and landmark data.

Every path the application touches is constructed here. Routes and the analysis
pipeline ask this module for a path and never join one themselves, which means
the switch to object storage in a later version is a change to this file plus
its interface, not a hunt through the codebase for `data_dir / ...`.

The filename-sanitising and traversal checks matter more than they look: an
analysis id arrives from the URL, and using it to build a path without
validation is how `GET /video/../../etc/passwd` becomes a file read.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from app.config import Settings, get_settings
from app.core.exceptions import NotFoundError, PayloadTooLargeError, ValidationError
from app.logging_config import get_logger

logger = get_logger(__name__)

# Anything outside this set is replaced with an underscore when we echo a
# user-supplied filename back into a path.
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")

# Analysis ids are uuid4 hex — 32 lowercase hex characters, nothing else.
_VALID_ID = re.compile(r"^[0-9a-f]{32}$")

# Copied in bounded chunks so a large upload never sits in memory in full.
_CHUNK_SIZE = 1024 * 1024


def new_analysis_id() -> str:
    """Generate an unguessable identifier for a new analysis."""
    return uuid.uuid4().hex


def validate_analysis_id(analysis_id: str) -> str:
    """Reject anything that is not a well-formed id.

    Called before an id is used in a filesystem path. This is the path
    traversal guard: `..` and `/` cannot survive the hex-only pattern.
    """
    if not _VALID_ID.match(analysis_id):
        raise ValidationError(
            "Malformed analysis identifier.",
            detail={"analysis_id": analysis_id},
        )
    return analysis_id


def sanitize_filename(filename: str) -> str:
    """Strip a user-supplied filename down to something safe to store."""
    # `Path(...).name` discards any directory component, including Windows
    # backslash paths that a POSIX-minded check would miss.
    base = Path(filename.replace("\\", "/")).name
    cleaned = _SAFE_FILENAME.sub("_", base).lstrip(".")
    return cleaned[:200] or "upload"


class LocalStorage:
    """Filesystem-backed artefact store."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

    # -- Path construction ---------------------------------------------------

    def video_path(self, stored_filename: str) -> Path:
        return self.settings.upload_dir / stored_filename

    def overlay_path(self, overlay_filename: str) -> Path:
        return self.settings.overlay_dir / overlay_filename

    def landmark_path(self, landmark_filename: str) -> Path:
        return self.settings.landmark_dir / landmark_filename

    def build_video_filename(self, analysis_id: str, original_filename: str) -> str:
        """Compose the on-disk name for an upload.

        Keeps the original extension so tooling (and the browser) can infer the
        container format, but prefixes the analysis id so names are unique and
        the file is traceable back to its record.
        """
        validate_analysis_id(analysis_id)
        suffix = Path(sanitize_filename(original_filename)).suffix.lower()
        if suffix not in self.settings.allowed_video_extensions:
            suffix = ".mp4"
        return f"{analysis_id}{suffix}"

    def build_overlay_filename(self, analysis_id: str) -> str:
        validate_analysis_id(analysis_id)
        return f"{analysis_id}_overlay.mp4"

    def build_landmark_filename(self, analysis_id: str) -> str:
        validate_analysis_id(analysis_id)
        return f"{analysis_id}_landmarks.json.gz"

    # -- Writing -------------------------------------------------------------

    def save_upload(self, source: BinaryIO, destination: Path) -> int:
        """Stream an upload to disk, enforcing the size cap as we go.

        The cap is checked *during* the copy rather than after. Trusting
        Content-Length, or writing the whole file and then measuring it, both
        let a client fill the disk before anyone objects.
        """
        limit = self.settings.max_upload_bytes
        written = 0
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with destination.open("wb") as handle:
                while chunk := source.read(_CHUNK_SIZE):
                    written += len(chunk)
                    if written > limit:
                        raise PayloadTooLargeError(
                            f"Upload exceeds the {limit / 1024 / 1024:.0f} MB limit.",
                            detail={"limit_bytes": limit},
                        )
                    handle.write(chunk)
        except Exception:
            # Never leave a partial file behind: it would look like a valid
            # upload to anything that only checks for existence.
            destination.unlink(missing_ok=True)
            raise

        if written == 0:
            destination.unlink(missing_ok=True)
            raise ValidationError("The uploaded file is empty.")

        logger.info("Stored upload at %s (%d bytes)", destination, written)
        return written

    def write_landmarks(self, path: Path, payload: dict[str, Any]) -> None:
        """Persist full-resolution landmark data, gzipped.

        This is the one genuinely large artefact — 33 landmarks x 4 floats per
        frame. It is written for reproducibility and future re-analysis, and is
        never sent to the browser. Gzip typically shrinks it by roughly 10x.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        logger.info("Wrote landmarks to %s (%d bytes)", path, path.stat().st_size)

    def read_landmarks(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise NotFoundError("Landmark data not found.", detail={"path": path.name})
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    # -- Reading and cleanup -------------------------------------------------

    def require_file(self, path: Path, description: str) -> Path:
        """Return ``path`` or raise a 404 describing what was missing."""
        if not path.is_file():
            logger.warning("%s missing at %s", description, path)
            raise NotFoundError(f"{description} not found.")
        return path

    def delete_artifacts(self, *paths: Path | None) -> None:
        """Best-effort cleanup. Never raises — used on failure paths where the
        original error matters more than a stray file."""
        for path in paths:
            if path is None:
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - platform dependent
                logger.warning("Could not remove %s: %s", path, exc)


def get_storage() -> LocalStorage:
    """FastAPI dependency providing the storage backend."""
    return LocalStorage()
