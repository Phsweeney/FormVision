"""SQLAlchemy ORM models."""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, Float, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for every model."""


class UtcDateTime(TypeDecorator):
    """A datetime column that is always timezone-aware UTC in Python.

    SQLite has no native timestamp type and silently discards tzinfo, so a
    value written as aware UTC reads back naive. That ambiguity would reach the
    API and leave the browser guessing what timezone a timestamp is in.

    This decorator normalises both directions: incoming values are converted to
    UTC before storage, outgoing naive values are re-tagged as UTC. Every
    consumer can then assume aware datetimes unconditionally.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class AnalysisStatus(enum.StrEnum):
    """Lifecycle of an analysis record.

    A ``StrEnum`` so members serialise directly to JSON and compare equal to
    plain strings in tests and route handlers.
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp.

    Used instead of ``datetime.utcnow`` (deprecated in 3.12) and instead of a
    database-side default, so timestamps are consistent regardless of the
    database backend.
    """
    return datetime.now(UTC)


class Analysis(Base):
    """One uploaded video and everything derived from it.

    Deliberately a single table. The analysis result is stored as a JSON blob
    rather than normalised into per-rep and per-frame tables because it is
    always read as a whole, never queried across, and its shape will change as
    metrics are added. A rigid schema here would mean a migration every time a
    new metric appears, for zero query benefit.

    ``user_id`` is intentionally absent in V1; when accounts arrive it is a
    nullable column plus an index, with no restructuring of this table.
    """

    __tablename__ = "analyses"

    # UUID hex string rather than an autoincrement integer: ids appear in URLs
    # and on disk, and sequential ids would let anyone enumerate other people's
    # uploads.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # -- Source file ---------------------------------------------------------
    original_filename: Mapped[str] = mapped_column(String(512))
    stored_filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)

    # -- Video properties, populated by probing on upload --------------------
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # -- Processing state ----------------------------------------------------
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus, native_enum=False, length=16),
        default=AnalysisStatus.UPLOADED,
        index=True,
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Results -------------------------------------------------------------
    # Serialised AnalysisResult. Read whole, never queried into.
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    overlay_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    landmark_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # -- Timestamps ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Analysis id={self.id} status={self.status.value}>"
