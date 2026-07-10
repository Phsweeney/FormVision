"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base
from app.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine.

    Two SQLite-specific concessions are made here:

    ``check_same_thread=False`` — analysis runs in a background task on a
    worker thread while the request that started it returns on another. SQLite's
    default same-thread guard would reject that.

    WAL journal mode — allows a reader (a status poll) and a writer (the
    background analysis) to proceed concurrently. Under the default rollback
    journal the poll would block or fail with "database is locked", which is
    exactly the pattern this app produces every 1.5 seconds while processing.
    """
    settings = get_settings()
    url = settings.resolved_database_url
    is_sqlite = url.startswith("sqlite")

    engine = create_engine(
        url,
        connect_args={"check_same_thread": False} if is_sqlite else {},
        pool_pre_ping=True,
        echo=False,
    )

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    logger.info("Database engine created for %s", url)
    return engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,  # allow reading attributes after commit
    )


def init_database() -> None:
    """Create any missing tables.

    Adequate for V1's single table. A schema migration tool (Alembic) becomes
    worthwhile once the schema changes against data someone cares about.
    """
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database schema ready")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for use outside a request.

    Background analysis tasks have no request lifecycle to hang a dependency
    off, so they use this instead: commit on success, roll back on exception,
    always close.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Drop cached engine and session factory. Used by tests after repointing
    the database URL."""
    get_engine.cache_clear()
    get_session_factory.cache_clear()
