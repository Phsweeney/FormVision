"""Pose estimator registry.

The single place that maps a configuration string to a `PoseEstimator`
implementation. Adding a real-time or ML-based estimator in V2 means writing
the class and calling `register_estimator` — no other module changes.

Implementations are registered as *factories* rather than instances so that
each analysis run gets its own estimator. MediaPipe landmarkers hold native
state and are not safe to share across threads, and analysis runs on a
background worker.
"""

from __future__ import annotations

from collections.abc import Callable

from app.analysis.pose.base import PoseEstimator
from app.config import Settings, get_settings
from app.core.exceptions import PoseEstimationError
from app.logging_config import get_logger

logger = get_logger(__name__)

EstimatorFactory = Callable[[Settings], PoseEstimator]

_REGISTRY: dict[str, EstimatorFactory] = {}


def register_estimator(name: str, factory: EstimatorFactory) -> None:
    """Register an estimator implementation under ``name``."""
    _REGISTRY[name] = factory
    logger.debug("Registered pose estimator '%s'", name)


def available_estimators() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_estimator(
    name: str | None = None, settings: Settings | None = None
) -> PoseEstimator:
    """Build the configured estimator.

    Args:
        name: Override the configured implementation. Used by tests to inject
            a synthetic estimator.
        settings: Override application settings.
    """
    settings = settings or get_settings()
    key = name or settings.pose_estimator

    factory = _REGISTRY.get(key)
    if factory is None:
        raise PoseEstimationError(
            f"Unknown pose estimator '{key}'.",
            detail={"available": list(available_estimators())},
        )
    return factory(settings)


def _mediapipe_factory(settings: Settings) -> PoseEstimator:
    """Import MediaPipe lazily.

    Keeps the ~200 MB import (and its native library loading) out of processes
    that never run analysis — test collection, linting, and the API's own
    startup all import this module.
    """
    from app.analysis.pose.mediapipe_estimator import MediaPipePoseEstimator

    return MediaPipePoseEstimator(settings)


register_estimator("mediapipe", _mediapipe_factory)
