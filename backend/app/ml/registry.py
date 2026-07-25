"""Fault predictor registry.

The same shape as `analysis/pose/registry.py`, for the same reasons: factories
rather than instances, so each analysis run gets its own predictor and nothing
native or stateful is shared across the background worker threads.

`set_predictor_override` is the test seam, matching
`services/analysis_service.set_estimator_override`. It lets a test inject a stub
predictor and assert on the rules without a trained artifact anywhere near it.
"""

from __future__ import annotations

from collections.abc import Callable

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.ml.predictor import FaultPredictor, NullFaultPredictor, SklearnFaultPredictor

logger = get_logger(__name__)

PredictorFactory = Callable[[Settings], FaultPredictor]

_REGISTRY: dict[str, PredictorFactory] = {}
_OVERRIDE: FaultPredictor | None = None


def register_predictor(name: str, factory: PredictorFactory) -> None:
    """Register a predictor implementation under ``name``."""
    _REGISTRY[name] = factory
    logger.debug("Registered fault predictor '%s'", name)


def available_predictors() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def set_predictor_override(predictor: FaultPredictor | None) -> None:
    """Force `create_predictor` to return ``predictor``. Tests only."""
    global _OVERRIDE
    _OVERRIDE = predictor


def create_predictor(settings: Settings | None = None) -> FaultPredictor:
    """Build the configured predictor.

    Unlike `create_estimator`, this never raises. Pose estimation failing is
    fatal to an analysis; the ML layer failing is not, and an unknown name in
    configuration should cost the model's opinion rather than the whole upload.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE

    settings = settings or get_settings()
    if not settings.ml_enabled:
        return NullFaultPredictor()

    factory = _REGISTRY.get(settings.ml_predictor)
    if factory is None:
        logger.warning(
            "Unknown fault predictor '%s'; model feedback is disabled. Available: %s",
            settings.ml_predictor,
            ", ".join(available_predictors()) or "none",
        )
        return NullFaultPredictor()

    return factory(settings)


def _sklearn_factory(settings: Settings) -> FaultPredictor:
    return SklearnFaultPredictor(settings.ml_model_path)


register_predictor("sklearn", _sklearn_factory)
register_predictor("null", lambda _settings: NullFaultPredictor())
