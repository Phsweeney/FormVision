"""Scoring reps with the trained fault detectors.

Mirrors `analysis/pose/base.py`: an abstract interface, a real implementation,
and a null one. Everything downstream consumes `FaultPrediction` and does not
know or care which produced it.

**The null case is the important one.** With no artifact on disk, with the ML
layer switched off, or with scikit-learn not installed, `NullFaultPredictor`
returns nothing and the pipeline produces exactly what it produced before this
package existed. That is not a fallback bolted on afterwards, it is the reason
the interface exists: the rules are the floor, and the model is only ever
allowed to add.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.analysis.types import AngleSeries, FaultPrediction, Rep
from app.config import Settings
from app.logging_config import get_logger
from app.ml.adapter import clip_reference, clip_samples, rep_frame_range
from app.ml.features import SHIPPED_FAULTS, build_frame_features, feature_row

logger = get_logger(__name__)


class FaultPredictor(ABC):
    """Judges reps against the trained fault detectors."""

    #: Stable identifier, recorded alongside results.
    name: str = "base"

    @abstractmethod
    def predict(
        self, angles: AngleSeries, reps: Sequence[Rep], settings: Settings
    ) -> list[FaultPrediction]:
        """Score every rep against every shipped detector.

        Implementations must never raise. A model that cannot answer returns
        nothing, because the caller is a background task whose failure would
        otherwise leave an analysis stuck in `processing` forever.
        """


class NullFaultPredictor(FaultPredictor):
    """Says nothing. The behaviour of the whole app before the ML layer."""

    name = "null"

    def predict(
        self,
        angles: AngleSeries,  # noqa: ARG002 - interface
        reps: Sequence[Rep],  # noqa: ARG002 - interface
        settings: Settings,  # noqa: ARG002 - interface
    ) -> list[FaultPrediction]:
        return []


class SklearnFaultPredictor(FaultPredictor):
    """Scores reps with the artifact produced by `training/train.py`.

    The bundle is loaded once, lazily, behind a lock. Lazily because analysis
    runs on a background thread and the API process should not pay for scipy at
    startup if no one ever uploads a video; behind a lock because two uploads
    can be processed concurrently and unpickling twice would be wasteful at best.
    """

    name = "sklearn"

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._lock = threading.Lock()
        self._bundle: dict[str, Any] | None = None
        self._load_failed = False

    def _faults(self) -> dict[str, Any]:
        """The loaded per-fault detectors, or an empty mapping if unavailable."""
        if self._bundle is not None:
            return self._bundle
        if self._load_failed:
            return {}

        with self._lock:
            # Re-check inside the lock: another thread may have loaded it while
            # this one waited.
            if self._bundle is not None:
                return self._bundle
            if self._load_failed:
                return {}

            self._bundle = self._load()
            return self._bundle

    def _load(self) -> dict[str, Any]:
        """Read the artifact, degrading to silence on any problem."""
        try:
            import joblib
            import sklearn
        except ImportError:
            logger.info("scikit-learn is not installed; model feedback is disabled")
            self._load_failed = True
            return {}

        if not self._model_path.exists():
            logger.info(
                "No model artifact at %s; model feedback is disabled", self._model_path
            )
            self._load_failed = True
            return {}

        try:
            bundle = joblib.load(self._model_path)
        except Exception:
            # Broad by intent. An unreadable artifact must cost the ML feedback
            # and nothing else; the analysis itself is unaffected.
            logger.exception("Could not load model artifact %s", self._model_path)
            self._load_failed = True
            return {}

        trained_with = bundle.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            # Not fatal: scikit-learn pickles usually load across minor versions.
            # Worth saying out loud, because when they do not the failure is
            # silently wrong output rather than an exception.
            logger.warning(
                "Model artifact was trained with scikit-learn %s but %s is "
                "installed; predictions may be unreliable",
                trained_with,
                sklearn.__version__,
            )

        faults = bundle.get("faults", {})
        logger.info(
            "Loaded %d fault detector(s) from %s: %s",
            len(faults),
            self._model_path.name,
            ", ".join(sorted(faults)) or "none",
        )
        return faults

    def predict(
        self, angles: AngleSeries, reps: Sequence[Rep], settings: Settings
    ) -> list[FaultPrediction]:
        faults = self._faults()
        if not faults or not reps or not len(angles):
            return []

        try:
            return self._predict(faults, angles, reps, settings)
        except Exception:
            # Same reasoning as `_load`: the model is an addition, and an
            # addition must not be able to take the analysis down with it.
            logger.exception("Fault prediction failed; continuing without it")
            return []

    def _predict(
        self,
        faults: dict[str, Any],
        angles: AngleSeries,
        reps: Sequence[Rep],
        settings: Settings,
    ) -> list[FaultPrediction]:
        import numpy as np

        samples = clip_samples(angles)

        # The reference is the whole clip, standing frames included. That is the
        # closest inference can get to the corpus's "this video's normal
        # movement", which was whole videos of people squatting. See `corpus.py`
        # for why training builds it differently and what that costs.
        #
        # Restricting it to frames inside reps was tried, for a real reason:
        # ranking is relative, so time spent standing dilutes the distribution
        # and desensitises every detector. On a synthetic one-sided squat, a
        # half-second pause between reps had asymmetry firing on 5 of 5 reps and
        # a four-second pause on none, for identical movement. Excluding
        # standing frames fixed that cleanly.
        #
        # It also made real footage worse, which is what settles it. On a
        # deliberately one-sided clip the asymmetry detector dropped from 0.55
        # to 0.21 affected and stopped firing, while valgus rose from 0.00 to
        # 0.45 and started, a false positive replacing a true one. Real clips
        # are mostly movement already, so removing their standing frames takes
        # away the neutral end of the range rather than a dead weight.
        #
        # The dilution is therefore a documented limitation, not a bug to fix
        # here: long rests between reps make the detectors quieter. See
        # `docs/ml.md`.
        reference = clip_reference(samples)
        features = [build_frame_features(sample, reference) for sample in samples]

        predictions: list[FaultPrediction] = []

        for fault_id in SHIPPED_FAULTS:
            detector = faults.get(fault_id)
            if detector is None:
                continue

            model = detector["model"]
            names = detector["features"]
            threshold = float(detector["threshold"])

            for rep in reps:
                indices = rep_frame_range(rep, len(samples))
                rows: list[list[float | None]] = []
                completeness: list[float] = []

                for index in indices:
                    row, complete = feature_row(features[index], names)
                    if complete <= 0.0:
                        continue
                    rows.append(row)
                    completeness.append(complete)

                if not rows:
                    continue

                mean_completeness = sum(completeness) / len(completeness)
                probabilities = model.predict_proba(np.array(rows, dtype=float))[:, 1]

                probability = float(probabilities.mean())
                affected = float((probabilities >= threshold).mean())

                predictions.append(
                    FaultPrediction(
                        fault_id=fault_id,
                        rep_index=rep.index,
                        probability=probability,
                        affected_fraction=affected,
                        threshold=threshold,
                        feature_completeness=mean_completeness,
                        fired=(
                            mean_completeness >= settings.ml_min_feature_completeness
                            and affected >= settings.ml_min_affected_fraction
                        ),
                    )
                )

        return predictions
