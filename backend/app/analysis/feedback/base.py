"""The coaching rule interface.

Each piece of feedback is a self-contained class. Adding advice to FormVision
means writing one class and registering it; no existing rule, and no part of the
pipeline, changes.

That structure is what keeps V1 honest about being rule-based and V2 open to
being smarter. An ML-backed judgement is just a `FeedbackRule` whose `evaluate`
consults a model — the engine, the API, and the frontend never learn the
difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.analysis.types import (
    AngleSeries,
    FaultPrediction,
    FeedbackItem,
    FeedbackSource,
    Metrics,
    Rep,
    Severity,
)
from app.config import Settings


@dataclass(frozen=True, slots=True)
class FeedbackContext:
    """Everything a rule may inspect.

    Passed as one object rather than as separate arguments so that adding a new
    input for a future rule does not change the signature of every existing one.
    Model predictions arriving here without touching any of the eight original
    rules is that design paying for itself.
    """

    reps: tuple[Rep, ...]
    metrics: Metrics
    angles: AngleSeries
    settings: Settings

    #: Per-rep model verdicts. Empty whenever the ML layer is off or silent.
    predictions: tuple[FaultPrediction, ...] = ()

    def fired_predictions(self, fault_id: str) -> tuple[FaultPrediction, ...]:
        """Reps where this fault cleared the model's bar, in rep order."""
        return tuple(
            prediction
            for prediction in self.predictions
            if prediction.fault_id == fault_id and prediction.fired
        )

    def scored_predictions(self, fault_id: str) -> tuple[FaultPrediction, ...]:
        """Every rep this fault was able to judge, fired or not.

        The denominator for "3 of 5 reps": a rep the camera could not judge is
        not a rep the lifter got right.
        """
        return tuple(
            prediction
            for prediction in self.predictions
            if prediction.fault_id == fault_id
        )


class FeedbackRule(ABC):
    """One coaching heuristic."""

    #: Stable identifier. Sent to the frontend and safe to key UI logic on;
    #: unlike the message text, it must not change casually.
    rule_id: str = "rule"

    #: Lower numbers surface first. Safety and data-quality warnings are given
    #: low values so they appear above technique commentary that may be based
    #: on unreliable measurements.
    priority: int = 100

    @abstractmethod
    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        """Judge the set, or return None to stay silent.

        Rules must return None rather than a neutral item when they have
        nothing to say. A dashboard listing every rule that did not fire is
        noise, and it buries the advice that matters.
        """

    #: Whether this rule's advice comes from a measurement or from a model.
    #: Declared on the class so a rule cannot forget to say, and so the answer
    #: is visible when reading the class rather than buried in a call.
    source: FeedbackSource = FeedbackSource.RULE

    def _item(
        self,
        severity: Severity,
        title: str,
        message: str,
        explanation: str,
        confidence: float | None = None,
    ) -> FeedbackItem:
        """Build a `FeedbackItem` carrying this rule's id and provenance."""
        return FeedbackItem(
            rule_id=self.rule_id,
            severity=severity,
            title=title,
            message=message,
            explanation=explanation,
            source=self.source,
            confidence=confidence,
        )
