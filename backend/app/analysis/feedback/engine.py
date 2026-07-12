"""Runs the coaching rules and orders the results."""

from __future__ import annotations

from collections.abc import Sequence

from app.analysis.feedback.base import FeedbackContext, FeedbackRule
from app.analysis.feedback.rules import DEFAULT_RULES
from app.analysis.types import AngleSeries, FeedbackItem, Metrics, Rep, Severity
from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Order severities are presented in. Problems first, praise last — a lifter
#: scanning the panel should meet what needs fixing before what went well.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
    Severity.GOOD: 3,
}


class FeedbackEngine:
    """Evaluates a set of rules against one analysis."""

    def __init__(self, rules: Sequence[type[FeedbackRule]] | None = None) -> None:
        self._rules = [rule() for rule in (rules or DEFAULT_RULES)]

    def generate(
        self,
        reps: Sequence[Rep],
        metrics: Metrics,
        angles: AngleSeries,
        settings: Settings,
    ) -> list[FeedbackItem]:
        """Run every rule and return the items that fired, ordered for display."""
        context = FeedbackContext(
            reps=tuple(reps), metrics=metrics, angles=angles, settings=settings
        )

        items: list[tuple[int, int, FeedbackItem]] = []
        for rule in self._rules:
            try:
                item = rule.evaluate(context)
            except Exception:
                # One malformed rule must not cost the lifter their entire
                # analysis. Log it and carry on with the rest.
                logger.exception("Feedback rule '%s' failed", rule.rule_id)
                continue

            if item is not None:
                items.append((_SEVERITY_ORDER[item.severity], rule.priority, item))

        # Sort by severity first, then by the rule's own priority, so that
        # (for example) a depth problem outranks a tempo problem but any
        # warning outranks any praise.
        items.sort(key=lambda entry: (entry[0], entry[1]))

        result = [item for _, _, item in items]
        logger.info(
            "Generated %d feedback items (%s)",
            len(result),
            ", ".join(item.rule_id for item in result) or "none",
        )
        return result


def generate_feedback(
    reps: Sequence[Rep],
    metrics: Metrics,
    angles: AngleSeries,
    settings: Settings,
) -> list[FeedbackItem]:
    """Convenience wrapper using the default rule set."""
    return FeedbackEngine().generate(reps, metrics, angles, settings)
