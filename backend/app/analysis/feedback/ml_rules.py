"""Coaching rules backed by a trained model rather than a threshold.

These are ordinary `FeedbackRule`s. The engine, the API, and the frontend do not
special-case them; what marks them out is `source = FeedbackSource.MODEL`, which
travels with each item so the UI can say plainly where the advice came from.

**Why these three faults and not the other two.** The corpus labels five, but
FormVision already measures depth and torso lean directly, in its own units,
with thresholds a coach can read and argue with. A model fitted on synthetic
offsets is strictly worse than a correct measurement, so depth and lean stay
rule-based and the two detectors trained for them are reported in the model card
and never consulted. What is left is exactly the set FormVision could not judge
at all before: knees caving, heels lifting, and left/right asymmetry.

**What these detectors can and cannot see.** They score how far a moment stands
out from the lifter's own movement in that clip, which is what lets a model
trained on one feature convention work on another. The cost is that a fault
present in *every* rep raises the baseline it would be measured against and
stops standing out. These rules find the rep that broke the pattern; they are
not a substitute for an absolute standard, and they say so in their own copy.
"""

from __future__ import annotations

from math import ceil

from app.analysis.feedback.base import FeedbackContext, FeedbackRule
from app.analysis.types import FaultPrediction, FeedbackItem, FeedbackSource, Severity
from app.config import Settings


def _required_reps(judged: int, settings: Settings) -> int:
    """How many flagged reps it takes to say something, for a clip this long.

    Proportional rather than a fixed count, and never below one. A fixed count
    gets *stricter* as the clip gets shorter, which is the wrong way round: one
    flagged rep in ten is as likely to be a tracking artefact as a fault, while
    one in two is half the evidence available.
    """
    if judged <= 0:
        return 1
    return max(1, ceil(settings.ml_min_affected_rep_fraction * judged))


def _describe_reps(predictions: tuple[FaultPrediction, ...]) -> str:
    """Name the affected reps the way a lifter counts them."""
    numbers = [str(prediction.rep_index) for prediction in predictions]
    if len(numbers) == 1:
        return f"rep {numbers[0]}"
    return f"reps {', '.join(numbers[:-1])} and {numbers[-1]}"


class ModelFaultRule(FeedbackRule):
    """Shared behaviour for the model-backed rules.

    Subclasses supply the fault id and the wording. Everything about *when* to
    speak lives here, so the three rules cannot drift apart on the question that
    matters most.
    """

    source = FeedbackSource.MODEL

    #: Which detector in the artifact this rule reads.
    fault_id: str = ""

    title: str = ""
    #: Filled with the affected rep description and the count.
    body: str = ""
    explanation: str = ""

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        scored = context.scored_predictions(self.fault_id)
        if not scored:
            # No prediction at all: the layer is off, the artifact is missing, or
            # this camera angle cannot see the fault. All three are reasons to
            # say nothing, and none of them is reason to say the lifter is fine.
            return None

        fired = context.fired_predictions(self.fault_id)
        if len(fired) < _required_reps(len(scored), context.settings):
            return None

        confidence = sum(prediction.probability or 0.0 for prediction in fired) / len(
            fired
        )

        return self._item(
            Severity.WARNING,
            self.title,
            self.body.format(
                reps=_describe_reps(fired),
                count=len(fired),
                total=len(scored),
            ),
            self.explanation,
            confidence=confidence,
        )


class KneeValgusModelRule(ModelFaultRule):
    """Knees travelling inward under load. Front-on footage only."""

    rule_id = "ml_knee_valgus"
    priority = 15
    fault_id = "knee_valgus"

    title = "Knees tracking inward"
    body = (
        "On {count} of {total} judged repetitions ({reps}), your knees drifted "
        "toward the midline relative to how they tracked across the rest of the set."
    )
    explanation = (
        "Knees collapsing inward as you drive out of the bottom puts the load "
        "across the knee rather than through it, and it usually means the hips "
        "are not contributing what they should. Cue driving the knees out over "
        "the middle of the foot through the whole ascent. This reading comes "
        "from a model comparing each repetition against the rest of your set, "
        "so it finds the reps that broke your pattern rather than judging your "
        "stance against a fixed standard."
    )


class HeelLiftModelRule(ModelFaultRule):
    """Heels coming off the floor. Side-on footage only."""

    rule_id = "ml_heel_lift"
    priority = 16
    fault_id = "heel_lift"

    title = "Heels coming up"
    body = (
        "On {count} of {total} judged repetitions ({reps}), your shin travelled "
        "further over the foot than it did elsewhere in the set, which is what "
        "a heel leaving the floor looks like from the side."
    )
    explanation = (
        "Once the heel lifts, the load shifts onto the front of the foot and you "
        "lose the floor to push against, which is both a stability and a knee "
        "problem. It is most often limited ankle mobility rather than a cue you "
        "are missing; a slightly wider stance, more toe-out, or a raised heel "
        "shoe usually fixes it faster than trying harder. This reading comes "
        "from a model comparing each repetition against the rest of your set."
    )


class AsymmetryModelRule(ModelFaultRule):
    """One side working harder than the other."""

    rule_id = "ml_asymmetry"
    priority = 35
    fault_id = "asymmetry"

    title = "Sides loading unevenly"
    body = (
        "On {count} of {total} judged repetitions ({reps}), your left and right "
        "legs moved through noticeably different angles."
    )
    explanation = (
        "A consistent side-to-side difference means one leg is taking more of "
        "the bar than the other, which limits what you can load and is worth "
        "addressing before it becomes an injury. Filming from the front and "
        "watching whether the bar stays level is the quickest confirmation. "
        "This reading comes from a model comparing each repetition against the "
        "rest of your set, so a difference present on every single rep may not "
        "register: it has nothing within the set to stand out against."
    )


#: The model-backed rules, appended to `DEFAULT_RULES` in `rules.py`.
MODEL_RULES: tuple[type[FeedbackRule], ...] = (
    KneeValgusModelRule,
    HeelLiftModelRule,
    AsymmetryModelRule,
)
