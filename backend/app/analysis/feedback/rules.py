"""The V1 coaching rules.

Each rule is deliberately simple and explainable — a lifter should be able to
read the message, look at the graph above it, and see exactly why it fired.
Every threshold comes from `Settings`, so the coaching standard is
configuration rather than code.

**Every rule states its own limits.** Torso lean is only meaningful from a
side-on camera; left/right asymmetry is only meaningful from the front.
`analysis/view.py` detects which one it is, and the signal the camera cannot see
arrives here as None — so those rules stay silent by construction rather than by
remembering to check. `CameraViewRule` then explains the silence, because a
measurement that is simply absent reads as a bug unless something says why.
Confidently wrong coaching is worse than coaching that admits its assumptions.
"""

from __future__ import annotations

from app.analysis.feedback.base import FeedbackContext, FeedbackRule
from app.analysis.feedback.ml_rules import MODEL_RULES
from app.analysis.types import FeedbackItem, Severity, ViewOrientation


class NoRepsDetectedRule(FeedbackRule):
    """Nothing was counted — explain why before anything else."""

    rule_id = "no_reps_detected"
    priority = 0

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        if context.reps:
            return None

        # Distinguish "we could not see you" from "you did not squat". They
        # need completely different corrective action from the user.
        if context.metrics.tracking_quality < context.settings.min_tracking_quality:
            return self._item(
                Severity.CRITICAL,
                "No repetitions detected",
                "We could not track your body clearly enough to count repetitions.",
                "Film with your whole body in frame, in even lighting, against an "
                "uncluttered background, with only one person visible.",
            )

        return self._item(
            Severity.WARNING,
            "No repetitions detected",
            "Your body was tracked successfully, but no complete squat was found.",
            "A repetition is counted when you descend and then return to a "
            "standing position. Partial reps, or a clip that ends before you "
            "stand back up, are not counted.",
        )


class TrackingQualityRule(FeedbackRule):
    """Warn when the measurements themselves are unreliable.

    Deliberately high priority: if this fires, everything below it is suspect,
    and the lifter deserves to know that before reading advice derived from it.
    """

    rule_id = "tracking_quality"
    priority = 1

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        quality = context.metrics.tracking_quality
        if quality >= context.settings.min_tracking_quality:
            return None
        if not context.reps:
            return None  # NoRepsDetectedRule already covers this case.

        return self._item(
            Severity.WARNING,
            "Limited tracking quality",
            f"Your body was clearly visible in only {quality * 100:.0f}% of frames, "
            "so the measurements below are approximate.",
            "Pose tracking degrades with poor lighting, loose clothing, body parts "
            "leaving frame, or other people in shot. Re-filming with the whole "
            "body visible will make the analysis more reliable.",
        )


class CameraViewRule(FeedbackRule):
    """Say which camera angle was detected, and what it can and cannot measure.

    Sits just under the tracking-quality warning because it frames everything
    below it. Without this, a torso-lean card reading "—" on front-on footage
    looks like a broken feature rather than the honest answer: that angle does
    not contain the information, so nothing was reported.
    """

    rule_id = "camera_view"
    priority = 2

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        view = context.angles.view

        if view is ViewOrientation.SIDE:
            return self._item(
                Severity.INFO,
                "Filmed from the side",
                "Squat depth and torso lean are measured from this angle.",
                "A side-on camera is the best view for depth and for how far "
                "your torso pitches forward. It cannot compare your left and "
                "right sides, because one leg is hidden behind the other — for "
                "that, film a set from the front.",
            )

        if view is ViewOrientation.FRONT:
            return self._item(
                Severity.INFO,
                "Filmed from the front",
                "Left/right symmetry is measured from this angle; torso lean is not.",
                "A front-on camera shows both legs separately, which is what "
                "makes a side-to-side comparison meaningful. It sees your torso "
                "almost edge-on, so forward lean cannot be measured from it at "
                "all — film a set from the side for that.",
            )

        if view is ViewOrientation.OBLIQUE:
            return self._item(
                Severity.WARNING,
                "Camera at an angle",
                "The camera was neither square-on nor side-on to you, so the "
                "measurements below are approximate.",
                "Filming at a diagonal foreshortens everything the analysis "
                "measures. Placing the camera directly to your side, or "
                "directly in front, at about hip height and far enough back to "
                "keep your whole body in frame, makes the numbers comparable "
                "from session to session.",
            )

        return None  # UNKNOWN: tracking failed, and that rule speaks first.


class DepthRule(FeedbackRule):
    """Judge squat depth — the headline coaching point for a squat."""

    rule_id = "squat_depth"
    priority = 10

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        average = context.metrics.avg_depth_percent
        if average is None:
            return None

        settings = context.settings
        total = len(context.reps)

        # Count against the same standard the depth percentage is derived from.
        # `hip_below_knee` is a different, stricter geometric test, and mixing
        # the two in one sentence produces contradictions like "97% of target
        # depth, reaching parallel on 0 of 5 reps".
        on_target = sum(
            1
            for rep in context.reps
            if rep.depth_percent is not None and rep.depth_percent >= 100.0
        )
        hips_below = sum(1 for rep in context.reps if rep.hip_below_knee)

        # Mentioned only when it happened; it is extra credit beyond the target,
        # not a second criterion the lifter is being judged against.
        hips_note = (
            f" Your hips passed below knee level on {hips_below} of {total}."
            if hips_below
            else ""
        )

        if average >= settings.good_depth_percent:
            return self._item(
                Severity.GOOD,
                "Good squat depth",
                f"You averaged {average:.0f}% of target depth, hitting the target "
                f"on {on_target} of {total} repetitions.{hips_note}",
                "Squatting to at least parallel - hip crease level with the top of "
                "the knee - trains the quadriceps and glutes through their full "
                "range and is the standard most strength programmes assume.",
            )

        if average < settings.shallow_depth_percent:
            return self._item(
                Severity.CRITICAL,
                "Not reaching full depth",
                f"You averaged {average:.0f}% of target depth. Your knees bent to "
                f"{context.metrics.min_knee_angle_deg:.0f} degrees at the deepest point.",
                "Partial squats limit the range the glutes and hamstrings work "
                "through. If depth is restricted by ankle or hip mobility rather "
                "than by load, reducing the weight and working on mobility usually "
                "resolves it faster than pushing through.",
            )

        return self._item(
            Severity.WARNING,
            "Close to full depth",
            f"You averaged {average:.0f}% of target depth, hitting the target on "
            f"{on_target} of {total} repetitions.{hips_note}",
            "You are near the target. Descending slightly further, or slowing the "
            "descent so you can control the bottom position, will usually close "
            "the gap.",
        )


class ForwardLeanRule(FeedbackRule):
    """Flag excessive forward inclination of the torso."""

    rule_id = "forward_lean"
    priority = 20

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        max_lean = context.metrics.max_torso_lean_deg
        average = context.metrics.avg_torso_lean_deg
        if max_lean is None or average is None:
            return None

        limit = context.settings.max_torso_lean_deg

        if average > limit:
            return self._item(
                Severity.CRITICAL,
                "Excessive forward lean",
                f"Your torso averaged {average:.0f} degrees from vertical, peaking at "
                f"{max_lean:.0f} degrees.",
                "A torso that pitches this far forward shifts load from the legs "
                "onto the lower back and turns the squat into something closer to a "
                "good morning. It is commonly caused by limited ankle mobility, an "
                "overly narrow stance, or simply too much weight. "
                "Note: this measurement assumes the camera is side-on to you.",
            )

        if max_lean > limit:
            return self._item(
                Severity.WARNING,
                "Forward lean on the deepest reps",
                f"Your torso stayed at {average:.0f} degrees on average but reached "
                f"{max_lean:.0f} degrees at its worst.",
                "Leaning further as you tire, or on your deepest repetitions, "
                "usually points to the load being near your limit. "
                "Note: this measurement assumes the camera is side-on to you.",
            )

        return self._item(
            Severity.GOOD,
            "Good torso position",
            f"Your torso stayed within {max_lean:.0f} degrees of vertical throughout.",
            "Keeping the torso relatively upright keeps the load over your midfoot "
            "and the work in your legs rather than your lower back.",
        )


class AsymmetryRule(FeedbackRule):
    """Flag uneven loading between the left and right leg."""

    rule_id = "knee_asymmetry"
    priority = 30

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        average = context.metrics.avg_knee_asymmetry_deg
        if average is None:
            return None

        limit = context.settings.max_knee_asymmetry_deg

        if average > limit:
            worst = max(
                (rep for rep in context.reps if rep.knee_asymmetry_deg is not None),
                key=lambda rep: rep.knee_asymmetry_deg,
                default=None,
            )
            worst_text = (
                f" The largest difference was {worst.knee_asymmetry_deg:.0f} degrees "
                f"on repetition {worst.index}."
                if worst
                else ""
            )
            return self._item(
                Severity.WARNING,
                "Uneven squat mechanics",
                f"Your left and right knee angles differed by {average:.0f} degrees "
                f"on average.{worst_text}",
                "A consistent side-to-side difference usually means you are shifting "
                "weight onto one leg, often from a mobility restriction or a past "
                "injury on the other side. "
                "Note: this is measured most reliably from a front-on camera.",
            )

        return self._item(
            Severity.GOOD,
            "Balanced left and right",
            f"Your knee angles matched to within {average:.0f} degrees on average.",
            "Even loading through both legs distributes the work as intended and "
            "avoids overdeveloping one side.",
        )


class ConsistencyRule(FeedbackRule):
    """Judge how repeatable the set was, rep to rep."""

    rule_id = "rep_consistency"
    priority = 40

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        # Consistency is a claim about variation between reps, so it needs at
        # least three to say anything meaningful about a set.
        if len(context.reps) < 3:
            return None

        depth_spread = context.metrics.depth_consistency_percent
        duration_spread = context.metrics.duration_consistency_s
        if depth_spread is None or duration_spread is None:
            return None

        settings = context.settings
        depth_inconsistent = depth_spread > settings.max_depth_variation_percent
        timing_inconsistent = duration_spread > settings.max_duration_variation_s

        if depth_inconsistent or timing_inconsistent:
            problems = []
            if depth_inconsistent:
                problems.append(f"depth varied by {depth_spread:.0f} percentage points")
            if timing_inconsistent:
                problems.append(f"rep duration varied by {duration_spread:.1f}s")

            return self._item(
                Severity.WARNING,
                "Inconsistent repetition quality",
                f"Across {len(context.reps)} repetitions, {' and '.join(problems)}.",
                "Reps that drift in depth or tempo through a set usually indicate "
                "fatigue, or a weight heavy enough that form degrades before the set "
                "ends. Consistent reps are what make a set comparable week to week.",
            )

        return self._item(
            Severity.GOOD,
            "Consistent repetition quality",
            f"All {len(context.reps)} repetitions were closely matched in depth "
            f"(within {depth_spread:.0f}%) and timing (within {duration_spread:.1f}s).",
            "Repeatable reps mean you are working within your capacity and that "
            "your set-to-set comparisons are meaningful.",
        )


class TempoRule(FeedbackRule):
    """Comment on how fast the reps were performed."""

    rule_id = "rep_tempo"
    priority = 50

    def evaluate(self, context: FeedbackContext) -> FeedbackItem | None:
        average = context.metrics.avg_rep_duration_s
        if average is None:
            return None

        minimum = context.settings.min_rep_tempo_s
        eccentric = context.metrics.avg_eccentric_s
        concentric = context.metrics.avg_concentric_s

        if average < minimum:
            return self._item(
                Severity.WARNING,
                "Rushed repetitions",
                f"Your repetitions averaged {average:.1f}s "
                f"({eccentric:.1f}s down, {concentric:.1f}s up).",
                "Fast reps rely on momentum out of the bottom rather than on the "
                "muscles working through the range. A controlled descent of around "
                "two seconds increases time under tension and gives you a chance to "
                "correct position before driving back up.",
            )

        return self._item(
            Severity.GOOD,
            "Controlled tempo",
            f"Your repetitions averaged {average:.1f}s "
            f"({eccentric:.1f}s down, {concentric:.1f}s up).",
            "A controlled tempo keeps tension on the muscle through the full range "
            "and gives you time to hold position at the bottom.",
        )


#: The rules evaluated for every analysis, in registration order. The engine
#: sorts by `priority`, so this list only needs to be complete, not ordered.
#:
#: The model-backed rules are appended rather than interleaved, and they stay
#: silent whenever the ML layer has nothing to say, so this tuple describes the
#: same behaviour as before when no artifact is present.
DEFAULT_RULES: tuple[type[FeedbackRule], ...] = (
    NoRepsDetectedRule,
    TrackingQualityRule,
    CameraViewRule,
    DepthRule,
    ForwardLeanRule,
    AsymmetryRule,
    ConsistencyRule,
    TempoRule,
    *MODEL_RULES,
)
