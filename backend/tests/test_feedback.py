"""Tests for the coaching rules and the engine that runs them.

Each rule is tested both firing and staying quiet, because a rule that fires on
everything is as useless as one that never fires. The engine tests cover
ordering and — importantly — that one broken rule cannot take down the analysis.
"""

from __future__ import annotations

import pytest

from app.analysis.angles import compute_angles
from app.analysis.feedback.base import FeedbackContext, FeedbackRule
from app.analysis.feedback.engine import FeedbackEngine, generate_feedback
from app.analysis.feedback.rules import (
    DEFAULT_RULES,
    AsymmetryRule,
    ConsistencyRule,
    DepthRule,
    ForwardLeanRule,
    NoRepsDetectedRule,
    TempoRule,
    TrackingQualityRule,
)
from app.analysis.metrics import compute_metrics
from app.analysis.reps import detect_reps
from app.analysis.types import Severity
from app.config import Settings
from tests.synthetic import build_squat_series, build_standing_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


def context_for(series, settings: Settings) -> FeedbackContext:
    angles = compute_angles(series, settings)
    reps = detect_reps(angles, settings)
    metrics = compute_metrics(reps, angles, series.metadata.duration_s, settings)
    return FeedbackContext(
        reps=tuple(reps), metrics=metrics, angles=angles, settings=settings
    )


def ids_for(series, settings: Settings) -> list[str]:
    ctx = context_for(series, settings)
    return [
        item.rule_id
        for item in generate_feedback(ctx.reps, ctx.metrics, ctx.angles, settings)
    ]


def context_with_metrics(settings: Settings, reps: int = 4, **metric_overrides):
    """Build a context around hand-specified metrics.

    Some rule branches compare an average against a peak. Synthetic reps are
    identical to each other, so the average equals the peak and those branches
    are unreachable through the full pipeline — even though they are perfectly
    reachable with real footage, where reps differ. Constructing the metrics
    directly tests the rule's own logic without contorting the generator.
    """
    from app.analysis.types import Metrics

    base = {
        "total_reps": reps,
        "video_duration_s": 20.0,
        "total_workout_time_s": 15.0,
        "avg_depth_percent": 95.0,
        "max_depth_percent": 100.0,
        "min_knee_angle_deg": 85.0,
        "avg_rep_duration_s": 2.5,
        "fastest_rep_s": 2.3,
        "slowest_rep_s": 2.7,
        "avg_eccentric_s": 1.3,
        "avg_concentric_s": 1.2,
        "reps_per_minute": 16.0,
        "avg_torso_lean_deg": 15.0,
        "max_torso_lean_deg": 20.0,
        "avg_knee_asymmetry_deg": 3.0,
        "depth_consistency_percent": 2.0,
        "duration_consistency_s": 0.1,
        "tracking_quality": 1.0,
    }
    base.update(metric_overrides)

    series = build_squat_series(reps=reps)
    angles = compute_angles(series, settings)
    detected = detect_reps(angles, settings)
    return FeedbackContext(
        reps=tuple(detected),
        metrics=Metrics(**base),
        angles=angles,
        settings=settings,
    )


class TestNoRepsDetectedRule:
    def test_fires_when_nothing_was_counted(self, settings):
        item = NoRepsDetectedRule().evaluate(
            context_for(build_standing_series(seconds=4.0), settings)
        )
        assert item is not None
        assert item.rule_id == "no_reps_detected"

    def test_silent_when_reps_exist(self, settings):
        assert (
            NoRepsDetectedRule().evaluate(
                context_for(build_squat_series(reps=3), settings)
            )
            is None
        )

    def test_distinguishes_untracked_from_motionless(self, settings):
        """These need opposite advice: fix your filming vs actually squat."""
        motionless = NoRepsDetectedRule().evaluate(
            context_for(build_standing_series(seconds=4.0), settings)
        )
        untracked = NoRepsDetectedRule().evaluate(
            context_for(
                build_squat_series(reps=3, undetected_frames=tuple(range(500))),
                settings,
            )
        )
        assert motionless.severity is Severity.WARNING
        assert untracked.severity is Severity.CRITICAL
        assert "track" in untracked.message.lower()


class TestTrackingQualityRule:
    def test_silent_on_clean_footage(self, settings):
        assert (
            TrackingQualityRule().evaluate(
                context_for(build_squat_series(reps=3), settings)
            )
            is None
        )

    def test_fires_on_poor_tracking(self, settings):
        series = build_squat_series(reps=3, undetected_frames=tuple(range(0, 110)))
        item = TrackingQualityRule().evaluate(context_for(series, settings))
        assert item is not None
        assert item.severity is Severity.WARNING

    def test_defers_to_no_reps_rule_when_nothing_counted(self, settings):
        """Avoids telling the lifter the same thing twice."""
        series = build_squat_series(reps=3, undetected_frames=tuple(range(500)))
        assert TrackingQualityRule().evaluate(context_for(series, settings)) is None


class TestDepthRule:
    def test_praises_full_depth(self, settings):
        item = DepthRule().evaluate(
            context_for(build_squat_series(reps=3, depth_fraction=1.0), settings)
        )
        assert item.severity is Severity.GOOD

    def test_flags_shallow_squats(self, settings):
        item = DepthRule().evaluate(
            context_for(build_squat_series(reps=3, depth_fraction=0.22), settings)
        )
        assert item.severity is Severity.CRITICAL
        assert "depth" in item.title.lower()

    def test_warns_on_borderline_depth(self, settings):
        item = DepthRule().evaluate(
            context_for(build_squat_series(reps=3, depth_fraction=0.32), settings)
        )
        assert item.severity is Severity.WARNING

    def test_threshold_is_configurable(self, settings):
        """The coaching standard must be a setting, not a constant."""
        series = build_squat_series(reps=3, depth_fraction=0.4)

        settings.good_depth_percent = 80.0
        lenient = DepthRule().evaluate(context_for(series, settings))

        settings.good_depth_percent = 99.0
        settings.shallow_depth_percent = 95.0
        strict = DepthRule().evaluate(context_for(series, settings))

        assert lenient.severity is Severity.GOOD
        assert strict.severity is Severity.CRITICAL

    def test_message_reports_the_measured_value(self, settings):
        item = DepthRule().evaluate(
            context_for(build_squat_series(reps=4, depth_fraction=1.0), settings)
        )
        assert "%" in item.message
        assert item.explanation

    def test_message_does_not_contradict_itself(self, settings):
        """The rep count must use the same standard as the percentage.

        Regression: the count previously came from `hip_below_knee`, a stricter
        geometric test than the knee-angle standard `depth_percent` is derived
        from. That produced sentences like "97% of target depth, reaching
        parallel on 0 of 5 repetitions" — both numbers correct, read together
        nonsense.
        """
        ctx = context_for(build_squat_series(reps=5, depth_fraction=0.55), settings)
        item = DepthRule().evaluate(ctx)

        on_target = sum(1 for rep in ctx.reps if (rep.depth_percent or 0) >= 100.0)
        assert f"{on_target} of {len(ctx.reps)}" in item.message
        # High average depth must never be paired with a zero count.
        if ctx.metrics.avg_depth_percent >= settings.good_depth_percent:
            assert on_target > 0

    def test_hips_below_knee_mentioned_only_when_it_happened(self, settings):
        deep = DepthRule().evaluate(
            context_for(build_squat_series(reps=4, depth_fraction=1.0), settings)
        )
        shallow = DepthRule().evaluate(
            context_for(build_squat_series(reps=4, depth_fraction=0.55), settings)
        )
        assert "below knee level" in deep.message
        assert "below knee level" not in shallow.message


class TestForwardLeanRule:
    def test_praises_an_upright_torso(self, settings):
        item = ForwardLeanRule().evaluate(
            context_for(
                build_squat_series(reps=3, torso_lean_deg=8.0, bottom_lean_deg=15.0),
                settings,
            )
        )
        assert item.severity is Severity.GOOD

    def test_flags_sustained_lean(self, settings):
        item = ForwardLeanRule().evaluate(
            context_for(
                build_squat_series(reps=3, torso_lean_deg=55.0, bottom_lean_deg=65.0),
                settings,
            )
        )
        assert item.severity is Severity.CRITICAL

    def test_warns_when_only_the_worst_reps_lean(self, settings):
        """Average within limits but a peak beyond it: a warning, not a failure.

        This is the "leaning as you fatigue" case, which needs reps that differ
        from one another - hence explicit metrics rather than the generator.
        """
        item = ForwardLeanRule().evaluate(
            context_with_metrics(
                settings, avg_torso_lean_deg=30.0, max_torso_lean_deg=58.0
            )
        )
        assert item.severity is Severity.WARNING

    def test_escalates_when_the_average_itself_is_excessive(self, settings):
        item = ForwardLeanRule().evaluate(
            context_with_metrics(
                settings, avg_torso_lean_deg=55.0, max_torso_lean_deg=62.0
            )
        )
        assert item.severity is Severity.CRITICAL

    def test_states_its_camera_assumption(self, settings):
        """Lean is only meaningful side-on. Saying so beats being confidently
        wrong about someone filming from the front."""
        item = ForwardLeanRule().evaluate(
            context_for(
                build_squat_series(reps=3, torso_lean_deg=55.0, bottom_lean_deg=65.0),
                settings,
            )
        )
        assert "side-on" in item.explanation


class TestAsymmetryRule:
    def test_praises_a_balanced_squat(self, settings):
        item = AsymmetryRule().evaluate(context_for(build_squat_series(reps=3), settings))
        assert item.severity is Severity.GOOD

    def test_flags_uneven_mechanics(self, settings):
        item = AsymmetryRule().evaluate(
            context_for(build_squat_series(reps=3, left_right_bias=0.35), settings)
        )
        assert item.severity is Severity.WARNING
        assert "repetition" in item.message

    def test_threshold_is_configurable(self, settings):
        series = build_squat_series(reps=3, left_right_bias=0.2)
        settings.max_knee_asymmetry_deg = 50.0
        assert AsymmetryRule().evaluate(context_for(series, settings)).severity is (
            Severity.GOOD
        )
        settings.max_knee_asymmetry_deg = 1.0
        assert AsymmetryRule().evaluate(context_for(series, settings)).severity is (
            Severity.WARNING
        )

    def test_states_its_camera_assumption(self, settings):
        item = AsymmetryRule().evaluate(
            context_for(build_squat_series(reps=3, left_right_bias=0.35), settings)
        )
        assert "front-on" in item.explanation


class TestConsistencyRule:
    def test_silent_below_three_reps(self, settings):
        """Two reps cannot evidence a claim about consistency across a set."""
        for count in (1, 2):
            assert (
                ConsistencyRule().evaluate(
                    context_for(build_squat_series(reps=count), settings)
                )
                is None
            )

    def test_praises_uniform_reps(self, settings):
        item = ConsistencyRule().evaluate(
            context_for(build_squat_series(reps=5, depth_jitter=0.0), settings)
        )
        assert item.severity is Severity.GOOD

    def test_flags_drifting_depth(self, settings):
        item = ConsistencyRule().evaluate(
            context_with_metrics(settings, reps=6, depth_consistency_percent=20.0)
        )
        assert item.severity is Severity.WARNING
        assert "depth" in item.message.lower()

    def test_flags_drifting_tempo(self, settings):
        item = ConsistencyRule().evaluate(
            context_with_metrics(settings, reps=6, duration_consistency_s=1.5)
        )
        assert item.severity is Severity.WARNING
        assert "duration" in item.message.lower()

    def test_reports_both_problems_together(self, settings):
        item = ConsistencyRule().evaluate(
            context_with_metrics(
                settings,
                reps=6,
                depth_consistency_percent=20.0,
                duration_consistency_s=1.5,
            )
        )
        assert "and" in item.message


class TestTempoRule:
    def test_praises_a_controlled_tempo(self, settings):
        item = TempoRule().evaluate(
            context_for(build_squat_series(reps=3, rep_duration_s=3.0), settings)
        )
        assert item.severity is Severity.GOOD

    def test_flags_rushed_reps(self, settings):
        item = TempoRule().evaluate(
            context_for(build_squat_series(reps=3, rep_duration_s=0.8), settings)
        )
        assert item.severity is Severity.WARNING

    def test_reports_both_phases(self, settings):
        item = TempoRule().evaluate(
            context_for(build_squat_series(reps=3, rep_duration_s=2.5), settings)
        )
        assert "down" in item.message
        assert "up" in item.message


class TestEngine:
    def test_a_good_set_produces_only_praise(self, settings):
        ctx = context_for(
            build_squat_series(
                reps=5, depth_fraction=1.0, torso_lean_deg=8.0, rep_duration_s=2.5
            ),
            settings,
        )
        items = generate_feedback(ctx.reps, ctx.metrics, ctx.angles, settings)
        assert items
        assert all(item.severity is Severity.GOOD for item in items)

    def test_problems_are_listed_before_praise(self, settings):
        ctx = context_for(
            build_squat_series(
                reps=5,
                depth_fraction=0.22,
                torso_lean_deg=55.0,
                bottom_lean_deg=65.0,
            ),
            settings,
        )
        items = generate_feedback(ctx.reps, ctx.metrics, ctx.angles, settings)
        order = [item.severity for item in items]
        ranks = {
            Severity.CRITICAL: 0,
            Severity.WARNING: 1,
            Severity.INFO: 2,
            Severity.GOOD: 3,
        }
        assert [ranks[s] for s in order] == sorted(ranks[s] for s in order)

    def test_every_item_is_fully_populated(self, settings):
        """The spec requires each message to carry an explanation."""
        ctx = context_for(build_squat_series(reps=4), settings)
        for item in generate_feedback(ctx.reps, ctx.metrics, ctx.angles, settings):
            assert item.rule_id
            assert item.title
            assert item.message
            assert item.explanation
            assert item.severity in set(Severity)

    def test_rule_ids_are_unique(self):
        ids = [rule.rule_id for rule in DEFAULT_RULES]
        assert len(ids) == len(set(ids))

    def test_a_broken_rule_does_not_lose_the_analysis(self, settings):
        """One bad rule must not cost the lifter their entire result."""

        class ExplodingRule(FeedbackRule):
            rule_id = "exploding"
            priority = 5

            def evaluate(self, context):
                raise RuntimeError("boom")

        engine = FeedbackEngine(rules=(ExplodingRule, DepthRule))
        ctx = context_for(build_squat_series(reps=3), settings)
        items = engine.generate(ctx.reps, ctx.metrics, ctx.angles, settings)

        assert [item.rule_id for item in items] == ["squat_depth"]

    def test_custom_rule_set_is_honoured(self, settings):
        """The extension point: a new rule is one class and one registration."""
        engine = FeedbackEngine(rules=(DepthRule,))
        ctx = context_for(build_squat_series(reps=3), settings)
        items = engine.generate(ctx.reps, ctx.metrics, ctx.angles, settings)
        assert len(items) == 1

    def test_empty_video_still_produces_guidance(self, settings):
        ids = ids_for(build_standing_series(seconds=4.0), settings)
        assert "no_reps_detected" in ids
