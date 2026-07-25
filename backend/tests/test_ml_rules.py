"""Tests for the inference layer and the model-backed coaching rules.

Almost everything here runs with no artifact and no scikit-learn call, because
the interesting behaviour is *when the model is allowed to speak*, and that is
pure logic over `FaultPrediction` objects a test can build by hand.

The degradation tests matter most. The whole design rests on the claim that with
the ML layer off, or its artifact missing, FormVision behaves exactly as it did
before the layer existed. That claim is worth pinning down rather than trusting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.angles import compute_angles
from app.analysis.feedback.engine import FeedbackEngine, generate_feedback
from app.analysis.feedback.ml_rules import MODEL_RULES, AsymmetryModelRule
from app.analysis.feedback.rules import DEFAULT_RULES
from app.analysis.metrics import compute_metrics
from app.analysis.pipeline import run_pipeline
from app.analysis.reps import detect_reps
from app.analysis.types import FaultPrediction, FeedbackSource, ViewOrientation
from app.config import Settings
from app.ml.predictor import NullFaultPredictor, SklearnFaultPredictor
from app.ml.registry import create_predictor, set_predictor_override
from tests.synthetic import SyntheticPoseEstimator, build_squat_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def clear_override():
    """No test may leak a predictor into the next one."""
    yield
    set_predictor_override(None)


def prediction(
    fault_id: str = "knee_valgus",
    rep_index: int = 1,
    fired: bool = True,
    probability: float = 0.8,
    completeness: float = 1.0,
) -> FaultPrediction:
    return FaultPrediction(
        fault_id=fault_id,
        rep_index=rep_index,
        probability=probability,
        affected_fraction=0.6 if fired else 0.05,
        threshold=0.5,
        feature_completeness=completeness,
        fired=fired,
    )


class TestGracefulDegradation:
    """With no model, FormVision must be exactly what it was before."""

    def test_disabled_layer_yields_the_null_predictor(self):
        """Constructed explicitly rather than taken from the default.

        The default is now on, and a test that silently inherits it would stop
        exercising the off path the moment someone flipped it back.
        """
        assert isinstance(
            create_predictor(Settings(ml_enabled=False)), NullFaultPredictor
        )

    def test_unknown_predictor_name_is_not_fatal(self):
        """A typo in configuration costs the model's opinion, not the upload.

        Deliberately unlike `create_estimator`, which raises: pose estimation
        failing means there is no analysis at all, where this failing means an
        analysis without one optional section.
        """
        settings = Settings(ml_enabled=True, ml_predictor="does-not-exist")
        assert isinstance(create_predictor(settings), NullFaultPredictor)

    def test_missing_artifact_is_silent_not_an_error(self, tmp_path, settings):
        predictor = SklearnFaultPredictor(tmp_path / "absent.joblib")
        angles = compute_angles(build_squat_series(reps=2), settings)
        reps = detect_reps(angles, settings)

        assert predictor.predict(angles, reps, settings) == []

    def test_unreadable_artifact_is_silent_not_an_error(self, tmp_path, settings):
        broken = tmp_path / "broken.joblib"
        broken.write_bytes(b"this is not a pickle")

        predictor = SklearnFaultPredictor(broken)
        angles = compute_angles(build_squat_series(reps=2), settings)
        reps = detect_reps(angles, settings)

        assert predictor.predict(angles, reps, settings) == []

    def test_null_predictor_says_nothing(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        reps = detect_reps(angles, settings)
        assert NullFaultPredictor().predict(angles, reps, settings) == []

    def test_pipeline_completes_with_the_layer_off(self, tmp_path):
        """Switching the layer off returns exactly the pre-V3 behaviour."""
        disabled = Settings(ml_enabled=False)
        estimator = SyntheticPoseEstimator(build_squat_series(reps=3))
        output = run_pipeline(tmp_path / "clip.mp4", None, estimator, disabled)

        assert output.result.predictions == ()
        assert output.result.metrics.total_reps > 0
        assert output.result.feedback
        assert all(item.source is FeedbackSource.RULE for item in output.result.feedback)

    def test_a_clean_clip_produces_no_model_feedback_on_the_default_config(
        self, tmp_path, settings
    ):
        """The layer is on by default, so silence on a good squat is the contract.

        A correct lifter must not be told they have a fault. This is the
        assertion that justifies shipping enabled at all, and it runs against
        the real artifact on the real default configuration.
        """
        if not settings.ml_model_path.exists():
            pytest.skip("no trained artifact; run `python -m training.train`")

        assert settings.ml_enabled
        estimator = SyntheticPoseEstimator(build_squat_series(reps=4))
        output = run_pipeline(tmp_path / "clip.mp4", None, estimator, settings)

        # Scored, but nothing rose to a claim about the lifter.
        assert output.result.predictions
        assert not [p for p in output.result.predictions if p.fired]
        assert not [
            item for item in output.result.feedback if item.source is FeedbackSource.MODEL
        ]


class TestModelRulesStaySilent:
    """Abstention is the default. A rule speaks only when it has grounds."""

    def _context_feedback(self, settings, predictions):
        angles = compute_angles(build_squat_series(reps=4), settings)
        reps = detect_reps(angles, settings)
        metrics = compute_metrics(reps, angles, 10.0, settings)
        return generate_feedback(reps, metrics, angles, settings, predictions)

    def test_no_predictions_means_no_model_feedback(self, settings):
        items = self._context_feedback(settings, [])
        assert not [i for i in items if i.source is FeedbackSource.MODEL]

    def test_predictions_that_did_not_fire_stay_quiet(self, settings):
        items = self._context_feedback(
            settings,
            [prediction(rep_index=index, fired=False) for index in range(1, 5)],
        )
        assert not [i for i in items if i.source is FeedbackSource.MODEL]

    def test_one_flagged_rep_out_of_four_is_not_enough(self, settings):
        """One rep in a longer set is as likely a tracking artefact as a fault."""
        items = self._context_feedback(
            settings,
            [prediction(rep_index=1, fired=True)]
            + [prediction(rep_index=index, fired=False) for index in (2, 3, 4)],
        )
        assert not [i for i in items if i.source is FeedbackSource.MODEL]

    def test_one_flagged_rep_out_of_two_is_enough(self, settings):
        """Half the clip is not an artefact, and a fixed count could not say so.

        This is the case that shipped broken: a genuinely one-sided squat filmed
        for two reps, flagged at full confidence on rep one, silenced because an
        absolute threshold of two reps demanded unanimity from a two-rep clip.
        """
        items = self._context_feedback(
            settings,
            [
                prediction("asymmetry", 1, fired=True),
                prediction("asymmetry", 2, fired=False),
            ],
        )
        model_items = [i for i in items if i.source is FeedbackSource.MODEL]
        assert [i.rule_id for i in model_items] == ["ml_asymmetry"]

    def test_the_bar_scales_with_the_length_of_the_clip(self, settings):
        from app.analysis.feedback.ml_rules import _required_reps

        assert _required_reps(2, settings) == 1
        assert _required_reps(4, settings) == 2
        assert _required_reps(10, settings) == 4
        # Never zero, however short or empty the set.
        assert _required_reps(1, settings) == 1
        assert _required_reps(0, settings) == 1

    def test_enough_flagged_reps_produces_one_item(self, settings):
        items = self._context_feedback(
            settings,
            [prediction(rep_index=index, fired=True) for index in (1, 2, 3)],
        )
        model_items = [i for i in items if i.source is FeedbackSource.MODEL]
        assert len(model_items) == 1
        assert model_items[0].rule_id == "ml_knee_valgus"

    def test_each_fault_is_judged_independently(self, settings):
        """A rep can carry several faults, which is why these are not one model."""
        items = self._context_feedback(
            settings,
            [prediction("knee_valgus", index, True) for index in (1, 2)]
            + [prediction("heel_lift", index, True) for index in (1, 2, 3)],
        )
        fired = {i.rule_id for i in items if i.source is FeedbackSource.MODEL}
        assert fired == {"ml_knee_valgus", "ml_heel_lift"}


class TestProvenance:
    """ "Which of these came from a model" must be answerable without parsing copy."""

    def test_model_items_are_marked_and_carry_a_confidence(self, settings):
        angles = compute_angles(build_squat_series(reps=3), settings)
        reps = detect_reps(angles, settings)
        metrics = compute_metrics(reps, angles, 10.0, settings)

        items = FeedbackEngine([AsymmetryModelRule]).generate(
            reps,
            metrics,
            angles,
            settings,
            [
                prediction("asymmetry", 1, True, probability=0.7),
                prediction("asymmetry", 2, True, probability=0.9),
            ],
        )

        assert len(items) == 1
        assert items[0].source is FeedbackSource.MODEL
        assert items[0].confidence == pytest.approx(0.8)
        assert items[0].rule_id.startswith("ml_")

    def test_rule_items_are_marked_as_rules_and_have_no_confidence(self, settings):
        """The eight original rules were not edited, so this is the default."""
        angles = compute_angles(build_squat_series(reps=3), settings)
        reps = detect_reps(angles, settings)
        metrics = compute_metrics(reps, angles, 10.0, settings)

        items = generate_feedback(reps, metrics, angles, settings)
        assert items
        for item in items:
            assert item.source is FeedbackSource.RULE
            assert item.confidence is None

    def test_every_model_rule_declares_itself(self):
        for rule in MODEL_RULES:
            assert rule.source is FeedbackSource.MODEL
            assert rule.rule_id.startswith("ml_")

    def test_model_rules_are_registered(self):
        for rule in MODEL_RULES:
            assert rule in DEFAULT_RULES

    def test_rule_ids_are_unique(self):
        ids = [rule.rule_id for rule in DEFAULT_RULES]
        assert len(ids) == len(set(ids))


class TestPredictorScoring:
    """The real artifact, if it has been trained."""

    @pytest.fixture
    def predictor(self, settings) -> SklearnFaultPredictor:
        if not settings.ml_model_path.exists():
            pytest.skip("no trained artifact; run `python -m training.train`")
        return SklearnFaultPredictor(settings.ml_model_path)

    def test_front_on_clip_is_scored_for_valgus_not_heel_lift(self, predictor, settings):
        """View gating must reach the model as missing features, not zeros."""
        angles = compute_angles(
            build_squat_series(reps=3, view=ViewOrientation.FRONT), settings
        )
        reps = detect_reps(angles, settings)
        predictions = predictor.predict(angles, reps, settings)

        scored = {p.fault_id for p in predictions if p.feature_completeness >= 0.75}
        assert "knee_valgus" in scored
        assert "heel_lift" not in scored

    def test_side_on_clip_is_scored_for_heel_lift_not_valgus(self, predictor, settings):
        angles = compute_angles(
            build_squat_series(reps=3, view=ViewOrientation.SIDE), settings
        )
        reps = detect_reps(angles, settings)
        predictions = predictor.predict(angles, reps, settings)

        scored = {p.fault_id for p in predictions if p.feature_completeness >= 0.75}
        assert "heel_lift" in scored
        assert "knee_valgus" not in scored

    def test_predictions_are_per_rep_and_well_formed(self, predictor, settings):
        angles = compute_angles(build_squat_series(reps=3), settings)
        reps = detect_reps(angles, settings)
        predictions = predictor.predict(angles, reps, settings)

        assert predictions
        rep_indices = {rep.index for rep in reps}
        for item in predictions:
            assert item.rep_index in rep_indices
            assert 0.0 <= item.probability <= 1.0
            assert 0.0 <= item.affected_fraction <= 1.0
            assert 0.0 <= item.feature_completeness <= 1.0

    def test_a_clip_with_no_reps_yields_no_predictions(self, predictor, settings):
        from tests.synthetic import build_standing_series

        angles = compute_angles(build_standing_series(seconds=3.0), settings)
        assert predictor.predict(angles, [], settings) == []

    def test_asymmetry_discriminates_on_formvisions_own_data(self, predictor, settings):
        """The transfer test: trained on one corpus, applied to our own pose data.

        Cross-validated accuracy is evidence about the training corpus. This is
        the separate and harder question of whether the detector says anything
        true about landmarks FormVision extracted itself, in its own angle
        conventions.

        It also guards a bug that shipped once. The operating threshold is
        chosen from out-of-fold scores; if those are produced by the bare
        estimator while the model that ships is wrapped in probability
        calibration, the two scales differ. AUC is unchanged, because the
        scales are monotone with each other, so nothing looks wrong anywhere in
        training, and the chosen cut lands in the wrong place. It turned this
        detector completely silent.
        """

        def fired(bias: float) -> int:
            angles = compute_angles(
                build_squat_series(reps=4, left_right_bias=bias), settings
            )
            reps = detect_reps(angles, settings)
            return sum(
                1
                for p in predictor.predict(angles, reps, settings)
                if p.fault_id == "asymmetry" and p.fired
            )

        assert fired(0.0) == 0, "a symmetric lifter must not be flagged"
        assert fired(0.7) > 0, "a clearly one-sided lifter must be flagged"

    def test_loading_happens_once(self, predictor, settings):
        """The bundle is cached; a second clip must not re-read the pickle."""
        angles = compute_angles(build_squat_series(reps=2), settings)
        reps = detect_reps(angles, settings)

        predictor.predict(angles, reps, settings)
        first = predictor._bundle
        predictor.predict(angles, reps, settings)

        assert predictor._bundle is first


class TestApiPayload:
    """The result blob is stored shaped for the API, so both ends must agree."""

    def test_predictions_and_provenance_survive_the_round_trip(self, settings):
        from app.analysis.types import AnalysisResult, VideoMetadata
        from app.schemas.analysis import build_result_payload

        angles = compute_angles(build_squat_series(reps=3), settings)
        reps = detect_reps(angles, settings)
        metrics = compute_metrics(reps, angles, 10.0, settings)
        predictions = [prediction("asymmetry", index, True) for index in (1, 2)]
        feedback = generate_feedback(reps, metrics, angles, settings, predictions)

        result = AnalysisResult(
            metadata=VideoMetadata(720, 1280, 30.0, len(angles), 10.0),
            angles=angles,
            reps=tuple(reps),
            metrics=metrics,
            feedback=tuple(feedback),
            estimator_name="synthetic",
            predictions=tuple(predictions),
        )

        payload = build_result_payload(result, settings.max_series_points)
        assert len(payload["predictions"]) == 2

        model_items = [i for i in payload["feedback"] if i["source"] == "model"]
        assert model_items
        assert model_items[0]["confidence"] is not None
        assert all(
            i["confidence"] is None for i in payload["feedback"] if i["source"] == "rule"
        )

    def test_a_result_stored_before_the_ml_layer_still_decodes(self, settings):
        """Old rows have no `predictions` key and no `source` on feedback.

        The result blob is never queried into, so adding fields needed no
        migration. The cost of that is exactly this: the reader must tolerate
        rows written by an older version, or every existing analysis 500s.
        """
        from datetime import UTC, datetime

        from app.analysis.types import AnalysisResult, VideoMetadata
        from app.schemas.analysis import AnalysisResponse, build_result_payload

        angles = compute_angles(build_squat_series(reps=2), settings)
        reps = detect_reps(angles, settings)
        metrics = compute_metrics(reps, angles, 10.0, settings)

        payload = build_result_payload(
            AnalysisResult(
                metadata=VideoMetadata(720, 1280, 30.0, len(angles), 10.0),
                angles=angles,
                reps=tuple(reps),
                metrics=metrics,
                feedback=tuple(generate_feedback(reps, metrics, angles, settings)),
                estimator_name="synthetic",
            ),
            settings.max_series_points,
        )

        # Strip everything V3 added, which is precisely what a row written by
        # the previous version looks like. Building it this way rather than by
        # hand means the test cannot drift out of date as other fields change.
        del payload["predictions"]
        for item in payload["feedback"]:
            del item["source"]
            del item["confidence"]

        class FakeRecord:
            id = "0" * 32
            status = "completed"
            original_filename = "old.mp4"
            created_at = updated_at = datetime.now(UTC)
            error_code = error_message = None
            processing_seconds = 1.0
            overlay_filename = None

        response = AnalysisResponse.from_record(FakeRecord(), payload)

        assert response.predictions == []
        assert response.feedback
        assert all(item.source == "rule" for item in response.feedback)
        assert all(item.confidence is None for item in response.feedback)


class TestOverrideSeam:
    def test_override_replaces_the_configured_predictor(self, settings):
        class Stub(NullFaultPredictor):
            name = "stub"

        set_predictor_override(Stub())
        assert create_predictor(settings).name == "stub"

    def test_pipeline_uses_the_override(self, tmp_path, settings):
        class Stub(NullFaultPredictor):
            name = "stub"

            def predict(self, angles, reps, settings):  # noqa: ARG002
                return [prediction("heel_lift", rep.index, True) for rep in reps]

        set_predictor_override(Stub())
        estimator = SyntheticPoseEstimator(build_squat_series(reps=3))
        output = run_pipeline(Path(tmp_path / "clip.mp4"), None, estimator, settings)

        assert output.result.predictions
        assert any(item.rule_id == "ml_heel_lift" for item in output.result.feedback)
