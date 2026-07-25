"""Tests for the ML feature layer.

No scikit-learn, no model artifact, no video. The feature builder is plain
arithmetic over plain floats, so everything here is checkable by hand.

The sign-convention tests are the important ones. A flipped signal produces no
error and no warning: the model simply learns the fault backwards and then
reports it backwards, confidently, forever.
"""

from __future__ import annotations

import pytest

from app.analysis.angles import compute_angles
from app.analysis.types import ViewOrientation
from app.config import Settings
from app.ml.adapter import clip_samples, frame_sample, rep_frame_range
from app.ml.features import (
    FAULT_FEATURES,
    FEATURE_NAMES,
    QUANTITY_NAMES,
    SHIPPED_FAULTS,
    ClipReference,
    FrameSample,
    build_frame_features,
    feature_row,
)
from tests.synthetic import build_squat_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestFrameSample:
    def test_mean_falls_back_to_the_one_visible_side(self):
        """Side-on footage tracks one leg. Requiring both would discard it."""
        sample = FrameSample(knee_flexion_left=-90.0, knee_flexion_right=None)
        assert sample.mean_of("knee_flexion") == pytest.approx(-90.0)

    def test_mean_of_two_sides_is_their_average(self):
        sample = FrameSample(knee_flexion_left=-80.0, knee_flexion_right=-100.0)
        assert sample.mean_of("knee_flexion") == pytest.approx(-90.0)

    def test_gap_requires_both_sides(self):
        """A left/right difference cannot be inferred from one side.

        Falling back the way `mean_of` does would report perfect symmetry for a
        lifter whose other leg was never seen, which is the most misleading
        possible answer to an asymmetry question.
        """
        assert FrameSample(knee_flexion_left=-90.0).gap_of("knee_flexion") is None
        both = FrameSample(knee_flexion_left=-80.0, knee_flexion_right=-100.0)
        assert both.gap_of("knee_flexion") == pytest.approx(20.0)

    def test_absent_signal_is_none_not_zero(self):
        assert FrameSample().mean_of("valgus") is None
        assert FrameSample().gap_of("valgus") is None


class TestClipReference:
    def test_rank_places_a_value_in_its_clip(self):
        samples = [FrameSample(torso_lean=value) for value in (0.0, 10.0, 20.0, 30.0)]
        reference = ClipReference.from_samples(samples)

        assert reference.rank("torso_lean", -5.0) == pytest.approx(0.0)
        assert reference.rank("torso_lean", 15.0) == pytest.approx(0.5)
        assert reference.rank("torso_lean", 99.0) == pytest.approx(1.0)

    def test_rank_survives_any_monotone_rescaling(self):
        """The property the whole cross-corpus bridge rests on.

        The training corpus quotes its angles on a different scale to
        FormVision's. Ranks are invariant to that as long as the relationship is
        monotone increasing, where a z-score would only survive an affine one.
        """
        raw = [1.0, 2.0, 3.0, 4.0, 5.0]
        # A deliberately non-affine but increasing remap.
        warped = [value**3 + 7.0 for value in raw]

        plain = ClipReference.from_samples([FrameSample(torso_lean=v) for v in raw])
        remapped = ClipReference.from_samples([FrameSample(torso_lean=v) for v in warped])

        for value, warped_value in zip(raw, warped, strict=True):
            assert plain.rank("torso_lean", value) == pytest.approx(
                remapped.rank("torso_lean", warped_value)
            )

    def test_a_signal_that_never_varies_ranks_as_unremarkable(self):
        """Ties resolve to the middle of their run, not the top.

        This is a real bug that shipped and was caught in testing. Taking the
        upper edge of a tied run ranks a constant signal at 1.0 in every frame,
        i.e. maximally extreme, when a quantity that never moves is the least
        remarkable thing in the clip. It made a perfectly symmetric lifter
        register as asymmetric on every single repetition, because a left/right
        gap pinned at zero ranked at the top of its own distribution.
        """
        reference = ClipReference.from_samples(
            [FrameSample(torso_lean=5.0) for _ in range(50)]
        )
        assert reference.rank("torso_lean", 5.0) == pytest.approx(0.5)

    def test_ties_inside_a_varying_signal_land_mid_run(self):
        samples = [FrameSample(torso_lean=v) for v in (0.0, 1.0, 1.0, 1.0, 2.0)]
        reference = ClipReference.from_samples(samples)
        # The three tied 1.0s occupy positions 1-3 of 5, so their midpoint is 2.5/5.
        assert reference.rank("torso_lean", 1.0) == pytest.approx(0.5)

    def test_unmeasured_quantity_ranks_as_none(self):
        """A front-on clip never measures the ankle. That must stay missing."""
        reference = ClipReference.from_samples([FrameSample(torso_lean=1.0)])
        assert reference.rank("ankle_openness_mean", 90.0) is None

    def test_missing_value_ranks_as_none(self):
        reference = ClipReference.from_samples([FrameSample(torso_lean=1.0)])
        assert reference.rank("torso_lean", None) is None


class TestFeatureVector:
    def test_every_quantity_gets_an_absolute_and_a_ranked_feature(self):
        assert len(FEATURE_NAMES) == 2 * len(QUANTITY_NAMES)
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_builder_emits_exactly_the_declared_names(self):
        sample = FrameSample(torso_lean=5.0)
        reference = ClipReference.from_samples([sample])
        assert set(build_frame_features(sample, reference)) == set(FEATURE_NAMES)

    def test_completeness_reports_what_was_actually_measured(self):
        sample = FrameSample(torso_lean=5.0, depth_phase=None)
        reference = ClipReference.from_samples([sample])
        features = build_frame_features(sample, reference)

        row, completeness = feature_row(features, ("abs_torso_lean", "abs_depth_phase"))
        assert row[0] == pytest.approx(5.0)
        assert row[1] is None
        assert completeness == pytest.approx(0.5)

    def test_every_fault_selects_only_declared_features(self):
        """A typo in a fault's feature list must not silently become a None column."""
        for fault, names in FAULT_FEATURES.items():
            unknown = set(names) - set(FEATURE_NAMES)
            assert not unknown, f"{fault} references unknown features: {unknown}"

    def test_shipped_faults_are_all_trained(self):
        assert set(SHIPPED_FAULTS) <= set(FAULT_FEATURES)


class TestCanonicalOrientation:
    """Larger must mean "more fault" for every signal, from FormVision's data.

    These run the real analysis stack over the synthetic figure and check the
    direction of each signal after adaptation. They are the guard against the
    one class of bug that produces no symptom other than backwards coaching.
    """

    def test_deeper_squat_raises_knee_flexion_and_depth_phase(self, settings):
        angles = compute_angles(build_squat_series(reps=1), settings)
        samples = clip_samples(angles)

        # The bottom of the rep is where hip height is lowest.
        heights = [
            (value, index)
            for index, value in enumerate(angles.hip_height)
            if value is not None
        ]
        bottom = min(heights)[1]
        top = max(heights)[1]

        assert samples[bottom].mean_of("knee_flexion") > samples[top].mean_of(
            "knee_flexion"
        )
        assert samples[bottom].depth_phase > samples[top].depth_phase

    def test_more_lean_raises_torso_lean(self, settings):
        upright = compute_angles(
            build_squat_series(
                reps=1,
                torso_lean_deg=5.0,
                bottom_lean_deg=5.0,
                view=ViewOrientation.SIDE,
            ),
            settings,
        )
        folded = compute_angles(
            build_squat_series(
                reps=1,
                torso_lean_deg=5.0,
                bottom_lean_deg=45.0,
                view=ViewOrientation.SIDE,
            ),
            settings,
        )

        def peak_lean(angles) -> float:
            values = [s.torso_lean for s in clip_samples(angles) if s.torso_lean]
            return max(values)

        assert peak_lean(folded) > peak_lean(upright)

    def test_caving_knees_raise_valgus(self, settings):
        neutral = compute_angles(
            build_squat_series(reps=1, view=ViewOrientation.FRONT), settings
        )
        caving = compute_angles(
            build_squat_series(reps=1, view=ViewOrientation.FRONT, knee_valgus=0.35),
            settings,
        )

        def mean_valgus(angles) -> float:
            values = [
                s.mean_of("valgus")
                for s in clip_samples(angles)
                if s.mean_of("valgus") is not None
            ]
            return sum(values) / len(values)

        assert mean_valgus(caving) > mean_valgus(neutral)

    def test_asymmetric_figure_raises_the_knee_gap(self, settings):
        even = compute_angles(build_squat_series(reps=1, left_right_bias=0.0), settings)
        uneven = compute_angles(build_squat_series(reps=1, left_right_bias=0.6), settings)

        def peak_gap(angles) -> float:
            gaps = [
                s.gap_of("knee_flexion")
                for s in clip_samples(angles)
                if s.gap_of("knee_flexion") is not None
            ]
            return max(gaps)

        assert peak_gap(uneven) > peak_gap(even)


class TestAdapter:
    def test_view_gating_reaches_the_samples(self, settings):
        """What the camera cannot see must arrive at the model as missing."""
        front = clip_samples(
            compute_angles(
                build_squat_series(reps=1, view=ViewOrientation.FRONT), settings
            )
        )
        side = clip_samples(
            compute_angles(
                build_squat_series(reps=1, view=ViewOrientation.SIDE), settings
            )
        )

        assert all(s.ankle_openness_left is None for s in front)
        assert any(s.valgus_left is not None for s in front)

        assert all(s.valgus_left is None for s in side)
        assert any(s.ankle_openness_left is not None for s in side)

    def test_rep_range_is_inclusive_of_lockout(self):
        from app.analysis.types import Rep

        rep = Rep(
            index=1,
            start_frame=10,
            bottom_frame=20,
            end_frame=30,
            start_time_s=0.0,
            bottom_time_s=1.0,
            end_time_s=2.0,
        )
        assert rep_frame_range(rep, 100) == range(10, 31)

    def test_rep_range_is_clamped_to_the_series(self):
        from app.analysis.types import Rep

        rep = Rep(
            index=1,
            start_frame=10,
            bottom_frame=20,
            end_frame=999,
            start_time_s=0.0,
            bottom_time_s=1.0,
            end_time_s=2.0,
        )
        assert rep_frame_range(rep, 40) == range(10, 40)

    def test_sample_count_matches_the_series(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        assert len(clip_samples(angles)) == len(angles)

    def test_frame_sample_reads_the_requested_index(self, settings):
        angles = compute_angles(build_squat_series(reps=1), settings)
        sample = frame_sample(angles, 5)
        assert sample.knee_flexion_left == pytest.approx(-angles.left_knee_deg[5])
