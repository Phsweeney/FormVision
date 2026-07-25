"""Guards on the corpus layer.

Every test here encodes a trap that was found by profiling the data and would
otherwise be invisible: the model would train, cross-validate beautifully, and
be wrong. Comments do not stop someone re-adding a leaked column a year from
now. These do.

They build their own miniature CSV rather than reading the real 11 MB corpus,
so they are fast and do not depend on the dataset being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from app.ml.features import FAULT_FEATURES, SHIPPED_FAULTS
from training.corpus import (
    EXCLUDED_COLUMNS,
    FAULT_LABELS,
    LABEL_CORRECT,
    build_matrix,
    clip_references,
    dataset_digest,
    load_rows,
)
from training.train import (
    N_SPLITS,
    TARGET_FALSE_POSITIVE_RATE,
    _choose_threshold,
)

COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "spine_angle",
    "torso_lean",
    "left_knee_lateral",
    "right_knee_lateral",
    "symmetry_score",
    "hip_depth",
    "video_file",
    "frame",
    "label",
]


def write_corpus(path, rows) -> None:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def row(video="a.mp4", frame=0, label=0, **overrides):
    values = {
        "left_knee_angle": 100.0,
        "right_knee_angle": 100.0,
        "left_hip_angle": 120.0,
        "right_hip_angle": 120.0,
        "left_ankle_angle": 80.0,
        "right_ankle_angle": 80.0,
        "spine_angle": 120.0,
        "torso_lean": 95.0,
        "left_knee_lateral": 0.05,
        "right_knee_lateral": 0.05,
        "symmetry_score": 3.0,
        "hip_depth": 0.5,
        "video_file": video,
        "frame": frame,
        "label": label,
    }
    values.update(overrides)
    return values


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus.csv"
    write_corpus(
        path,
        [
            row(video="a.mp4", frame=index, label=label, hip_depth=0.4 + index * 0.01)
            for index in range(20)
            for label in range(6)
        ],
    )
    return path


class TestLeakColumns:
    """The two columns that make this corpus lie, and why they are dropped."""

    def test_the_excluded_set_is_not_quietly_emptied(self):
        assert set(EXCLUDED_COLUMNS) == {"spine_angle", "symmetry_score"}
        for reason in EXCLUDED_COLUMNS.values():
            assert reason.strip()

    def test_no_shipped_detector_can_reach_an_excluded_column(self):
        """Feature names are canonical, so an excluded column has no route in.

        `spine_angle` is a byte-for-byte duplicate of `left_hip_angle` that the
        augmentation perturbs separately, making their difference an exact
        identifier of the label group. `symmetry_score` is never recomputed
        after perturbation, so it describes the *correct* frame. Neither may
        appear under any name.
        """
        for fault in SHIPPED_FAULTS:
            for feature in FAULT_FEATURES[fault]:
                for excluded in EXCLUDED_COLUMNS:
                    assert excluded not in feature

    def test_loader_drops_them_before_anything_downstream_sees_them(self, corpus_path):
        rows = load_rows(corpus_path)
        assert rows
        # FrameSample has no field that could carry either column through.
        assert not hasattr(rows[0].sample, "spine_angle")
        assert not hasattr(rows[0].sample, "symmetry_score")

    def test_a_corpus_missing_the_columns_fails_loudly(self, tmp_path):
        """Silently succeeding would mean the guard had stopped guarding."""
        path = tmp_path / "short.csv"
        frame = pd.DataFrame([row()], columns=COLUMNS).drop(columns=["spine_angle"])
        frame.to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing expected columns"):
            load_rows(path)


class TestCanonicalOrientation:
    """The corpus adapter must agree with `app/ml/adapter.py` on direction.

    A disagreement here trains the model on one convention and applies it to the
    mirror image. Nothing at runtime would notice.
    """

    @pytest.mark.parametrize(
        ("column", "field", "raises_canonical"),
        [
            ("left_knee_angle", "knee_flexion_left", False),
            ("right_knee_angle", "knee_flexion_right", False),
            ("left_hip_angle", "hip_flexion_left", False),
            ("right_hip_angle", "hip_flexion_right", False),
            ("left_ankle_angle", "ankle_openness_left", True),
            ("right_ankle_angle", "ankle_openness_right", True),
            ("left_knee_lateral", "valgus_left", True),
            ("right_knee_lateral", "valgus_right", True),
            ("torso_lean", "torso_lean", True),
            ("hip_depth", "depth_phase", True),
        ],
    )
    def test_each_column_points_the_declared_way(
        self, tmp_path, column, field, raises_canonical
    ):
        path = tmp_path / "pair.csv"
        write_corpus(
            path,
            [
                row(frame=0, **{column: 10.0}),
                row(frame=1, **{column: 90.0}),
            ],
        )
        low, high = load_rows(path)

        moved = getattr(high.sample, field) - getattr(low.sample, field)
        if raises_canonical:
            assert moved > 0, f"{column} should raise {field}"
        else:
            assert moved < 0, f"{column} should lower {field} (a joint angle falls"
            " as the joint closes)"

    def test_depth_runs_opposite_to_formvision_and_is_flipped_for_it(self, tmp_path):
        """The one place the two sources genuinely disagree.

        The corpus's `hip_depth` grows as the lifter descends. FormVision's
        `hip_height` shrinks. `app/ml/adapter.py` negates its own; this adapter
        must not negate the corpus's. Both must arrive as `depth_phase` growing
        with depth, and a rank transform would carry the mistake through
        silently, because it preserves an increasing remap and inverts a
        decreasing one.
        """
        path = tmp_path / "depth.csv"
        write_corpus(path, [row(frame=0, hip_depth=0.3), row(frame=1, hip_depth=0.8)])
        shallow, deep = load_rows(path)
        assert deep.sample.depth_phase > shallow.sample.depth_phase


class TestClipReference:
    def test_reference_is_built_from_unperturbed_frames_only(self, corpus_path):
        """Perturbed rows must not set the scale used to judge them.

        Five sixths of the corpus is synthetic, which no real clip resembles.
        The reference describes the video's real movement, so only label 0
        contributes.
        """
        rows = load_rows(corpus_path)
        references = clip_references(rows)

        correct = [r for r in rows if r.label == LABEL_CORRECT]
        assert len(references["a.mp4"].distributions["depth_phase"]) == len(correct)

    def test_a_video_with_no_correct_frames_gets_no_reference(self, tmp_path):
        path = tmp_path / "faulty.csv"
        write_corpus(path, [row(video="b.mp4", frame=i, label=3) for i in range(4)])
        assert clip_references(load_rows(path)) == {}

    def test_rows_without_a_reference_are_skipped_not_guessed(self, tmp_path):
        path = tmp_path / "mixed.csv"
        write_corpus(
            path,
            [row(video="a.mp4", frame=i, label=0) for i in range(4)]
            + [row(video="b.mp4", frame=i, label=3) for i in range(4)],
        )
        rows = load_rows(path)
        matrix, _, groups = build_matrix(
            rows, clip_references(rows), ("rank_depth_phase",)
        )
        assert set(groups) == {"a.mp4"}
        assert len(matrix) == 4


class TestGroupedSplits:
    """No video may appear on both sides of a split.

    Rows derived from one frame share their untouched columns exactly, so a
    random split lets a model recognise siblings it has already seen. Grouping
    by source video is what makes the reported metric mean anything.
    """

    def test_no_video_straddles_a_fold(self):
        videos = np.array([f"v{index % 15}.mp4" for index in range(600)])
        target = np.array([index % 2 for index in range(600)])
        features = np.zeros((600, 3))

        for train_index, test_index in GroupKFold(n_splits=N_SPLITS).split(
            features, target, videos
        ):
            overlap = set(videos[train_index]) & set(videos[test_index])
            assert not overlap, f"video seen on both sides: {overlap}"

    def test_every_video_is_held_out_exactly_once(self):
        videos = np.array([f"v{index % 15}.mp4" for index in range(600)])
        target = np.array([index % 2 for index in range(600)])
        features = np.zeros((600, 3))

        held_out: list[str] = []
        for _, test_index in GroupKFold(n_splits=N_SPLITS).split(
            features, target, videos
        ):
            held_out.extend(set(videos[test_index]))

        assert sorted(held_out) == sorted(set(videos))


class TestThresholdSelection:
    """The operating point is set by the false-positive budget, not by precision.

    Precision moves with the base rate of faults, and the corpus is a 50/50 mix
    where a real clip is overwhelmingly correct. Tuning for 90% precision on the
    corpus produced a threshold of 0.043 that flagged 96% of a clean clip's
    frames. A false-positive rate is a property of the negative class alone, so
    it means the same thing in both places.
    """

    def test_separable_classes_give_a_threshold_that_catches_everything(self):
        target = np.array([0] * 100 + [1] * 100)
        scores = np.concatenate([np.linspace(0.0, 0.4, 100), np.linspace(0.6, 1.0, 100)])

        threshold, precision, recall = _choose_threshold(target, scores)
        assert threshold is not None
        assert recall == pytest.approx(1.0)
        # Not exactly 1.0: the threshold sits at the negatives' 98th percentile,
        # so it spends its full budget and lets the top 2% of them through.
        assert precision > 0.95

    def test_the_threshold_holds_the_false_positive_budget(self):
        """The property the whole choice exists to guarantee."""
        rng = np.random.default_rng(0)
        target = np.array([0] * 1000 + [1] * 1000)
        scores = np.concatenate([rng.normal(0.3, 0.1, 1000), rng.normal(0.7, 0.1, 1000)])

        threshold, _, _ = _choose_threshold(target, scores)
        assert threshold is not None

        false_positive_rate = float((scores[target == 0] >= threshold).mean())
        assert false_positive_rate <= TARGET_FALSE_POSITIVE_RATE + 0.005

    def test_budget_is_held_even_when_the_model_is_useless(self):
        """Random scores must not become confident coaching.

        A detector with no signal still gets a threshold, because 2% of anything
        clears its own 98th percentile. What protects the lifter here is not this
        function but `MIN_AUC`, which withholds the detector entirely, and the
        rep-level gates above it. What this function must guarantee is only that
        the false-positive budget is respected, and it does.
        """
        rng = np.random.default_rng(0)
        target = np.array([0] * 1000 + [1] * 1000)
        scores = rng.random(2000)

        threshold, _, recall = _choose_threshold(target, scores)
        assert threshold is not None
        assert recall < 0.05
        assert float((scores[target == 0] >= threshold).mean()) <= 0.03

    def test_a_detector_that_never_reaches_a_fault_is_withheld(self):
        """Every positive below every negative: nothing to ship."""
        target = np.array([0] * 100 + [1] * 100)
        scores = np.concatenate([np.linspace(0.6, 1.0, 100), np.linspace(0.0, 0.4, 100)])

        threshold, _, _ = _choose_threshold(target, scores)
        assert threshold is None


class TestDigest:
    def test_digest_pins_the_exact_file(self, tmp_path):
        first = tmp_path / "one.csv"
        second = tmp_path / "two.csv"
        write_corpus(first, [row()])
        write_corpus(second, [row(hip_depth=0.9)])

        assert dataset_digest(first) == dataset_digest(first)
        assert dataset_digest(first) != dataset_digest(second)


def test_fault_labels_cover_every_trained_fault():
    assert set(FAULT_LABELS) == set(FAULT_FEATURES)
