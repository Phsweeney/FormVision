"""Fit the per-fault detectors and export the artifact.

Run from the backend directory:

    .venv/Scripts/python -m training.train

**Why one binary detector per fault rather than one six-class model.** The
obvious framing scores badly and for a structural reason: the corpus is
per-frame with no notion of where in a rep a frame sits, so a correct standing
frame and a shallow bottom frame occupy the same point in feature space. A
six-way model fitted on it reaches roughly 38% accuracy under a grouped split
with the leak columns removed, and recognises the *correct* class about 8% of
the time. Split into one-versus-correct detectors, the same features reach a
usable AUC on most faults, each detector is independently evaluable, and a rep
can carry several faults at once, which real reps do.

Every split is grouped by source video. That is not a refinement, it is the
difference between a real number and a fiction: see `corpus.py` for the two
columns that make a random split leak.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

logger = logging.getLogger("training")

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = BACKEND_ROOT.parent / "squat_dataset" / "squat_features_augmented.csv"
DEFAULT_OUTPUT = BACKEND_ROOT / "app" / "ml" / "artifacts" / "squat_faults_v1.joblib"

#: Folds for cross-validation. Five over fifteen videos leaves three unseen
#: videos in every fold, which is the smallest holdout that still varies the
#: filming conditions rather than just the frames.
N_SPLITS = 5

#: Share of *correct* frames a detector is allowed to flag. This, and not
#: precision, is what sets the operating threshold.
#:
#: Precision looked like the obvious target and is the wrong one, for a reason
#: worth recording. Precision depends on the base rate of faults, and the corpus
#: is a 50/50 mix of correct and faulty frames while a real clip is
#: overwhelmingly correct. A threshold tuned for 90% precision on the corpus
#: came out at 0.043, and applied to a clean synthetic clip it flagged 96% of
#: frames: at a 50% prior almost any elevated score really was a fault, and at a
#: realistic prior almost none of them are.
#:
#: A false-positive rate is a property of the negative class alone, so it does
#: not move with the prior and it transfers. Read directly: at most 2% of a
#: clean lifter's frames may be flagged.
TARGET_FALSE_POSITIVE_RATE = 0.02

#: Below this out-of-fold AUC a detector is not exported at all. 0.70 is the
#: point where a probability starts carrying enough signal to be worth showing;
#: under it the honest description is a coin weighted slightly.
MIN_AUC = 0.70


@dataclass(frozen=True, slots=True)
class FaultReport:
    """Cross-validated evaluation of one detector."""

    fault: str
    estimator: str
    auc: float
    average_precision: float
    threshold: float | None
    precision_at_threshold: float | None
    recall_at_threshold: float | None
    positives: int
    negatives: int
    shippable: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "fault": self.fault,
            "estimator": self.estimator,
            "roc_auc": round(self.auc, 4),
            "average_precision": round(self.average_precision, 4),
            "threshold": None if self.threshold is None else round(self.threshold, 4),
            "precision_at_threshold": (
                None
                if self.precision_at_threshold is None
                else round(self.precision_at_threshold, 4)
            ),
            "recall_at_threshold": (
                None
                if self.recall_at_threshold is None
                else round(self.recall_at_threshold, 4)
            ),
            "positives": self.positives,
            "negatives": self.negatives,
            "shippable": self.shippable,
        }


def build_estimators() -> dict[str, Pipeline]:
    """Candidate pipelines, tried per fault and chosen on cross-validated AUC.

    The median imputer matters at inference rather than here: the corpus has no
    missing values, but real footage does, and a frame that lost one leg should
    still be scoreable. The completeness gate in the predictor, not the imputer,
    is what decides whether the result is trustworthy enough to show.

    Scaling is affine and fitted once, so it is baked into the artifact and
    applies identically to any input. It is safe in a way that a *global*
    rank transform would not be, since that would renormalise across clips and
    undo the whole point of ranking within one.
    """
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=4, max_iter=200, learning_rate=0.1, random_state=0
                    ),
                ),
            ]
        ),
    }


def _out_of_fold_scores(
    estimator: Pipeline,
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Predict every row from a *calibrated* model that never saw its video.

    Out-of-fold rather than a single holdout so every row contributes to the
    reported metric while none contributes to the model that scored it.

    **The calibration has to happen inside the loop**, which makes this nested
    cross-validation and is the reason it is slower than it looks like it should
    be. The scores this returns are what the operating threshold is chosen from,
    and the threshold is then applied to the calibrated model that actually
    ships. Scoring here with the bare estimator would pick a threshold on the
    uncalibrated scale and apply it on the calibrated one: the two are monotone
    with each other, so AUC would be unchanged and nothing would look wrong,
    while the chosen cut sat at completely the wrong place. That bug shipped
    once and turned a working asymmetry detector silent.
    """
    scores = np.zeros(len(target), dtype=float)
    outer = GroupKFold(n_splits=N_SPLITS)

    for train_index, test_index in outer.split(features, target, groups):
        fold = _calibrate(
            estimator,
            features[train_index],
            target[train_index],
            groups[train_index],
        )
        scores[test_index] = fold.predict_proba(features[test_index])[:, 1]

    return scores


def _calibrate(
    estimator: Pipeline,
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> CalibratedClassifierCV:
    """Wrap an estimator in isotonic calibration, grouped by video.

    `ensemble=False` fits one model on all the data and learns the calibration
    from cross-validated predictions, rather than keeping one fitted model per
    fold and averaging them. Same calibration, a fifth of the artifact: the
    default takes the bundle to 8 MB, which is a lot to commit permanently for
    no gain in accuracy.
    """
    distinct = len(set(groups.tolist()))
    folds = list(
        GroupKFold(n_splits=min(N_SPLITS, distinct)).split(features, target, groups)
    )
    calibrated = CalibratedClassifierCV(
        sklearn.base.clone(estimator),
        method="isotonic",
        cv=folds,
        ensemble=False,
    )
    calibrated.fit(features, target)
    return calibrated


def _choose_threshold(
    target: np.ndarray, scores: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Threshold flagging at most `TARGET_FALSE_POSITIVE_RATE` of correct frames.

    Taken as a quantile of the *negative* scores, so the operating point is set
    entirely by what the model does on correct movement and carries no
    assumption about how often faults occur. That is what makes it survive the
    move from a 50/50 corpus to a real clip, where nearly every frame is fine.

    Returns the threshold with the precision and recall it achieves, or all-None
    when it cannot reach a single real fault inside its false-positive budget,
    which is the signal to withhold the detector rather than ship it loose.
    """
    negatives = scores[target == 0]
    positives = scores[target == 1]
    if not len(negatives) or not len(positives):
        return (None, None, None)

    threshold = float(
        np.quantile(negatives, 1.0 - TARGET_FALSE_POSITIVE_RATE, method="higher")
    )

    flagged_positives = int((positives >= threshold).sum())
    flagged_negatives = int((negatives >= threshold).sum())
    if flagged_positives == 0:
        return (None, None, None)

    precision = flagged_positives / (flagged_positives + flagged_negatives)
    recall = flagged_positives / len(positives)
    return (threshold, precision, recall)


def evaluate_fault(
    fault: str,
    rows,
    references,
    feature_names: Sequence[str],
) -> tuple[FaultReport, Pipeline, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-validate every candidate for one fault and keep the best."""
    label = FAULT_LABELS[fault]
    subset = [row for row in rows if row.label in (LABEL_CORRECT, label)]

    matrix, labels, groups = build_matrix(subset, references, feature_names)
    features = np.array(matrix, dtype=float)
    target = np.array([1 if value == label else 0 for value in labels], dtype=int)
    group_array = np.array(groups)

    best_report: FaultReport | None = None
    best_estimator: Pipeline | None = None

    for name, estimator in build_estimators().items():
        scores = _out_of_fold_scores(estimator, features, target, group_array)
        auc = float(roc_auc_score(target, scores))
        ap = float(average_precision_score(target, scores))
        threshold, precise, recalled = _choose_threshold(target, scores)

        report = FaultReport(
            fault=fault,
            estimator=name,
            auc=auc,
            average_precision=ap,
            threshold=threshold,
            precision_at_threshold=precise,
            recall_at_threshold=recalled,
            positives=int(target.sum()),
            negatives=int((1 - target).sum()),
            shippable=auc >= MIN_AUC and threshold is not None,
        )
        logger.info(
            "  %-18s %-18s auc=%.3f ap=%.3f threshold=%s recall=%s",
            fault,
            name,
            auc,
            ap,
            "none" if threshold is None else f"{threshold:.3f}",
            "n/a" if recalled is None else f"{recalled:.3f}",
        )

        if best_report is None or auc > best_report.auc:
            best_report, best_estimator = report, estimator

    assert best_report is not None and best_estimator is not None
    return best_report, best_estimator, features, target, group_array


def fit_final(
    estimator: Pipeline,
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> CalibratedClassifierCV:
    """Refit on everything, calibrated so the probability means something.

    Calibration folds are grouped by video like every other split here. An
    uncalibrated score is monotone with confidence but is not a probability, and
    the whole confidence gate downstream assumes it can compare one against a
    fixed number.

    Uses the same `_calibrate` helper as the scoring loop, so the model that
    ships and the model the threshold was chosen from are built identically.
    """
    return _calibrate(estimator, features, target, groups)


def train(corpus_path: Path, output_path: Path) -> dict[str, object]:
    """Train every fault, export the artifact, and return the model card."""
    logger.info("Loading corpus from %s", corpus_path)
    rows = load_rows(corpus_path)
    references = clip_references(rows)
    logger.info("  %d rows across %d videos", len(rows), len(references))

    bundle: dict[str, object] = {}
    reports: list[FaultReport] = []

    for fault, feature_names in FAULT_FEATURES.items():
        report, estimator, features, target, groups = evaluate_fault(
            fault, rows, references, feature_names
        )
        reports.append(report)

        if not report.shippable:
            logger.warning(
                "  %s withheld: auc=%.3f, no threshold reaches a fault within a "
                "%.0f%% false-positive budget",
                fault,
                report.auc,
                TARGET_FALSE_POSITIVE_RATE * 100,
            )
            continue

        bundle[fault] = {
            "model": fit_final(estimator, features, target, groups),
            "features": tuple(feature_names),
            "threshold": report.threshold,
        }

    card = {
        "created_utc": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "corpus": {
            "path": corpus_path.name,
            "sha256": dataset_digest(corpus_path),
            "rows": len(rows),
            "videos": sorted(references),
            "excluded_columns": dict(EXCLUDED_COLUMNS),
        },
        "policy": {
            "n_splits": N_SPLITS,
            "grouped_by": "video_file",
            "target_false_positive_rate": TARGET_FALSE_POSITIVE_RATE,
            "min_auc": MIN_AUC,
        },
        "faults": [report.as_dict() for report in reports],
        "shipped": sorted(set(bundle) & set(SHIPPED_FAULTS)),
        "trained_not_shipped": sorted(set(bundle) - set(SHIPPED_FAULTS)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": 1,
            "sklearn_version": sklearn.__version__,
            "faults": bundle,
        },
        output_path,
    )
    card_path = output_path.with_name("model_card.json")
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", card_path)
    return card


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not arguments.corpus.exists():
        logger.error("Corpus not found at %s", arguments.corpus)
        return 1

    card = train(arguments.corpus, arguments.output)

    logger.info("")
    logger.info("%-16s %-18s %6s %8s %8s", "fault", "estimator", "auc", "recall", "ship")
    for entry in card["faults"]:
        assert isinstance(entry, Mapping)
        recall = entry["recall_at_threshold"]
        logger.info(
            "%-16s %-18s %6.3f %8s %8s",
            entry["fault"],
            entry["estimator"],
            entry["roc_auc"],
            "n/a" if recall is None else f"{recall:.3f}",
            "yes" if entry["shippable"] else "NO",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
