"""Export the shipped detectors to JSON the browser can score.

Run from the backend directory, after `training.train`:

    .venv/Scripts/python -m training.export_web

The live webcam mode runs its analysis entirely client-side, so getting model
output on screen there means the model has to go to the browser. This writes the
weights as plain JSON and a matching parity fixture; `frontend/src/lib/ml/`
implements the scorer and asserts against that fixture, which is what keeps the
two implementations from drifting apart silently.

**Why this is a port and not a re-fit.** A cheaper option was retraining as
logistic regression, which exports to a handful of numbers. It costs real
accuracy on the one detector proven to transfer to real pose data (asymmetry,
0.79 to 0.73 AUC), and it would leave live and upload disagreeing about the same
movement. Exporting the fitted trees is a few hundred more lines once and no
divergence ever.

**What makes the export tractable.** `train.py` calibrates with
`ensemble=False`, so each fault has exactly one `calibrated_classifiers_` entry
rather than one per fold: there is no fold-averaging to reproduce. And the
calibrator consumes `decision_function`, not `predict_proba`, which was verified
rather than assumed. That means the browser never needs a sigmoid: raw score
straight into the isotonic step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from app.ml.features import SHIPPED_FAULTS

logger = logging.getLogger("export_web")

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BACKEND_ROOT / "app" / "ml" / "artifacts" / "squat_faults_v1.joblib"
DEFAULT_OUTPUT = BACKEND_ROOT / "app" / "ml" / "artifacts" / "squat_faults_web.json"

#: Rows in the parity fixture. Enough to exercise every branch of every tree
#: many times over; small enough to read.
FIXTURE_ROWS = 400

#: Decimal places kept for weights. Tree thresholds and leaf values carry no
#: meaningful precision past this, and it roughly halves the file.
PRECISION = 7


def _round(value: float) -> float:
    return round(float(value), PRECISION)


def _export_trees(model: Any) -> dict[str, Any]:
    """Flatten a HistGradientBoostingClassifier into arrays.

    Only the fields the tree walk needs are kept. `count`, `gain`, `depth`, and
    the binning fields are training bookkeeping and are dropped.

    The traversal being reproduced is sklearn's own: NaN follows
    `missing_go_to_left`, otherwise `value <= num_threshold` goes left.
    """
    trees: list[dict[str, list[float]]] = []

    for predictors in model._predictors:
        # Binary classification, so exactly one predictor per boosting iteration.
        nodes = predictors[0].nodes
        trees.append(
            {
                "isLeaf": [int(n) for n in nodes["is_leaf"]],
                "value": [_round(v) for v in nodes["value"]],
                "featureIdx": [int(i) for i in nodes["feature_idx"]],
                "threshold": [_round(t) for t in nodes["num_threshold"]],
                "left": [int(i) for i in nodes["left"]],
                "right": [int(i) for i in nodes["right"]],
                "missingGoToLeft": [int(b) for b in nodes["missing_go_to_left"]],
            }
        )

    return {
        "kind": "trees",
        "baseline": _round(np.asarray(model._baseline_prediction).ravel()[0]),
        "trees": trees,
    }


def _export_linear(model: Any) -> dict[str, Any]:
    return {
        "kind": "linear",
        "coefficients": [_round(c) for c in np.asarray(model.coef_).ravel()],
        "intercept": _round(np.asarray(model.intercept_).ravel()[0]),
    }


def _export_pipeline(pipeline: Any) -> dict[str, Any]:
    """Export the imputer, the optional scaler, and the estimator itself."""
    steps = dict(pipeline.steps)

    imputer = steps.get("impute")
    scaler = steps.get("scale")
    final = pipeline.steps[-1][1]

    exported: dict[str, Any] = {
        # The corpus has no missing values, so these medians only ever matter at
        # inference, where a frame that lost a leg still deserves a score.
        "imputerMedians": [_round(v) for v in np.asarray(imputer.statistics_).ravel()]
        if imputer is not None
        else None,
        "scalerMean": [_round(v) for v in np.asarray(scaler.mean_).ravel()]
        if scaler is not None
        else None,
        "scalerScale": [_round(v) for v in np.asarray(scaler.scale_).ravel()]
        if scaler is not None
        else None,
    }

    if hasattr(final, "_predictors"):
        exported["estimator"] = _export_trees(final)
    elif hasattr(final, "coef_"):
        exported["estimator"] = _export_linear(final)
    else:
        raise TypeError(f"cannot export estimator of type {type(final).__name__}")

    return exported


def _export_calibrator(calibrator: Any) -> dict[str, Any]:
    """Isotonic regression as its knots, interpolated linearly at scoring time."""
    return {
        "x": [_round(v) for v in np.asarray(calibrator.X_thresholds_).ravel()],
        "y": [_round(v) for v in np.asarray(calibrator.y_thresholds_).ravel()],
    }


def export_fault(detector: dict[str, Any]) -> dict[str, Any]:
    """Everything the browser needs to reproduce one detector's probability."""
    model = detector["model"]
    if len(model.calibrated_classifiers_) != 1:
        raise ValueError(
            "expected a single calibrated classifier (train.py uses "
            f"ensemble=False), found {len(model.calibrated_classifiers_)}"
        )

    calibrated = model.calibrated_classifiers_[0]
    exported = _export_pipeline(calibrated.estimator)
    exported["features"] = list(detector["features"])
    exported["threshold"] = _round(detector["threshold"])
    exported["calibrator"] = _export_calibrator(calibrated.calibrators[0])
    return exported


def build_fixture(
    bundle: dict[str, Any], rows: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Feature rows with the probability Python produces for each.

    Every shipped feature is a within-clip rank, so uniform [0, 1] is exactly
    the distribution the model meets in production. A tenth of the values are
    made missing so the TypeScript imputation path is covered too.
    """
    rng = np.random.default_rng(seed)
    fixture: list[dict[str, Any]] = []

    for fault_id, detector in bundle["faults"].items():
        if fault_id not in SHIPPED_FAULTS:
            continue

        width = len(detector["features"])
        features = rng.random((rows, width))
        features[rng.random((rows, width)) < 0.1] = np.nan

        probabilities = detector["model"].predict_proba(features)[:, 1]

        fixture.append(
            {
                "faultId": fault_id,
                "features": [
                    [None if np.isnan(v) else _round(v) for v in row] for row in features
                ],
                "expected": [_round(p) for p in probabilities],
            }
        )

    return fixture


def export(input_path: Path, output_path: Path) -> dict[str, Any]:
    bundle = joblib.load(input_path)

    faults = {
        fault_id: export_fault(detector)
        for fault_id, detector in bundle["faults"].items()
        if fault_id in SHIPPED_FAULTS
    }
    missing = set(SHIPPED_FAULTS) - set(faults)
    if missing:
        logger.warning(
            "shipped faults absent from the artifact and therefore from the "
            "browser bundle: %s",
            ", ".join(sorted(missing)),
        )

    payload = {
        "version": 1,
        "sklearnVersion": bundle.get("sklearn_version", sklearn.__version__),
        "faults": faults,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    fixture_path = output_path.with_name(f"{output_path.stem}_fixture.json")
    fixture_path.write_text(
        json.dumps(build_fixture(bundle, FIXTURE_ROWS), separators=(",", ":")),
        encoding="utf-8",
    )

    logger.info("Wrote %s (%.0f KB)", output_path, output_path.stat().st_size / 1024)
    logger.info("Wrote %s", fixture_path)
    for fault_id, detector in faults.items():
        kind = detector["estimator"]["kind"]
        size = (
            len(detector["estimator"]["trees"])
            if kind == "trees"
            else len(detector["estimator"]["coefficients"])
        )
        logger.info(
            "  %-14s %-7s %4d %-12s %d features",
            fault_id,
            kind,
            size,
            "trees" if kind == "trees" else "coefficients",
            len(detector["features"]),
        )

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not arguments.input.exists():
        logger.error(
            "No trained artifact at %s. Run `python -m training.train` first.",
            arguments.input,
        )
        return 1

    export(arguments.input, arguments.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
