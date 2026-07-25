/**
 * Scoring the exported squat fault detectors in the browser.
 *
 * The TypeScript half of a port. `backend/training/export_web.py` writes the
 * fitted scikit-learn models as plain JSON; this reproduces their arithmetic.
 * Pure functions over plain arrays (no DOM, no fetch) so it runs under the
 * Node test environment, and `model.test.ts` asserts it reproduces Python's
 * probabilities on a generated fixture to 1e-6. That test is the whole reason
 * a second implementation of a model is defensible: it cannot drift silently.
 *
 * The scoring chain, in order:
 *
 *   missing -> median      the imputer, so a frame that lost a leg still scores
 *   (optional) standardise the scaler, present only on the linear detector
 *   trees or dot product   the estimator's raw decision value
 *   isotonic knots         calibration, turning that into a probability
 *
 * Note what is *absent*: there is no sigmoid. scikit-learn calibrates from
 * `decision_function` rather than `predict_proba`, which was verified against
 * the real artifact rather than assumed, so the raw score goes straight into the
 * isotonic step.
 */

/** One boosted tree, as flat parallel arrays indexed by node. */
export interface TreeNodes {
  isLeaf: number[];
  value: number[];
  featureIdx: number[];
  threshold: number[];
  left: number[];
  right: number[];
  missingGoToLeft: number[];
}

export type Estimator =
  | { kind: "trees"; baseline: number; trees: TreeNodes[] }
  | { kind: "linear"; coefficients: number[]; intercept: number };

export interface Calibrator {
  x: number[];
  y: number[];
}

export interface FaultModel {
  features: string[];
  threshold: number;
  imputerMedians: number[] | null;
  scalerMean: number[] | null;
  scalerScale: number[] | null;
  estimator: Estimator;
  calibrator: Calibrator;
}

export interface FaultModelBundle {
  version: number;
  sklearnVersion: string;
  faults: Record<string, FaultModel>;
}

/**
 * Walk one tree to its leaf.
 *
 * Reproduces scikit-learn's own traversal exactly: a missing value follows
 * `missingGoToLeft`, and otherwise `value <= threshold` goes left. The
 * `<=` matters: a strict `<` sends every value that sits exactly on a split
 * the wrong way, which the parity fixture would catch but only by luck.
 */
function walkTree(tree: TreeNodes, features: readonly (number | null)[]): number {
  let node = 0;
  // Bounded rather than `while (true)`: a malformed export with a cycle in it
  // would otherwise hang the render loop rather than fail.
  for (let step = 0; step <= tree.isLeaf.length; step++) {
    if (tree.isLeaf[node]) return tree.value[node];

    const value = features[tree.featureIdx[node]];
    const goLeft =
      value === null || Number.isNaN(value)
        ? tree.missingGoToLeft[node] === 1
        : value <= tree.threshold[node];
    node = goLeft ? tree.left[node] : tree.right[node];
  }
  throw new Error("tree traversal did not reach a leaf");
}

/** The estimator's raw decision value, before calibration. */
function decisionValue(
  estimator: Estimator,
  features: readonly (number | null)[],
): number {
  if (estimator.kind === "linear") {
    let total = estimator.intercept;
    for (let i = 0; i < estimator.coefficients.length; i++) {
      total += estimator.coefficients[i] * (features[i] ?? 0);
    }
    return total;
  }

  let total = estimator.baseline;
  for (const tree of estimator.trees) total += walkTree(tree, features);
  return total;
}

/**
 * Piecewise-linear interpolation over the isotonic knots, clamped at both ends.
 *
 * scikit-learn's `IsotonicRegression` predicts by interpolating between the
 * thresholds it kept and holding the end values beyond them, which is what the
 * clamping reproduces.
 */
export function interpolateIsotonic(calibrator: Calibrator, value: number): number {
  const { x, y } = calibrator;
  if (x.length === 0) return 0;
  if (value <= x[0]) return y[0];
  if (value >= x[x.length - 1]) return y[y.length - 1];

  // Binary search for the segment containing `value`.
  let low = 0;
  let high = x.length - 1;
  while (high - low > 1) {
    const mid = (low + high) >> 1;
    if (x[mid] <= value) low = mid;
    else high = mid;
  }

  const span = x[high] - x[low];
  if (span === 0) return y[low];
  return y[low] + ((value - x[low]) * (y[high] - y[low])) / span;
}

/**
 * Score one feature row, returning a calibrated probability in [0, 1].
 *
 * `features` is positional and must match the model's own `features` order.
 * Nulls are imputed with the training median, matching the pipeline: the
 * decision about whether enough was really measured to trust the answer belongs
 * upstream, in `classify.ts`, not here.
 */
export function scoreFault(
  model: FaultModel,
  features: readonly (number | null)[],
): number {
  const prepared: number[] = [];

  for (let i = 0; i < model.features.length; i++) {
    const raw = features[i];
    let value =
      raw === null || raw === undefined || Number.isNaN(raw)
        ? (model.imputerMedians?.[i] ?? 0)
        : raw;

    if (model.scalerMean && model.scalerScale) {
      const scale = model.scalerScale[i] || 1;
      value = (value - model.scalerMean[i]) / scale;
    }
    prepared.push(value);
  }

  return interpolateIsotonic(model.calibrator, decisionValue(model.estimator, prepared));
}

/**
 * Validate a parsed bundle.
 *
 * Returns null rather than throwing for anything unrecognisable, because the
 * caller is a camera loop: a corrupt model must cost the model readout and
 * nothing else.
 */
export function parseBundle(raw: unknown): FaultModelBundle | null {
  if (typeof raw !== "object" || raw === null) return null;
  const bundle = raw as Partial<FaultModelBundle>;
  if (!bundle.faults || typeof bundle.faults !== "object") return null;

  for (const model of Object.values(bundle.faults)) {
    if (!Array.isArray(model?.features) || typeof model?.threshold !== "number") {
      return null;
    }
    if (!model.estimator || !model.calibrator) return null;
  }

  return bundle as FaultModelBundle;
}
