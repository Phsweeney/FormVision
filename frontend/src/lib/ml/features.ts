/**
 * The model's feature contract, in the browser.
 *
 * The TypeScript mirror of `backend/app/ml/features.py` and
 * `backend/app/ml/adapter.py`. Two devices carry a model trained on one corpus
 * onto pose data FormVision extracted itself, and both are reproduced here:
 *
 * 1. **Canonical orientation.** Every signal is restated so larger always means
 *    more of the fault. Four of the ten are negations of how the angle is
 *    normally quoted, and every one of those flips is load-bearing: get one
 *    wrong and the model still runs, still returns confident probabilities, and
 *    is silently backwards for that signal forever.
 *
 * 2. **Within-clip ranking.** Each value becomes its percentile within the
 *    clip's own distribution, which is unitless and so survives the fact that
 *    the training corpus quotes its angles on a different scale.
 *
 * **What differs from the backend, unavoidably.** The batch pipeline ranks each
 * frame against the whole clip. Live, only the past exists, so `RunningRanks`
 * ranks against the session so far. That asks the same question (is this
 * moment unusual compared with how you have been moving today) with less data
 * early on. It also means live and upload will not produce identical verdicts
 * for the same movement, and cannot.
 */

import type { RawFrameAngles } from "@/lib/analysis/angles";

/** Signals measured on both sides of the body. */
export const PAIRED_SIGNALS = [
  "knee_flexion",
  "hip_flexion",
  "ankle_openness",
  "valgus",
] as const;

/** Signals with a single value for the whole body. */
export const SINGLE_SIGNALS = ["torso_lean", "depth_phase"] as const;

export type PairedSignal = (typeof PAIRED_SIGNALS)[number];

/**
 * One frame's signals in canonical orientation. `null` means *not measured*,
 * never zero. A front-on clip has no ankle reading at all, and a zero there
 * would describe a perfectly neutral ankle rather than the absence of one.
 *
 * Larger means: knee/hip flexion, more closed; ankleOpenness, shin further over
 * the foot (heel lifting); valgus, knee further toward the midline; torsoLean,
 * further from upright; depthPhase, lower in the movement.
 */
export interface FrameSample {
  knee_flexion_left: number | null;
  knee_flexion_right: number | null;
  hip_flexion_left: number | null;
  hip_flexion_right: number | null;
  ankle_openness_left: number | null;
  ankle_openness_right: number | null;
  valgus_left: number | null;
  valgus_right: number | null;
  torso_lean: number | null;
  depth_phase: number | null;
}

const negate = (value: number | null): number | null =>
  value === null ? null : -value;

/**
 * Build the canonical sample for one frame of live analysis.
 *
 * A direct port of `adapter.py::frame_sample`. The four negations are the whole
 * point:
 *
 * - Knee and hip angles *fall* as the joint closes, so flexion is their
 *   negative. Left as-is, "more bent" would rank as "less".
 * - Hip height is measured upward from the ankles, so it falls as the lifter
 *   descends, whereas `depth_phase` must grow. The training corpus's own depth
 *   column runs the other way, which is exactly the trap: a rank transform
 *   carries an increasing reparameterisation through unchanged and inverts a
 *   decreasing one without complaint.
 *
 * Ankle openness and valgus need no flip: `angles.ts` already defines the ankle
 * angle as opening when the heel lifts, and knee lateral offset as positive
 * when the knee travels medially.
 */
export function frameSample(raw: RawFrameAngles): FrameSample {
  return {
    knee_flexion_left: negate(raw.leftKneeDeg),
    knee_flexion_right: negate(raw.rightKneeDeg),
    hip_flexion_left: negate(raw.leftHipDeg),
    hip_flexion_right: negate(raw.rightHipDeg),
    ankle_openness_left: raw.leftAnkleDeg,
    ankle_openness_right: raw.rightAnkleDeg,
    valgus_left: raw.leftKneeLateral,
    valgus_right: raw.rightKneeLateral,
    torso_lean: raw.torsoLeanDeg,
    depth_phase: negate(raw.hipHeight),
  };
}

/**
 * Average of a paired signal, tolerating one missing side.
 *
 * Side-on footage genuinely only tracks one leg, so requiring both would
 * discard most of a usable clip.
 */
export function meanOf(sample: FrameSample, signal: PairedSignal): number | null {
  const left = sample[`${signal}_left`];
  const right = sample[`${signal}_right`];
  const present = [left, right].filter((v): v is number => v !== null);
  if (present.length === 0) return null;
  return present.reduce((a, b) => a + b, 0) / present.length;
}

/**
 * Absolute left/right difference, or null if either side is missing.
 *
 * Unlike `meanOf` this cannot fall back to one side: a difference needs both
 * terms, and inventing the missing one would manufacture symmetry that was
 * never observed, the most misleading possible answer to an asymmetry question.
 */
export function gapOf(sample: FrameSample, signal: PairedSignal): number | null {
  const left = sample[`${signal}_left`];
  const right = sample[`${signal}_right`];
  if (left === null || right === null) return null;
  return Math.abs(left - right);
}

/** Flatten a sample into every rankable quantity, keyed as the model names them. */
export function derivedValues(sample: FrameSample): Record<string, number | null> {
  const values: Record<string, number | null> = {};

  for (const signal of PAIRED_SIGNALS) {
    values[`${signal}_left`] = sample[`${signal}_left`];
    values[`${signal}_right`] = sample[`${signal}_right`];
    values[`${signal}_mean`] = meanOf(sample, signal);
    values[`${signal}_gap`] = gapOf(sample, signal);
  }
  for (const signal of SINGLE_SIGNALS) {
    values[signal] = sample[signal];
  }

  return values;
}

/** Index of the first element not less than `value`. */
function lowerBound(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid] < value) low = mid + 1;
    else high = mid;
  }
  return low;
}

/** Index of the first element greater than `value`. */
function upperBound(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid] <= value) low = mid + 1;
    else high = mid;
  }
  return low;
}

/**
 * The session's own distribution of each quantity, grown one frame at a time.
 *
 * The live counterpart to the backend's `ClipReference`. Values are kept sorted
 * by binary-search insertion so a rank is a binary search rather than a scan,
 * which matters when ten quantities are ranked on every frame at 30fps.
 */
export class RunningRanks {
  private readonly distributions = new Map<string, number[]>();
  private count = 0;

  /** Frames observed so far, however few of their signals were measurable. */
  get frames(): number {
    return this.count;
  }

  /** Record one frame's values into the session distribution. */
  insert(values: Record<string, number | null>): void {
    this.count += 1;
    for (const [name, value] of Object.entries(values)) {
      if (value === null || Number.isNaN(value)) continue;
      let sorted = this.distributions.get(name);
      if (!sorted) {
        sorted = [];
        this.distributions.set(name, sorted);
      }
      sorted.splice(lowerBound(sorted, value), 0, value);
    }
  }

  /**
   * Where `value` falls in the session's distribution, as a 0-1 fraction, or
   * null if it is missing or the session never measured this quantity.
   *
   * Ties resolve to the **midpoint** of the tied run, and that detail is
   * load-bearing. Taking the upper edge would rank a signal that never varies at
   * 1.0 in every frame, i.e. maximally extreme, when the truth is the opposite:
   * a quantity that never moves has nothing remarkable about it anywhere. That
   * bug made a perfectly symmetric lifter register as asymmetric on every
   * repetition, because a left/right gap pinned at zero ranked at the top of its
   * own distribution.
   */
  rank(name: string, value: number | null): number | null {
    if (value === null || Number.isNaN(value)) return null;
    const sorted = this.distributions.get(name);
    if (!sorted || sorted.length === 0) return null;
    return (lowerBound(sorted, value) + upperBound(sorted, value)) / (2 * sorted.length);
  }

  reset(): void {
    this.distributions.clear();
    this.count = 0;
  }
}

/**
 * Build one detector's feature row from a frame.
 *
 * Feature names are read off the model rather than hardcoded, so retraining
 * with a different feature set needs no change here. Every shipped feature is a
 * `rank_` of a derived quantity; anything else is reported missing rather than
 * guessed at, which surfaces as low completeness and makes the detector abstain.
 *
 * Returns the row alongside the share of it that was actually measured.
 */
export function featureRow(
  featureNames: readonly string[],
  values: Record<string, number | null>,
  ranks: RunningRanks,
): { row: (number | null)[]; completeness: number } {
  const row = featureNames.map((name) => {
    if (!name.startsWith("rank_")) return null;
    return ranks.rank(name.slice("rank_".length), values[name.slice("rank_".length)]);
  });

  if (row.length === 0) return { row, completeness: 0 };
  const present = row.filter((value) => value !== null).length;
  return { row, completeness: present / row.length };
}
