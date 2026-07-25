/**
 * Turning per-fault probabilities into the one line the live box shows.
 *
 * The backend's coaching rules abstain by default and speak only when they have
 * grounds. This is deliberately different: it is a live readout, always on
 * screen, showing what the model currently thinks. That is a reasonable thing
 * for a status box and a bad thing for coaching, which is why this never feeds
 * the voice coach or the cue banner, which keep their existing rule-based
 * behaviour untouched.
 *
 * The honesty requirement carries over though. No single camera angle can see
 * all three faults: valgus and asymmetry need a front-on view, heel lift needs a
 * side-on one. A box that reports "looks correct" without saying what it did not
 * check would be claiming more than it knows, so the verdict carries both lists.
 */

import { featureRow, type RunningRanks } from "./features";
import { scoreFault, type FaultModelBundle } from "./model";

/** Human labels, and which view each fault needs to be visible at all. */
const FAULTS: Record<string, { label: string; short: string }> = {
  knee_valgus: { label: "Knees caving in", short: "knees" },
  heel_lift: { label: "Heels coming up", short: "heels" },
  asymmetry: { label: "Sides loading unevenly", short: "symmetry" },
};

/**
 * Share of a detector's inputs that must have been measured before it counts as
 * assessable. Mirrors `ml_min_feature_completeness` on the backend: a valgus
 * reading from a camera that cannot see the frontal plane is not a reading.
 */
const MIN_COMPLETENESS = 0.75;

/**
 * Share of a repetition's frames that must clear a detector's threshold before
 * the fault is reported. Mirrors `ml_min_affected_fraction` on the backend.
 *
 * The distinction between this and a threshold on the *mean* probability is not
 * academic, and getting it wrong silenced the asymmetry detector entirely.
 * The backend asks "what fraction of frames scored above the threshold", where
 * averaging the probabilities first and comparing that to the same threshold is
 * a far stricter test: a fault present in a third of a rep pulls the mean
 * nowhere near a threshold of 0.875. Thresholding per frame and averaging the
 * *indicator* is what the model's operating point was actually chosen for.
 */
const MIN_AFFECTED_FRACTION = 0.25;

export interface FaultReading {
  faultId: string;
  label: string;
  short: string;
  /** Smoothed probability, for display. */
  probability: number;
  /** Share of the recent window that cleared the threshold. */
  affectedFraction: number;
  threshold: number;
  fired: boolean;
}

export interface MlVerdict {
  /** Which repetition this verdict describes, as a lifter would count it. */
  repIndex: number;
  /** The fault being reported, or null when nothing cleared its threshold. */
  fault: FaultReading | null;
  /** 0-1. The fault's probability, or the confidence that nothing is wrong. */
  confidence: number;
  /** Short names of the faults this camera angle could judge. */
  checking: string[];
  /** Short names of the faults it could not. */
  notChecking: string[];
}

/** Running totals for one fault across the repetition being performed. */
interface Accumulator {
  scored: number;
  cleared: number;
  probabilitySum: number;
}

/**
 * Scores every shipped detector per frame, and reports once per repetition.
 *
 * **Why per rep and not per frame.** A frame-by-frame readout updates thirty
 * times a second, and a box that changes its mind that often cannot be read:
 * by the time you have stood up you have no idea what it said about the part
 * of the rep that mattered. Aggregating over the whole repetition and reporting
 * at lockout gives one stable answer per rep, which is the unit a lifter thinks
 * in anyway.
 *
 * It is also how the model was trained and evaluated. The backend scores frames
 * within a rep window and fires when a quarter of them clear the threshold; the
 * same rule here means live and upload now agree on the *decision*, not merely
 * on the weights.
 */
export class LiveClassifier {
  private readonly totals = new Map<string, Accumulator>();
  /** Faults that were assessable for at least one frame of this rep. */
  private readonly seen = new Set<string>();

  // Note there is no camera-view parameter. Which faults are assessable is
  // decided by feature completeness alone, because that is the thing actually
  // being asked: a detector whose inputs the camera could not see fails the
  // completeness gate regardless of what the view was classified as.
  constructor(private readonly bundle: FaultModelBundle) {
    this.resetRep();
  }

  /**
   * Fold one frame into the repetition in progress.
   *
   * Call only for frames where the lifter is actually moving. Standing between
   * reps is a near-constant pose whose ranks say nothing about technique, and
   * including it would dilute the fraction that decides the verdict.
   */
  observe(values: Record<string, number | null>, ranks: RunningRanks): void {
    for (const [faultId, model] of Object.entries(this.bundle.faults)) {
      const { row, completeness } = featureRow(model.features, values, ranks);
      if (completeness < MIN_COMPLETENESS) continue;

      this.seen.add(faultId);
      const total = this.totals.get(faultId)!;
      const probability = scoreFault(model, row);
      total.scored += 1;
      total.probabilitySum += probability;
      if (probability >= model.threshold) total.cleared += 1;
    }
  }

  /**
   * Close out the repetition and produce its verdict, then start a fresh one.
   *
   * `hasPriorRep` is a gate, not a nicety. Ranking is causal here: a frame is
   * compared against the session so far, so during the very first repetition
   * the distribution is built almost entirely from standing frames. That rep's
   * descent produces values the session has never seen, they all rank near 1.0,
   * and the model reads an extreme. Measured on a clean synthetic squat, that
   * flagged 63% of frames as knees caving. The first rep therefore establishes
   * the range of motion and is not itself judged.
   */
  completeRep(repIndex: number, hasPriorRep: boolean): MlVerdict | null {
    const verdict = hasPriorRep ? this.buildVerdict(repIndex) : null;
    this.resetRep();
    return verdict;
  }

  /** Discard the repetition in progress, e.g. when tracking is lost. */
  resetRep(): void {
    this.seen.clear();
    for (const faultId of Object.keys(this.bundle.faults)) {
      this.totals.set(faultId, { scored: 0, cleared: 0, probabilitySum: 0 });
    }
  }

  private buildVerdict(repIndex: number): MlVerdict | null {
    const readings: FaultReading[] = [];
    const checking: string[] = [];
    const notChecking: string[] = [];

    for (const [faultId, model] of Object.entries(this.bundle.faults)) {
      const meta = FAULTS[faultId] ?? { label: faultId, short: faultId };
      const total = this.totals.get(faultId)!;

      if (!this.seen.has(faultId) || total.scored === 0) {
        notChecking.push(meta.short);
        continue;
      }

      checking.push(meta.short);
      const affectedFraction = total.cleared / total.scored;
      readings.push({
        faultId,
        label: meta.label,
        short: meta.short,
        probability: total.probabilitySum / total.scored,
        affectedFraction,
        threshold: model.threshold,
        fired: affectedFraction >= MIN_AFFECTED_FRACTION,
      });
    }

    // A rep the camera could not judge at all is still worth reporting, so the
    // box can say the angle saw nothing rather than implying the rep was clean.
    // Most affected first, so a rep showing two faults reports the dominant one.
    const fired = readings
      .filter((reading) => reading.fired)
      .sort((a, b) => b.affectedFraction - a.affectedFraction);

    if (fired.length > 0) {
      return { repIndex, fault: fired[0], confidence: fired[0].probability, checking, notChecking };
    }

    // Nothing fired. Confidence in "clean" is the complement of how close the
    // nearest detector came, so a rep that nearly triggered reads as less
    // certain than one nothing came near.
    const closest = readings.reduce(
      (worst, r) => Math.max(worst, r.affectedFraction / MIN_AFFECTED_FRACTION),
      0,
    );
    return {
      repIndex,
      fault: null,
      confidence: readings.length > 0 ? 1 - Math.min(closest, 1) : 0,
      checking,
      notChecking,
    };
  }
}

/**
 * Fetch the exported detectors.
 *
 * Follows `fetchConfig`'s pattern in `lib/analysis/config.ts`: swallows every
 * error and resolves to null rather than rejecting, so a missing or corrupt
 * model costs the readout and nothing else. Live mode must keep working with no
 * model at all.
 */
export async function fetchFaultModels(
  signal?: AbortSignal,
): Promise<FaultModelBundle | null> {
  try {
    const response = await fetch("/models/squat_faults_web.json", {
      cache: "force-cache",
      signal,
    });
    if (!response.ok) return null;
    const { parseBundle } = await import("./model");
    return parseBundle(await response.json());
  } catch {
    return null;
  }
}
