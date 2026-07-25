/**
 * Repetition detection — the TS mirror of `reps.py`.
 *
 * A hysteresis state machine over the hip-height signal. Two thresholds, a lower
 * one to enter the descent and a higher one to close the rep, so the signal must
 * cross a band to advance and a wobble at the boundary cannot register many reps.
 *
 * The batch `detectReps` derives its thresholds from the 90th/10th percentiles
 * of the whole clip, exactly like Python. The state machine itself is factored
 * into `RepStateMachine`, which is already incremental — the live driver feeds
 * it one frame at a time with thresholds set from calibration instead.
 */

import type { AnalysisConfig } from "./config";
import { linearScale } from "./geometry";
import { percentile } from "./smoothing";
import {
  type AngleSeries,
  type Rep,
  type Signal,
} from "./types";

type State = "standing" | "descending" | "ascending";

export interface RepTriple {
  start: number;
  bottom: number;
  end: number;
}

/**
 * The incremental core of rep detection. `push` one frame value at a time; it
 * returns a completed `(start, bottom, end)` triple the instant a rep closes,
 * or null. Untracked frames (null) do not advance it, so a brief dropout does
 * not split a rep in two.
 */
export class RepStateMachine {
  private state: State = "standing";
  private startFrame = 0;
  private bottomFrame = 0;
  private bottomValue = Infinity;
  private lastStandingFrame = 0;

  /**
   * @param turnaroundBand How far the hip must reverse, in torso lengths,
   *   before the bottom counts as passed. Hysteresis, for the same reason the
   *   descend and ascend thresholds are separated: the signal is nearly flat at
   *   the bottom of a rep, so reacting to any flicker makes the phase oscillate
   *   on real tracking noise. Defaults to zero so existing callers and the
   *   parity tests keep their exact previous behaviour.
   */
  constructor(
    private descendThreshold: number,
    private ascendThreshold: number,
    private readonly turnaroundBand = 0,
  ) {}

  /** Update the thresholds mid-stream (the live driver adapts them per rep). */
  setThresholds(descend: number, ascend: number): void {
    this.descendThreshold = descend;
    this.ascendThreshold = ascend;
  }

  /** The current phase, for a live state readout. */
  get phase(): State {
    return this.state;
  }

  push(index: number, value: number | null): RepTriple | null {
    if (value === null) return null;

    if (this.state === "standing") {
      if (value >= this.ascendThreshold) {
        this.lastStandingFrame = index;
      } else if (value < this.descendThreshold) {
        this.state = "descending";
        this.startFrame = this.lastStandingFrame;
        this.bottomFrame = index;
        this.bottomValue = value;
      }
      return null;
    }

    if (this.state === "descending") {
      if (value < this.bottomValue) {
        this.bottomValue = value;
        this.bottomFrame = index;
      } else if (value > this.bottomValue + this.turnaroundBand) {
        // The hip has started back up: the bottom is behind us.
        this.state = "ascending";
      }
      return null;
    }

    // ascending
    if (value < this.bottomValue - this.turnaroundBand) {
      // Dipped again before locking out, so still the same descent. The band
      // keeps a noisy sample from reopening a descent that is genuinely over.
      this.bottomValue = value;
      this.bottomFrame = index;
      this.state = "descending";
      return null;
    }
    if (value >= this.ascendThreshold) {
      const triple: RepTriple = {
        start: this.startFrame,
        bottom: this.bottomFrame,
        end: index,
      };
      this.state = "standing";
      this.lastStandingFrame = index;
      this.bottomValue = Infinity;
      return triple;
    }
    return null;
  }
}

/** Find every repetition in a clip. Empty when there is no detectable movement. */
export function detectReps(angles: AngleSeries, config: AnalysisConfig): Rep[] {
  const signal = angles.hipHeight;
  if (signal.length === 0) return [];

  const baseline = percentile(signal, 90);
  const bottomReference = percentile(signal, 10);
  if (baseline === null || bottomReference === null) return [];

  const travel = baseline - bottomReference;
  if (travel < config.min_rep_range) return [];

  const descend = baseline - config.rep_descent_fraction * travel;
  const ascend = baseline - config.rep_ascent_fraction * travel;

  const machine = new RepStateMachine(descend, ascend, config.rep_turnaround_band);
  const triples: RepTriple[] = [];
  signal.forEach((value, index) => {
    const triple = machine.push(index, value);
    if (triple) triples.push(triple);
  });

  return finalise(triples, angles, config);
}

/** Turn frame triples into `Rep`s, discarding implausibly fast candidates. */
export function finalise(
  triples: RepTriple[],
  angles: AngleSeries,
  config: AnalysisConfig,
): Rep[] {
  const reps: Rep[] = [];
  for (const { start, bottom, end } of triples) {
    const startTime = angles.timestampsS[start];
    const endTime = angles.timestampsS[end];
    if (endTime - startTime < config.min_rep_duration_s) continue;
    reps.push(buildRep(reps.length + 1, start, bottom, end, angles, config));
  }
  return reps;
}

/** Measure one repetition over its frame window. */
export function buildRep(
  index: number,
  start: number,
  bottom: number,
  end: number,
  angles: AngleSeries,
  config: AnalysisConfig,
): Rep {
  const leftMin = windowMin(angles.leftKneeDeg, start, end);
  const rightMin = windowMin(angles.rightKneeDeg, start, end);
  const kneeValues = [leftMin, rightMin].filter((v): v is number => v !== null);
  const minKnee = kneeValues.length ? Math.min(...kneeValues) : null;

  // Asymmetry needs a front-on camera and is the worst instantaneous gap across
  // the rep, not the gap between each leg's independent minimum (that cancels a
  // real one-sided shift). Matches the fix made on the backend.
  const asymmetry =
    angles.view === "front"
      ? windowMaxGap(angles.leftKneeDeg, angles.rightKneeDeg, start, end)
      : null;

  const depthPercent =
    minKnee !== null
      ? linearScale(
          minKnee,
          config.standing_knee_angle_deg,
          config.parallel_knee_angle_deg,
        ) * 100
      : null;

  const offset = windowMax(angles.hipKneeOffset, start, end);

  return {
    index,
    startFrame: start,
    bottomFrame: bottom,
    endFrame: end,
    startTimeS: angles.timestampsS[start],
    bottomTimeS: angles.timestampsS[bottom],
    endTimeS: angles.timestampsS[end],
    minKneeAngleDeg: minKnee,
    minLeftKneeDeg: leftMin,
    minRightKneeDeg: rightMin,
    minHipAngleDeg: windowMin(angles.hipDeg, start, end),
    maxTorsoLeanDeg: windowMax(angles.torsoLeanDeg, start, end),
    kneeAsymmetryDeg: asymmetry,
    depthPercent,
    hipBelowKnee: offset !== null && offset >= 0,
  };
}

export function windowMin(values: Signal, start: number, end: number): number | null {
  let min: number | null = null;
  for (let i = start; i <= end && i < values.length; i++) {
    const value = values[i];
    if (value !== null && (min === null || value < min)) min = value;
  }
  return min;
}

export function windowMax(values: Signal, start: number, end: number): number | null {
  let max: number | null = null;
  for (let i = start; i <= end && i < values.length; i++) {
    const value = values[i];
    if (value !== null && (max === null || value > max)) max = value;
  }
  return max;
}

/** Largest |a - b| over frames where both are present, or null. */
export function windowMaxGap(
  a: Signal,
  b: Signal,
  start: number,
  end: number,
): number | null {
  let max: number | null = null;
  for (let i = start; i <= end && i < a.length && i < b.length; i++) {
    const av = a[i];
    const bv = b[i];
    if (av !== null && bv !== null) {
      const gap = Math.abs(av - bv);
      if (max === null || gap > max) max = gap;
    }
  }
  return max;
}
