import { describe, expect, it } from "vitest";

import { computeAngles } from "./angles";
import { DEFAULT_CONFIG } from "./config";
import { RepStateMachine, detectReps } from "./reps";
import { buildSquatSeries, buildStandingSeries, type SyntheticSeries } from "./synthetic";
import type { Rep } from "./types";

function reps(series: SyntheticSeries): Rep[] {
  const angles = computeAngles(series.frames, series.fps, DEFAULT_CONFIG);
  return detectReps(angles, DEFAULT_CONFIG);
}

describe("detectReps", () => {
  it.each([1, 2, 3, 5])("counts exactly %i reps", (n) => {
    expect(reps(buildSquatSeries({ reps: n })).length).toBe(n);
  });

  it("counts zero on standing-still footage", () => {
    expect(reps(buildStandingSeries()).length).toBe(0);
    // Even with a wobble that a naive counter would amplify into phantom reps.
    expect(reps(buildStandingSeries(4, 30, 0.01)).length).toBe(0);
  });

  it("describes each rep with ordered times and a depth", () => {
    for (const rep of reps(buildSquatSeries({ reps: 2 }))) {
      expect(rep.startTimeS).toBeLessThan(rep.bottomTimeS);
      expect(rep.bottomTimeS).toBeLessThan(rep.endTimeS);
      expect(rep.depthPercent).not.toBeNull();
    }
  });

  it("measures asymmetry front-on, stays near zero when symmetric, null side-on", () => {
    const biased = reps(buildSquatSeries({ reps: 2, leftRightBias: 0.3, view: "front" }));
    expect(biased.every((r) => (r.kneeAsymmetryDeg ?? 0) > 5)).toBe(true);

    const symmetric = reps(buildSquatSeries({ reps: 2, view: "front" }));
    expect(symmetric.every((r) => (r.kneeAsymmetryDeg ?? 99) < 2)).toBe(true);

    const sideOn = reps(
      buildSquatSeries({ reps: 2, view: "side", farSideVisibility: 0.4 }),
    );
    expect(sideOn.every((r) => r.kneeAsymmetryDeg === null)).toBe(true);
  });
});

/**
 * The bottom of a rep is where the hip signal is flattest, so without
 * hysteresis any upward flicker flips descending to ascending and any downward
 * one flips it back. On live webcam footage that made the phase badge oscillate
 * several times per rep. This is the same hysteresis idea as the gap between
 * the descend and ascend thresholds, applied to the turnaround.
 */
describe("RepStateMachine turnaround band", () => {
  /** Feed a signal and return the phase after each sample. */
  function phases(values: number[], band: number): string[] {
    const machine = new RepStateMachine(0.7, 0.9, band);
    return values.map((v, i) => {
      machine.push(i, v);
      return machine.phase;
    });
  }

  // Descend to 0.5, then wobble by 0.01 either side of the bottom.
  const wobble = [1.0, 0.8, 0.6, 0.5, 0.51, 0.5, 0.505, 0.5, 0.51, 0.5];

  it("oscillates without a band", () => {
    const seen = phases(wobble, 0);
    expect(new Set(seen.slice(3)).size).toBeGreaterThan(1);
  });

  it("holds the phase steady through noise at the bottom", () => {
    const seen = phases(wobble, 0.04);
    expect(new Set(seen.slice(3))).toEqual(new Set(["descending"]));
  });

  it("still turns around on a real ascent", () => {
    const seen = phases([1.0, 0.8, 0.6, 0.5, 0.55, 0.65, 0.75], 0.04);
    expect(seen.at(-1)).toBe("ascending");
  });

  it("still closes a rep, and only one", () => {
    const machine = new RepStateMachine(0.7, 0.9, 0.04);
    const signal = [1.0, 0.8, 0.6, 0.5, 0.51, 0.5, 0.52, 0.7, 0.85, 0.95, 1.0];
    const triples = signal
      .map((v, i) => machine.push(i, v))
      .filter((t) => t !== null);
    expect(triples).toHaveLength(1);
  });
});
