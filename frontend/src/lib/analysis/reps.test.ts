import { describe, expect, it } from "vitest";

import { computeAngles } from "./angles";
import { DEFAULT_CONFIG } from "./config";
import { detectReps } from "./reps";
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
