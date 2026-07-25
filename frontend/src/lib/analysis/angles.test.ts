import { describe, expect, it } from "vitest";

import { computeAngles } from "./angles";
import { DEFAULT_CONFIG } from "./config";
import { buildSquatSeries } from "./synthetic";
import { validFraction } from "./types";

const present = (values: (number | null)[]): number[] =>
  values.filter((v): v is number => v !== null);

describe("computeAngles", () => {
  it("recovers a knee-angle dip per rep and measures body scale", () => {
    const { frames, fps } = buildSquatSeries({ reps: 2, view: "front" });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);

    expect(angles.view).toBe("front");
    expect(angles.torsoScale).not.toBeNull();
    expect(validFraction(angles)).toBeGreaterThan(0.9);

    const knees = present(angles.leftKneeDeg);
    const top = Math.max(...knees);
    const bottom = Math.min(...knees);
    expect(top).toBeGreaterThan(145); // near-extended standing (figure stands at ~97%)
    expect(bottom).toBeLessThan(top - 20); // a clear dip at the bottom of each rep
  });

  it("records torso lean as unmeasurable front-on", () => {
    const { frames, fps } = buildSquatSeries({ view: "front", torsoLeanDeg: 20 });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);
    expect(angles.torsoLeanDeg.every((v) => v === null)).toBe(true);
  });

  it("measures torso lean side-on", () => {
    const { frames, fps } = buildSquatSeries({
      view: "side",
      torsoLeanDeg: 25,
      farSideVisibility: 0.4,
    });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);
    const leans = present(angles.torsoLeanDeg);
    expect(leans.length).toBeGreaterThan(0);
    expect(Math.max(...leans)).toBeGreaterThan(10);
  });
});

/**
 * The signals added so the browser can feed the model. These mirror
 * `backend/tests/test_angles.py`, and the parity matters: the model is trained
 * on one and applied to the other.
 */
describe("per-side signals", () => {
  const SIGNALS = [
    "leftHipDeg",
    "rightHipDeg",
    "leftAnkleDeg",
    "rightAnkleDeg",
    "leftKneeLateral",
    "rightKneeLateral",
  ] as const;

  it("keeps every new signal frame-aligned", () => {
    const { frames, fps } = buildSquatSeries({ reps: 2 });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);
    for (const name of SIGNALS) {
      expect(angles[name]).toHaveLength(frames.length);
    }
  });

  it("reports a long dropout as missing, not zero", () => {
    // The gap must exceed `max_interpolation_gap_frames`, or smoothing fills it
    // — which is correct behaviour for a blink-length dropout, and precisely
    // why this test uses a dropout too long to bridge. Zero is a real reading
    // (a fully folded joint, a knee exactly on its hip-ankle line), so it can
    // never double as "no measurement".
    const gap = DEFAULT_CONFIG.max_interpolation_gap_frames * 2 + 1;
    const undetectedFrames = Array.from({ length: gap }, (_, i) => 20 + i);
    const { frames, fps } = buildSquatSeries({ reps: 1, undetectedFrames });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);

    const middle = 20 + Math.floor(gap / 2);
    for (const name of SIGNALS) {
      expect(angles[name][middle]).toBeNull();
    }
  });

  it("brackets the midpoint hip angle with the per-side ones", () => {
    // Not identical, and should not be: the shoulders are wider than the hips,
    // so each side's torso segment tilts a couple of degrees off the centre line.
    const { frames, fps } = buildSquatSeries({ reps: 2 });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);

    let compared = 0;
    for (let i = 0; i < frames.length; i++) {
      const mid = angles.hipDeg[i];
      const left = angles.leftHipDeg[i];
      const right = angles.rightHipDeg[i];
      if (mid === null || left === null || right === null) continue;
      expect(Math.abs(left - mid)).toBeLessThan(10);
      expect(Math.abs(right - mid)).toBeLessThan(10);
      compared += 1;
    }
    expect(compared).toBeGreaterThan(0);
  });
});

describe("view gating of the new signals", () => {
  it("measures valgus front-on and withholds it side-on", () => {
    const front = computeAngles(
      buildSquatSeries({ reps: 1, view: "front" }).frames,
      30,
      DEFAULT_CONFIG,
    );
    const side = computeAngles(
      buildSquatSeries({ reps: 1, view: "side" }).frames,
      30,
      DEFAULT_CONFIG,
    );

    expect(present(front.leftKneeLateral).length).toBeGreaterThan(0);
    // Side-on a knee projects onto its own hip-to-ankle line however far it has
    // actually collapsed, so a number here would be a confident zero.
    expect(side.leftKneeLateral.every((v) => v === null)).toBe(true);
    expect(side.rightKneeLateral.every((v) => v === null)).toBe(true);
  });

  it("measures the ankle side-on and withholds it front-on", () => {
    const front = computeAngles(
      buildSquatSeries({ reps: 1, view: "front" }).frames,
      30,
      DEFAULT_CONFIG,
    );
    const side = computeAngles(
      buildSquatSeries({ reps: 1, view: "side" }).frames,
      30,
      DEFAULT_CONFIG,
    );

    expect(present(side.leftAnkleDeg).length).toBeGreaterThan(0);
    // Front-on the foot points at the lens. Same reasoning as torso lean.
    expect(front.leftAnkleDeg.every((v) => v === null)).toBe(true);
    expect(front.rightAnkleDeg.every((v) => v === null)).toBe(true);
  });
});

describe("valgus sign", () => {
  it("reads knees caving inward as positive on both sides", () => {
    // Measured as a change against the same clip without valgus rather than as
    // an absolute: the synthetic figure bends both knees toward +x in the image,
    // which lands medially on one leg and laterally on the other. The difference
    // isolates the medial travel, which is what the sign convention is about.
    const neutral = computeAngles(
      buildSquatSeries({ reps: 2, view: "front" }).frames,
      30,
      DEFAULT_CONFIG,
    );
    const caving = computeAngles(
      buildSquatSeries({ reps: 2, view: "front", kneeValgus: 0.35 }).frames,
      30,
      DEFAULT_CONFIG,
    );

    const mean = (values: number[]) =>
      values.reduce((a, b) => a + b, 0) / values.length;

    for (const name of ["leftKneeLateral", "rightKneeLateral"] as const) {
      const before = present(neutral[name]);
      const after = present(caving[name]);
      expect(before.length).toBeGreaterThan(0);
      expect(after.length).toBeGreaterThan(0);
      // A sign error on either side would flip this negative.
      expect(mean(after) - mean(before)).toBeGreaterThan(0.02);
    }
  });
});
