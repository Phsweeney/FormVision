import { describe, expect, it } from "vitest";

import { computeRawFrameAngles, viewGates } from "@/lib/analysis/angles";
import { DEFAULT_CONFIG } from "@/lib/analysis/config";
import { buildSquatSeries } from "@/lib/analysis/synthetic";

import {
  RunningRanks,
  derivedValues,
  featureRow,
  frameSample,
  gapOf,
  meanOf,
  type FrameSample,
} from "./features";

const EMPTY: FrameSample = {
  knee_flexion_left: null,
  knee_flexion_right: null,
  hip_flexion_left: null,
  hip_flexion_right: null,
  ankle_openness_left: null,
  ankle_openness_right: null,
  valgus_left: null,
  valgus_right: null,
  torso_lean: null,
  depth_phase: null,
};

describe("paired signals", () => {
  it("falls back to the one visible side for a mean", () => {
    // Side-on footage tracks one leg. Requiring both would discard it.
    const sample = { ...EMPTY, knee_flexion_left: -90 };
    expect(meanOf(sample, "knee_flexion")).toBeCloseTo(-90, 10);
  });

  it("averages both sides when both are present", () => {
    const sample = { ...EMPTY, knee_flexion_left: -80, knee_flexion_right: -100 };
    expect(meanOf(sample, "knee_flexion")).toBeCloseTo(-90, 10);
  });

  it("refuses to infer a gap from one side", () => {
    // Falling back the way `meanOf` does would report perfect symmetry for a
    // lifter whose other leg was never seen, which is the most misleading
    // possible answer to an asymmetry question.
    expect(gapOf({ ...EMPTY, knee_flexion_left: -90 }, "knee_flexion")).toBeNull();
    const both = { ...EMPTY, knee_flexion_left: -80, knee_flexion_right: -100 };
    expect(gapOf(both, "knee_flexion")).toBeCloseTo(20, 10);
  });

  it("reports an absent signal as missing, not zero", () => {
    expect(meanOf(EMPTY, "valgus")).toBeNull();
    expect(gapOf(EMPTY, "valgus")).toBeNull();
  });
});

describe("RunningRanks", () => {
  it("places a value within the session so far", () => {
    const ranks = new RunningRanks();
    for (const value of [0, 10, 20, 30]) {
      ranks.insert({ ...emptyValues(), torso_lean: value });
    }
    expect(ranks.rank("torso_lean", -5)).toBeCloseTo(0, 10);
    expect(ranks.rank("torso_lean", 15)).toBeCloseTo(0.5, 10);
    expect(ranks.rank("torso_lean", 99)).toBeCloseTo(1, 10);
  });

  it("ranks a signal that never varies as unremarkable", () => {
    // The bug this pins shipped once on the backend. Taking the upper edge of a
    // tied run ranks a constant signal at 1.0 in every frame, i.e. maximally
    // extreme, when a quantity that never moves is the least remarkable thing
    // in the session. It made a perfectly symmetric lifter register as
    // asymmetric on every repetition.
    const ranks = new RunningRanks();
    for (let i = 0; i < 50; i++) ranks.insert({ ...emptyValues(), torso_lean: 5 });
    expect(ranks.rank("torso_lean", 5)).toBeCloseTo(0.5, 10);
  });

  it("is invariant to any monotone rescaling", () => {
    // The property the whole cross-corpus bridge rests on: the training corpus
    // quotes its angles on a different scale to FormVision's.
    const raw = [1, 2, 3, 4, 5];
    const warped = raw.map((v) => v ** 3 + 7);

    const plain = new RunningRanks();
    const remapped = new RunningRanks();
    for (const v of raw) plain.insert({ ...emptyValues(), torso_lean: v });
    for (const v of warped) remapped.insert({ ...emptyValues(), torso_lean: v });

    for (let i = 0; i < raw.length; i++) {
      expect(plain.rank("torso_lean", raw[i])).toBeCloseTo(
        remapped.rank("torso_lean", warped[i])!,
        10,
      );
    }
  });

  it("returns null for a quantity the session never measured", () => {
    const ranks = new RunningRanks();
    ranks.insert({ ...emptyValues(), torso_lean: 1 });
    expect(ranks.rank("ankle_openness_mean", 90)).toBeNull();
    expect(ranks.rank("torso_lean", null)).toBeNull();
  });

  it("counts frames even when nothing in them was measurable", () => {
    const ranks = new RunningRanks();
    ranks.insert(emptyValues());
    ranks.insert(emptyValues());
    expect(ranks.frames).toBe(2);
  });

  it("keeps ranks correct as values arrive out of order", () => {
    const ranks = new RunningRanks();
    for (const value of [50, 10, 90, 30, 70]) {
      ranks.insert({ ...emptyValues(), torso_lean: value });
    }
    // 10 is the lowest of five, so its tied-midpoint rank is 0.5/5.
    expect(ranks.rank("torso_lean", 10)).toBeCloseTo(0.1, 10);
    expect(ranks.rank("torso_lean", 90)).toBeCloseTo(0.9, 10);
  });
});

describe("featureRow", () => {
  it("reports completeness from what was actually measured", () => {
    const ranks = new RunningRanks();
    for (let i = 0; i < 10; i++) {
      ranks.insert({ ...emptyValues(), torso_lean: i, depth_phase: i });
    }
    const values = { ...emptyValues(), torso_lean: 5, depth_phase: null };
    const { row, completeness } = featureRow(
      ["rank_torso_lean", "rank_depth_phase"],
      values,
      ranks,
    );
    expect(row[0]).not.toBeNull();
    expect(row[1]).toBeNull();
    expect(completeness).toBeCloseTo(0.5, 10);
  });
});

/**
 * Larger must mean "more fault" for every signal, read off the real analysis
 * stack. These guard the one class of bug that produces no symptom other than
 * backwards coaching.
 */
describe("canonical orientation", () => {
  function samplesFor(options: Parameters<typeof buildSquatSeries>[0]) {
    const { frames } = buildSquatSeries(options);
    const gates = viewGates(options?.view ?? "front");
    return frames.map((frame) =>
      frameSample(
        computeRawFrameAngles(
          frame,
          0.22,
          0.18,
          gates,
          DEFAULT_CONFIG.landmark_visibility_threshold,
        ),
      ),
    );
  }

  it("raises knee flexion and depth phase at the bottom of a rep", () => {
    const samples = samplesFor({ reps: 1 });
    const depths = samples
      .map((s, i) => ({ depth: s.depth_phase, i }))
      .filter((d): d is { depth: number; i: number } => d.depth !== null);

    const deepest = depths.reduce((a, b) => (b.depth > a.depth ? b : a));
    const highest = depths.reduce((a, b) => (b.depth < a.depth ? b : a));

    expect(meanOf(samples[deepest.i], "knee_flexion")!).toBeGreaterThan(
      meanOf(samples[highest.i], "knee_flexion")!,
    );
  });

  it("raises valgus when the knees cave", () => {
    const mean = (samples: FrameSample[]) => {
      const values = samples
        .map((s) => meanOf(s, "valgus"))
        .filter((v): v is number => v !== null);
      return values.reduce((a, b) => a + b, 0) / values.length;
    };
    expect(mean(samplesFor({ reps: 1, kneeValgus: 0.35 }))).toBeGreaterThan(
      mean(samplesFor({ reps: 1 })),
    );
  });

  it("raises the knee gap when the lifter is one-sided", () => {
    const peak = (samples: FrameSample[]) =>
      Math.max(
        ...samples
          .map((s) => gapOf(s, "knee_flexion"))
          .filter((v): v is number => v !== null),
      );
    expect(peak(samplesFor({ reps: 1, leftRightBias: 0.6 }))).toBeGreaterThan(
      peak(samplesFor({ reps: 1 })),
    );
  });

  it("carries view gating through to the samples", () => {
    // What the camera cannot see must arrive at the model as missing.
    const front = samplesFor({ reps: 1, view: "front" });
    const side = samplesFor({ reps: 1, view: "side" });

    expect(front.every((s) => s.ankle_openness_left === null)).toBe(true);
    expect(front.some((s) => s.valgus_left !== null)).toBe(true);
    expect(side.every((s) => s.valgus_left === null)).toBe(true);
    expect(side.some((s) => s.ankle_openness_left !== null)).toBe(true);
  });
});

describe("derivedValues", () => {
  it("emits left, right, mean and gap for every paired signal", () => {
    const values = derivedValues({
      ...EMPTY,
      valgus_left: 0.1,
      valgus_right: 0.3,
    });
    expect(values.valgus_left).toBeCloseTo(0.1, 10);
    expect(values.valgus_right).toBeCloseTo(0.3, 10);
    expect(values.valgus_mean).toBeCloseTo(0.2, 10);
    expect(values.valgus_gap).toBeCloseTo(0.2, 10);
  });
});

function emptyValues(): Record<string, number | null> {
  return derivedValues(EMPTY);
}
