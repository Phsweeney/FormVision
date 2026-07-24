import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "@/lib/analysis/config";
import { buildSquatSeries } from "@/lib/analysis/synthetic";
import type { AngleSeries, Rep } from "@/lib/analysis/types";

import { LiveAnalyzer } from "./live-analyzer";
import { analyzeRep } from "./rep-analysis";

/** Build a minimal AngleSeries carrying only the signals analyzeRep reads. */
function bufferFrom(hipHeight: (number | null)[], fps = 30): AngleSeries {
  return {
    timestampsS: hipHeight.map((_, i) => i / fps),
    leftKneeDeg: [],
    rightKneeDeg: [],
    hipDeg: [],
    torsoLeanDeg: [],
    hipHeight,
    hipKneeOffset: [],
    valid: [],
    leftLegValid: [],
    rightLegValid: [],
    torsoScale: 1,
    thighScale: 1,
    view: "front",
  };
}

/** Descend over `down` frames, hold `hold`, ascend over `up`, top=1.6 bottom=0.4. */
function repSignal(down: number, hold: number, up: number): (number | null)[] {
  const top = 1.6;
  const bottom = 0.4;
  const values: number[] = [];
  for (let i = 0; i < down; i++) values.push(top - (top - bottom) * (i / (down - 1)));
  for (let i = 0; i < hold; i++) values.push(bottom);
  for (let i = 0; i < up; i++) values.push(bottom + (top - bottom) * (i / (up - 1)));
  return values;
}

describe("analyzeRep", () => {
  it("splits a paused rep into eccentric / pause / concentric", () => {
    const signal = repSignal(30, 36, 30); // ~1s down, ~1.2s hold, ~1s up @30fps
    const angles = bufferFrom(signal);
    const rep = {
      index: 1,
      startFrame: 0,
      bottomFrame: 45,
      endFrame: signal.length - 1,
      startTimeS: 0,
      bottomTimeS: 45 / 30,
      endTimeS: (signal.length - 1) / 30,
      minKneeAngleDeg: 95,
      minLeftKneeDeg: 95,
      minRightKneeDeg: 95,
      minHipAngleDeg: 40,
      maxTorsoLeanDeg: null,
      kneeAsymmetryDeg: null,
      depthPercent: 95,
      hipBelowKnee: true,
    } satisfies Rep;

    const analyzed = analyzeRep(rep, angles, DEFAULT_CONFIG);
    expect(analyzed.pauseKind).toBe("competition"); // >= 1.0s hold
    expect(analyzed.eccentricS).toBeGreaterThan(0.5);
    expect(analyzed.concentricS).toBeGreaterThan(0.5);
    expect(analyzed.tempo).toMatch(/^\d+-\d+-\d+$/);
    expect(analyzed.halfRep).toBe(false);
  });

  it("reports no pause for a continuous rep", () => {
    const signal = repSignal(30, 1, 30);
    const analyzed = analyzeRep(
      {
        index: 1,
        startFrame: 0,
        bottomFrame: 30,
        endFrame: signal.length - 1,
        startTimeS: 0,
        bottomTimeS: 1,
        endTimeS: (signal.length - 1) / 30,
        minKneeAngleDeg: 95,
        minLeftKneeDeg: 95,
        minRightKneeDeg: 95,
        minHipAngleDeg: 40,
        maxTorsoLeanDeg: null,
        kneeAsymmetryDeg: null,
        depthPercent: 95,
        hipBelowKnee: true,
      },
      bufferFrom(signal),
      DEFAULT_CONFIG,
    );
    expect(analyzed.pauseKind).toBe("none");
  });

  it("flags a rep below the shallow-depth threshold as a half rep", () => {
    const signal = repSignal(30, 1, 30);
    const angles = bufferFrom(signal);
    const base = {
      index: 1,
      startFrame: 0,
      bottomFrame: 30,
      endFrame: signal.length - 1,
      startTimeS: 0,
      bottomTimeS: 1,
      endTimeS: (signal.length - 1) / 30,
      minKneeAngleDeg: 120,
      minLeftKneeDeg: 120,
      minRightKneeDeg: 120,
      minHipAngleDeg: 60,
      maxTorsoLeanDeg: null,
      kneeAsymmetryDeg: null,
      hipBelowKnee: false,
    };
    // shallow_depth_percent defaults to 70.
    const deep = analyzeRep({ ...base, depthPercent: 95 }, angles, DEFAULT_CONFIG);
    const shallow = analyzeRep({ ...base, depthPercent: 55 }, angles, DEFAULT_CONFIG);
    expect(deep.halfRep).toBe(false);
    expect(shallow.halfRep).toBe(true);
  });

  it("enriches the analyzer's completed reps with tempo", () => {
    const series = buildSquatSeries({ reps: 2, standingPauseS: 4, view: "front" });
    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
    for (const frame of series.frames) analyzer.push(frame);
    expect(analyzer.reps.length).toBe(2);
    for (const rep of analyzer.reps) {
      expect(rep.tempo).toMatch(/^\d+-\d+-\d+$/);
    }
  });
});
