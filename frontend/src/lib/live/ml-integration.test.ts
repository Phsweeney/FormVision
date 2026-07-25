import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "@/lib/analysis/config";
import { buildSquatSeries } from "@/lib/analysis/synthetic";
import type { ViewOrientation } from "@/lib/analysis/types";
import { parseBundle, type FaultModelBundle } from "@/lib/ml/model";

import { LiveAnalyzer, type LiveState } from "./live-analyzer";

const bundle = parseBundle(
  JSON.parse(
    readFileSync(
      join(
        process.cwd(),
        "..",
        "backend",
        "app",
        "ml",
        "artifacts",
        "squat_faults_web.json",
      ),
      "utf8",
    ),
  ),
) as FaultModelBundle;

/**
 * Drive the analyzer over a whole synthetic clip and keep every state.
 *
 * The standing pause has to exceed `live_calibration_seconds`, or calibration
 * never completes, no reps are detected, and every assertion here fails for a
 * reason that has nothing to do with the model.
 */
function run(
  options: Parameters<typeof buildSquatSeries>[0],
  models: FaultModelBundle | null = bundle,
): LiveState[] {
  const { frames } = buildSquatSeries({ standingPauseS: 4, ...options });
  const analyzer = new LiveAnalyzer(DEFAULT_CONFIG, models);
  return frames.map((frame) => analyzer.push(frame));
}

const verdicts = (states: LiveState[]) =>
  states.map((s) => s.mlVerdict).filter((v): v is NonNullable<typeof v> => v !== null);

/**
 * The live model path, driven through the real analyzer.
 *
 * `ml/features.test.ts` and `ml/model.test.ts` cover the pieces; this covers
 * them assembled, which is where the interesting failures live — a signal that
 * never reaches the sample, a classifier built before the view is known, a
 * verdict that never appears because the warm-up gate is never satisfied.
 */
describe("live model integration", () => {
  it("produces verdicts once the session has enough history", () => {
    const states = run({ reps: 3, view: "front" });
    const seen = verdicts(states);

    expect(seen.length).toBeGreaterThan(0);
    // Nothing before the warm-up gate, and it must actually be reached.
    expect(states[0].mlVerdict).toBeNull();
    expect(states.at(-1)!.mlVerdict).not.toBeNull();
  });

  describe("verdicts are per repetition", () => {
    it("changes only when a rep completes", () => {
      // The whole point of the per-rep design. A readout that updates every
      // frame is unreadable: by the time you have stood up you have no idea
      // what it said about the part of the rep that mattered.
      const states = run({ reps: 4, view: "front" });

      let changes = 0;
      let previous: (typeof states)[number]["mlVerdict"] = null;
      const repCountsAtChange: number[] = [];

      for (const state of states) {
        if (state.mlVerdict !== previous) {
          if (previous !== null) repCountsAtChange.push(state.repCount);
          previous = state.mlVerdict;
          changes += 1;
        }
      }

      // One change per judged rep, not hundreds.
      expect(changes).toBeLessThanOrEqual(states.at(-1)!.repCount);
      // Every change lands on a distinct rep count, i.e. at a rep boundary.
      expect(new Set(repCountsAtChange).size).toBe(repCountsAtChange.length);
    });

    it("labels the verdict with the rep it describes", () => {
      const states = run({ reps: 4, view: "front" });
      for (const state of states) {
        if (state.mlVerdict === null) continue;
        // The verdict describes a rep that has already been counted.
        expect(state.mlVerdict.repIndex).toBeGreaterThan(0);
        expect(state.mlVerdict.repIndex).toBeLessThanOrEqual(state.repCount);
      }
    });

    it("does not judge the first rep", () => {
      // The first rep establishes the range of motion the ranks are measured
      // against. Judging it would compare a descent against a distribution
      // built entirely from standing frames.
      const states = run({ reps: 3, view: "front" });
      const firstWithVerdict = states.find((s) => s.mlVerdict !== null);
      expect(firstWithVerdict).toBeDefined();
      expect(firstWithVerdict!.mlVerdict!.repIndex).toBeGreaterThanOrEqual(2);
    });

    it("holds the verdict steady through the following rep", () => {
      const states = run({ reps: 4, view: "front" });
      const withVerdict = states.filter((s) => s.mlVerdict !== null);

      // Group consecutive states by the rep their verdict describes; each group
      // must carry one identical verdict object throughout.
      for (const state of withVerdict) {
        const sameRep = withVerdict.filter(
          (s) => s.mlVerdict!.repIndex === state.mlVerdict!.repIndex,
        );
        const distinct = new Set(sameRep.map((s) => s.mlVerdict));
        expect(distinct.size).toBe(1);
      }
    });
  });

  it("says nothing at all when no model was supplied", () => {
    // Live mode has to work with the artifact missing, which is what happens
    // before `npm run setup:live` or if the export was never built.
    const states = run({ reps: 2, view: "front" }, null);
    expect(states.every((s) => s.mlVerdict === null)).toBe(true);
    // Everything else keeps working.
    expect(states.at(-1)!.repCount).toBeGreaterThan(0);
  });

  it("does not disturb rep counting", () => {
    const withModel = run({ reps: 3, view: "front" });
    const without = run({ reps: 3, view: "front" }, null);
    expect(withModel.at(-1)!.repCount).toBe(without.at(-1)!.repCount);
    expect(withModel.at(-1)!.maxDepthPercent).toBeCloseTo(
      without.at(-1)!.maxDepthPercent!,
      10,
    );
  });

  it("reports confidence as a probability", () => {
    for (const verdict of verdicts(run({ reps: 3, view: "front" }))) {
      expect(verdict.confidence).toBeGreaterThanOrEqual(0);
      expect(verdict.confidence).toBeLessThanOrEqual(1);
    }
  });

  describe("camera angle decides what is assessable", () => {
    const lastVerdict = (view: ViewOrientation) => {
      const seen = verdicts(run({ reps: 3, view }));
      expect(seen.length).toBeGreaterThan(0);
      return seen.at(-1)!;
    };

    it("checks knees and symmetry front-on, not heels", () => {
      const verdict = lastVerdict("front");
      expect(verdict.checking).toContain("knees");
      expect(verdict.notChecking).toContain("heels");
    });

    it("checks heels side-on, not knees", () => {
      const verdict = lastVerdict("side");
      expect(verdict.checking).toContain("heels");
      expect(verdict.notChecking).toContain("knees");
    });

    it("never claims to check and not check the same fault", () => {
      for (const view of ["front", "side"] as const) {
        const verdict = lastVerdict(view);
        const overlap = verdict.checking.filter((f) => verdict.notChecking.includes(f));
        expect(overlap).toEqual([]);
      }
    });
  });

  it("stays quiet on a clean side-on squat", () => {
    // The same standard the backend detectors are held to: a correct lifter
    // must not be told they have a fault.
    //
    // Side-on rather than front-on, deliberately. The synthetic figure cannot
    // represent a front-on view faithfully for valgus: it displaces both knees
    // toward +x in the image as they flex, which reads as medial on the left
    // leg and lateral on the right, manufacturing a left/right knee gap that
    // reaches 1.8 torso lengths. That is roughly ten times anatomically
    // plausible, and the valgus detector reads it — correctly — as knees doing
    // something extraordinary. On real front-on footage the same detector
    // measured 0.000 at full feature completeness, so this is an artefact of
    // the stick figure rather than of the model. Asserting on it would be
    // testing the fixture. See docs/ml.md.
    const seen = verdicts(run({ reps: 4, view: "side" }));
    const flagged = seen.filter((v) => v.fault !== null);
    expect(flagged.length / seen.length).toBeLessThan(0.1);
  });

  it("holds its last verdict through the standing pause", () => {
    // Standing between reps is a near-constant pose whose ranks say nothing
    // about technique, so those frames are not scored. The last rep's verdict
    // must stay on screen rather than blanking at lockout.
    const { frames } = buildSquatSeries({
      reps: 3,
      standingPauseS: 4,
      view: "front",
    });
    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG, bundle);
    const states = frames.map((frame) => analyzer.push(frame));

    // The tail is the long standing pause after the final rep.
    const trailing = states.slice(-60).map((s) => s.mlVerdict);
    expect(trailing.every((v) => v !== null)).toBe(true);
    expect(new Set(trailing).size).toBe(1);
  });

  it("clears its verdict on reset", () => {
    const { frames } = buildSquatSeries({ reps: 2, view: "front" });
    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG, bundle);
    for (const frame of frames) analyzer.push(frame);

    analyzer.reset();
    expect(analyzer.push(frames[0]).mlVerdict).toBeNull();
  });
});
