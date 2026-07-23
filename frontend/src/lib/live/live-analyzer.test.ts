import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "@/lib/analysis/config";
import {
  buildFrame,
  buildSquatSeries,
  buildStandingSeries,
} from "@/lib/analysis/synthetic";

import { LiveAnalyzer } from "./live-analyzer";

describe("LiveAnalyzer", () => {
  it("calibrates during the lead-in, then counts reps online", () => {
    // A 3s stand-still lead-in covers the 2.5s calibration before the first rep.
    const series = buildSquatSeries({ reps: 3, standingPauseS: 3, view: "front" });
    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);

    let last = analyzer.push(series.frames[0]);
    for (let i = 1; i < series.frames.length; i++) {
      last = analyzer.push(series.frames[i]);
    }

    expect(last.ready).toBe(true);
    expect(last.view).toBe("front");
    expect(analyzer.reps.length).toBe(3);
    expect(last.repCount).toBe(3);
    expect(last.maxDepthPercent).not.toBeNull();
  });

  it("survives a tracking glitch that a running-minimum would never recover from", () => {
    // Regression: a few frames where the hip is tracked far below the ankle (a
    // real MediaPipe blip) produce a wildly negative hip height. A session-long
    // running minimum would latch onto it, push the descend threshold below
    // anything reachable, and stop counting reps for good — the exact symptom
    // reported (stuck on "Standing" at full depth). The per-rep reset must
    // absorb it.
    const series = buildSquatSeries({ reps: 3, standingPauseS: 3, view: "front" });
    // Corrupt three consecutive frames in the standing pause after rep 1.
    for (const i of [199, 200, 201]) {
      series.frames[i] = buildFrame(i, series.frames[i].timestampS, 2.0, {
        view: "front",
      });
    }

    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
    for (const frame of series.frames) analyzer.push(frame);

    expect(analyzer.reps.length).toBe(3);
  });

  it("counts nothing while the lifter just stands there", () => {
    const series = buildStandingSeries(6, 30);
    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
    let last = analyzer.push(series.frames[0]);
    for (let i = 1; i < series.frames.length; i++) {
      last = analyzer.push(series.frames[i]);
    }
    expect(last.ready).toBe(true);
    expect(analyzer.reps.length).toBe(0);
    expect(last.phase).toBe("standing");
  });
});
