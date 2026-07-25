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
    // A 4s stand-still lead-in comfortably covers the stillness gate plus the
    // 2.5s calibration before the first rep.
    const series = buildSquatSeries({ reps: 3, standingPauseS: 4, view: "front" });
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
    const series = buildSquatSeries({ reps: 3, standingPauseS: 4, view: "front" });
    // Corrupt three consecutive frames in the standing pause after rep 1
    // (lead-in 0-119, rep 1 120-179, pause 180-299).
    for (const i of [240, 241, 242]) {
      series.frames[i] = buildFrame(i, series.frames[i].timestampS, 2.0, {
        view: "front",
      });
    }

    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
    for (const frame of series.frames) analyzer.push(frame);

    expect(analyzer.reps.length).toBe(3);
  });

  it("waits out a walk-in so movement never poisons calibration", () => {
    // Reproduces the real failure: starting the camera, then walking into
    // position. The first ~1.5s are moving frames (hip drifting well beyond the
    // stillness tolerance); calibration must ignore them and only measure once
    // the lifter settles, then still count every rep.
    const series = buildSquatSeries({ reps: 3, standingPauseS: 6, view: "front" });
    for (let i = 0; i < 45; i++) {
      const drifting = 0.57 + 0.08 * Math.sin(i * 0.9);
      series.frames[i] = buildFrame(i, series.frames[i].timestampS, drifting, {
        view: "front",
      });
    }

    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
    let last = analyzer.push(series.frames[0]);
    for (let i = 1; i < series.frames.length; i++) {
      last = analyzer.push(series.frames[i]);
    }

    expect(last.ready).toBe(true);
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

  // Every rep used to be counted twice, the second landing about half a second
  // after the first as the lifter finished standing up. A rep closes when the
  // hip crosses back above `baseline - 0.25 * travel`, which is short of
  // upright; the standing reference then snapped to the baseline, collapsing
  // `travel` to `min_rep_range` and lifting the descend threshold above where
  // the hip actually was. The machine re-entered a descent on the spot and the
  // rest of the ascent tripped a second rep.
  //
  // Only slow reps showed it, which is why the suite missed it: the phantom has
  // to outlast `min_rep_duration_s` to survive the filter, so a brisk rep hid
  // it. These cases are therefore deliberately slow.
  describe("does not double-count a rep at lockout", () => {
    for (const repDurationS of [4, 6, 8]) {
      it(`counts a ${repDurationS}s rep once`, () => {
        const series = buildSquatSeries({
          reps: 3,
          standingPauseS: 4,
          repDurationS,
          view: "front",
        });
        const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
        for (const frame of series.frames) analyzer.push(frame);

        expect(analyzer.reps.length).toBe(3);
      });
    }

    it("keeps counting when reps get shallower", () => {
      // The standing reference carries the last rep's depth forward, so a rep
      // shallower than the one before it must still cross the descend
      // threshold. If it did not, the fix above would trade a double count for
      // a missed rep.
      const series = buildSquatSeries({
        reps: 4,
        standingPauseS: 4,
        repDurationS: 6,
        depthFraction: 0.75,
        depthJitter: 0.25,
        view: "front",
      });
      const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
      for (const frame of series.frames) analyzer.push(frame);

      expect(analyzer.reps.length).toBe(4);
    });
  });
});
