import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "@/lib/analysis/config";
import { buildSquatSeries, buildStandingSeries } from "@/lib/analysis/synthetic";

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
