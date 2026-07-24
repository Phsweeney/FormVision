import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "@/lib/analysis/config";
import { buildSquatSeries } from "@/lib/analysis/synthetic";

import { LiveAnalyzer } from "./live-analyzer";
import { summarizeSession } from "./session";

describe("summarizeSession", () => {
  it("folds a session's reps into a summary", () => {
    const series = buildSquatSeries({ reps: 3, standingPauseS: 4, view: "front" });
    const analyzer = new LiveAnalyzer(DEFAULT_CONFIG);
    for (const frame of series.frames) analyzer.push(frame);

    const summary = summarizeSession(analyzer.reps, 42);
    expect(summary.totalReps).toBe(3);
    expect(summary.fullReps + summary.halfReps).toBe(3);
    expect(summary.bestDepthPercent).not.toBeNull();
    expect(summary.avgTempo).toMatch(/^\d+-\d+-\d+$/);
    expect(summary.workingTimeS).toBeGreaterThan(0);
    expect(summary.durationS).toBe(42);
  });

  it("returns an empty summary with nulls, never zeros, for no reps", () => {
    const summary = summarizeSession([], 10);
    expect(summary.totalReps).toBe(0);
    expect(summary.bestDepthPercent).toBeNull();
    expect(summary.avgTempo).toBeNull();
    expect(summary.workingTimeS).toBeNull();
    expect(summary.durationS).toBe(10);
  });
});
