import { describe, expect, it } from "vitest";

import { computeAngles } from "./angles";
import { DEFAULT_CONFIG } from "./config";
import { computeMetrics } from "./metrics";
import { detectReps } from "./reps";
import { buildSquatSeries } from "./synthetic";

describe("computeMetrics", () => {
  it("aggregates a full set", () => {
    const series = buildSquatSeries({ reps: 3 });
    const angles = computeAngles(series.frames, series.fps, DEFAULT_CONFIG);
    const reps = detectReps(angles, DEFAULT_CONFIG);
    const metrics = computeMetrics(reps, angles, series.durationS);

    expect(metrics.totalReps).toBe(3);
    expect(metrics.avgDepthPercent).not.toBeNull();
    expect(metrics.depthConsistencyPercent).not.toBeNull(); // needs >= 2 reps
    expect(metrics.repsPerMinute).not.toBeNull();
    expect(metrics.trackingQuality).toBeGreaterThan(0.9);
    expect(metrics.cameraView).toBe("front");
    expect(metrics.fastestRepS).toBeLessThanOrEqual(metrics.slowestRepS ?? 0);
  });

  it("returns zeroed metrics with nulls, never 0, when there are no reps", () => {
    const angles = computeAngles([], 30, DEFAULT_CONFIG);
    const metrics = computeMetrics([], angles, 0);
    expect(metrics.totalReps).toBe(0);
    expect(metrics.avgDepthPercent).toBeNull();
    expect(metrics.depthConsistencyPercent).toBeNull();
    expect(metrics.trackingQuality).toBe(0);
  });
});
