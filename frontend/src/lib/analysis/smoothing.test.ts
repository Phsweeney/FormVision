import { describe, expect, it } from "vitest";

import {
  interpolateGaps,
  movingAverage,
  percentile,
  windowSizeForFps,
} from "./smoothing";

describe("windowSizeForFps", () => {
  it("returns an odd frame count so the window is phase-neutral", () => {
    expect(windowSizeForFps(30, 0.15)).toBe(5);
    expect(windowSizeForFps(0, 0.15)).toBe(3); // floor
    expect(windowSizeForFps(60, 0.15) % 2).toBe(1);
  });
});

describe("interpolateGaps", () => {
  it("bridges short gaps and leaves long ones and edges alone", () => {
    // one-frame gap filled; a three-frame gap with maxGap 2 left as null.
    expect(interpolateGaps([0, null, 2], 2)).toEqual([0, 1, 2]);
    expect(interpolateGaps([0, null, null, null, 4], 2)).toEqual([
      0,
      null,
      null,
      null,
      4,
    ]);
    // leading/trailing never filled.
    expect(interpolateGaps([null, 1, 2, null], 5)).toEqual([null, 1, 2, null]);
  });
});

describe("movingAverage", () => {
  it("averages present values and keeps missing positions missing", () => {
    // window 3, centred: middle value is mean of its neighbours + itself.
    expect(movingAverage([0, 3, 0], 3)).toEqual([1.5, 1, 1.5]);
    // a null position stays null even after smoothing.
    const out = movingAverage([1, null, 3], 3);
    expect(out[1]).toBeNull();
  });
});

describe("percentile", () => {
  it("interpolates linearly and ignores missing values", () => {
    expect(percentile([1, 2, 3, 4], 10)).toBeCloseTo(1.3, 6);
    expect(percentile([1, 2, 3, 4], 90)).toBeCloseTo(3.7, 6);
    expect(percentile([null, null], 50)).toBeNull();
  });
});
