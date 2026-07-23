import { describe, expect, it } from "vitest";

import {
  formatDuration,
  formatPercent,
  formatSeconds,
  NOT_MEASURED,
} from "./format";

describe("format helpers", () => {
  it("renders an em dash for missing values, and never for a real zero", () => {
    // The load-bearing distinction: a value that could not be measured must not
    // read as 0, which on a dashboard looks like a real (and alarming) result.
    expect(formatPercent(null)).toBe(NOT_MEASURED);
    expect(formatSeconds(undefined)).toBe(NOT_MEASURED);
    expect(formatPercent(Number.NaN)).toBe(NOT_MEASURED);
    expect(formatPercent(0)).toBe("0%");
  });

  it("formats durations as raw seconds below a minute and m:ss above", () => {
    expect(formatDuration(45)).toBe("45.0s");
    expect(formatDuration(90)).toBe("1:30");
    expect(formatDuration(605)).toBe("10:05");
  });
});
