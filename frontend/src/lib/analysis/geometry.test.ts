import { describe, expect, it } from "vitest";

import {
  angleBetweenPoints,
  angleFromVertical,
  linearScale,
} from "./geometry";

describe("angleBetweenPoints", () => {
  it("measures a right angle and a straight line", () => {
    expect(angleBetweenPoints([1, 0], [0, 0], [0, 1])).toBeCloseTo(90, 6);
    expect(angleBetweenPoints([1, 0], [0, 0], [-1, 0])).toBeCloseTo(180, 6);
  });

  it("returns null for a degenerate arm rather than a fake zero", () => {
    // A coincident point gives a zero-length vector: the angle is undefined,
    // and reporting 0 would look like a fully-bent joint.
    expect(angleBetweenPoints([0, 0], [0, 0], [0, 1])).toBeNull();
  });
});

describe("angleFromVertical", () => {
  it("is 0 upright and 90 horizontal, always non-negative", () => {
    // upper directly above lower (image y grows down, so upper.y < lower.y).
    expect(angleFromVertical([0, 1], [0, 0])).toBeCloseTo(0, 6);
    expect(angleFromVertical([0, 0], [1, 0])).toBeCloseTo(90, 6);
    // Leaning either way reads the same magnitude.
    const left = angleFromVertical([0, 1], [-0.5, 0]);
    const right = angleFromVertical([0, 1], [0.5, 0]);
    expect(left).toBeCloseTo(right ?? -1, 6);
  });
});

describe("linearScale", () => {
  it("maps endpoints to 0 and 1 and runs in either direction", () => {
    // Depth: knee angle decreases as the squat deepens, 170deg -> 0%, 90deg -> 100%.
    expect(linearScale(170, 170, 90)).toBeCloseTo(0, 6);
    expect(linearScale(90, 170, 90)).toBeCloseTo(1, 6);
    expect(linearScale(130, 170, 90)).toBeCloseTo(0.5, 6);
  });

  it("clamps out-of-range values by default", () => {
    expect(linearScale(200, 170, 90)).toBe(0);
    expect(linearScale(45, 170, 90)).toBe(1);
    expect(linearScale(45, 170, 90, false)).toBeGreaterThan(1);
  });
});
