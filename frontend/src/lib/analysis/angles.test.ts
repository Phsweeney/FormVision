import { describe, expect, it } from "vitest";

import { computeAngles } from "./angles";
import { DEFAULT_CONFIG } from "./config";
import { buildSquatSeries } from "./synthetic";
import { validFraction } from "./types";

const present = (values: (number | null)[]): number[] =>
  values.filter((v): v is number => v !== null);

describe("computeAngles", () => {
  it("recovers a knee-angle dip per rep and measures body scale", () => {
    const { frames, fps } = buildSquatSeries({ reps: 2, view: "front" });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);

    expect(angles.view).toBe("front");
    expect(angles.torsoScale).not.toBeNull();
    expect(validFraction(angles)).toBeGreaterThan(0.9);

    const knees = present(angles.leftKneeDeg);
    const top = Math.max(...knees);
    const bottom = Math.min(...knees);
    expect(top).toBeGreaterThan(145); // near-extended standing (figure stands at ~97%)
    expect(bottom).toBeLessThan(top - 20); // a clear dip at the bottom of each rep
  });

  it("records torso lean as unmeasurable front-on", () => {
    const { frames, fps } = buildSquatSeries({ view: "front", torsoLeanDeg: 20 });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);
    expect(angles.torsoLeanDeg.every((v) => v === null)).toBe(true);
  });

  it("measures torso lean side-on", () => {
    const { frames, fps } = buildSquatSeries({
      view: "side",
      torsoLeanDeg: 25,
      farSideVisibility: 0.4,
    });
    const angles = computeAngles(frames, fps, DEFAULT_CONFIG);
    const leans = present(angles.torsoLeanDeg);
    expect(leans.length).toBeGreaterThan(0);
    expect(Math.max(...leans)).toBeGreaterThan(10);
  });
});
