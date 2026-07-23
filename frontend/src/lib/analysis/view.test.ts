import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "./config";
import { buildSquatSeries } from "./synthetic";
import { detectView } from "./view";

describe("detectView", () => {
  it("classifies front-on and side-on clips", () => {
    const front = buildSquatSeries({ view: "front" }).frames;
    const side = buildSquatSeries({ view: "side" }).frames;
    expect(detectView(front, DEFAULT_CONFIG)).toBe("front");
    expect(detectView(side, DEFAULT_CONFIG)).toBe("side");
  });

  it("reports unknown when nobody is tracked", () => {
    expect(detectView([], DEFAULT_CONFIG)).toBe("unknown");
  });
});
