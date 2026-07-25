import { describe, expect, it } from "vitest";

import {
  OVERLAY_TRACKS,
  indexForTime,
  projectTrack,
  resolveDomain,
  stackPanels,
  type Box,
} from "./overlay-series";

const BOX: Box = { x: 0, y: 0, width: 100, height: 50 };

describe("resolveDomain", () => {
  it("passes a fixed domain through untouched", () => {
    // Fixed domains are copied from the full charts so a value read off the
    // overlay matches the section below. Padding them would break that.
    expect(resolveDomain([[0, 500]], [40, 190])).toEqual([40, 190]);
  });

  it("spans every signal in the track for an auto domain", () => {
    // The knee track carries two lines; a domain fitted to one would clip the
    // other.
    const [low, high] = resolveDomain(
      [
        [1, 2],
        [0, 9],
      ],
      ["auto", "auto"],
    );
    expect(low).toBeLessThan(0);
    expect(high).toBeGreaterThan(9);
  });

  it("ignores missing values when fitting", () => {
    const [low, high] = resolveDomain([[null, 4, null, 6]], ["auto", "auto"]);
    expect(low).toBeLessThan(4);
    expect(high).toBeGreaterThan(6);
  });

  it("gives a flat signal a usable range", () => {
    // A zero-height domain divides by zero when projecting.
    const [low, high] = resolveDomain([[3, 3, 3]], ["auto", "auto"]);
    expect(high).toBeGreaterThan(low);
  });

  it("survives a track with nothing measured at all", () => {
    const [low, high] = resolveDomain([[null, null]], ["auto", "auto"]);
    expect(high).toBeGreaterThan(low);
  });

  it("mixes a fixed end with an auto one", () => {
    const [low, high] = resolveDomain([[5, 25]], [0, "auto"]);
    expect(low).toBe(0);
    expect(high).toBeGreaterThan(25);
  });
});

describe("projectTrack", () => {
  const timeS = [0, 1, 2, 3];

  it("maps time across the box width and value up its height", () => {
    const points = projectTrack([0, 10, 20, 30], timeS, [0, 30], BOX);

    expect(points[0]).toEqual({ x: 0, y: 50 });
    expect(points[3]).toEqual({ x: 100, y: 0 });
  });

  it("puts the top of the domain at the top of the box", () => {
    // Canvas y grows downward, so this is the flip that is easy to get wrong.
    const points = projectTrack([30], [0], [0, 30], BOX);
    expect(points[0]!.y).toBe(0);
  });

  it("keeps a gap where the value was never measured", () => {
    // Joining across would draw a line through data that was never observed,
    // which is what `connectNulls={false}` prevents on the real charts.
    const points = projectTrack([1, null, 3, 4], timeS, [0, 10], BOX);
    expect(points[1]).toBeNull();
    expect(points[0]).not.toBeNull();
    expect(points[2]).not.toBeNull();
  });

  it("clamps a value outside a fixed domain to the edge", () => {
    const points = projectTrack([-100, 999], [0, 1], [0, 10], BOX);
    expect(points[0]!.y).toBe(50);
    expect(points[1]!.y).toBe(0);
  });

  it("keeps every point inside the box", () => {
    const box: Box = { x: 12, y: 7, width: 80, height: 40 };
    const points = projectTrack([-5, 3, 99, 7], timeS, [0, 10], box);

    for (const point of points) {
      if (point === null) continue;
      expect(point.x).toBeGreaterThanOrEqual(box.x);
      expect(point.x).toBeLessThanOrEqual(box.x + box.width);
      expect(point.y).toBeGreaterThanOrEqual(box.y);
      expect(point.y).toBeLessThanOrEqual(box.y + box.height);
    }
  });

  it("does not divide by zero on a single-sample clip", () => {
    const points = projectTrack([5], [2], [0, 10], BOX);
    expect(Number.isFinite(points[0]!.x)).toBe(true);
  });
});

describe("indexForTime", () => {
  const timeS = [0, 0.5, 1, 1.5, 2];

  it("lands on the last sample at or before the time", () => {
    expect(indexForTime(timeS, 0)).toBe(0);
    expect(indexForTime(timeS, 1)).toBe(2);
    expect(indexForTime(timeS, 1.4)).toBe(2);
    expect(indexForTime(timeS, 1.5)).toBe(3);
  });

  it("returns -1 before the clip starts", () => {
    // So the caller draws nothing rather than a stray point at the origin.
    expect(indexForTime(timeS, -1)).toBe(-1);
  });

  it("clamps past the end", () => {
    expect(indexForTime(timeS, 99)).toBe(4);
  });

  it("handles an empty series", () => {
    expect(indexForTime([], 1)).toBe(-1);
  });

  it("agrees with a linear scan across the whole clip", () => {
    // The binary search is the part most likely to be subtly off by one, and
    // the failure would be a trace lagging playback by a frame rather than
    // anything visible in a unit assertion.
    const dense = Array.from({ length: 600 }, (_, i) => i * 0.033);
    for (const t of [0, 0.01, 0.033, 5, 9.9, 19.767, 100]) {
      let expected = -1;
      for (let i = 0; i < dense.length; i++) if (dense[i] <= t) expected = i;
      expect(indexForTime(dense, t)).toBe(expected);
    }
  });
});

describe("stackPanels", () => {
  it("returns one box per track, stacked without overlapping", () => {
    const boxes = stackPanels(1000, 600, 3);
    expect(boxes).toHaveLength(3);
    for (let i = 1; i < boxes.length; i++) {
      expect(boxes[i].y).toBeGreaterThanOrEqual(boxes[i - 1].y + boxes[i - 1].height);
    }
  });

  it("pins them to the right edge with a margin", () => {
    const boxes = stackPanels(1000, 600, 3);
    for (const box of boxes) {
      expect(box.x + box.width).toBeLessThan(1000);
      expect(box.x).toBeGreaterThan(500);
    }
  });

  it("stays inside the container in portrait and landscape", () => {
    for (const [w, h] of [
      [1000, 600],
      [400, 900],
      [320, 240],
    ]) {
      for (const box of stackPanels(w, h, 3)) {
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.y).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(w);
        expect(box.y + box.height).toBeLessThanOrEqual(h);
      }
    }
  });

  it("never produces a zero-height panel", () => {
    // A tiny container must still yield something drawable rather than a
    // degenerate box that divides by zero downstream.
    for (const box of stackPanels(120, 80, 3)) {
      expect(box.height).toBeGreaterThan(0);
      expect(box.width).toBeGreaterThan(0);
    }
  });
});

describe("OVERLAY_TRACKS", () => {
  it("matches the domains used by the full charts", () => {
    // If these drift, a value read off the overlay stops matching the same
    // value in "Movement over time", which is the one thing this must not do.
    expect(OVERLAY_TRACKS[0].domain).toEqual([40, 190]);
    expect(OVERLAY_TRACKS[1].domain).toEqual([0, 190]);
    expect(OVERLAY_TRACKS[2].domain).toEqual(["auto", "auto"]);
  });

  it("gives every signal its own colour", () => {
    for (const track of OVERLAY_TRACKS) {
      expect(track.colors).toHaveLength(track.keys.length);
    }
    const all = OVERLAY_TRACKS.flatMap((t) => t.colors);
    expect(new Set(all).size).toBe(all.length);
  });
});
