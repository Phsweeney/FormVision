/**
 * Pure geometric primitives — the TS mirror of `geometry.py`.
 *
 * Numbers in, numbers out. No config, no I/O. Coordinate convention matches the
 * backend and MediaPipe: `x` grows right, `y` grows *downward*, so a larger `y`
 * is lower in the frame. Functions that care about real-world "up" account for
 * it internally.
 */

import type { Landmark } from "./types";

/** Vectors shorter than this (normalised units) are treated as degenerate. */
const EPSILON = 1e-9;

export type Point = readonly [number, number];

export function midpoint(a: Landmark, b: Landmark): Point {
  return [(a.x + b.x) / 2, (a.y + b.y) / 2];
}

export function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/**
 * Interior angle at `vertex`, in degrees within [0, 180], or null if either arm
 * is degenerate. Uses atan2 of cross/dot (not acos), which keeps full precision
 * near 0 and 180 degrees — where a locked-out knee sits — and can never domain-error.
 */
export function angleBetweenPoints(
  first: Point,
  vertex: Point,
  second: Point,
): number | null {
  const ax = first[0] - vertex[0];
  const ay = first[1] - vertex[1];
  const bx = second[0] - vertex[0];
  const by = second[1] - vertex[1];

  if (Math.hypot(ax, ay) < EPSILON || Math.hypot(bx, by) < EPSILON) return null;

  const cross = ax * by - ay * bx;
  const dot = ax * bx + ay * by;
  return radiansToDegrees(Math.atan2(Math.abs(cross), dot));
}

/**
 * Interior angle at the `vertex` landmark. For a knee, call
 * `jointAngle(hip, knee, ankle)`: a straight leg is ~180, falling as it flexes.
 */
export function jointAngle(
  first: Landmark,
  vertex: Landmark,
  second: Landmark,
): number | null {
  return angleBetweenPoints(
    [first.x, first.y],
    [vertex.x, vertex.y],
    [second.x, second.y],
  );
}

/**
 * Tilt of the `lower -> upper` segment away from vertical, in degrees. 0 is
 * upright, rising to 90 at horizontal. Always non-negative: this is *how much*
 * lean, not which way. Note the y inversion — image y grows downward, so the
 * vertical component is `lower.y - upper.y`.
 */
export function angleFromVertical(lower: Point, upper: Point): number | null {
  const horizontal = upper[0] - lower[0];
  const vertical = lower[1] - upper[1];
  if (Math.hypot(horizontal, vertical) < EPSILON) return null;
  return radiansToDegrees(Math.atan2(Math.abs(horizontal), Math.abs(vertical)));
}

/**
 * Signed horizontal gap from `point` to the `start -> end` line, sampled at the
 * point's own height.
 *
 * Answers "how far sideways is the knee from where the hip-to-ankle line puts
 * it", which is what a coach means by a knee tracking inward. Positive means the
 * point lies to the right of the line in image coordinates (larger x); callers
 * decide what that means anatomically.
 *
 * Returns null when `start` and `end` share a height, since there is then no
 * line to interpolate along. That is not pathological — it is a frame cropped
 * mid-shin, or a badly tracked ankle.
 */
export function horizontalOffsetFromLine(
  point: Point,
  start: Point,
  end: Point,
): number | null {
  const verticalSpan = end[1] - start[1];
  if (Math.abs(verticalSpan) < EPSILON) return null;

  // Deliberately not clamped to [0, 1]: a knee above the hip or below the ankle
  // is a tracking failure, and the visibility gates upstream will already have
  // suppressed it.
  const ratio = (point[1] - start[1]) / verticalSpan;
  const expectedX = start[0] + ratio * (end[0] - start[0]);
  return point[0] - expectedX;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

/**
 * Map `value` onto [0, 1] where `fromValue` is 0 and `toValue` is 1. Works in
 * either direction, which matters for depth: knee angles *decrease* as the squat
 * deepens (170 degrees -> 0%, 90 degrees -> 100%).
 */
export function linearScale(
  value: number,
  fromValue: number,
  toValue: number,
  clampResult = true,
): number {
  const span = toValue - fromValue;
  if (Math.abs(span) < EPSILON) return 0;
  const result = (value - fromValue) / span;
  return clampResult ? clamp(result, 0, 1) : result;
}

function radiansToDegrees(radians: number): number {
  return (radians * 180) / Math.PI;
}
