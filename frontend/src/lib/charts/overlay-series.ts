/**
 * Geometry for the mini charts drawn over the analysis video.
 *
 * Pure functions over plain numbers: no canvas, no DOM, no React. The drawing
 * itself lives in `components/dashboard/video-overlay-charts.tsx`, which is the
 * part that cannot be tested in this project's Node test environment. Splitting
 * them the same way `skeleton.ts` and `pose-runner.ts` are split keeps the
 * arithmetic under test and quarantines the runtime.
 *
 * **The domains here are copies of the ones in `analysis-charts.tsx`, and that
 * is the point.** The overlay and the full "Movement over time" section plot the
 * same numbers, so a reading taken off one has to match the other. Different
 * scales would make them quietly disagree at a glance, which is worse than not
 * having the overlay at all.
 */

import { SERIES_COLORS } from "@/lib/chart-theme";
import type { Series } from "@/lib/types";

/** A y-domain, matching the shape `TimeSeriesChart` takes. */
export type Domain = [number | "auto", number | "auto"];

export interface OverlayTrack {
  /** Keys into `Series`, in draw order. Two for the knees, one for the rest. */
  keys: (keyof Series)[];
  colors: string[];
  label: string;
  unit: string;
  domain: Domain;
  /** Decimal places for the printed current value. */
  digits: number;
}

/**
 * The three tracks, in the order they stack down the right edge.
 *
 * Deliberately the same three, in the same order, with the same colours and
 * domains as the section below the video.
 */
export const OVERLAY_TRACKS: OverlayTrack[] = [
  {
    keys: ["left_knee_deg", "right_knee_deg"],
    colors: [SERIES_COLORS.leftKnee, SERIES_COLORS.rightKnee],
    label: "KNEE",
    unit: "°",
    domain: [40, 190],
    digits: 0,
  },
  {
    keys: ["hip_deg"],
    colors: [SERIES_COLORS.hip],
    label: "HIP",
    unit: "°",
    domain: [0, 190],
    digits: 0,
  },
  {
    keys: ["hip_height"],
    colors: [SERIES_COLORS.hipHeight],
    label: "HIP HEIGHT",
    unit: "",
    domain: ["auto", "auto"],
    digits: 2,
  },
];

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

/** Fraction of the observed range added above and below an auto domain. */
const AUTO_PAD = 0.08;

/**
 * Turn a possibly-auto domain into concrete numbers.
 *
 * `series` may hold several arrays (the knee track has two), and an auto domain
 * has to span all of them or one line would clip. A flat or empty signal gets an
 * arbitrary unit range rather than a zero-height one, which would divide by zero
 * when projecting.
 */
export function resolveDomain(
  series: (number | null)[][],
  domain: Domain,
): [number, number] {
  const present = series.flat().filter((v): v is number => v !== null);

  const autoLow = present.length > 0 ? Math.min(...present) : 0;
  const autoHigh = present.length > 0 ? Math.max(...present) : 1;
  const pad = (autoHigh - autoLow) * AUTO_PAD;

  const low = domain[0] === "auto" ? autoLow - pad : domain[0];
  const high = domain[1] === "auto" ? autoHigh + pad : domain[1];

  if (high - low < 1e-9) return [low - 0.5, low + 0.5];
  return [low, high];
}

/**
 * Project one signal into pixel space, once.
 *
 * Returns a point per sample, or `null` where the value was never measured.
 * Nulls have to survive as gaps rather than being dropped: joining across one
 * would draw a line through data that was never observed, which is exactly what
 * `connectNulls={false}` prevents on the real charts.
 *
 * Computed once per canvas size rather than per frame, so drawing a frame is
 * walking a prefix of this array.
 */
export function projectTrack(
  values: (number | null)[],
  timeS: number[],
  domain: [number, number],
  box: Box,
): (Point | null)[] {
  const [low, high] = domain;
  const span = high - low;
  const first = timeS[0] ?? 0;
  const last = timeS[timeS.length - 1] ?? first;
  const duration = last - first;

  return values.map((value, index) => {
    if (value === null) return null;

    const time = timeS[index] ?? first;
    const tx = duration > 0 ? (time - first) / duration : 0;
    // Clamped so a value outside a fixed domain sits on the edge rather than
    // being drawn outside the panel it belongs to.
    const ty = clamp((value - low) / span, 0, 1);

    return {
      x: box.x + tx * box.width,
      // Canvas y grows downward, so the high end of the domain is the top.
      y: box.y + (1 - ty) * box.height,
    };
  });
}

/**
 * Index of the last sample at or before `currentTime`.
 *
 * Binary search rather than a scan because this runs on every animation frame.
 * Returns -1 before the first sample, so a caller can draw nothing rather than
 * a single stray point at time zero.
 */
export function indexForTime(timeS: number[], currentTime: number): number {
  if (timeS.length === 0 || currentTime < timeS[0]) return -1;

  let low = 0;
  let high = timeS.length - 1;
  while (low < high) {
    // Upper midpoint, so the loop always advances and settles on the last
    // qualifying sample rather than spinning on the lower one.
    const mid = Math.ceil((low + high) / 2);
    if (timeS[mid] <= currentTime) low = mid;
    else high = mid - 1;
  }
  return low;
}

/**
 * Stack `count` equal panels down the right-hand edge of a container.
 *
 * Sized from the shorter dimension so the panels stay legible whether the clip
 * is portrait or landscape, and capped so they never dominate a wide frame.
 */
export function stackPanels(
  containerWidth: number,
  containerHeight: number,
  count: number,
): Box[] {
  const shortest = Math.min(containerWidth, containerHeight);
  const margin = Math.round(shortest * 0.03);
  const width = Math.round(Math.min(containerWidth * 0.3, shortest * 0.42));
  const gap = Math.round(shortest * 0.02);
  const available = containerHeight - margin * 2 - gap * (count - 1);
  const height = Math.max(1, Math.floor(available / count / 2.1));

  return Array.from({ length: count }, (_, index) => ({
    x: containerWidth - margin - width,
    y: margin + index * (height + gap),
    width,
    height,
  }));
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}
