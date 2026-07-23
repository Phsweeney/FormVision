/**
 * Camera view detection — the TS mirror of `view.py`.
 *
 * Decides whether the camera is side-on or front-on so the engine can return
 * null for what that angle physically cannot see (lean front-on, asymmetry
 * side-on) instead of a confident wrong number. The signal is shoulder
 * separation divided by torso length: scale-free, and an order of magnitude
 * apart between the two views (≈0.06 side-on, ≥0.40 front-on).
 */

import type { AnalysisConfig } from "./config";
import { median } from "./stats";
import {
  CORE_LANDMARKS,
  PoseLandmarkIndex,
  landmarkAt,
  type FramePose,
  type ViewOrientation,
} from "./types";

/** A torso this short in normalised units is division noise, not a subject. */
const MIN_TORSO = 1e-6;

/**
 * Median shoulder separation in torso lengths across the frames, or null if the
 * subject was never tracked well enough. Median (not mean) because the torso
 * foreshortens at the bottom of a rep and one bad frame should not decide the view.
 */
export function shoulderRatio(
  frames: FramePose[],
  visibilityThreshold: number,
): number | null {
  const ratios: number[] = [];

  for (const frame of frames) {
    const core = CORE_LANDMARKS.map((i) => landmarkAt(frame, i));
    if (core.some((l) => l === null || l.visibility < visibilityThreshold)) {
      continue;
    }
    const ls = landmarkAt(frame, PoseLandmarkIndex.LEFT_SHOULDER)!;
    const rs = landmarkAt(frame, PoseLandmarkIndex.RIGHT_SHOULDER)!;
    const lh = landmarkAt(frame, PoseLandmarkIndex.LEFT_HIP)!;
    const rh = landmarkAt(frame, PoseLandmarkIndex.RIGHT_HIP)!;

    const shoulderMidX = (ls.x + rs.x) / 2;
    const shoulderMidY = (ls.y + rs.y) / 2;
    const hipMidX = (lh.x + rh.x) / 2;
    const hipMidY = (lh.y + rh.y) / 2;

    const torso = Math.hypot(shoulderMidX - hipMidX, shoulderMidY - hipMidY);
    if (torso < MIN_TORSO) continue;

    ratios.push(Math.abs(ls.x - rs.x) / torso);
  }

  return ratios.length ? median(ratios) : null;
}

/** Classify the camera angle from the shoulder-separation ratio. */
export function detectView(
  frames: FramePose[],
  config: AnalysisConfig,
): ViewOrientation {
  const ratio = shoulderRatio(frames, config.landmark_visibility_threshold);
  if (ratio === null) return "unknown";
  if (ratio <= config.view_side_max_shoulder_ratio) return "side";
  if (ratio >= config.view_front_min_shoulder_ratio) return "front";
  return "oblique";
}
