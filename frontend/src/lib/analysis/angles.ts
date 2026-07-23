/**
 * Landmarks -> joint angles and derived signals — the TS mirror of `angles.py`.
 *
 * The central idea is scale normalisation: every distance is divided by the
 * subject's own torso length (median over the frames), so depth thresholds mean
 * the same thing whatever the camera distance or body size. Torso is the
 * reference because it is the one segment whose projected length barely changes
 * during a squat.
 *
 * This is the batch form, over a whole frame array — used by the parity tests
 * and as the reference the live driver's per-frame path is checked against.
 */

import type { AnalysisConfig } from "./config";
import {
  angleBetweenPoints,
  angleFromVertical,
  jointAngle,
  type Point,
} from "./geometry";
import { smoothSeries } from "./smoothing";
import { median } from "./stats";
import { detectView } from "./view";
import {
  CORE_LANDMARKS,
  LEFT_LEG_LANDMARKS,
  PoseLandmarkIndex as LM,
  RIGHT_LEG_LANDMARKS,
  landmarkAt,
  type AngleSeries,
  type FramePose,
  type Landmark,
} from "./types";

const MIN_SCALE = 1e-6;

/** Raw (un-smoothed) per-frame angles and validity, shared by batch and live. */
export interface RawFrameAngles {
  usable: boolean;
  leftLeg: boolean;
  rightLeg: boolean;
  leftKneeDeg: number | null;
  rightKneeDeg: number | null;
  hipDeg: number | null;
  torsoLeanDeg: number | null;
  hipHeight: number | null;
  hipKneeOffset: number | null;
}

/**
 * Compute one frame's raw angles given already-known body scale and view.
 *
 * The batch pipeline calls this in a loop then smooths the arrays; the live
 * driver calls it per frame (with scale and view fixed at calibration) and
 * smooths causally. Extracting it keeps the trig in exactly one place.
 */
export function computeRawFrameAngles(
  frame: FramePose,
  torsoScale: number | null,
  thighScale: number | null,
  leanIsMeasurable: boolean,
  threshold: number,
): RawFrameAngles {
  const usable = frameIsUsable(frame, threshold);
  const leftLeg = usable && groupVisible(frame, LEFT_LEG_LANDMARKS, threshold);
  const rightLeg = usable && groupVisible(frame, RIGHT_LEG_LANDMARKS, threshold);

  if (!usable) {
    return {
      usable: false,
      leftLeg: false,
      rightLeg: false,
      leftKneeDeg: null,
      rightKneeDeg: null,
      hipDeg: null,
      torsoLeanDeg: null,
      hipHeight: null,
      hipKneeOffset: null,
    };
  }

  const shoulderMid = pairPoint(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, threshold);
  const hipMid = pairPoint(frame, LM.LEFT_HIP, LM.RIGHT_HIP, threshold);
  const kneeMid = pairPoint(frame, LM.LEFT_KNEE, LM.RIGHT_KNEE, threshold);
  const ankleMid = pairPoint(frame, LM.LEFT_ANKLE, LM.RIGHT_ANKLE, threshold);

  return {
    usable: true,
    leftLeg,
    rightLeg,
    leftKneeDeg: leftLeg
      ? jointAngle(
          landmarkAt(frame, LM.LEFT_HIP)!,
          landmarkAt(frame, LM.LEFT_KNEE)!,
          landmarkAt(frame, LM.LEFT_ANKLE)!,
        )
      : null,
    rightKneeDeg: rightLeg
      ? jointAngle(
          landmarkAt(frame, LM.RIGHT_HIP)!,
          landmarkAt(frame, LM.RIGHT_KNEE)!,
          landmarkAt(frame, LM.RIGHT_ANKLE)!,
        )
      : null,
    hipDeg:
      shoulderMid && hipMid && kneeMid
        ? angleBetweenPoints(shoulderMid, hipMid, kneeMid)
        : null,
    torsoLeanDeg:
      leanIsMeasurable && hipMid && shoulderMid
        ? angleFromVertical(hipMid, shoulderMid)
        : null,
    hipHeight:
      torsoScale && ankleMid && hipMid ? (ankleMid[1] - hipMid[1]) / torsoScale : null,
    hipKneeOffset:
      thighScale && hipMid && kneeMid ? (hipMid[1] - kneeMid[1]) / thighScale : null,
  };
}

function visible(
  frame: FramePose,
  index: number,
  threshold: number,
): Landmark | null {
  const landmark = landmarkAt(frame, index);
  if (landmark === null || landmark.visibility < threshold) return null;
  return landmark;
}

function groupVisible(
  frame: FramePose,
  group: readonly number[],
  threshold: number,
): boolean {
  return group.every((index) => visible(frame, index, threshold) !== null);
}

/** Core torso points, plus at least one complete leg (not both — side-on hides one). */
function frameIsUsable(frame: FramePose, threshold: number): boolean {
  return (
    groupVisible(frame, CORE_LANDMARKS, threshold) &&
    (groupVisible(frame, LEFT_LEG_LANDMARKS, threshold) ||
      groupVisible(frame, RIGHT_LEG_LANDMARKS, threshold))
  );
}

/** Midpoint of a left/right pair, or the one visible side (they coincide side-on). */
function pairPoint(
  frame: FramePose,
  left: number,
  right: number,
  threshold: number,
): Point | null {
  const l = visible(frame, left, threshold);
  const r = visible(frame, right, threshold);
  if (l && r) return [(l.x + r.x) / 2, (l.y + r.y) / 2];
  if (l) return [l.x, l.y];
  if (r) return [r.x, r.y];
  return null;
}

function computeScales(
  frames: FramePose[],
  threshold: number,
): { torso: number | null; thigh: number | null } {
  const torsoLengths: number[] = [];
  const thighLengths: number[] = [];

  for (const frame of frames) {
    if (!frameIsUsable(frame, threshold)) continue;
    const shoulderMid = pairPoint(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, threshold);
    const hipMid = pairPoint(frame, LM.LEFT_HIP, LM.RIGHT_HIP, threshold);
    const kneeMid = pairPoint(frame, LM.LEFT_KNEE, LM.RIGHT_KNEE, threshold);
    if (!shoulderMid || !hipMid || !kneeMid) continue;
    torsoLengths.push(Math.hypot(shoulderMid[0] - hipMid[0], shoulderMid[1] - hipMid[1]));
    thighLengths.push(Math.hypot(hipMid[0] - kneeMid[0], hipMid[1] - kneeMid[1]));
  }

  let torso = median(torsoLengths);
  let thigh = median(thighLengths);
  if (torso !== null && torso < MIN_SCALE) torso = null;
  if (thigh !== null && thigh < MIN_SCALE) thigh = null;
  return { torso, thigh };
}

/**
 * Build the full `AngleSeries` for a clip. Raw values are computed per frame,
 * then every signal is gap-filled and smoothed with one shared window so they
 * stay phase-aligned.
 */
export function computeAngles(
  frames: FramePose[],
  fps: number,
  config: AnalysisConfig,
): AngleSeries {
  const threshold = config.landmark_visibility_threshold;
  const { torso: torsoScale, thigh: thighScale } = computeScales(frames, threshold);
  const view = detectView(frames, config);

  const series: AngleSeries = {
    timestampsS: [],
    leftKneeDeg: [],
    rightKneeDeg: [],
    hipDeg: [],
    torsoLeanDeg: [],
    hipHeight: [],
    hipKneeOffset: [],
    valid: [],
    leftLegValid: [],
    rightLegValid: [],
    torsoScale,
    thighScale,
    view,
  };

  // Front-on, the torso hinges toward the lens and its lean barely projects, so
  // it is recorded as unmeasurable rather than a flattering ~1 degree.
  const leanIsMeasurable = view !== "front";

  for (const frame of frames) {
    series.timestampsS.push(frame.timestampS);
    const raw = computeRawFrameAngles(
      frame,
      torsoScale,
      thighScale,
      leanIsMeasurable,
      threshold,
    );
    series.valid.push(raw.usable);
    series.leftLegValid.push(raw.leftLeg);
    series.rightLegValid.push(raw.rightLeg);
    series.leftKneeDeg.push(raw.leftKneeDeg);
    series.rightKneeDeg.push(raw.rightKneeDeg);
    series.hipDeg.push(raw.hipDeg);
    series.torsoLeanDeg.push(raw.torsoLeanDeg);
    series.hipHeight.push(raw.hipHeight);
    series.hipKneeOffset.push(raw.hipKneeOffset);
  }

  smoothInPlace(series, fps, config);
  return series;
}

function smoothInPlace(
  series: AngleSeries,
  fps: number,
  config: AnalysisConfig,
): void {
  const seconds = config.smoothing_window_seconds;
  const maxGap = config.max_interpolation_gap_frames;
  const smooth = (values: (number | null)[]) => smoothSeries(values, fps, seconds, maxGap);
  series.leftKneeDeg = smooth(series.leftKneeDeg);
  series.rightKneeDeg = smooth(series.rightKneeDeg);
  series.hipDeg = smooth(series.hipDeg);
  series.torsoLeanDeg = smooth(series.torsoLeanDeg);
  series.hipHeight = smooth(series.hipHeight);
  series.hipKneeOffset = smooth(series.hipKneeOffset);
}
