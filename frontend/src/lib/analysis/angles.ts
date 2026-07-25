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
  horizontalOffsetFromLine,
  jointAngle,
  type Point,
} from "./geometry";
import { smoothSeries } from "./smoothing";
import { median } from "./stats";
import { detectView } from "./view";
import {
  CORE_LANDMARKS,
  LEFT_ANKLE_ANGLE_LANDMARKS,
  LEFT_LEG_LANDMARKS,
  PoseLandmarkIndex as LM,
  RIGHT_ANKLE_ANGLE_LANDMARKS,
  RIGHT_LEG_LANDMARKS,
  landmarkAt,
  type AngleSeries,
  type FramePose,
  type Landmark,
  type ViewOrientation,
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
  leftHipDeg: number | null;
  rightHipDeg: number | null;
  leftAnkleDeg: number | null;
  rightAnkleDeg: number | null;
  leftKneeLateral: number | null;
  rightKneeLateral: number | null;
}

/**
 * Which signals a given camera angle can actually see.
 *
 * The two new families split along the same axis, for the same reason. Ankle
 * travel is a sagittal movement and shares torso lean's gate. Knee valgus is a
 * frontal-plane movement and is the mirror image: side-on, the knee projects
 * onto its own hip-to-ankle line however far it collapses. Valgus takes the
 * stricter test because a wrongly-signed or flattened reading would accuse a
 * lifter of a fault they do not have, and silence is the cheaper error.
 */
export interface ViewGates {
  leanIsMeasurable: boolean;
  ankleIsMeasurable: boolean;
  valgusIsMeasurable: boolean;
}

/**
 * Every view-gated signal suppressed. Used before the camera angle is known,
 * which is the live analyzer's state during calibration: reporting a valgus
 * reading before we know whether the camera can see the frontal plane would be
 * guessing, and guessing is what the gates exist to prevent.
 */
export const NO_GATES: ViewGates = {
  leanIsMeasurable: false,
  ankleIsMeasurable: false,
  valgusIsMeasurable: false,
};

export function viewGates(view: ViewOrientation): ViewGates {
  const leanIsMeasurable = view !== "front";
  return {
    leanIsMeasurable,
    ankleIsMeasurable: leanIsMeasurable,
    valgusIsMeasurable: view === "front",
  };
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
  gates: ViewGates,
  threshold: number,
): RawFrameAngles {
  const usable = frameIsUsable(frame, threshold);
  const leftLeg = usable && groupVisible(frame, LEFT_LEG_LANDMARKS, threshold);
  const rightLeg = usable && groupVisible(frame, RIGHT_LEG_LANDMARKS, threshold);

  if (!usable) return emptyFrameAngles();

  const shoulderMid = pairPoint(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, threshold);
  const hipMid = pairPoint(frame, LM.LEFT_HIP, LM.RIGHT_HIP, threshold);
  const kneeMid = pairPoint(frame, LM.LEFT_KNEE, LM.RIGHT_KNEE, threshold);
  const ankleMid = pairPoint(frame, LM.LEFT_ANKLE, LM.RIGHT_ANKLE, threshold);

  const medialSign = gates.valgusIsMeasurable ? medialSignFor(frame, threshold) : null;
  const left = sideSignals(frame, LEFT_SIDE, {
    threshold,
    legVisible: leftLeg,
    torsoScale,
    medialSign,
    ankleIsMeasurable: gates.ankleIsMeasurable,
  });
  const right = sideSignals(frame, RIGHT_SIDE, {
    threshold,
    legVisible: rightLeg,
    torsoScale,
    medialSign,
    ankleIsMeasurable: gates.ankleIsMeasurable,
  });

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
      gates.leanIsMeasurable && hipMid && shoulderMid
        ? angleFromVertical(hipMid, shoulderMid)
        : null,
    hipHeight:
      torsoScale && ankleMid && hipMid ? (ankleMid[1] - hipMid[1]) / torsoScale : null,
    hipKneeOffset:
      thighScale && hipMid && kneeMid ? (hipMid[1] - kneeMid[1]) / thighScale : null,
    leftHipDeg: left.hipDeg,
    rightHipDeg: right.hipDeg,
    leftAnkleDeg: left.ankleDeg,
    rightAnkleDeg: right.ankleDeg,
    leftKneeLateral: left.kneeLateral,
    rightKneeLateral: right.kneeLateral,
  };
}

function emptyFrameAngles(): RawFrameAngles {
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
    leftHipDeg: null,
    rightHipDeg: null,
    leftAnkleDeg: null,
    rightAnkleDeg: null,
    leftKneeLateral: null,
    rightKneeLateral: null,
  };
}

/** One half of the body, so the per-side maths is written once. */
interface Side {
  shoulder: number;
  hip: number;
  knee: number;
  ankle: number;
  ankleGroup: readonly number[];
  /** Relates this side to `medialSignFor`, which is defined for the left. */
  medialOrientation: number;
}

const LEFT_SIDE: Side = {
  shoulder: LM.LEFT_SHOULDER,
  hip: LM.LEFT_HIP,
  knee: LM.LEFT_KNEE,
  ankle: LM.LEFT_ANKLE,
  ankleGroup: LEFT_ANKLE_ANGLE_LANDMARKS,
  medialOrientation: 1,
};

const RIGHT_SIDE: Side = {
  shoulder: LM.RIGHT_SHOULDER,
  hip: LM.RIGHT_HIP,
  knee: LM.RIGHT_KNEE,
  ankle: LM.RIGHT_ANKLE,
  ankleGroup: RIGHT_ANKLE_ANGLE_LANDMARKS,
  medialOrientation: -1,
};

/**
 * Which image direction counts as medial for the *left* leg, as +1 or -1.
 *
 * Derived per frame from where the two hips sit rather than assumed, because
 * MediaPipe labels landmarks anatomically: a lifter facing away from the camera
 * has their left hip on the left of the image, and one facing the camera has it
 * on the right. Hard-coding either would silently invert the valgus sign for
 * half of all clips.
 */
function medialSignFor(frame: FramePose, threshold: number): number | null {
  const leftHip = visible(frame, LM.LEFT_HIP, threshold);
  const rightHip = visible(frame, LM.RIGHT_HIP, threshold);
  if (!leftHip || !rightHip) return null;

  const separation = rightHip.x - leftHip.x;
  // Side-on the hips project onto each other and there is no left-right axis to
  // speak of. The caller gates this to front-on footage anyway; this guard is
  // what stops a near-zero separation from picking a sign at random.
  if (Math.abs(separation) < MIN_SCALE) return null;
  return separation > 0 ? 1 : -1;
}

/**
 * Hip angle, ankle angle, and medial knee offset for one side of one frame.
 *
 * All three are independently nullable. A frame can easily yield a hip angle but
 * no ankle angle, because the foot left the bottom of the shot, and that must
 * cost the ankle angle only.
 */
function sideSignals(
  frame: FramePose,
  side: Side,
  options: {
    threshold: number;
    legVisible: boolean;
    torsoScale: number | null;
    medialSign: number | null;
    ankleIsMeasurable: boolean;
  },
): { hipDeg: number | null; ankleDeg: number | null; kneeLateral: number | null } {
  const { threshold, legVisible, torsoScale, medialSign, ankleIsMeasurable } = options;

  const shoulder = visible(frame, side.shoulder, threshold);
  const hip = visible(frame, side.hip, threshold);
  const knee = visible(frame, side.knee, threshold);
  const ankle = visible(frame, side.ankle, threshold);

  const hipDeg =
    legVisible && shoulder && hip && knee ? jointAngle(shoulder, hip, knee) : null;

  let ankleDeg: number | null = null;
  if (ankleIsMeasurable && groupVisible(frame, side.ankleGroup, threshold)) {
    const foot = landmarkAt(frame, side.ankleGroup[side.ankleGroup.length - 1]);
    if (knee && ankle && foot) ankleDeg = jointAngle(knee, ankle, foot);
  }

  let kneeLateral: number | null = null;
  if (legVisible && medialSign !== null && torsoScale && hip && knee && ankle) {
    const offset = horizontalOffsetFromLine(
      [knee.x, knee.y],
      [hip.x, hip.y],
      [ankle.x, ankle.y],
    );
    if (offset !== null) {
      kneeLateral = (offset * medialSign * side.medialOrientation) / torsoScale;
    }
  }

  return { hipDeg, ankleDeg, kneeLateral };
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

  const series = emptyAngleSeries(torsoScale, thighScale, view);

  // Front-on, the torso hinges toward the lens and its lean barely projects, so
  // it is recorded as unmeasurable rather than a flattering ~1 degree.
  const gates = viewGates(view);

  for (const frame of frames) {
    series.timestampsS.push(frame.timestampS);
    const raw = computeRawFrameAngles(frame, torsoScale, thighScale, gates, threshold);
    appendFrameAngles(series, raw);
  }

  smoothInPlace(series, fps, config);
  return series;
}

/**
 * An `AngleSeries` with every signal empty.
 *
 * Exported because the live analyzer needs the same shape for its session
 * buffer, and a hand-written literal in two places is exactly how a new signal
 * ends up added to one and forgotten in the other.
 */
export function emptyAngleSeries(
  torsoScale: number | null = null,
  thighScale: number | null = null,
  view: ViewOrientation = "unknown",
): AngleSeries {
  return {
    timestampsS: [],
    leftKneeDeg: [],
    rightKneeDeg: [],
    hipDeg: [],
    torsoLeanDeg: [],
    hipHeight: [],
    hipKneeOffset: [],
    leftHipDeg: [],
    rightHipDeg: [],
    leftAnkleDeg: [],
    rightAnkleDeg: [],
    leftKneeLateral: [],
    rightKneeLateral: [],
    valid: [],
    leftLegValid: [],
    rightLegValid: [],
    torsoScale,
    thighScale,
    view,
  };
}

/** Push one frame's raw angles onto every parallel array of a series. */
export function appendFrameAngles(series: AngleSeries, raw: RawFrameAngles): void {
  series.valid.push(raw.usable);
  series.leftLegValid.push(raw.leftLeg);
  series.rightLegValid.push(raw.rightLeg);
  series.leftKneeDeg.push(raw.leftKneeDeg);
  series.rightKneeDeg.push(raw.rightKneeDeg);
  series.hipDeg.push(raw.hipDeg);
  series.torsoLeanDeg.push(raw.torsoLeanDeg);
  series.hipHeight.push(raw.hipHeight);
  series.hipKneeOffset.push(raw.hipKneeOffset);
  series.leftHipDeg.push(raw.leftHipDeg);
  series.rightHipDeg.push(raw.rightHipDeg);
  series.leftAnkleDeg.push(raw.leftAnkleDeg);
  series.rightAnkleDeg.push(raw.rightAnkleDeg);
  series.leftKneeLateral.push(raw.leftKneeLateral);
  series.rightKneeLateral.push(raw.rightKneeLateral);
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
  series.leftHipDeg = smooth(series.leftHipDeg);
  series.rightHipDeg = smooth(series.rightHipDeg);
  series.leftAnkleDeg = smooth(series.leftAnkleDeg);
  series.rightAnkleDeg = smooth(series.rightAnkleDeg);
  series.leftKneeLateral = smooth(series.leftKneeLateral);
  series.rightKneeLateral = smooth(series.rightKneeLateral);
}
