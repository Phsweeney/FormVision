/**
 * Framework-free analysis types — the browser mirror of the backend's
 * `analysis/types.py`.
 *
 * These are the contract the live engine is built on: pose in, angles and reps
 * out, with nothing from MediaPipe, React, or the DOM leaking in. Keeping them
 * pure is what lets the whole engine be unit-tested in Node exactly like the
 * Python side, by building a `FramePose[]` by hand.
 *
 * Note this is distinct from `@/lib/types.ts`, which mirrors the REST API
 * (snake_case, one shape per response). The live path never touches that API,
 * so it has its own internal, camelCase vocabulary.
 */

/** One body point. `x`/`y` are normalised [0,1]; `y` grows downward. */
export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

/** Every landmark for a single video frame, plus whether a body was found. */
export interface FramePose {
  frameIndex: number;
  timestampS: number;
  landmarks: Landmark[];
  detected: boolean;
}

/** Indices into the 33-point MediaPipe Pose topology (only the named points). */
export const PoseLandmarkIndex = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

/**
 * Landmarks required in every usable frame, whatever the camera angle. The
 * shoulders and hips survive occlusion even side-on, so they anchor scale and
 * view detection.
 */
export const CORE_LANDMARKS: readonly number[] = [
  PoseLandmarkIndex.LEFT_SHOULDER,
  PoseLandmarkIndex.RIGHT_SHOULDER,
  PoseLandmarkIndex.LEFT_HIP,
  PoseLandmarkIndex.RIGHT_HIP,
];

/** The legs, per side. A usable frame needs *one* of these complete, not both. */
export const LEFT_LEG_LANDMARKS: readonly number[] = [
  PoseLandmarkIndex.LEFT_KNEE,
  PoseLandmarkIndex.LEFT_ANKLE,
];
export const RIGHT_LEG_LANDMARKS: readonly number[] = [
  PoseLandmarkIndex.RIGHT_KNEE,
  PoseLandmarkIndex.RIGHT_ANKLE,
];

/**
 * Pairs of landmark indices to connect when drawing a skeleton. Copied verbatim
 * from the backend, which declares them locally because MediaPipe 0.10.x no
 * longer exports `POSE_CONNECTIONS`.
 */
export const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  // Face
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  // Arms
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  // Torso
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
  // Legs
  [23, 25],
  [25, 27],
  [27, 29],
  [27, 31],
  [29, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [28, 32],
  [30, 32],
];

/** Where the camera stood relative to the lifter. Decides which signals mean anything. */
export type ViewOrientation = "side" | "front" | "oblique" | "unknown";

/** How a piece of coaching feedback should be presented. */
export type Severity = "good" | "info" | "warning" | "critical";

/** Read a landmark from a frame, or `null` if the frame has no usable body. */
export function landmarkAt(frame: FramePose, index: number): Landmark | null {
  if (!frame.detected || index >= frame.landmarks.length) return null;
  return frame.landmarks[index];
}

/** A per-frame signal: one value per frame, null where it could not be measured. */
export type Signal = (number | null)[];

/**
 * Per-frame joint angles and derived signals — the TS mirror of `AngleSeries`.
 * Every array is index-aligned with the frames it was built from.
 */
export interface AngleSeries {
  timestampsS: number[];
  leftKneeDeg: Signal;
  rightKneeDeg: Signal;
  hipDeg: Signal;
  torsoLeanDeg: Signal;
  hipHeight: Signal;
  hipKneeOffset: Signal;
  valid: boolean[];
  leftLegValid: boolean[];
  rightLegValid: boolean[];
  torsoScale: number | null;
  thighScale: number | null;
  view: ViewOrientation;
}

/** Fraction of frames that were usable — the tracking-quality signal. */
export function validFraction(angles: AngleSeries): number {
  if (angles.valid.length === 0) return 0;
  const usable = angles.valid.reduce((n, v) => n + (v ? 1 : 0), 0);
  return usable / angles.valid.length;
}

/** One measured repetition — the TS mirror of `Rep`. */
export interface Rep {
  index: number;
  startFrame: number;
  bottomFrame: number;
  endFrame: number;
  startTimeS: number;
  bottomTimeS: number;
  endTimeS: number;
  minKneeAngleDeg: number | null;
  minLeftKneeDeg: number | null;
  minRightKneeDeg: number | null;
  minHipAngleDeg: number | null;
  maxTorsoLeanDeg: number | null;
  kneeAsymmetryDeg: number | null;
  depthPercent: number | null;
  hipBelowKnee: boolean;
}

export function repDurationS(rep: Rep): number {
  return rep.endTimeS - rep.startTimeS;
}
/** Descent time: start to the bottom. */
export function repEccentricS(rep: Rep): number {
  return rep.bottomTimeS - rep.startTimeS;
}
/** Ascent time: the bottom to lockout. */
export function repConcentricS(rep: Rep): number {
  return rep.endTimeS - rep.bottomTimeS;
}

/** Workout-level summary — the TS mirror of `Metrics`. */
export interface Metrics {
  totalReps: number;
  videoDurationS: number;
  totalWorkoutTimeS: number;
  maxDepthPercent: number | null;
  avgDepthPercent: number | null;
  minKneeAngleDeg: number | null;
  avgRepDurationS: number | null;
  fastestRepS: number | null;
  slowestRepS: number | null;
  avgEccentricS: number | null;
  avgConcentricS: number | null;
  repsPerMinute: number | null;
  avgTorsoLeanDeg: number | null;
  maxTorsoLeanDeg: number | null;
  avgKneeAsymmetryDeg: number | null;
  depthConsistencyPercent: number | null;
  durationConsistencyS: number | null;
  trackingQuality: number;
  cameraView: ViewOrientation;
}
