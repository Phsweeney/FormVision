/**
 * Synthetic pose data for tests — a faithful TS port of `tests/synthetic.py`.
 *
 * Builds a stick figure whose hips and knees follow a controlled trajectory, so
 * a test can assert the engine recovers exactly what was put in: the knee angles
 * are real consequences of the geometry, not values written by hand. This is the
 * parity harness — the same fixtures the Python suite uses, so a divergence
 * between the two engines shows up as a failing assertion.
 *
 * Not imported by any app code; test-only.
 */

import {
  PoseLandmarkIndex as LM,
  type FramePose,
  type Landmark,
  type ViewOrientation,
} from "./types";

const THIGH = 0.18;
const SHIN = 0.18;
const TORSO = 0.22;
const SHOULDER_WIDTH = 0.09;
const HIP_WIDTH = 0.07;
const ANKLE_Y = 0.92;
const CENTER_X = 0.5;
const LANDMARK_COUNT = 33;

const LATERAL_SCALE: Record<ViewOrientation, number> = {
  front: 1.0,
  oblique: 0.65,
  side: 0.05,
  unknown: 1.0,
};

const FAR_SIDE_LANDMARKS = new Set<number>([
  LM.LEFT_KNEE,
  LM.LEFT_ANKLE,
  LM.LEFT_HEEL,
  LM.LEFT_FOOT_INDEX,
]);

export interface SyntheticSeries {
  frames: FramePose[];
  fps: number;
  durationS: number;
}

/** Place the knee by solving the two-link leg between hip and ankle. */
function legGeometry(
  hip: [number, number],
  ankle: [number, number],
  kneeForward: number,
): [number, number] {
  const [hipX, hipY] = hip;
  const [ankleX, ankleY] = ankle;
  const dx = ankleX - hipX;
  const dy = ankleY - hipY;
  const span = Math.hypot(dx, dy);

  if (span < 1e-9 || span >= THIGH + SHIN) {
    const scale = span > 1e-9 ? THIGH / span : 0;
    return [hipX + dx * scale + kneeForward * 0.02, hipY + dy * scale];
  }

  const along = (span * span + THIGH * THIGH - SHIN * SHIN) / (2 * span);
  const perpendicular = Math.sqrt(Math.max(THIGH * THIGH - along * along, 0));
  const ux = dx / span;
  const uy = dy / span;
  const baseX = hipX + ux * along;
  const baseY = hipY + uy * along;
  const sign = kneeForward >= 0 ? 1 : -1;
  const perpX = uy * sign;
  const perpY = -ux * sign;
  const magnitude = perpendicular * Math.min(Math.abs(kneeForward), 1);
  return [baseX + perpX * magnitude, baseY + perpY * magnitude];
}

interface FrameOptions {
  torsoLeanDeg?: number;
  kneeForward?: number;
  leftRightBias?: number;
  hipSetback?: number;
  /** Medial knee travel at full depth, as a fraction of a thigh length. */
  kneeValgus?: number;
  /** How much of `kneeValgus` applies this frame; normally the frame's depth. */
  valgusDepthScale?: number;
  visibility?: number;
  detected?: boolean;
  view?: ViewOrientation;
  farSideVisibility?: number | null;
}

export function buildFrame(
  frameIndex: number,
  timestampS: number,
  hipY: number,
  options: FrameOptions = {},
): FramePose {
  const {
    torsoLeanDeg = 10,
    kneeForward = 1,
    leftRightBias = 0,
    hipSetback = 0,
    kneeValgus = 0,
    valgusDepthScale = 1,
    visibility = 0.95,
    detected = true,
    view = "front",
    farSideVisibility = null,
  } = options;

  if (!detected) {
    return { frameIndex, timestampS, landmarks: [], detected: false };
  }

  const lateral = LATERAL_SCALE[view];
  const shoulderWidth = SHOULDER_WIDTH * lateral;
  const hipWidth = HIP_WIDTH * lateral;

  const confidence = (index: number): number =>
    farSideVisibility !== null && FAR_SIDE_LANDMARKS.has(index)
      ? farSideVisibility
      : visibility;

  const points: Landmark[] = Array.from({ length: LANDMARK_COUNT }, () => ({
    x: CENTER_X,
    y: 0.5,
    z: 0,
    visibility,
  }));
  const place = (index: number, x: number, y: number): void => {
    points[index] = { x, y, z: 0, visibility: confidence(index) };
  };

  const hipX = CENTER_X - hipSetback;
  const leanRad = (torsoLeanDeg * Math.PI) / 180;
  const shoulderX = hipX + TORSO * Math.sin(leanRad);
  const shoulderY = hipY - TORSO * Math.cos(leanRad);

  place(LM.NOSE, shoulderX, shoulderY - 0.08);
  place(LM.LEFT_SHOULDER, shoulderX - shoulderWidth / 2, shoulderY);
  place(LM.RIGHT_SHOULDER, shoulderX + shoulderWidth / 2, shoulderY);
  place(LM.LEFT_ELBOW, shoulderX - shoulderWidth, shoulderY + 0.09);
  place(LM.RIGHT_ELBOW, shoulderX + shoulderWidth, shoulderY + 0.09);
  place(LM.LEFT_WRIST, shoulderX - shoulderWidth, shoulderY + 0.18);
  place(LM.RIGHT_WRIST, shoulderX + shoulderWidth, shoulderY + 0.18);

  place(LM.LEFT_HIP, hipX - hipWidth / 2, hipY);
  place(LM.RIGHT_HIP, hipX + hipWidth / 2, hipY);

  const legs: Array<[number, number, number, number]> = [
    [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, leftRightBias],
    [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, 0],
  ];
  for (const [sideHip, sideKnee, sideAnkle, bias] of legs) {
    const offset = sideHip === LM.LEFT_HIP ? -hipWidth / 2 : hipWidth / 2;
    const ankleX = CENTER_X + offset + bias * THIGH;
    const ankleY = ANKLE_Y;
    const [kneeX, kneeY] = legGeometry(
      [hipX + offset, hipY],
      [ankleX, ankleY],
      kneeForward,
    );
    // Medial is toward the midline: +x for the left leg, -x for the right.
    // Scaled by `lateral` so a side-on figure, whose two sides collapse onto
    // each other in the image, does not acquire a displacement the camera could
    // never have seen. Scaled by depth because real valgus appears under load
    // and disappears at lockout; nobody's knees cave while they are standing.
    const medial = sideHip === LM.LEFT_HIP ? 1 : -1;
    place(sideKnee, kneeX + medial * kneeValgus * valgusDepthScale * THIGH * lateral, kneeY);
    place(sideAnkle, ankleX, ankleY);
  }

  place(LM.LEFT_HEEL, CENTER_X - hipWidth / 2 - 0.01, ANKLE_Y + 0.02);
  place(LM.RIGHT_HEEL, CENTER_X + hipWidth / 2 - 0.01, ANKLE_Y + 0.02);
  place(LM.LEFT_FOOT_INDEX, CENTER_X - hipWidth / 2 + 0.05, ANKLE_Y + 0.03);
  place(LM.RIGHT_FOOT_INDEX, CENTER_X + hipWidth / 2 + 0.05, ANKLE_Y + 0.03);

  return { frameIndex, timestampS, landmarks: points, detected: true };
}

export interface SquatOptions {
  reps?: number;
  fps?: number;
  repDurationS?: number;
  standingPauseS?: number;
  depthFraction?: number;
  torsoLeanDeg?: number;
  bottomLeanDeg?: number | null;
  leftRightBias?: number;
  /** Medial knee travel at full depth, scaled through the rep by depth. */
  kneeValgus?: number;
  depthJitter?: number;
  undetectedFrames?: number[];
  view?: ViewOrientation;
  farSideVisibility?: number | null;
}

/** Build a full synthetic squat clip with a raised-cosine hip trajectory. */
export function buildSquatSeries(options: SquatOptions = {}): SyntheticSeries {
  const {
    reps = 3,
    fps = 30,
    repDurationS = 2,
    standingPauseS = 0.5,
    depthFraction = 1,
    torsoLeanDeg = 12,
    bottomLeanDeg = null,
    leftRightBias = 0,
    kneeValgus = 0,
    depthJitter = 0,
    undetectedFrames = [],
    view = "front",
    farSideVisibility = null,
  } = options;

  const standingHipY = ANKLE_Y - (THIGH + SHIN) * 0.97;
  const fullDepthHipY = ANKLE_Y - THIGH * 0.85;
  const fullDepthSetback = THIGH * 0.78;
  const undetected = new Set(undetectedFrames);

  const frames: FramePose[] = [];
  let frameIndex = 0;
  const emit = (
    hipY: number,
    lean: number,
    setback: number,
    depth = 0,
  ): void => {
    frames.push(
      buildFrame(frameIndex, frameIndex / fps, hipY, {
        torsoLeanDeg: lean,
        leftRightBias,
        hipSetback: setback,
        kneeValgus,
        valgusDepthScale: depth,
        detected: !undetected.has(frameIndex),
        view,
        farSideVisibility,
      }),
    );
    frameIndex += 1;
  };

  const pauseFrames = Math.max(1, Math.floor(standingPauseS * fps));
  const repFrames = Math.max(2, Math.floor(repDurationS * fps));
  const topLean = torsoLeanDeg;
  const lowLean = bottomLeanDeg ?? torsoLeanDeg;

  for (let i = 0; i < pauseFrames; i++) emit(standingHipY, topLean, 0);

  for (let rep = 0; rep < reps; rep++) {
    let repDepth = depthFraction + depthJitter * (rep % 2 ? 1 : -1);
    repDepth = Math.max(0.05, Math.min(1.2, repDepth));
    const bottomHipY = standingHipY + (fullDepthHipY - standingHipY) * repDepth;
    const bottomSetback = fullDepthSetback * repDepth;

    for (let step = 0; step < repFrames; step++) {
      const phase = 0.5 * (1 - Math.cos((2 * Math.PI * step) / repFrames));
      const hipY = standingHipY + (bottomHipY - standingHipY) * phase;
      emit(hipY, topLean + (lowLean - topLean) * phase, bottomSetback * phase, phase);
    }
    for (let i = 0; i < pauseFrames; i++) emit(standingHipY, topLean, 0);
  }

  return { frames, fps, durationS: frames.length / fps };
}

/** A clip of someone standing still. Must yield zero reps. */
export function buildStandingSeries(seconds = 4, fps = 30, noise = 0): SyntheticSeries {
  const standingHipY = ANKLE_Y - (THIGH + SHIN) * 0.97;
  const count = Math.floor(seconds * fps);
  const frames: FramePose[] = [];
  for (let i = 0; i < count; i++) {
    frames.push(
      buildFrame(i, i / fps, standingHipY + noise * Math.sin(i * 0.7), {
        torsoLeanDeg: 8,
      }),
    );
  }
  return { frames, fps, durationS: count / fps };
}
