/**
 * Skeleton drawing on a 2D canvas.
 *
 * A direct port of the backend's `overlay.py` skeleton pass, with `cv2.line` /
 * `cv2.circle` replaced by Canvas 2D. The colours match (converted from the
 * backend's BGR to RGB), and so does the load-bearing idea: **two visibility
 * thresholds, not one**. Below the draw threshold a landmark is a guess and is
 * not drawn; between the draw and confident thresholds it is drawn dimmed. That
 * keeps the skeleton whole through an occluded far-side limb while staying
 * honest about which parts were inferred rather than observed.
 *
 * Pure and framework-free: it takes a context and a `FramePose`, and is the
 * only reason this file knows the canvas exists.
 */

import { POSE_CONNECTIONS, PoseLandmarkIndex, type FramePose } from "@/lib/analysis/types";

const BONE = "rgb(135, 206, 235)";
const BONE_DIM = "rgb(70, 105, 120)";
const JOINT = "rgb(255, 255, 255)";
const JOINT_DIM = "rgb(140, 140, 140)";
const ACCENT = "rgb(120, 220, 120)";
const ACCENT_DIM = "rgb(70, 120, 70)";

/** The eight joints the analysis actually uses; drawn larger and accented. */
const KEY_JOINTS = new Set<number>([
  PoseLandmarkIndex.LEFT_SHOULDER,
  PoseLandmarkIndex.RIGHT_SHOULDER,
  PoseLandmarkIndex.LEFT_HIP,
  PoseLandmarkIndex.RIGHT_HIP,
  PoseLandmarkIndex.LEFT_KNEE,
  PoseLandmarkIndex.RIGHT_KNEE,
  PoseLandmarkIndex.LEFT_ANKLE,
  PoseLandmarkIndex.RIGHT_ANKLE,
]);

export interface SkeletonThresholds {
  /** At or above this visibility a landmark is drawn solid (backend: 0.5). */
  confidentVisibility: number;
  /** Below this a landmark is not drawn at all (backend: 0.10). */
  drawVisibility: number;
}

/**
 * Clear the canvas and draw the skeleton for one frame.
 *
 * `width`/`height` are the canvas's pixel dimensions; normalised landmarks
 * ([0,1]) scale onto them. An undetected frame just clears — the video shows
 * through and the caller can display a "searching" state.
 */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  frame: FramePose,
  width: number,
  height: number,
  thresholds: SkeletonThresholds,
): void {
  ctx.clearRect(0, 0, width, height);
  if (!frame.detected) return;

  const points = new Map<number, [number, number]>();
  const confident = new Set<number>();
  frame.landmarks.forEach((landmark, i) => {
    if (landmark.visibility < thresholds.drawVisibility) return;
    points.set(i, [landmark.x * width, landmark.y * height]);
    if (landmark.visibility >= thresholds.confidentVisibility) confident.add(i);
  });

  const shortest = Math.min(width, height);
  const thickness = Math.max(2, Math.floor(shortest / 250));
  const radius = Math.max(3, Math.floor(shortest / 200));

  ctx.lineWidth = thickness;
  ctx.lineCap = "round";
  for (const [start, end] of POSE_CONNECTIONS) {
    const a = points.get(start);
    const b = points.get(end);
    if (!a || !b) continue;
    // A bone is only as trustworthy as its least certain end.
    const solid = confident.has(start) && confident.has(end);
    ctx.strokeStyle = solid ? BONE : BONE_DIM;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }

  for (const [i, [x, y]] of points) {
    const key = KEY_JOINTS.has(i);
    let colour: string;
    if (confident.has(i)) colour = key ? ACCENT : JOINT;
    else colour = key ? ACCENT_DIM : JOINT_DIM;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(x, y, key ? radius : Math.max(2, radius - 2), 0, Math.PI * 2);
    ctx.fill();
  }
}
