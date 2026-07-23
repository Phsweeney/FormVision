/**
 * Live pose estimation.
 *
 * Wraps MediaPipe's `PoseLandmarker` in VIDEO running mode — the same mode and
 * the same `pose_landmarker_lite` model the backend uses, so the landmarks the
 * live engine sees match what the offline pipeline would produce. Everything
 * else in the live code consumes the framework-free `FramePose`; MediaPipe is
 * quarantined to this file, exactly as it is to one module on the backend.
 */

import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

import type { FramePose } from "@/lib/analysis/types";

/** Served from our own origin by `npm run setup:live` (see scripts/). */
const WASM_PATH = "/mediapipe/wasm";
const MODEL_PATH = "/models/pose_landmarker_lite.task";

export class PoseRunner {
  private landmarker: PoseLandmarker | null = null;
  private frameIndex = 0;
  private lastTimestampMs = -1;

  /** Whether the underlying task is loaded and ready to detect. */
  get ready(): boolean {
    return this.landmarker !== null;
  }

  /**
   * Load the WASM runtime and the pose model.
   *
   * Tries the GPU delegate first and falls back to CPU: the GPU path is much
   * faster but is not available in every browser/driver combination, and a
   * hard failure there should degrade to "slower" rather than "broken".
   */
  async load(): Promise<void> {
    if (this.landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);

    const options = (delegate: "GPU" | "CPU") =>
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate },
        runningMode: "VIDEO" as const,
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

    try {
      this.landmarker = await options("GPU");
    } catch {
      this.landmarker = await options("CPU");
    }
  }

  /**
   * Detect a pose in the current video frame.
   *
   * `timestampMs` is coerced to be strictly increasing: MediaPipe's VIDEO mode
   * rejects a timestamp that does not advance, which a paused or stuttering
   * feed can otherwise produce. An undetected frame is returned as
   * `detected: false` with no landmarks rather than being dropped, so the
   * downstream signal keeps a slot for every frame.
   */
  detect(video: HTMLVideoElement, timestampMs: number): FramePose {
    if (!this.landmarker) {
      throw new Error("PoseRunner.detect called before load()");
    }
    const stamp = Math.max(timestampMs, this.lastTimestampMs + 1);
    this.lastTimestampMs = stamp;

    const result = this.landmarker.detectForVideo(video, stamp);
    const points = result.landmarks[0];
    const detected = Array.isArray(points) && points.length > 0;

    return {
      frameIndex: this.frameIndex++,
      timestampS: stamp / 1000,
      landmarks: detected ? points.map(toLandmark) : [],
      detected,
    };
  }

  /** Release the native task. Idempotent. */
  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}

function toLandmark(point: NormalizedLandmark) {
  return {
    x: point.x,
    y: point.y,
    z: point.z,
    // MediaPipe always reports visibility for pose; default guards older builds.
    visibility: point.visibility ?? 1,
  };
}
