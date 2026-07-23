/**
 * The live analysis driver.
 *
 * Turns a stream of `FramePose`s into live state: a calibration phase, then
 * online rep counting and metrics. It reuses the ported engine wholesale — the
 * per-frame angle maths (`computeRawFrameAngles`), the rep state machine
 * (`RepStateMachine`), and `buildRep` — and adds only what a live setting needs
 * that a file does not:
 *
 *   1. Calibration. The batch pipeline sets body scale and camera view from the
 *      whole clip; live has no whole clip, so the first couple of seconds of
 *      standing still establish them (and a standing hip-height baseline).
 *   2. Online thresholds. Batch derives rep thresholds from clip-wide
 *      percentiles; live seeds them from the baseline and adapts the bottom
 *      reference as the lifter descends.
 *   3. Causal smoothing. A trailing average, because the future frames a centred
 *      window needs do not exist yet.
 */

import { computeAngles, computeRawFrameAngles } from "@/lib/analysis/angles";
import type { AnalysisConfig } from "@/lib/analysis/config";
import { linearScale } from "@/lib/analysis/geometry";
import { RepStateMachine, buildRep } from "@/lib/analysis/reps";
import { windowSizeForFps } from "@/lib/analysis/smoothing";
import { median } from "@/lib/analysis/stats";
import type {
  AngleSeries,
  FramePose,
  Rep,
  ViewOrientation,
} from "@/lib/analysis/types";

import { TrailingAverage } from "./trailing-average";

export type LivePhase =
  | "calibrating"
  | "standing"
  | "descending"
  | "bottom"
  | "ascending";

export interface LiveState {
  phase: LivePhase;
  /** 0..1 through the stand-still calibration. */
  calibrationProgress: number;
  /** True once body scale and baseline are established. */
  ready: boolean;
  view: ViewOrientation;
  repCount: number;
  currentKneeAngleDeg: number | null;
  currentDepthPercent: number | null;
  maxDepthPercent: number | null;
  currentTorsoLeanDeg: number | null;
  /** Seconds since the current rep's descent began, or null between reps. */
  currentRepElapsedS: number | null;
  /** The most recently completed rep, for tempo/feedback consumers. */
  lastRep: Rep | null;
}

/** Minimum detected frames before calibration is trusted. */
const MIN_CALIBRATION_FRAMES = 5;

function emptySeries(view: ViewOrientation): AngleSeries {
  return {
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
    torsoScale: null,
    thighScale: null,
    view,
  };
}

export class LiveAnalyzer {
  private calibrating = true;
  private calibrationFrames: FramePose[] = [];
  private startTs: number | null = null;
  private lastFrameTs: number | null = null;
  private fpsEstimate = 30;

  private torsoScale: number | null = null;
  private thighScale: number | null = null;
  private view: ViewOrientation = "unknown";
  private leanIsMeasurable = true;
  private baseline = 0;
  private bottomReference = 0;
  private ready = false;

  private machine: RepStateMachine | null = null;
  private hipSmoother: TrailingAverage | null = null;
  private leftKneeSmoother: TrailingAverage | null = null;
  private rightKneeSmoother: TrailingAverage | null = null;
  private leanSmoother: TrailingAverage | null = null;
  private offsetSmoother: TrailingAverage | null = null;

  private buffer: AngleSeries = emptySeries("unknown");
  private liveIndex = 0;

  private readonly repList: Rep[] = [];
  private maxDepth: number | null = null;
  private phase: LivePhase = "calibrating";
  private repStartTs: number | null = null;

  constructor(private readonly config: AnalysisConfig) {}

  /** Completed reps so far this session. */
  get reps(): Rep[] {
    return this.repList;
  }

  /** Feed one frame; get the current live state back. */
  push(frame: FramePose): LiveState {
    this.updateFps(frame.timestampS);
    return this.calibrating ? this.calibrate(frame) : this.analyze(frame);
  }

  reset(): void {
    this.calibrating = true;
    this.calibrationFrames = [];
    this.startTs = null;
    this.lastFrameTs = null;
    this.ready = false;
    this.machine = null;
    this.buffer = emptySeries("unknown");
    this.liveIndex = 0;
    this.repList.length = 0;
    this.maxDepth = null;
    this.phase = "calibrating";
    this.repStartTs = null;
    this.view = "unknown";
  }

  // --- Calibration -------------------------------------------------------

  private calibrate(frame: FramePose): LiveState {
    if (this.startTs === null) this.startTs = frame.timestampS;
    if (frame.detected) this.calibrationFrames.push(frame);

    const elapsed = frame.timestampS - this.startTs;
    const progress = Math.min(elapsed / this.config.live_calibration_seconds, 1);

    if (progress >= 1 && this.calibrationFrames.length >= MIN_CALIBRATION_FRAMES) {
      this.finishCalibration();
    }

    // Show the live knee angle even while calibrating (it needs no scale).
    const raw = computeRawFrameAngles(
      frame,
      null,
      null,
      false,
      this.config.landmark_visibility_threshold,
    );
    const knee = smallestPresent(raw.leftKneeDeg, raw.rightKneeDeg);

    return {
      phase: this.calibrating ? "calibrating" : this.phase,
      calibrationProgress: progress,
      ready: this.ready,
      view: this.view,
      repCount: 0,
      currentKneeAngleDeg: knee,
      currentDepthPercent: null,
      maxDepthPercent: null,
      currentTorsoLeanDeg: null,
      currentRepElapsedS: null,
      lastRep: null,
    };
  }

  private finishCalibration(): void {
    const angles = computeAngles(this.calibrationFrames, this.fpsEstimate, this.config);
    this.torsoScale = angles.torsoScale;
    this.thighScale = angles.thighScale;
    this.view = angles.view;
    this.leanIsMeasurable = angles.view !== "front";

    const standingHeights = angles.hipHeight.filter((v): v is number => v !== null);
    this.baseline = median(standingHeights) ?? 0;
    this.bottomReference = this.baseline;
    this.ready = this.torsoScale !== null && standingHeights.length > 0;

    const window = windowSizeForFps(
      this.fpsEstimate,
      this.config.smoothing_window_seconds,
    );
    this.hipSmoother = new TrailingAverage(window);
    this.leftKneeSmoother = new TrailingAverage(window);
    this.rightKneeSmoother = new TrailingAverage(window);
    this.leanSmoother = new TrailingAverage(window);
    this.offsetSmoother = new TrailingAverage(window);

    const { descend, ascend } = this.thresholds();
    this.machine = new RepStateMachine(descend, ascend);
    this.buffer = emptySeries(this.view);
    this.liveIndex = 0;
    this.calibrating = false;
    this.phase = "standing";
  }

  /** Rep thresholds from the baseline and the deepest point seen so far. */
  private thresholds(): { descend: number; ascend: number } {
    const travel = Math.max(
      this.baseline - this.bottomReference,
      this.config.min_rep_range,
    );
    return {
      descend: this.baseline - this.config.rep_descent_fraction * travel,
      ascend: this.baseline - this.config.rep_ascent_fraction * travel,
    };
  }

  // --- Live analysis -----------------------------------------------------

  private analyze(frame: FramePose): LiveState {
    const raw = computeRawFrameAngles(
      frame,
      this.torsoScale,
      this.thighScale,
      this.leanIsMeasurable,
      this.config.landmark_visibility_threshold,
    );

    const hip = this.hipSmoother!.push(raw.hipHeight);
    const leftKnee = this.leftKneeSmoother!.push(raw.leftKneeDeg);
    const rightKnee = this.rightKneeSmoother!.push(raw.rightKneeDeg);
    const lean = this.leanSmoother!.push(raw.torsoLeanDeg);
    const offset = this.offsetSmoother!.push(raw.hipKneeOffset);

    // Append to the growing buffer so a completed rep can be measured over it.
    this.buffer.timestampsS.push(frame.timestampS);
    this.buffer.leftKneeDeg.push(leftKnee);
    this.buffer.rightKneeDeg.push(rightKnee);
    this.buffer.hipDeg.push(raw.hipDeg);
    this.buffer.torsoLeanDeg.push(lean);
    this.buffer.hipHeight.push(hip);
    this.buffer.hipKneeOffset.push(offset);
    this.buffer.valid.push(raw.usable);
    this.buffer.leftLegValid.push(raw.leftLeg);
    this.buffer.rightLegValid.push(raw.rightLeg);

    // Adapt the bottom reference downward as the lifter descends, then refresh
    // the thresholds from it.
    if (hip !== null && hip < this.bottomReference) this.bottomReference = hip;
    const { descend, ascend } = this.thresholds();
    this.machine!.setThresholds(descend, ascend);

    const previousPhase = this.machine!.phase;
    const triple = this.machine!.push(this.liveIndex, hip);
    const machinePhase = this.machine!.phase;

    // A descent has just begun: mark the rep's start time.
    if (previousPhase === "standing" && machinePhase === "descending") {
      this.repStartTs = frame.timestampS;
    }

    if (triple) {
      const startTime = this.buffer.timestampsS[triple.start];
      const endTime = this.buffer.timestampsS[triple.end];
      if (endTime - startTime >= this.config.min_rep_duration_s) {
        const rep = buildRep(
          this.repList.length + 1,
          triple.start,
          triple.bottom,
          triple.end,
          this.buffer,
          this.config,
        );
        this.repList.push(rep);
        if (rep.depthPercent !== null) {
          this.maxDepth = Math.max(this.maxDepth ?? 0, rep.depthPercent);
        }
      }
      this.repStartTs = null;
    }

    this.liveIndex += 1;

    const knee = smallestPresent(leftKnee, rightKnee);
    const depth =
      knee !== null
        ? linearScale(
            knee,
            this.config.standing_knee_angle_deg,
            this.config.parallel_knee_angle_deg,
          ) * 100
        : null;
    if (depth !== null) this.maxDepth = Math.max(this.maxDepth ?? 0, depth);

    this.phase = this.derivePhase(machinePhase, hip);

    return {
      phase: this.phase,
      calibrationProgress: 1,
      ready: this.ready,
      view: this.view,
      repCount: this.repList.length,
      currentKneeAngleDeg: knee,
      currentDepthPercent: depth,
      maxDepthPercent: this.maxDepth,
      currentTorsoLeanDeg: lean,
      currentRepElapsedS:
        this.repStartTs !== null ? frame.timestampS - this.repStartTs : null,
      lastRep: this.repList.at(-1) ?? null,
    };
  }

  /** Map the machine phase to the four display states, adding a "bottom" band. */
  private derivePhase(
    machinePhase: "standing" | "descending" | "ascending",
    hip: number | null,
  ): LivePhase {
    if (machinePhase === "standing") return "standing";
    // Near the deepest point of an active rep, call it the bottom.
    if (hip !== null) {
      const travel = this.baseline - this.bottomReference;
      if (travel > 0 && hip <= this.bottomReference + 0.12 * travel) return "bottom";
    }
    return machinePhase;
  }

  private updateFps(timestampS: number): void {
    if (this.lastFrameTs !== null) {
      const dt = timestampS - this.lastFrameTs;
      if (dt > 0) {
        const instant = 1 / dt;
        this.fpsEstimate = this.fpsEstimate * 0.9 + instant * 0.1;
      }
    }
    this.lastFrameTs = timestampS;
  }
}

function smallestPresent(a: number | null, b: number | null): number | null {
  if (a === null) return b;
  if (b === null) return a;
  return Math.min(a, b);
}
