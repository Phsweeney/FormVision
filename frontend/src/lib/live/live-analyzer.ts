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

import {
  NO_GATES,
  appendFrameAngles,
  computeAngles,
  computeRawFrameAngles,
  emptyAngleSeries,
  viewGates,
  type RawFrameAngles,
  type ViewGates,
} from "@/lib/analysis/angles";
import type { AnalysisConfig } from "@/lib/analysis/config";
import { linearScale } from "@/lib/analysis/geometry";
import { RepStateMachine, buildRep } from "@/lib/analysis/reps";
import { windowSizeForFps } from "@/lib/analysis/smoothing";
import { median } from "@/lib/analysis/stats";
import {
  PoseLandmarkIndex as LM,
  landmarkAt,
  type AngleSeries,
  type FramePose,
  type ViewOrientation,
} from "@/lib/analysis/types";

import { LiveClassifier, type MlVerdict } from "@/lib/ml/classify";
import { RunningRanks, derivedValues, frameSample } from "@/lib/ml/features";
import type { FaultModelBundle } from "@/lib/ml/model";

import { analyzeRep, type LiveRep } from "./rep-analysis";
import { TrailingAverage } from "./trailing-average";

export type LivePhase =
  | "waiting"
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
  /** The most recently completed rep, enriched with tempo/pause/half-rep. */
  lastRep: LiveRep | null;
  /**
   * What the model currently makes of the squat, or null when it has no model
   * loaded or too little session history for a rank to mean anything.
   *
   * Purely a readout. It never reaches the cue banner or the voice coach, which
   * stay rule-based: a per-frame classification is a reasonable thing to display
   * and a bad thing to shout at someone mid-rep.
   */
  mlVerdict: MlVerdict | null;
}

/** Minimum detected frames before calibration is trusted. */
const MIN_CALIBRATION_FRAMES = 5;

/** How long the state badge holds on "Bottom" after the turnaround. */
const BOTTOM_HOLD_S = 0.4;

/** Trailing window over which stillness is judged, before calibration begins. */
const STILL_WINDOW_S = 0.5;

/**
 * Maximum hip-height movement (normalised frame units) over the still window for
 * the lifter to count as "standing still". A settled stance sways well under
 * this; walking back into position, or bobbing, exceeds it. This gates the
 * *start* of calibration so getting into position never poisons the baseline.
 */
const STILLNESS_TOLERANCE = 0.03;

function emptySeries(view: ViewOrientation): AngleSeries {
  return emptyAngleSeries(null, null, view);
}

/** The per-frame signals added for the model, which need their own smoothers. */
type ModelSignal =
  | "leftHipDeg"
  | "rightHipDeg"
  | "leftAnkleDeg"
  | "rightAnkleDeg"
  | "leftKneeLateral"
  | "rightKneeLateral";

const MODEL_SIGNALS: readonly ModelSignal[] = [
  "leftHipDeg",
  "rightHipDeg",
  "leftAnkleDeg",
  "rightAnkleDeg",
  "leftKneeLateral",
  "rightKneeLateral",
];

export class LiveAnalyzer {
  private calibrating = true;
  private calibrationFrames: FramePose[] = [];
  /** When the current unbroken stretch of standing still began, or null. */
  private stillSinceTs: number | null = null;
  /** Recent hip-height samples for the stillness check. */
  private recentHip: Array<{ t: number; y: number }> = [];
  private lastFrameTs: number | null = null;
  private fpsEstimate = 30;

  private torsoScale: number | null = null;
  private thighScale: number | null = null;
  private view: ViewOrientation = "unknown";
  private gates: ViewGates = NO_GATES;
  private baseline = 0;
  private bottomReference = 0;
  private ready = false;

  private machine: RepStateMachine | null = null;
  private hipSmoother: TrailingAverage | null = null;
  private leftKneeSmoother: TrailingAverage | null = null;
  private rightKneeSmoother: TrailingAverage | null = null;
  private leanSmoother: TrailingAverage | null = null;
  private offsetSmoother: TrailingAverage | null = null;

  /**
   * Smoothers for the six signals only the model consumes. Grouped rather than
   * given a field each because none of them is referenced outside the ML path,
   * and eleven named smoother fields would bury the five that the live metrics
   * actually read.
   */
  private modelSmoothers: Record<ModelSignal, TrailingAverage> | null = null;

  private buffer: AngleSeries = emptySeries("unknown");
  private liveIndex = 0;

  private readonly repList: LiveRep[] = [];
  private maxDepth: number | null = null;
  private phase: LivePhase = "waiting";
  private repStartTs: number | null = null;
  private bottomHoldUntilS = 0;

  /** The session's own distribution of each quantity, for within-clip ranking. */
  private readonly ranks = new RunningRanks();
  private classifier: LiveClassifier | null = null;
  private mlVerdict: MlVerdict | null = null;

  /**
   * @param models Exported fault detectors, or null to run with no model at
   *   all. Optional so every existing caller and test is unchanged, and so the
   *   analyzer never depends on a network fetch having succeeded.
   */
  constructor(
    private readonly config: AnalysisConfig,
    private readonly models: FaultModelBundle | null = null,
  ) {}

  /** Completed reps so far this session. */
  get reps(): LiveRep[] {
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
    this.stillSinceTs = null;
    this.recentHip = [];
    this.lastFrameTs = null;
    this.ready = false;
    this.machine = null;
    this.buffer = emptySeries("unknown");
    this.liveIndex = 0;
    this.repList.length = 0;
    this.maxDepth = null;
    this.phase = "waiting";
    this.repStartTs = null;
    this.view = "unknown";
    this.ranks.reset();
    this.classifier = null;
    this.mlVerdict = null;
  }

  // --- Calibration -------------------------------------------------------

  private calibrate(frame: FramePose): LiveState {
    // Nothing is measurable yet: scale and view are exactly what calibration is
    // about to establish, so every view-gated signal is suppressed until it has.
    const raw = computeRawFrameAngles(
      frame,
      null,
      null,
      NO_GATES,
      this.config.landmark_visibility_threshold,
    );
    const knee = smallestPresent(raw.leftKneeDeg, raw.rightKneeDeg);
    const hipY = this.hipMidY(frame);

    // Calibration only runs while the lifter is fully in frame and holding
    // still. This is what lets someone start the camera at their keyboard and
    // walk into position: the moving frames are ignored, and the 2.5s of
    // measurement only accumulates once they have settled. Any motion resets it.
    const inFrame = raw.usable && hipY !== null;
    if (inFrame && this.isStill(frame.timestampS, hipY)) {
      if (this.stillSinceTs === null) {
        this.stillSinceTs = frame.timestampS;
        this.calibrationFrames = [];
      }
      this.calibrationFrames.push(frame);
      const elapsed = frame.timestampS - this.stillSinceTs;
      if (
        elapsed >= this.config.live_calibration_seconds &&
        this.calibrationFrames.length >= MIN_CALIBRATION_FRAMES
      ) {
        this.finishCalibration();
      }
    } else {
      this.stillSinceTs = null;
      this.calibrationFrames = [];
      // Out of frame entirely: drop stale stillness samples so returning starts fresh.
      if (!inFrame) this.recentHip = [];
    }

    const progress =
      this.stillSinceTs !== null
        ? Math.min(
            (frame.timestampS - this.stillSinceTs) /
              this.config.live_calibration_seconds,
            1,
          )
        : 0;
    const phase: LivePhase = this.calibrating
      ? this.stillSinceTs !== null
        ? "calibrating"
        : "waiting"
      : this.phase;

    return {
      phase,
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
      // Nothing to say until the camera view is known and the session has some
      // history to rank against.
      mlVerdict: null,
    };
  }

  /** Hip-midpoint y in normalised frame coords, or null if not both visible. */
  private hipMidY(frame: FramePose): number | null {
    const threshold = this.config.landmark_visibility_threshold;
    const left = landmarkAt(frame, LM.LEFT_HIP);
    const right = landmarkAt(frame, LM.RIGHT_HIP);
    if (!left || !right) return null;
    if (left.visibility < threshold || right.visibility < threshold) return null;
    return (left.y + right.y) / 2;
  }

  /** True once the hip has barely moved across the last STILL_WINDOW_S. */
  private isStill(timestampS: number, hipY: number): boolean {
    this.recentHip.push({ t: timestampS, y: hipY });
    const cutoff = timestampS - STILL_WINDOW_S;
    while (this.recentHip.length > 0 && this.recentHip[0].t < cutoff) {
      this.recentHip.shift();
    }
    // Need a full window of samples before judging stillness.
    if (this.recentHip[0].t > timestampS - STILL_WINDOW_S * 0.8) return false;
    let min = Infinity;
    let max = -Infinity;
    for (const { y } of this.recentHip) {
      if (y < min) min = y;
      if (y > max) max = y;
    }
    return max - min < STILLNESS_TOLERANCE;
  }

  private finishCalibration(): void {
    const angles = computeAngles(this.calibrationFrames, this.fpsEstimate, this.config);
    this.torsoScale = angles.torsoScale;
    this.thighScale = angles.thighScale;
    this.view = angles.view;
    this.gates = viewGates(angles.view);

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
    this.modelSmoothers = Object.fromEntries(
      MODEL_SIGNALS.map((name) => [name, new TrailingAverage(window)]),
    ) as Record<ModelSignal, TrailingAverage>;

    // Built here rather than in the constructor because it needs the camera
    // view, which is exactly what calibration has just established, and the
    // view is what decides which faults are assessable at all.
    this.classifier = this.models ? new LiveClassifier(this.models) : null;

    const { descend, ascend } = this.thresholds();
    this.machine = new RepStateMachine(
      descend,
      ascend,
      this.config.rep_turnaround_band,
    );
    this.buffer = emptySeries(this.view);
    this.liveIndex = 0;
    this.calibrating = false;
    this.phase = "standing";
  }

  /**
   * Score this frame with the model, and fold it into the session distribution.
   *
   * Ranking against the session so far is the live counterpart of the backend's
   * ranking against a whole clip: mid-session, the past is all there is. The
   * value is inserted *before* it is ranked so a frame is always ranked within a
   * population that includes itself, matching the batch behaviour.
   *
   * Wrapped so a model failure costs the readout and nothing else. This runs
   * inside the camera loop, and an exception here would take rep counting and
   * the coach down with it.
   */
  private updateModel(smoothed: RawFrameAngles, moving: boolean): void {
    if (!this.classifier) return;
    try {
      const values = derivedValues(frameSample(smoothed));
      // Every frame joins the distribution, standing ones included, matching
      // what the backend ranks against. Excluding them was tried and made real
      // footage measurably worse; see the note in `app/ml/predictor.py`. The
      // cost is that long rests between reps quieten the detectors, which is a
      // documented limitation rather than something fixed here.
      this.ranks.insert(values);

      // Scoring, though, is restricted to frames where the lifter is moving.
      // The backend does the same by scoring within rep windows, and the
      // difference is not cosmetic: on a clean synthetic squat, 41% of all
      // frames cleared the valgus threshold but only 2% of the frames inside a
      // rep did. Standing between reps is a near-constant pose whose ranks say
      // nothing about technique.
      if (!moving) return;
      this.classifier.observe(values, this.ranks);
    } catch {
      this.classifier = null;
      this.mlVerdict = null;
    }
  }

  /**
   * Close out the model's view of a repetition that has just been counted.
   *
   * The verdict then stands until the next rep finishes, which is the whole
   * point: one stable answer per rep rather than a readout that changes thirty
   * times a second and tells you nothing about what just happened.
   */
  private completeModelRep(repIndex: number, hasPriorRep: boolean): void {
    if (!this.classifier) return;
    try {
      const verdict = this.classifier.completeRep(repIndex, hasPriorRep);
      if (verdict !== null) this.mlVerdict = verdict;
    } catch {
      this.classifier = null;
      this.mlVerdict = null;
    }
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
      this.gates,
      this.config.landmark_visibility_threshold,
    );

    const hip = this.hipSmoother!.push(raw.hipHeight);
    const leftKnee = this.leftKneeSmoother!.push(raw.leftKneeDeg);
    const rightKnee = this.rightKneeSmoother!.push(raw.rightKneeDeg);
    const lean = this.leanSmoother!.push(raw.torsoLeanDeg);
    const offset = this.offsetSmoother!.push(raw.hipKneeOffset);

    // The live path stores *smoothed* values where the batch path smooths the
    // arrays afterwards. Rebuilding the frame record with the smoothed numbers
    // in place lets both share one append, so a new signal cannot be added to
    // the batch series and forgotten here.
    const smoothed: RawFrameAngles = {
      ...raw,
      leftKneeDeg: leftKnee,
      rightKneeDeg: rightKnee,
      torsoLeanDeg: lean,
      hipHeight: hip,
      hipKneeOffset: offset,
    };
    for (const name of MODEL_SIGNALS) {
      smoothed[name] = this.modelSmoothers![name].push(raw[name]);
    }

    // Append to the growing buffer so a completed rep can be measured over it.
    this.buffer.timestampsS.push(frame.timestampS);
    appendFrameAngles(this.buffer, smoothed);

    // Bottom reference: the deepest point of the *current* descent. While the
    // lifter is standing it is pinned to a reference depth, so each rep
    // re-derives its own depth from scratch. This is deliberately not a
    // session-long running minimum: one glitch frame (a tracking blip that
    // briefly puts the hip below the ankle, i.e. a spuriously low hip height)
    // would otherwise ratchet it down permanently, blow the descend threshold
    // below anything reachable, and silently kill rep detection for the rest of
    // the session. Flooring at zero guards against that same case within a rep.
    //
    // The collapse waits until the hip is genuinely back up, and that wait is
    // what stops every rep being counted twice. A rep closes when the hip
    // crosses `baseline - 0.25 * travel`, which is well short of upright.
    // Collapsing the reference there took `travel` down to `min_rep_range`,
    // which yanked the descend threshold up from around `baseline - 0.48` to
    // `baseline - 0.09`, above where the hip actually was. The machine
    // re-entered a descent on the spot and the rest of the ascent tripped a
    // second rep. It only showed on slow reps, because the phantom has to
    // outlast `min_rep_duration_s` to survive the filter.
    //
    // `settled` is deliberately the same threshold the collapse produces, so
    // the collapse can never itself start a descent: the hip is already above
    // the new descend threshold before that threshold exists.
    if (this.machine!.phase === "standing") {
      const settled =
        this.baseline - this.config.rep_descent_fraction * this.config.min_rep_range;
      if (hip !== null && hip >= settled) this.bottomReference = this.baseline;
    }
    if (hip !== null && hip < this.bottomReference) this.bottomReference = hip;
    this.bottomReference = Math.max(this.bottomReference, 0);
    const { descend, ascend } = this.thresholds();
    this.machine!.setThresholds(descend, ascend);

    const previousPhase = this.machine!.phase;
    const triple = this.machine!.push(this.liveIndex, hip);
    const machinePhase = this.machine!.phase;

    // A descent has just begun: mark the rep's start time.
    if (previousPhase === "standing" && machinePhase === "descending") {
      this.repStartTs = frame.timestampS;
    }
    // The hip has bottomed out and started rising: hold a "bottom" readout
    // briefly so the state badge shows the moment, which is otherwise instant.
    if (previousPhase === "descending" && machinePhase === "ascending") {
      this.bottomHoldUntilS = frame.timestampS + BOTTOM_HOLD_S;
    }

    // Score this frame before the rep is closed out below, so the frame that
    // completes a repetition is counted as part of it.
    this.updateModel(smoothed, machinePhase !== "standing" || triple !== null);

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
        this.repList.push(analyzeRep(rep, this.buffer, this.config));
        if (rep.depthPercent !== null) {
          this.maxDepth = Math.max(this.maxDepth ?? 0, rep.depthPercent);
        }
        // `length - 1` prior reps: the one just pushed is the one being judged.
        this.completeModelRep(rep.index, this.repList.length > 1);
      } else {
        // Too short to count as a repetition, so its frames must not leak into
        // the next one's verdict.
        this.classifier?.resetRep();
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

    this.phase = this.derivePhase(machinePhase, frame.timestampS);

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
      mlVerdict: this.mlVerdict,
    };
  }

  /** Map the machine phase to the four display states, adding a "bottom" band. */
  private derivePhase(
    machinePhase: "standing" | "descending" | "ascending",
    nowS: number,
  ): LivePhase {
    if (machinePhase === "standing") return "standing";
    if (machinePhase === "descending") return "descending";
    // Ascending: show "bottom" for a brief hold right after the turnaround.
    return nowS < this.bottomHoldUntilS ? "bottom" : "ascending";
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
