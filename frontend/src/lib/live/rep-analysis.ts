/**
 * Per-rep tempo, pause, and half-rep analysis.
 *
 * The base `Rep` records the descent and ascent as `bottomTime - startTime` and
 * `endTime - bottomTime` — but that lumps any pause at the bottom into one side.
 * Lifters think in three phases (a 3-1-2 tempo is eccentric-pause-concentric), so
 * this measures the bottom dwell separately by finding how long the hip stayed
 * within a small band of its lowest point, and splits the rep around it.
 */

import type { AnalysisConfig } from "@/lib/analysis/config";
import type { AngleSeries, Rep } from "@/lib/analysis/types";

export type PauseKind = "none" | "brief" | "competition";

/** A completed rep enriched with the live-only tempo/pause/half-rep read. */
export interface LiveRep extends Rep {
  eccentricS: number;
  pauseS: number;
  concentricS: number;
  /** Tempo notation, e.g. "3-1-2" (eccentric-pause-concentric, whole seconds). */
  tempo: string;
  pauseKind: PauseKind;
  /** True when the rep did not reach a real squat depth. */
  halfRep: boolean;
}

/**
 * Fraction of the rep's hip travel that counts as "at the bottom". Kept tight so
 * the pause reflects a genuine hold, not the moments a smooth rep naturally
 * spends passing through the turnaround.
 */
const BOTTOM_BAND_FRACTION = 0.05;

/**
 * Split a completed rep into eccentric / bottom-pause / concentric phases and
 * classify it. `angles` is the buffer the rep's frame indices point into.
 */
export function analyzeRep(
  rep: Rep,
  angles: AngleSeries,
  config: AnalysisConfig,
): LiveRep {
  const { startFrame, endFrame } = rep;
  const hip = angles.hipHeight;

  let bottom = Infinity;
  let top = -Infinity;
  for (let i = startFrame; i <= endFrame && i < hip.length; i++) {
    const value = hip[i];
    if (value === null) continue;
    if (value < bottom) bottom = value;
    if (value > top) top = value;
  }

  const { eccentricS, pauseS, concentricS } = phaseTimes(
    rep,
    angles,
    bottom,
    top,
  );

  const brief = config.bottom_pause_brief_s;
  const competition = config.bottom_pause_competition_s;
  let pauseKind: PauseKind = "none";
  if (pauseS >= competition) pauseKind = "competition";
  else if (pauseS >= brief) pauseKind = "brief";

  const halfRep =
    rep.depthPercent !== null && rep.depthPercent < config.shallow_depth_percent;

  return {
    ...rep,
    eccentricS,
    pauseS,
    concentricS,
    tempo: `${roundS(eccentricS)}-${roundS(pauseS)}-${roundS(concentricS)}`,
    pauseKind,
    halfRep,
  };
}

function phaseTimes(
  rep: Rep,
  angles: AngleSeries,
  bottom: number,
  top: number,
): { eccentricS: number; pauseS: number; concentricS: number } {
  const times = angles.timestampsS;
  const hip = angles.hipHeight;
  const travel = top - bottom;

  // Degenerate (flat) rep: fall back to the single-bottom-frame split.
  if (!Number.isFinite(travel) || travel <= 0) {
    return {
      eccentricS: rep.bottomTimeS - rep.startTimeS,
      pauseS: 0,
      concentricS: rep.endTimeS - rep.bottomTimeS,
    };
  }

  const band = bottom + BOTTOM_BAND_FRACTION * travel;
  let firstIn: number | null = null;
  let lastIn: number | null = null;
  for (let i = rep.startFrame; i <= rep.endFrame && i < hip.length; i++) {
    const value = hip[i];
    if (value !== null && value <= band) {
      if (firstIn === null) firstIn = i;
      lastIn = i;
    }
  }

  if (firstIn === null || lastIn === null) {
    return {
      eccentricS: rep.bottomTimeS - rep.startTimeS,
      pauseS: 0,
      concentricS: rep.endTimeS - rep.bottomTimeS,
    };
  }

  return {
    eccentricS: times[firstIn] - rep.startTimeS,
    pauseS: times[lastIn] - times[firstIn],
    concentricS: rep.endTimeS - times[lastIn],
  };
}

function roundS(seconds: number): number {
  return Math.max(0, Math.round(seconds));
}
