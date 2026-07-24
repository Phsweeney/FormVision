/**
 * Session statistics.
 *
 * A live session's summary is derived, not accumulated: given the completed reps
 * and the wall-clock duration, everything the summary shows is a fold over the
 * rep list. That keeps it a pure function — easy to test, and impossible to drift
 * from the reps it describes. It lives client-side only and is discarded when the
 * session ends.
 */

import { mean } from "@/lib/analysis/stats";

import type { LiveRep } from "./rep-analysis";

export interface SessionSummary {
  totalReps: number;
  fullReps: number;
  halfReps: number;
  bestDepthPercent: number | null;
  avgDepthPercent: number | null;
  /** Average tempo across reps, in ecc-pause-con whole-second notation. */
  avgTempo: string | null;
  /** First-descent to last-lockout, the actual working time. */
  workingTimeS: number | null;
  /** Wall-clock length of the session. */
  durationS: number;
}

export function summarizeSession(
  reps: LiveRep[],
  durationS: number,
): SessionSummary {
  if (reps.length === 0) {
    return {
      totalReps: 0,
      fullReps: 0,
      halfReps: 0,
      bestDepthPercent: null,
      avgDepthPercent: null,
      avgTempo: null,
      workingTimeS: null,
      durationS,
    };
  }

  const depths = reps
    .map((rep) => rep.depthPercent)
    .filter((value): value is number => value !== null);
  const halfReps = reps.filter((rep) => rep.halfRep).length;

  const avgEcc = mean(reps.map((rep) => rep.eccentricS)) ?? 0;
  const avgPause = mean(reps.map((rep) => rep.pauseS)) ?? 0;
  const avgCon = mean(reps.map((rep) => rep.concentricS)) ?? 0;
  const round = (s: number) => Math.max(0, Math.round(s));

  return {
    totalReps: reps.length,
    fullReps: reps.length - halfReps,
    halfReps,
    bestDepthPercent: depths.length ? Math.max(...depths) : null,
    avgDepthPercent: mean(depths),
    avgTempo: `${round(avgEcc)}-${round(avgPause)}-${round(avgCon)}`,
    workingTimeS: reps[reps.length - 1].endTimeS - reps[0].startTimeS,
    durationS,
  };
}
