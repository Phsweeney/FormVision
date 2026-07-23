/**
 * Workout-level metric aggregation — the TS mirror of `metrics.py`.
 *
 * Two conventions: missing stays missing (null, never 0 — a set with no reps has
 * no average depth, not 0%), and consistency is a standard deviation, not a
 * range, so one bad rep among twenty does not look like twenty erratic ones.
 */

import { mean, populationStd } from "./stats";
import {
  repConcentricS,
  repDurationS,
  repEccentricS,
  validFraction,
  type AngleSeries,
  type Metrics,
  type Rep,
} from "./types";

/** Present values of a per-rep field, skipping the unmeasured. */
function collect(reps: Rep[], pick: (rep: Rep) => number | null): number[] {
  return reps.map(pick).filter((v): v is number => v !== null);
}

const min = (values: number[]): number | null =>
  values.length ? Math.min(...values) : null;
const max = (values: number[]): number | null =>
  values.length ? Math.max(...values) : null;

/** Aggregate per-rep measurements into the dashboard summary. */
export function computeMetrics(
  reps: Rep[],
  angles: AngleSeries,
  videoDurationS: number,
): Metrics {
  const trackingQuality = validFraction(angles);

  if (reps.length === 0) {
    return {
      totalReps: 0,
      videoDurationS,
      totalWorkoutTimeS: 0,
      maxDepthPercent: null,
      avgDepthPercent: null,
      minKneeAngleDeg: null,
      avgRepDurationS: null,
      fastestRepS: null,
      slowestRepS: null,
      avgEccentricS: null,
      avgConcentricS: null,
      repsPerMinute: null,
      avgTorsoLeanDeg: null,
      maxTorsoLeanDeg: null,
      avgKneeAsymmetryDeg: null,
      depthConsistencyPercent: null,
      durationConsistencyS: null,
      trackingQuality,
      cameraView: angles.view,
    };
  }

  const depths = collect(reps, (r) => r.depthPercent);
  const durations = reps.map(repDurationS);
  const eccentrics = reps.map(repEccentricS);
  const concentrics = reps.map(repConcentricS);
  const leans = collect(reps, (r) => r.maxTorsoLeanDeg);
  const asymmetries = collect(reps, (r) => r.kneeAsymmetryDeg);
  const kneeAngles = collect(reps, (r) => r.minKneeAngleDeg);

  // Working time spans the first descent to the last lockout, excluding setup.
  const totalWorkoutTime = reps[reps.length - 1].endTimeS - reps[0].startTimeS;
  const repsPerMinute =
    totalWorkoutTime > 0 ? (reps.length / totalWorkoutTime) * 60 : null;

  return {
    totalReps: reps.length,
    videoDurationS,
    totalWorkoutTimeS: totalWorkoutTime,
    // Deepest rep = largest depth percent = smallest knee angle.
    maxDepthPercent: max(depths),
    avgDepthPercent: mean(depths),
    minKneeAngleDeg: min(kneeAngles),
    avgRepDurationS: mean(durations),
    fastestRepS: min(durations),
    slowestRepS: max(durations),
    avgEccentricS: mean(eccentrics),
    avgConcentricS: mean(concentrics),
    repsPerMinute,
    avgTorsoLeanDeg: mean(leans),
    maxTorsoLeanDeg: max(leans),
    avgKneeAsymmetryDeg: mean(asymmetries),
    depthConsistencyPercent: populationStd(depths),
    durationConsistencyS: populationStd(durations),
    trackingQuality,
    cameraView: angles.view,
  };
}
