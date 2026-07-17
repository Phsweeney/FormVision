"use client";

import { useMemo } from "react";

import { TimeSeriesChart } from "@/components/dashboard/time-series-chart";
import { SERIES_COLORS } from "@/lib/chart-theme";
import type { Rep, Series } from "@/lib/types";

interface AnalysisChartsProps {
  series: Series;
  reps: Rep[];
}

/**
 * The three time-series charts.
 *
 * The API returns column arrays (one per measurement); Recharts wants row
 * objects. That transposition happens once here and is memoised, rather than
 * three times inside three chart components.
 *
 * The charts are stacked full width rather than laid out in a two-column grid.
 * Two reasons, and the second is the one that settled it:
 *
 * 1. These are 15+ seconds of time series. More horizontal pixels per
 *    repetition is exactly what makes them readable.
 * 2. Recharts' ResponsiveContainer measures its parent element, and inside a
 *    grid track it can latch onto a width wider than the track it ends up
 *    occupying — rendering a chart that overflows into its neighbour. A
 *    single-column flow gives it an unambiguous width to measure.
 */
export function AnalysisCharts({ series, reps }: AnalysisChartsProps) {
  const rows = useMemo(
    () =>
      series.time_s.map((time, index) => ({
        time_s: time,
        left_knee_deg: series.left_knee_deg[index],
        right_knee_deg: series.right_knee_deg[index],
        hip_deg: series.hip_deg[index],
        torso_lean_deg: series.torso_lean_deg[index],
        hip_height: series.hip_height[index],
      })),
    [series],
  );

  if (rows.length === 0) return null;

  return (
    <div className="flex flex-col gap-4">
      <TimeSeriesChart
        title="Knee angle over time"
        description="Both knees, in degrees. Lower is deeper — 90° is parallel, 180° is a locked-out leg. Shaded bands are detected repetitions."
        data={rows}
        reps={reps}
        yLabel="degrees"
        domain={[40, 190]}
        series={[
          {
            key: "left_knee_deg",
            label: "Left knee",
            color: SERIES_COLORS.leftKnee,
            unit: "°",
          },
          {
            key: "right_knee_deg",
            label: "Right knee",
            color: SERIES_COLORS.rightKnee,
            unit: "°",
          },
        ]}
      />

      <TimeSeriesChart
        title="Hip angle over time"
        description="The angle between your torso and thigh. It closes as you hinge forward and descend."
        data={rows}
        reps={reps}
        yLabel="degrees"
        domain={[0, 190]}
        series={[
          {
            key: "hip_deg",
            label: "Hip angle",
            color: SERIES_COLORS.hip,
            unit: "°",
          },
        ]}
      />

      <TimeSeriesChart
        title="Hip height over time"
        description="Vertical hip position in torso lengths, so it is comparable regardless of camera distance. This is the signal repetitions are detected from."
        data={rows}
        reps={reps}
        yLabel="torso lengths"
        domain={["auto", "auto"]}
        valueDigits={2}
        series={[
          {
            key: "hip_height",
            label: "Hip height",
            color: SERIES_COLORS.hipHeight,
          },
        ]}
      />
    </div>
  );
}
