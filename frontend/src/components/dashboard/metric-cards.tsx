import { MetricCard } from "@/components/dashboard/metric-card";
import {
  formatDegrees,
  formatDuration,
  formatNumber,
  formatPercent,
  formatSeconds,
} from "@/lib/format";
import type { Metrics } from "@/lib/types";

interface MetricCardsProps {
  metrics: Metrics;
  /** Depth at or above this counts as full depth. Mirrors the backend default. */
  goodDepthPercent?: number;
  shallowDepthPercent?: number;
}

/**
 * The metric grid.
 *
 * Only depth and tracking quality are given a colour tone. Tempo, rep count,
 * and duration are descriptive, not judgements — colouring them would imply a
 * verdict the analysis is not making. The coaching panel is where judgements
 * belong, with an explanation attached.
 */
export function MetricCards({
  metrics,
  goodDepthPercent = 90,
  shallowDepthPercent = 70,
}: MetricCardsProps) {
  const depth = metrics.avg_depth_percent;
  const depthTone
    = depth === null
      ? "default"
      : depth >= goodDepthPercent
        ? "good"
        : depth < shallowDepthPercent
          ? "critical"
          : "warning";

  const quality = metrics.tracking_quality * 100;
  const qualityTone
    = quality >= 90 ? "good" : quality >= 60 ? "warning" : "critical";

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      <MetricCard
        label="Repetitions"
        value={String(metrics.total_reps)}
        detail={
          metrics.reps_per_minute !== null
            ? `${formatNumber(metrics.reps_per_minute, 0)} per minute`
            : undefined
        }
      />

      <MetricCard
        label="Average depth"
        value={formatPercent(depth)}
        detail={
          metrics.max_depth_percent !== null
            ? `Best rep ${formatPercent(metrics.max_depth_percent)}`
            : undefined
        }
        meter={depth}
        tone={depthTone}
      />

      <MetricCard
        label="Deepest knee angle"
        value={formatDegrees(metrics.min_knee_angle_deg)}
        detail="Lower is deeper · 90° is parallel"
      />

      <MetricCard
        label="Average rep time"
        value={formatSeconds(metrics.avg_rep_duration_s)}
        detail={
          metrics.avg_eccentric_s !== null && metrics.avg_concentric_s !== null
            ? `${formatSeconds(metrics.avg_eccentric_s)} down · ${formatSeconds(metrics.avg_concentric_s)} up`
            : undefined
        }
      />

      <MetricCard
        label="Fastest rep"
        value={formatSeconds(metrics.fastest_rep_s)}
        detail={
          metrics.slowest_rep_s !== null
            ? `Slowest ${formatSeconds(metrics.slowest_rep_s)}`
            : undefined
        }
      />

      <MetricCard
        label="Max forward lean"
        value={formatDegrees(metrics.max_torso_lean_deg)}
        detail={
          metrics.avg_torso_lean_deg !== null
            ? `${formatDegrees(metrics.avg_torso_lean_deg)} average`
            : undefined
        }
      />

      <MetricCard
        label="Working time"
        value={formatDuration(metrics.total_workout_time_s)}
        detail={`Video ${formatDuration(metrics.video_duration_s)}`}
      />

      <MetricCard
        label="Tracking quality"
        value={formatPercent(quality)}
        detail="Frames usable for analysis"
        meter={quality}
        tone={qualityTone}
      />
    </div>
  );
}
