import { MetricCard } from "@/components/dashboard/metric-card";
import { formatDuration, formatPercent } from "@/lib/format";
import type { SessionSummary } from "@/lib/live/session";

/** Shown after a live session ends: the set at a glance. */
export function WorkoutSummary({ summary }: { summary: SessionSummary }) {
  return (
    <div className="border-border/60 bg-card/40 space-y-4 rounded-xl border p-5">
      <div>
        <h2 className="text-base font-semibold">Workout summary</h2>
        <p className="text-muted-foreground text-xs">
          {summary.totalReps} rep{summary.totalReps === 1 ? "" : "s"}
          {summary.halfReps > 0
            ? ` · ${summary.halfReps} half rep${summary.halfReps === 1 ? "" : "s"}`
            : ""}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricCard label="Total reps" value={String(summary.totalReps)} />
        <MetricCard label="Full reps" value={String(summary.fullReps)} />
        <MetricCard
          label="Best depth"
          value={formatPercent(summary.bestDepthPercent)}
          meter={summary.bestDepthPercent}
          tone="good"
        />
        <MetricCard
          label="Average depth"
          value={formatPercent(summary.avgDepthPercent)}
          meter={summary.avgDepthPercent}
        />
        <MetricCard label="Average tempo" value={summary.avgTempo ?? "—"} />
        <MetricCard
          label="Working time"
          value={formatDuration(summary.workingTimeS)}
          detail={`Session ${formatDuration(summary.durationS)}`}
        />
      </div>
    </div>
  );
}
