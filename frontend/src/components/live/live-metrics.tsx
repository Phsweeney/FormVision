import { MetricCard } from "@/components/dashboard/metric-card";
import type { AnalysisConfig } from "@/lib/analysis/config";
import type { ViewOrientation } from "@/lib/analysis/types";
import { formatDegrees, formatPercent, formatSeconds } from "@/lib/format";
import type { LivePhase, LiveState } from "@/lib/live/live-analyzer";
import type { PauseKind } from "@/lib/live/rep-analysis";

const PAUSE_LABEL: Record<PauseKind, string> = {
  none: "No pause",
  brief: "Brief pause",
  competition: "Competition pause",
};

/** How each phase reads and colours in the state badge. */
const PHASE: Record<LivePhase, { label: string; className: string }> = {
  waiting: {
    label: "Get set",
    className: "border-amber-500/40 bg-amber-500/15 text-amber-300",
  },
  calibrating: {
    label: "Calibrating",
    className: "border-amber-500/40 bg-amber-500/15 text-amber-300",
  },
  standing: {
    label: "Standing",
    className: "border-white/20 bg-white/10 text-white/80",
  },
  descending: {
    label: "Descending",
    className: "border-sky-500/40 bg-sky-500/15 text-sky-300",
  },
  bottom: {
    label: "Bottom",
    className: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  },
  ascending: {
    label: "Ascending",
    className: "border-sky-500/40 bg-sky-500/15 text-sky-300",
  },
};

const VIEW_LABEL: Record<ViewOrientation, string> = {
  side: "Side-on",
  front: "Front-on",
  oblique: "Oblique",
  unknown: "Unknown",
};

/** Why the torso card is blank, given the camera view (mirrors V1's honesty). */
function leanDetail(view: ViewOrientation): string | undefined {
  if (view === "front") return "Not measurable front-on";
  if (view === "unknown") return "Camera view unknown";
  return undefined;
}

function depthTone(
  percent: number | null,
  config: AnalysisConfig,
): "default" | "good" | "warning" | "critical" {
  if (percent === null) return "default";
  if (percent >= config.good_depth_percent) return "good";
  if (percent >= config.shallow_depth_percent) return "warning";
  return "critical";
}

export function LiveMetrics({
  state,
  config,
}: {
  state: LiveState;
  config: AnalysisConfig;
}) {
  const phase = PHASE[state.phase];

  return (
    <div className="space-y-4">
      {/* Rep counter + current phase, given the most visual weight. */}
      <div className="border-border/60 bg-card/40 flex items-center justify-between rounded-xl border p-5">
        <div>
          <p className="text-muted-foreground text-xs font-medium">Reps</p>
          <p className="font-mono text-5xl font-semibold tabular-nums">
            {state.repCount}
          </p>
        </div>
        <span
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium ${phase.className}`}
        >
          {phase.label}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricCard
          label="Current depth"
          value={formatPercent(state.currentDepthPercent)}
          meter={state.currentDepthPercent}
          tone={depthTone(state.currentDepthPercent, config)}
        />
        <MetricCard
          label="Max depth"
          value={formatPercent(state.maxDepthPercent)}
          meter={state.maxDepthPercent}
          tone={depthTone(state.maxDepthPercent, config)}
        />
        <MetricCard
          label="Knee angle"
          value={formatDegrees(state.currentKneeAngleDeg)}
        />
        <MetricCard
          label="Torso angle"
          value={formatDegrees(state.currentTorsoLeanDeg)}
          detail={leanDetail(state.view)}
        />
        <MetricCard
          label="Rep time"
          value={formatSeconds(state.currentRepElapsedS)}
        />
        <MetricCard label="Camera view" value={VIEW_LABEL[state.view]} />
      </div>

      {state.lastRep && (
        <div className="border-border/60 bg-card/40 rounded-xl border p-4">
          <div className="flex items-center justify-between">
            <p className="text-muted-foreground text-xs font-medium">
              Last rep · #{state.lastRep.index}
            </p>
            {state.lastRep.halfRep && (
              <span className="inline-flex items-center rounded-full border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300">
                Half rep
              </span>
            )}
          </div>
          <div className="mt-2 flex items-end gap-5">
            <div>
              <p className="font-mono text-2xl font-semibold tabular-nums">
                {state.lastRep.tempo}
              </p>
              <p className="text-muted-foreground/80 text-[11px]">
                tempo · ecc-pause-con
              </p>
            </div>
            <p className="text-muted-foreground pb-1 text-xs">
              {PAUSE_LABEL[state.lastRep.pauseKind]}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
