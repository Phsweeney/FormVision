import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string;
  /** Secondary context, e.g. the range behind an average. */
  detail?: string;
  /** Optional 0–100 bar, for metrics that read naturally as a proportion. */
  meter?: number | null;
  tone?: "default" | "good" | "warning" | "critical";
}

const METER_TONE: Record<string, string> = {
  default: "bg-primary/70",
  good: "bg-emerald-500/80",
  warning: "bg-amber-500/80",
  critical: "bg-red-500/80",
};

const VALUE_TONE: Record<string, string> = {
  default: "text-foreground",
  good: "text-emerald-400",
  warning: "text-amber-400",
  critical: "text-red-400",
};

/**
 * A single dashboard statistic.
 *
 * The value is rendered in a tabular-figure font so numbers do not jitter
 * horizontally when they update, which they do while polling.
 */
export function MetricCard({
  label,
  value,
  detail,
  meter,
  tone = "default",
}: MetricCardProps) {
  return (
    <div className="border-border/60 bg-card/40 flex flex-col justify-between rounded-xl border p-4">
      <p className="text-muted-foreground text-xs font-medium">{label}</p>

      <p
        className={cn(
          "mt-2 font-mono text-2xl font-semibold tabular-nums",
          VALUE_TONE[tone],
        )}
      >
        {value}
      </p>

      {detail && (
        <p className="text-muted-foreground/80 mt-1 text-[11px]">{detail}</p>
      )}

      {meter !== null && meter !== undefined && (
        <div
          className="bg-muted/50 mt-3 h-1 overflow-hidden rounded-full"
          role="presentation"
        >
          <div
            className={cn("h-full rounded-full transition-all", METER_TONE[tone])}
            style={{ width: `${Math.max(0, Math.min(100, meter))}%` }}
          />
        </div>
      )}
    </div>
  );
}
