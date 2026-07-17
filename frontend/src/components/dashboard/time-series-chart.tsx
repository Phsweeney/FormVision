"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { CHART_GEOMETRY, CHART_INK } from "@/lib/chart-theme";
import type { Rep } from "@/lib/types";

export interface SeriesDefinition {
  key: string;
  label: string;
  color: string;
  /** Suffix shown in the tooltip, e.g. "°". */
  unit?: string;
}

interface TimeSeriesChartProps {
  title: string;
  description: string;
  data: Record<string, number | null>[];
  series: SeriesDefinition[];
  reps: Rep[];
  yLabel?: string;
  /** Fixed y-domain. Angles use a fixed range so charts stay comparable. */
  domain?: [number | "auto", number | "auto"];
  valueDigits?: number;
}

/**
 * One time-series chart, shared by all three plots on the dashboard.
 *
 * A single component rather than three, so the axes, grid, tooltip, rep
 * shading, and spacing are identical everywhere and the charts read as one
 * system.
 *
 * Detected repetitions are shaded as background bands. That is the feature
 * that makes these charts readable: without it a lifter sees a wave and has to
 * guess which part was a rep. With it, every dip is labelled.
 */
export function TimeSeriesChart({
  title,
  description,
  data,
  series,
  reps,
  yLabel,
  domain = ["auto", "auto"],
  valueDigits = 0,
}: TimeSeriesChartProps) {
  return (
    // `min-w-0` is load-bearing. Grid and flex children default to
    // `min-width: auto`, which lets their content set a floor wider than the
    // track. Recharts' ResponsiveContainer then measures that inflated width and
    // renders a chart that overflows into its neighbour. Every chart wrapper
    // needs this.
    <figure className="border-border/60 bg-card/40 min-w-0 rounded-xl border p-4">
      <figcaption className="mb-3">
        <h3 className="text-sm font-medium">{title}</h3>
        <p className="text-muted-foreground mt-0.5 text-xs">{description}</p>
      </figcaption>

      {/* A legend is always present for two or more series, so identity never
          depends on colour alone. */}
      {series.length > 1 && (
        <div className="mb-2 flex flex-wrap items-center gap-4">
          {series.map((entry) => (
            <span
              key={entry.key}
              className="text-muted-foreground flex items-center gap-1.5 text-xs"
            >
              <span
                aria-hidden
                className="h-0.5 w-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              {entry.label}
            </span>
          ))}
        </div>
      )}

      <div className="w-full min-w-0" style={{ height: CHART_GEOMETRY.height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 8, right: 12, bottom: 0, left: -14 }}
          >
            <CartesianGrid
              stroke={CHART_INK.grid}
              strokeDasharray="3 3"
              vertical={false}
            />

            {/* Rep bands are drawn before the lines so they sit behind them. */}
            {reps.map((rep) => (
              <ReferenceArea
                key={rep.index}
                x1={rep.start_time_s}
                x2={rep.end_time_s}
                fill={CHART_INK.repBand}
                fillOpacity={CHART_INK.repBandOpacity}
                stroke="none"
                ifOverflow="hidden"
              />
            ))}

            <XAxis
              dataKey="time_s"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value: number) => `${value.toFixed(0)}s`}
              stroke={CHART_INK.axis}
              tick={{ fontSize: CHART_GEOMETRY.fontSize, fill: CHART_INK.axis }}
              tickLine={false}
              axisLine={false}
              minTickGap={28}
            />

            <YAxis
              domain={domain}
              stroke={CHART_INK.axis}
              tick={{ fontSize: CHART_GEOMETRY.fontSize, fill: CHART_INK.axis }}
              tickLine={false}
              axisLine={false}
              width={48}
              label={
                yLabel
                  ? {
                      value: yLabel,
                      angle: -90,
                      position: "insideLeft",
                      style: {
                        fontSize: CHART_GEOMETRY.fontSize,
                        fill: CHART_INK.axis,
                      },
                    }
                  : undefined
              }
            />

            <Tooltip
              content={
                <ChartTooltip series={series} valueDigits={valueDigits} reps={reps} />
              }
              cursor={{ stroke: CHART_INK.axis, strokeWidth: 1 }}
            />

            {series.map((entry) => (
              <Line
                key={entry.key}
                type="monotone"
                dataKey={entry.key}
                name={entry.label}
                stroke={entry.color}
                strokeWidth={CHART_GEOMETRY.strokeWidth}
                dot={false}
                // Bigger than the 2px line so the hit target is comfortable.
                activeDot={{ r: 4, strokeWidth: 0 }}
                isAnimationActive={false}
                // Nulls mark frames where tracking failed. Leaving a gap is the
                // honest rendering; connecting across would draw a line through
                // data that was never observed.
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

interface TooltipProps {
  active?: boolean;
  label?: number;
  payload?: { dataKey?: string | number; value?: number | null }[];
  series: SeriesDefinition[];
  reps: Rep[];
  valueDigits: number;
}

/**
 * Custom tooltip.
 *
 * Values wear text tokens; the coloured swatch beside each carries identity, so
 * the numbers stay legible regardless of series colour. It also names the rep
 * the cursor is inside, which is what turns a position on a wave into something
 * a lifter can act on.
 */
function ChartTooltip({
  active,
  label,
  payload,
  series,
  reps,
  valueDigits,
}: TooltipProps) {
  if (!active || !payload?.length || label === undefined) return null;

  const rep = reps.find(
    (candidate) =>
      label >= candidate.start_time_s && label <= candidate.end_time_s,
  );

  return (
    <div
      className="rounded-lg border px-3 py-2 text-xs shadow-lg"
      style={{
        backgroundColor: CHART_INK.tooltipSurface,
        borderColor: CHART_INK.tooltipBorder,
      }}
    >
      <div className="text-muted-foreground flex items-center justify-between gap-4">
        <span className="font-mono tabular-nums">{label.toFixed(2)}s</span>
        {rep && <span>Rep {rep.index}</span>}
      </div>

      <div className="mt-1.5 space-y-1">
        {series.map((entry) => {
          const point = payload.find((item) => item.dataKey === entry.key);
          const value = point?.value;
          return (
            <div
              key={entry.key}
              className="flex items-center justify-between gap-4"
            >
              <span className="text-muted-foreground flex items-center gap-1.5">
                <span
                  aria-hidden
                  className="size-2 rounded-full"
                  style={{ backgroundColor: entry.color }}
                />
                {entry.label}
              </span>
              <span className="text-foreground font-mono tabular-nums">
                {value === null || value === undefined
                  ? "—"
                  : `${value.toFixed(valueDigits)}${entry.unit ?? ""}`}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
