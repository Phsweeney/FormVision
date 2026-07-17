import { formatDegrees, formatPercent, formatSeconds } from "@/lib/format";
import type { Rep } from "@/lib/types";

/**
 * Per-repetition breakdown.
 *
 * This doubles as the accessible table view for the charts above: every value
 * plotted per rep is also readable as text, so the analysis does not depend on
 * being able to interpret a line graph.
 *
 * Scrolls inside its own container so a narrow viewport never forces the page
 * body to scroll sideways.
 */
export function RepTable({ reps }: { reps: Rep[] }) {
  if (reps.length === 0) return null;

  return (
    <div className="border-border/60 bg-card/40 overflow-hidden rounded-xl border">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <caption className="sr-only">
            Per-repetition measurements: timing, depth, knee angles, lean, and
            left-right difference.
          </caption>
          <thead>
            <tr className="border-border/60 text-muted-foreground border-b text-xs">
              <th scope="col" className="px-4 py-2.5 text-left font-medium">
                Rep
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                Start
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                Duration
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                Down / Up
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                Depth
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                Knee
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                Lean
              </th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">
                L/R diff
              </th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {reps.map((rep) => (
              <tr
                key={rep.index}
                className="border-border/40 last:border-0 border-b"
              >
                <th
                  scope="row"
                  className="px-4 py-2.5 text-left font-sans font-medium"
                >
                  {rep.index}
                  {rep.hip_below_knee && (
                    <span
                      className="ml-1.5 text-[10px] text-emerald-400"
                      title="Hip reached or passed knee level"
                    >
                      ● parallel
                    </span>
                  )}
                </th>
                <td className="px-4 py-2.5 text-right">
                  {formatSeconds(rep.start_time_s)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {formatSeconds(rep.duration_s)}
                </td>
                <td className="text-muted-foreground px-4 py-2.5 text-right">
                  {formatSeconds(rep.eccentric_s)} / {formatSeconds(rep.concentric_s)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {formatPercent(rep.depth_percent)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {formatDegrees(rep.min_knee_angle_deg)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {formatDegrees(rep.max_torso_lean_deg)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {formatDegrees(rep.knee_asymmetry_deg)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
