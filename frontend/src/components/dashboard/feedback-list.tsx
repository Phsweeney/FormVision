import type { FeedbackItem, Severity } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Per-severity presentation.
 *
 * Each carries an icon and a text label alongside its colour, so severity is
 * never communicated by colour alone — which matters both for colour-blind
 * readers and for anyone glancing at the page.
 *
 * These are the reserved status colours, deliberately distinct from the
 * categorical series colours used in the charts, so a status can never be
 * mistaken for a data series.
 */
const SEVERITY: Record<
  Severity,
  { label: string; ring: string; text: string; icon: React.ReactNode }
> = {
  critical: {
    label: "Needs work",
    ring: "border-red-500/30 bg-red-500/5",
    text: "text-red-400",
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
        <path
          fillRule="evenodd"
          d="M10 2a8 8 0 1 0 0 16 8 8 0 0 0 0-16Zm.75 4a.75.75 0 0 0-1.5 0v4.5a.75.75 0 0 0 1.5 0V6ZM10 14.25a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
          clipRule="evenodd"
        />
      </svg>
    ),
  },
  warning: {
    label: "Watch this",
    ring: "border-amber-500/30 bg-amber-500/5",
    text: "text-amber-400",
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
        <path
          fillRule="evenodd"
          d="M8.7 2.9a1.5 1.5 0 0 1 2.6 0l6.3 11.1a1.5 1.5 0 0 1-1.3 2.25H3.7A1.5 1.5 0 0 1 2.4 14L8.7 2.9Zm2.05 4.35a.75.75 0 0 0-1.5 0v3.5a.75.75 0 0 0 1.5 0v-3.5ZM10 13.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
          clipRule="evenodd"
        />
      </svg>
    ),
  },
  info: {
    label: "Note",
    ring: "border-sky-500/30 bg-sky-500/5",
    text: "text-sky-400",
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
        <path
          fillRule="evenodd"
          d="M10 2a8 8 0 1 0 0 16 8 8 0 0 0 0-16Zm-.75 5a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0Zm1.5 3a.75.75 0 0 0-1.5 0v4a.75.75 0 0 0 1.5 0v-4Z"
          clipRule="evenodd"
        />
      </svg>
    ),
  },
  good: {
    label: "Good",
    ring: "border-emerald-500/30 bg-emerald-500/5",
    text: "text-emerald-400",
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
        <path
          fillRule="evenodd"
          d="M10 2a8 8 0 1 0 0 16 8 8 0 0 0 0-16Zm3.56 6.06a.75.75 0 1 0-1.12-1L9.2 10.74 7.55 9.09a.75.75 0 1 0-1.06 1.06l2.22 2.22a.75.75 0 0 0 1.09-.03l3.76-4.28Z"
          clipRule="evenodd"
        />
      </svg>
    ),
  },
};

/**
 * The coaching panel.
 *
 * The backend has already ordered these — problems above praise, and within
 * problems by rule priority — so this component renders the given order rather
 * than re-sorting. Ordering is a coaching decision and belongs on the server
 * with the rules that made it.
 */
export function FeedbackList({ items }: { items: FeedbackItem[] }) {
  if (items.length === 0) return null;

  return (
    <ul className="space-y-3">
      {items.map((item) => {
        const style = SEVERITY[item.severity] ?? SEVERITY.info;
        return (
          <li
            key={item.rule_id}
            className={cn("rounded-xl border p-4", style.ring)}
          >
            <div className="flex items-start gap-3">
              <span className={cn("mt-0.5 shrink-0", style.text)} aria-hidden>
                {style.icon}
              </span>

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <h3 className="text-sm font-medium">{item.title}</h3>
                  {/* The text label is what keeps severity from being
                      colour-only information. */}
                  <span
                    className={cn(
                      "rounded-full border px-1.5 py-0.5 text-[10px] font-medium",
                      style.ring,
                      style.text,
                    )}
                  >
                    {style.label}
                  </span>
                </div>

                <p className="mt-1.5 text-sm leading-relaxed">{item.message}</p>
                <p className="text-muted-foreground mt-2 text-xs leading-relaxed">
                  {item.explanation}
                </p>
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
