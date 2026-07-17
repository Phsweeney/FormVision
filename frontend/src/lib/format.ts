/**
 * Display formatting helpers.
 *
 * Every one of these treats `null` as "not measured" and renders an em dash,
 * never `0`. The backend is careful to distinguish the two — a set with no reps
 * has *no* average depth rather than 0% — and that distinction has to survive
 * all the way to the screen, or the UI reports a failure to measure as a
 * catastrophically bad lift.
 */

/** Placeholder for a value that could not be measured. */
export const NOT_MEASURED = "—";

export function formatNumber(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_MEASURED;
  }
  return value.toFixed(digits);
}

export function formatPercent(
  value: number | null | undefined,
  digits = 0,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_MEASURED;
  }
  return `${value.toFixed(digits)}%`;
}

export function formatDegrees(
  value: number | null | undefined,
  digits = 0,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_MEASURED;
  }
  return `${value.toFixed(digits)}°`;
}

export function formatSeconds(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_MEASURED;
  }
  return `${value.toFixed(digits)}s`;
}

/** Seconds as `m:ss`, for durations long enough that raw seconds read poorly. */
export function formatDuration(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return NOT_MEASURED;
  }
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** A `Date` rendered in the viewer's own locale and timezone. */
export function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return NOT_MEASURED;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
