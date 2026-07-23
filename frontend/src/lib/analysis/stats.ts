/**
 * Small numeric reductions the engine needs, matching numpy's conventions so
 * results line up with the Python side.
 */

/** Mean, or null for an empty list. */
export function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Median with numpy's even-length rule (average of the two middle values). */
export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Population standard deviation (divide by N, as numpy does), or null for fewer
 * than two values — one sample has zero deviation by definition, which would
 * read as perfect consistency the data does not support.
 */
export function populationStd(values: number[]): number | null {
  if (values.length < 2) return null;
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((sum, v) => sum + (v - m) * (v - m), 0) / values.length;
  return Math.sqrt(variance);
}
