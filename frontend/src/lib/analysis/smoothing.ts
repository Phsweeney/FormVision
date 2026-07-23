/**
 * Signal smoothing and gap filling — the TS mirror of `smoothing.py`.
 *
 * Raw landmarks jitter frame to frame, and rep detection compares a signal
 * against thresholds, so untreated jitter around a threshold invents reps.
 * Every function works on `(number | null)[]` where null means "not measured
 * this frame", and preserves length so series stay index-aligned.
 *
 * This is the batch (centred) treatment, matching the backend exactly, used by
 * `computeAngles` over a full frame array and by the parity tests. The live
 * driver applies a causal variant instead (see `live/`), because a centred
 * window needs frames from the future that a live feed does not have yet.
 */

export type Series = (number | null)[];

/** Convert a duration into an odd frame count (odd = symmetric, no phase shift). */
export function windowSizeForFps(fps: number, seconds: number, minimum = 3): number {
  if (fps <= 0 || seconds <= 0) return minimum;
  let size = Math.max(Math.round(fps * seconds), minimum);
  if (size % 2 === 0) size += 1;
  return size;
}

/**
 * Linearly fill runs of null no longer than `maxGap`. Short dropouts (a briefly
 * occluded limb) are bridged; long ones are left missing rather than fabricated.
 * Leading and trailing gaps are never filled — nothing to interpolate between.
 */
export function interpolateGaps(values: Series, maxGap: number): Series {
  if (maxGap <= 0) return [...values];

  const result = [...values];
  const known: number[] = [];
  result.forEach((value, i) => {
    if (value !== null) known.push(i);
  });
  if (known.length < 2) return result;

  for (let k = 0; k < known.length - 1; k++) {
    const left = known[k];
    const right = known[k + 1];
    const gap = right - left - 1;
    if (gap <= 0 || gap > maxGap) continue;
    const startValue = result[left] as number;
    const endValue = result[right] as number;
    const step = (endValue - startValue) / (gap + 1);
    for (let offset = 1; offset <= gap; offset++) {
      result[left + offset] = startValue + step * offset;
    }
  }
  return result;
}

/**
 * Centred moving average that tolerates missing values. Each output is the mean
 * of the present values in the window centred on that index; positions that were
 * null stay null (smoothing must not resurrect an untracked frame). Windows are
 * truncated at the ends rather than padded.
 */
export function movingAverage(values: Series, window: number): Series {
  if (window <= 1 || values.length === 0) return [...values];

  const half = Math.floor(window / 2);
  const n = values.length;
  const result: Series = new Array(n).fill(null);

  for (let i = 0; i < n; i++) {
    if (values[i] === null) continue;
    let sum = 0;
    let count = 0;
    const start = Math.max(i - half, 0);
    const end = Math.min(i + half, n - 1);
    for (let j = start; j <= end; j++) {
      const value = values[j];
      if (value !== null) {
        sum += value;
        count += 1;
      }
    }
    result[i] = count > 0 ? sum / count : null;
  }
  return result;
}

/** Fill short gaps, then smooth. The standard treatment for every batch signal. */
export function smoothSeries(
  values: Series,
  fps: number,
  seconds: number,
  maxGap: number,
): Series {
  return movingAverage(
    interpolateGaps(values, maxGap),
    windowSizeForFps(fps, seconds),
  );
}

/**
 * Percentile of the present values (linear interpolation, matching numpy's
 * default), or null if there are none. Rep detection uses the 10th/90th rather
 * than min/max so one mis-tracked extreme frame cannot set the whole scale.
 */
export function percentile(values: Series, q: number): number | null {
  const present = values
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b);
  if (present.length === 0) return null;
  if (present.length === 1) return present[0];

  const rank = (q / 100) * (present.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return present[lo];
  return present[lo] * (hi - rank) + present[hi] * (rank - lo);
}
