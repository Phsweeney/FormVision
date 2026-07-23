/**
 * A causal moving average over the most recent N values.
 *
 * The batch pipeline uses a *centred* window (it can see future frames); a live
 * feed cannot, so live signals are smoothed with this trailing window instead.
 * It keeps the same "missing stays missing" rule as the batch smoother: a null
 * input produces a null output, so an untracked frame is never invented.
 */
export class TrailingAverage {
  private readonly buffer: (number | null)[] = [];

  constructor(private readonly window: number) {}

  /** Push the next raw value; get back the trailing mean of present values. */
  push(value: number | null): number | null {
    this.buffer.push(value);
    if (this.buffer.length > this.window) this.buffer.shift();
    if (value === null) return null;

    let sum = 0;
    let count = 0;
    for (const v of this.buffer) {
      if (v !== null) {
        sum += v;
        count += 1;
      }
    }
    return count > 0 ? sum / count : null;
  }

  reset(): void {
    this.buffer.length = 0;
  }
}
