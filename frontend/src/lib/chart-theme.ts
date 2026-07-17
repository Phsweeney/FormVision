/**
 * Chart colour and geometry tokens.
 *
 * One definition for every chart, so the same measurement is the same colour
 * everywhere on the dashboard. Colour follows the *entity* — the left knee is
 * always blue, the right knee always orange — never the series' position in a
 * list, so a chart rendering fewer series never repaints the survivors.
 *
 * These are categorical slots stepped for a dark surface. The set was validated
 * against the actual chart surface (`#131313`, which is `bg-card/40` over the
 * page background) rather than assumed:
 *
 *   Lightness band      all inside L 0.48–0.67   PASS
 *   Chroma floor        all >= 0.1               PASS
 *   CVD separation      worst pair ΔE 9.4        PASS  (target >= 8)
 *   Normal-vision floor worst pair ΔE 24.6       PASS  (floor >= 15)
 *   Contrast vs surface all >= 3:1               PASS
 *
 * Every multi-series chart also carries a legend, so identity is never
 * conveyed by colour alone.
 */

export const SERIES_COLORS = {
  /** Categorical slot 1. */
  leftKnee: "#3987e5",
  /** Categorical slot 2. */
  rightKnee: "#d95926",
  /** Categorical slot 3. */
  hip: "#199e70",
  /** Categorical slot 7. */
  hipHeight: "#9085e9",
} as const;

/** Chart chrome. Recessive by design — the data should carry the emphasis. */
export const CHART_INK = {
  grid: "#2c2c2a",
  axis: "#898781",
  tooltipSurface: "#1a1a19",
  tooltipBorder: "#3a3a38",
  /** Shading behind each detected repetition. */
  repBand: "#ffffff",
  repBandOpacity: 0.05,
} as const;

/** Shared geometry, so all three charts read as one system. */
export const CHART_GEOMETRY = {
  /** 2px lines, per the mark spec. */
  strokeWidth: 2,
  height: 240,
  fontSize: 11,
} as const;
