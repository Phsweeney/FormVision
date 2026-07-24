/**
 * The continuous coaching engine.
 *
 * Evaluates each completed rep and emits at most one short cue for it, with a
 * per-cue cooldown so the same advice cannot repeat every rep. The copy is
 * deliberately terse — these are spoken aloud mid-set, so "Go deeper" beats a
 * sentence. Problems are prioritised over praise, and any signal the current
 * camera view cannot see (lean front-on, asymmetry side-on) is simply absent
 * from the rep, so those cues stay silent by construction — the same honesty as
 * the offline coaching rules.
 *
 * This is intentionally a small, rule-based engine behind a plain method. A
 * future ML judge could replace `evaluateRep` without any other module noticing.
 */

import type { AnalysisConfig } from "@/lib/analysis/config";
import type { Severity } from "@/lib/analysis/types";

import type { LiveRep } from "./rep-analysis";
import type { LiveState } from "./live-analyzer";

export interface Cue {
  /** Stable id, also the cooldown key. */
  id: string;
  text: string;
  severity: Severity;
}

export class Coach {
  private readonly lastFiredS: Record<string, number> = {};
  private lastRepIndex = 0;

  constructor(private readonly config: AnalysisConfig) {}

  /**
   * Feed the current live state. Returns a cue to surface and speak, or null.
   * Only fires when a new rep has just completed.
   */
  update(state: LiveState, nowS: number): Cue | null {
    const rep = state.lastRep;
    if (!rep || rep.index <= this.lastRepIndex) return null;
    this.lastRepIndex = rep.index;
    return this.evaluateRep(rep, nowS);
  }

  reset(): void {
    for (const key of Object.keys(this.lastFiredS)) delete this.lastFiredS[key];
    this.lastRepIndex = 0;
  }

  private evaluateRep(rep: LiveRep, nowS: number): Cue | null {
    // Problems first, most actionable first; then praise.
    if (rep.halfRep) return this.fire("go-deeper", "Go deeper", "warning", nowS);

    if (
      rep.maxTorsoLeanDeg !== null &&
      rep.maxTorsoLeanDeg > this.config.max_torso_lean_deg
    ) {
      return this.fire("chest-up", "Chest up", "warning", nowS);
    }

    if (
      rep.kneeAsymmetryDeg !== null &&
      rep.kneeAsymmetryDeg > this.config.max_knee_asymmetry_deg
    ) {
      return this.fire("balance", "Stay even", "warning", nowS);
    }

    const duration = rep.endTimeS - rep.startTimeS;
    if (duration < this.config.min_rep_tempo_s) {
      return this.fire("slow-down", "Control it", "info", nowS);
    }

    if (
      rep.depthPercent !== null &&
      rep.depthPercent >= this.config.good_depth_percent
    ) {
      return this.fire("nice-rep", "Nice rep", "good", nowS);
    }

    return null;
  }

  /** Emit a cue unless the same one fired within the cooldown window. */
  private fire(
    id: string,
    text: string,
    severity: Severity,
    nowS: number,
  ): Cue | null {
    const last = this.lastFiredS[id] ?? Number.NEGATIVE_INFINITY;
    if (nowS - last < this.config.coaching_cooldown_s) return null;
    this.lastFiredS[id] = nowS;
    return { id, text, severity };
  }
}
