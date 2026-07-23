/**
 * Analysis configuration — the thresholds the engine runs on.
 *
 * These come from the backend's `GET /config`, which reads them straight from
 * `config.py`. Keeping the field names identical to the wire (and to the Python
 * settings) is deliberate: it makes the fetch a zero-mapping decode, and it lets
 * a threshold be traced from `.env` to server to browser without a rename in the
 * middle. That is why this one boundary type uses snake_case rather than the
 * camelCase of the rest of the engine.
 *
 * `DEFAULT_CONFIG` is a fallback with the same defaults as `config.py`, so the
 * live engine can still run if the config request fails (offline, backend down).
 */

import { API_BASE_URL } from "@/lib/api";

export interface AnalysisConfig {
  // Squat depth
  standing_knee_angle_deg: number;
  parallel_knee_angle_deg: number;
  good_depth_percent: number;
  shallow_depth_percent: number;
  // Rep detection
  rep_descent_fraction: number;
  rep_ascent_fraction: number;
  min_rep_range: number;
  min_rep_duration_s: number;
  // Signal processing
  smoothing_window_seconds: number;
  max_interpolation_gap_frames: number;
  // Camera view detection
  view_side_max_shoulder_ratio: number;
  view_front_min_shoulder_ratio: number;
  landmark_visibility_threshold: number;
  // Coaching rule thresholds
  max_torso_lean_deg: number;
  max_knee_asymmetry_deg: number;
  min_rep_tempo_s: number;
  // Live coaching
  live_calibration_seconds: number;
  bottom_pause_brief_s: number;
  bottom_pause_competition_s: number;
  coaching_cooldown_s: number;
}

/** Mirrors the defaults in `backend/app/config.py`. */
export const DEFAULT_CONFIG: AnalysisConfig = {
  standing_knee_angle_deg: 170,
  parallel_knee_angle_deg: 90,
  good_depth_percent: 90,
  shallow_depth_percent: 70,
  rep_descent_fraction: 0.6,
  rep_ascent_fraction: 0.25,
  min_rep_range: 0.15,
  min_rep_duration_s: 0.4,
  smoothing_window_seconds: 0.15,
  max_interpolation_gap_frames: 5,
  view_side_max_shoulder_ratio: 0.2,
  view_front_min_shoulder_ratio: 0.32,
  landmark_visibility_threshold: 0.5,
  max_torso_lean_deg: 45,
  max_knee_asymmetry_deg: 12,
  min_rep_tempo_s: 1.2,
  live_calibration_seconds: 2.5,
  bottom_pause_brief_s: 0.3,
  bottom_pause_competition_s: 1.0,
  coaching_cooldown_s: 4.0,
};

/**
 * Fetch the live analysis thresholds. Falls back to the compiled-in defaults so
 * a missing or unreachable backend degrades to "uses defaults", never "live
 * mode is broken".
 */
export async function fetchConfig(signal?: AbortSignal): Promise<AnalysisConfig> {
  try {
    const response = await fetch(`${API_BASE_URL}/config`, {
      cache: "no-store",
      signal,
    });
    if (!response.ok) return DEFAULT_CONFIG;
    return (await response.json()) as AnalysisConfig;
  } catch {
    return DEFAULT_CONFIG;
  }
}
