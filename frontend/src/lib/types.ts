/**
 * API types, mirroring `backend/app/schemas/analysis.py`.
 *
 * This file is the single typed surface between frontend and backend. Nothing
 * else in the app should describe an API shape — if a field changes on the
 * server, it changes here once and TypeScript points at every affected line.
 *
 * Field names keep the backend's unit suffixes (`_s`, `_deg`, `_percent`)
 * rather than being camel-cased. Renaming would mean a translation layer that
 * could silently drift from the schema, and the suffixes are the reason nobody
 * has to guess whether a duration is seconds or milliseconds.
 */

export type AnalysisStatus = "uploaded" | "processing" | "completed" | "failed";

export type Severity = "good" | "info" | "warning" | "critical";

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_s: number;
}

export interface UploadResponse {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: AnalysisStatus;
  created_at: string;
}

export interface AnalyzeResponse {
  id: string;
  status: AnalysisStatus;
  message: string;
}

export interface Rep {
  index: number;
  start_time_s: number;
  bottom_time_s: number;
  end_time_s: number;
  duration_s: number;
  /** Descent phase, in seconds. */
  eccentric_s: number;
  /** Ascent phase, in seconds. */
  concentric_s: number;

  start_frame: number;
  bottom_frame: number;
  end_frame: number;

  min_knee_angle_deg: number | null;
  min_left_knee_deg: number | null;
  min_right_knee_deg: number | null;
  min_hip_angle_deg: number | null;
  max_torso_lean_deg: number | null;
  knee_asymmetry_deg: number | null;
  /** 0–100, where 100 means the configured depth target was reached. */
  depth_percent: number | null;
  hip_below_knee: boolean;
}

export interface Metrics {
  total_reps: number;
  video_duration_s: number;
  total_workout_time_s: number;

  max_depth_percent: number | null;
  avg_depth_percent: number | null;
  min_knee_angle_deg: number | null;

  avg_rep_duration_s: number | null;
  fastest_rep_s: number | null;
  slowest_rep_s: number | null;
  avg_eccentric_s: number | null;
  avg_concentric_s: number | null;
  reps_per_minute: number | null;

  avg_torso_lean_deg: number | null;
  max_torso_lean_deg: number | null;
  avg_knee_asymmetry_deg: number | null;

  /** Standard deviation of per-rep depth. Lower is more consistent. */
  depth_consistency_percent: number | null;
  /** Standard deviation of per-rep duration. Lower is more consistent. */
  duration_consistency_s: number | null;

  /** Fraction of frames usable for analysis, 0–1. */
  tracking_quality: number;

  /**
   * Which camera angle the clip was filmed from.
   *
   * Explains the nulls above rather than decorating them: a 2D analysis cannot
   * see forward lean from the front, nor compare left against right from the
   * side, so whichever one the camera missed comes back null by design.
   */
  camera_view: CameraView;
}

export type CameraView = "side" | "front" | "oblique" | "unknown";

export interface FeedbackItem {
  /** Stable identifier — safe to key UI logic on, unlike the message text. */
  rule_id: string;
  severity: Severity;
  title: string;
  message: string;
  explanation: string;
}

/**
 * Time series for charting.
 *
 * Every array is the same length and index-aligned with `time_s`. `null` marks
 * a frame where the value could not be measured; charts render those as gaps,
 * which is the honest representation of a tracking dropout rather than
 * drawing a line through data that was never observed.
 */
export interface Series {
  time_s: number[];
  left_knee_deg: (number | null)[];
  right_knee_deg: (number | null)[];
  hip_deg: (number | null)[];
  torso_lean_deg: (number | null)[];
  hip_height: (number | null)[];
  sample_count: number;
  source_frame_count: number;
}

/**
 * `GET /analysis/{id}`.
 *
 * One shape covers every status. While processing, the result fields are null
 * and only `status` carries information — so the polling client checks one
 * field rather than branching on which of several shapes came back.
 */
export interface Analysis {
  id: string;
  status: AnalysisStatus;
  filename: string;
  created_at: string;
  updated_at: string;

  error_code: string | null;
  error_message: string | null;
  processing_seconds: number | null;

  video: VideoInfo | null;
  metrics: Metrics | null;
  reps: Rep[] | null;
  feedback: FeedbackItem[] | null;
  series: Series | null;

  video_url: string | null;
  overlay_url: string | null;
  estimator: string | null;
}

/** Every non-2xx response from the API uses this envelope. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    detail: Record<string, unknown>;
  };
}

/** True once results are available to render. */
export function isComplete(analysis: Analysis): boolean {
  return analysis.status === "completed" && analysis.metrics !== null;
}

/** True while the client should keep polling. */
export function isPending(analysis: Analysis): boolean {
  return analysis.status === "uploaded" || analysis.status === "processing";
}
