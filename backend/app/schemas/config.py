"""Public analysis configuration.

The live webcam mode runs the entire analysis pipeline in the browser, so the
client needs the same thresholds the server uses. Rather than hard-coding a
second copy of every number in TypeScript (which would drift the moment someone
retunes a `.env`), the client fetches them from ``GET /config``.

This is deliberately a *curated* view of ``Settings``: only the analysis
thresholds the client actually consumes. Storage paths, model URLs, CORS origins
and other operational settings are not exposed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import Settings


class ConfigResponse(BaseModel):
    """Analysis thresholds the browser needs to reproduce the pipeline live."""

    # -- Squat depth ---------------------------------------------------------
    standing_knee_angle_deg: float = Field(description="Knee angle at 0% depth.")
    parallel_knee_angle_deg: float = Field(description="Knee angle at 100% depth.")
    good_depth_percent: float
    shallow_depth_percent: float

    # -- Rep detection -------------------------------------------------------
    rep_descent_fraction: float
    rep_ascent_fraction: float
    min_rep_range: float
    rep_turnaround_band: float
    min_rep_duration_s: float

    # -- Signal processing ---------------------------------------------------
    smoothing_window_seconds: float
    max_interpolation_gap_frames: int

    # -- Camera view detection -----------------------------------------------
    view_side_max_shoulder_ratio: float
    view_front_min_shoulder_ratio: float
    landmark_visibility_threshold: float

    # -- Coaching rule thresholds -------------------------------------------
    max_torso_lean_deg: float
    max_knee_asymmetry_deg: float
    min_rep_tempo_s: float

    # -- Live coaching -------------------------------------------------------
    live_calibration_seconds: float
    bottom_pause_brief_s: float
    bottom_pause_competition_s: float
    coaching_cooldown_s: float

    @classmethod
    def from_settings(cls, settings: Settings) -> ConfigResponse:
        """Project the settings singleton onto the public field set.

        Every field is named identically to its ``Settings`` attribute, so this
        stays a one-to-one copy: adding a field here means adding one line, and
        the value can only ever come from ``config.py``.
        """
        return cls(**{name: getattr(settings, name) for name in cls.model_fields})
