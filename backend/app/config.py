"""Application configuration.

Every tunable value in FormVision lives here — upload limits, analysis
thresholds, coaching rule sensitivities, storage locations. Analysis modules
receive these values as arguments; they never define thresholds themselves.

That rule exists so a coach can retune the app for, say, a powerlifting
standard versus a general-fitness standard by editing a `.env` file, without
touching or redeploying any analysis code.

Values are read from the environment (or a `.env` file) with the ``FORMVISION_``
prefix, e.g. ``FORMVISION_PARALLEL_KNEE_ANGLE_DEG=95``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory containing the `app` package, i.e. the `backend/` folder.
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings, validated at startup."""

    model_config = SettingsConfigDict(
        env_prefix="FORMVISION_",
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Application ---------------------------------------------------------
    app_name: str = "FormVision API"
    app_version: str = "1.0.0"
    environment: str = Field(
        default="development",
        description="One of: development, production.",
    )
    log_level: str = "INFO"

    # -- HTTP ----------------------------------------------------------------
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Origins permitted to call the API from a browser.",
    )

    # -- Storage -------------------------------------------------------------
    data_dir: Path = Field(
        default=BACKEND_ROOT / "data",
        description="Root for all runtime artefacts (videos, overlays, landmarks).",
    )
    database_url: str = Field(
        default="",
        description="SQLAlchemy URL. Defaults to a SQLite file inside data_dir.",
    )

    # -- Upload validation ---------------------------------------------------
    max_upload_bytes: int = Field(
        default=200 * 1024 * 1024,
        description="Reject uploads larger than this (default 200 MB).",
    )
    max_video_duration_s: float = Field(
        default=60.0,
        description="Reject videos longer than this. Bounds worst-case analysis time.",
    )
    allowed_video_extensions: list[str] = Field(default=[".mp4", ".mov"])
    allowed_content_types: list[str] = Field(
        default=[
            "video/mp4",
            "video/quicktime",
            "video/x-quicktime",
            "application/octet-stream",  # some browsers send this for .mov
        ]
    )

    # -- Pose estimation -----------------------------------------------------
    pose_estimator: str = Field(
        default="mediapipe",
        description="Which registered PoseEstimator implementation to use.",
    )
    pose_model_path: Path = Field(
        default=BACKEND_ROOT / "models" / "pose_landmarker_lite.task",
        description="Local path to the MediaPipe pose landmarker bundle.",
    )
    pose_model_url: str = Field(
        default=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        ),
        description="Downloaded once on first run if pose_model_path is missing.",
    )
    pose_inference_width: int = Field(
        default=640,
        description=(
            "Longest side, in pixels, that frames are downscaled to before pose "
            "inference. Landmarks are normalised (0-1), so they map back to the "
            "original resolution for free. Purely a speed lever."
        ),
    )
    landmark_visibility_threshold: float = Field(
        default=0.5,
        description="Below this visibility a landmark is treated as not detected.",
    )
    min_tracking_quality: float = Field(
        default=0.6,
        description="Fraction of usable frames below which results are flagged.",
    )

    # -- Camera view detection -----------------------------------------------
    # Shoulder separation divided by torso length: scale-free, so it is
    # independent of camera distance and body size. Measured across the sample
    # footage, the two orientations are nowhere near each other — 0.06 to 0.07
    # filmed side-on, 0.40 to 1.27 filmed front-on — so these thresholds sit in
    # a wide empty band rather than on a knife edge.
    view_side_max_shoulder_ratio: float = Field(
        default=0.20,
        description=(
            "Shoulder separation, in torso lengths, at or below which the "
            "camera is treated as side-on. Side-on clips measure around 0.06."
        ),
    )
    view_front_min_shoulder_ratio: float = Field(
        default=0.32,
        description=(
            "Shoulder separation, in torso lengths, at or above which the "
            "camera is treated as front-on. Front-on clips measure 0.40 and up. "
            "Between the two thresholds the view is reported as oblique."
        ),
    )

    # -- Signal processing ---------------------------------------------------
    smoothing_window_seconds: float = Field(
        default=0.15,
        description="Width of the centred moving average applied to signals.",
    )
    max_interpolation_gap_frames: int = Field(
        default=5,
        description="Longest run of untracked frames that is bridged by interpolation.",
    )

    # -- Rep detection -------------------------------------------------------
    rep_descent_fraction: float = Field(
        default=0.60,
        description=(
            "Fraction of the hip-height range that must be travelled downward to "
            "count as descending. Higher = stricter, fewer false reps."
        ),
    )
    rep_ascent_fraction: float = Field(
        default=0.25,
        description=(
            "Hip must return above this fraction of the range to close a rep. "
            "The gap between this and rep_descent_fraction is the hysteresis "
            "band that stops a wobble at the threshold registering many reps."
        ),
    )
    min_rep_range: float = Field(
        default=0.15,
        description=(
            "Minimum hip-height travel, in torso lengths, for a clip to contain "
            "any reps at all. Guards against noise in a standing-still video "
            "being amplified into phantom repetitions."
        ),
    )
    min_rep_duration_s: float = Field(
        default=0.40,
        description="Reps faster than this are treated as tracking jitter.",
    )

    # -- Squat depth ---------------------------------------------------------
    standing_knee_angle_deg: float = Field(
        default=170.0,
        description="Knee angle treated as fully extended, i.e. 0% depth.",
    )
    parallel_knee_angle_deg: float = Field(
        default=90.0,
        description="Knee angle treated as parallel, i.e. 100% depth.",
    )
    good_depth_percent: float = Field(
        default=90.0,
        description="Depth at or above this is praised as full depth.",
    )
    shallow_depth_percent: float = Field(
        default=70.0,
        description="Depth below this triggers the insufficient-depth rule.",
    )

    # -- Coaching rule thresholds -------------------------------------------
    max_torso_lean_deg: float = Field(
        default=45.0,
        description="Forward lean beyond this at the bottom is flagged.",
    )
    max_knee_asymmetry_deg: float = Field(
        default=12.0,
        description="Left/right knee angle difference beyond this is flagged.",
    )
    max_depth_variation_percent: float = Field(
        default=12.0,
        description="Std-dev of per-rep depth above this is called inconsistent.",
    )
    max_duration_variation_s: float = Field(
        default=0.75,
        description="Std-dev of per-rep duration above this is called inconsistent.",
    )
    min_rep_tempo_s: float = Field(
        default=1.2,
        description="Reps quicker than this on average are called rushed.",
    )

    # -- Overlay rendering ---------------------------------------------------
    overlay_dim_visibility_threshold: float = Field(
        default=0.10,
        description=(
            "Landmarks below this visibility are not drawn at all. Between this "
            "and landmark_visibility_threshold they are drawn dimmed, which is "
            "how an occluded far-side limb stays on screen without pretending "
            "to be as well observed as the near side."
        ),
    )

    # -- API response shaping ------------------------------------------------
    max_series_points: int = Field(
        default=600,
        description=(
            "Time series are decimated to at most this many points before being "
            "returned. Full-resolution landmarks stay on disk; a 60 s clip at "
            "60 fps would otherwise ship 3600 points per series to the browser."
        ),
    )

    # -- Derived paths -------------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def overlay_dir(self) -> Path:
        return self.data_dir / "overlays"

    @property
    def landmark_dir(self) -> Path:
        return self.data_dir / "landmarks"

    @property
    def resolved_database_url(self) -> str:
        """Database URL, defaulting to a SQLite file under ``data_dir``."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'formvision.db').as_posix()}"

    @field_validator("parallel_knee_angle_deg")
    @classmethod
    def _parallel_below_standing(cls, value: float, info) -> float:
        standing = info.data.get("standing_knee_angle_deg")
        if standing is not None and value >= standing:
            raise ValueError(
                "parallel_knee_angle_deg must be smaller than "
                "standing_knee_angle_deg (knees bend to a smaller angle)"
            )
        return value

    @field_validator("view_front_min_shoulder_ratio")
    @classmethod
    def _front_above_side(cls, value: float, info) -> float:
        side = info.data.get("view_side_max_shoulder_ratio")
        if side is not None and value <= side:
            raise ValueError(
                "view_front_min_shoulder_ratio must be larger than "
                "view_side_max_shoulder_ratio; the gap between them is the "
                "band classified as oblique"
            )
        return value

    @field_validator("overlay_dim_visibility_threshold")
    @classmethod
    def _dim_below_visible(cls, value: float, info) -> float:
        visible = info.data.get("landmark_visibility_threshold")
        if visible is not None and value > visible:
            raise ValueError(
                "overlay_dim_visibility_threshold must not exceed "
                "landmark_visibility_threshold, or confidently-detected "
                "landmarks would be dropped from the overlay entirely"
            )
        return value

    @field_validator("rep_ascent_fraction")
    @classmethod
    def _ascent_below_descent(cls, value: float, info) -> float:
        descent = info.data.get("rep_descent_fraction")
        if descent is not None and value >= descent:
            raise ValueError(
                "rep_ascent_fraction must be smaller than rep_descent_fraction "
                "so the two thresholds form a hysteresis band"
            )
        return value

    def ensure_directories(self) -> None:
        """Create every runtime directory. Safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.upload_dir,
            self.overlay_dir,
            self.landmark_dir,
            self.pose_model_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the `.env` file is parsed once. Tests clear the cache via
    ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()
