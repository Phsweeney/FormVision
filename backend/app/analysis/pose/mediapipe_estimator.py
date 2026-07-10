"""MediaPipe Pose implementation of `PoseEstimator`.

This is the only module in the project that imports MediaPipe. Everything the
library's API quirks impose — model bundles, BGR/RGB conversion, millisecond
timestamps — is absorbed here so the rest of the codebase sees only plain
dataclasses.

A note on the API used: MediaPipe **0.10.35 removed the legacy
``mp.solutions.pose`` API entirely**. The Tasks API is the only option, which
means the model weights are not bundled with the package and must be fetched
once and cached on disk.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.analysis.pose.base import PoseEstimator
from app.analysis.types import FramePose, Landmark, PoseSeries, VideoMetadata
from app.config import Settings, get_settings
from app.core.exceptions import PoseEstimationError, VideoProcessingError
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Used when a video reports a nonsensical frame rate (some phone exports and
#: most webcam captures do). 30 fps is the safest guess for phone footage.
_FALLBACK_FPS = 30.0

#: MediaPipe rejects a non-monotonic timestamp in VIDEO mode, and constant-rate
#: video can produce duplicate integer milliseconds at high frame rates.
_MIN_TIMESTAMP_STEP_MS = 1


class MediaPipePoseEstimator(PoseEstimator):
    """Extracts 33-point body landmarks using MediaPipe Pose Landmarker."""

    name = "mediapipe"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._landmarker: vision.PoseLandmarker | None = None

    # -- Model management ----------------------------------------------------

    def _ensure_model(self) -> Path:
        """Return the model bundle path, downloading it if absent.

        The download happens lazily on first use rather than at startup so the
        API can boot (and report its health) on a machine with no network.
        """
        path = self.settings.pose_model_path
        if path.exists() and path.stat().st_size > 0:
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        url = self.settings.pose_model_url
        logger.info("Pose model missing; downloading from %s", url)

        # Download to a temporary name and rename on success, so an interrupted
        # download can never leave a truncated file that looks valid.
        temp_path = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, temp_path)
            temp_path.replace(path)
        except (urllib.error.URLError, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            raise PoseEstimationError(
                "Could not download the pose model. Check network connectivity, "
                "or place the model bundle at the configured path manually.",
                detail={"url": url, "path": str(path)},
            ) from exc

        logger.info("Pose model cached at %s (%d bytes)", path, path.stat().st_size)
        return path

    def _create_landmarker(self) -> vision.PoseLandmarker:
        model_path = self._ensure_model()
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                # VIDEO mode (rather than IMAGE) lets MediaPipe carry tracking
                # state between frames, which is both faster and steadier than
                # detecting from scratch every frame.
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            return vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise PoseEstimationError(
                "Failed to initialise the MediaPipe pose landmarker.",
                detail={"model_path": str(model_path)},
            ) from exc

    # -- Estimation ----------------------------------------------------------

    def estimate(self, video_path: Path) -> PoseSeries:
        """Run pose estimation over every frame of ``video_path``."""
        if not video_path.is_file():
            raise VideoProcessingError(
                "Video file not found.", detail={"path": video_path.name}
            )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoProcessingError(
                "The video could not be opened. It may be corrupt or use an "
                "unsupported codec.",
                detail={"path": video_path.name},
            )

        started = time.perf_counter()
        try:
            metadata = self._read_metadata(capture)
            frames = self._process_frames(capture, metadata)
        finally:
            capture.release()
            self.close()

        series = PoseSeries(
            frames=tuple(frames),
            metadata=metadata,
            estimator_name=self.name,
        )
        logger.info(
            "Pose estimation finished: %d frames, %.1f%% detected, %.1fs elapsed",
            len(frames),
            series.detection_rate * 100,
            time.perf_counter() - started,
        )
        return series

    def _read_metadata(self, capture: cv2.VideoCapture) -> VideoMetadata:
        """Read video properties, repairing values that containers get wrong."""
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # A missing or absurd fps is common in phone and webcam exports, and
        # would otherwise propagate into every duration and tempo metric.
        if not fps or fps <= 0 or fps > 480:
            logger.warning(
                "Video reported fps=%s; falling back to %s", fps, _FALLBACK_FPS
            )
            fps = _FALLBACK_FPS

        if frame_count <= 0:
            frame_count = 0  # unknown; the real count is measured while decoding

        if width <= 0 or height <= 0:
            raise VideoProcessingError("The video reports invalid dimensions.")

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=frame_count / fps if frame_count else 0.0,
        )

    def _process_frames(
        self, capture: cv2.VideoCapture, metadata: VideoMetadata
    ) -> list[FramePose]:
        """Decode and run inference frame by frame."""
        self._landmarker = self._create_landmarker()
        scale = self._inference_scale(metadata)

        frames: list[FramePose] = []
        frame_index = 0
        last_timestamp_ms = -1

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            timestamp_s = frame_index / metadata.fps
            # Strictly increasing integer milliseconds: MediaPipe's VIDEO mode
            # raises if a timestamp repeats, which happens above 1000 fps and
            # after rounding on some variable-rate files.
            timestamp_ms = max(
                int(timestamp_s * 1000),
                last_timestamp_ms + _MIN_TIMESTAMP_STEP_MS,
            )
            last_timestamp_ms = timestamp_ms

            frames.append(
                self._detect_frame(
                    frame_bgr, frame_index, timestamp_s, timestamp_ms, scale
                )
            )
            frame_index += 1

        if not frames:
            raise VideoProcessingError("The video contained no readable frames.")

        return frames

    def _inference_scale(self, metadata: VideoMetadata) -> float:
        """Factor to shrink frames by before inference.

        Landmarks come back normalised to [0, 1], so downscaling costs no
        accuracy in the coordinate system and only affects how much detail the
        model sees. Upscaling is never useful, hence the clamp at 1.0.
        """
        longest = max(metadata.width, metadata.height)
        target = self.settings.pose_inference_width
        return min(1.0, target / longest) if longest > target else 1.0

    def _detect_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        timestamp_ms: int,
        scale: float,
    ) -> FramePose:
        """Run the landmarker on one frame."""
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (
                    max(1, int(frame_bgr.shape[1] * scale)),
                    max(1, int(frame_bgr.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        # OpenCV decodes BGR; MediaPipe expects RGB. Getting this backwards does
        # not error, it just quietly degrades detection accuracy.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            # One bad frame must not abandon the whole video; record it as
            # undetected and carry on.
            logger.warning("Pose detection failed on frame %d: %s", frame_index, exc)
            return FramePose(frame_index, timestamp_s, (), detected=False)

        if not result.pose_landmarks:
            return FramePose(frame_index, timestamp_s, (), detected=False)

        landmarks = tuple(
            Landmark(
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
                # `visibility` can be None on some builds; treat that as fully
                # visible rather than discarding an otherwise good frame.
                visibility=(
                    float(point.visibility) if point.visibility is not None else 1.0
                ),
            )
            for point in result.pose_landmarks[0]
        )
        return FramePose(frame_index, timestamp_s, landmarks, detected=True)

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:  # pragma: no cover - native teardown
                logger.debug("Landmarker close raised: %s", exc)
            self._landmarker = None
