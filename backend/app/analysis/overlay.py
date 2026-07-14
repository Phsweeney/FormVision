"""Skeleton overlay video rendering.

Redraws the source video with the detected skeleton, live joint angles, a rep
counter, and a depth bar burned in.

**Encoding is the part worth reading.** OpenCV's `VideoWriter` with the default
``mp4v`` fourcc produces MPEG-4 Part 2. That writes a perfectly valid `.mp4`
file and raises no error — and then Chrome refuses to play it, giving you a
black `<video>` element with no diagnostic. Browsers want H.264.

OpenCV's pip wheels cannot encode H.264 (the `avc1` codec is omitted for
licensing reasons). So frames are piped to the ffmpeg binary bundled with
`imageio-ffmpeg`, which encodes H.264 / yuv420p with `+faststart`. The
`yuv420p` pixel format matters as much as the codec: browsers will not decode
`yuv444p`. `+faststart` relocates the index to the front of the file so
playback can begin before the download finishes.

The OpenCV writer is kept as a logged fallback for environments without the
ffmpeg binary — a possibly-unplayable overlay beats no overlay at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.analysis.types import (
    POSE_CONNECTIONS,
    AngleSeries,
    PoseSeries,
    Rep,
)
from app.analysis.types import (
    PoseLandmarkIndex as LM,
)
from app.config import Settings
from app.core.exceptions import VideoProcessingError
from app.logging_config import get_logger

logger = get_logger(__name__)

# BGR colours (OpenCV convention, not RGB).
_COLOR_BONE = (235, 206, 135)  # warm blue
_COLOR_JOINT = (255, 255, 255)
_COLOR_ACCENT = (120, 220, 120)  # green
_COLOR_WARN = (80, 120, 255)  # red-orange
_COLOR_TEXT = (255, 255, 255)
_COLOR_PANEL = (32, 28, 24)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_overlay(
    video_path: Path,
    output_path: Path,
    pose: PoseSeries,
    angles: AngleSeries,
    reps: list[Rep],
    settings: Settings,
) -> Path:
    """Render the annotated video and return the written path."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoProcessingError("Could not reopen the video to render the overlay.")

    # Dimensions come from the capture we are actually decoding, not from
    # `pose.metadata`. The encoder is fed raw frames, so it must be told their
    # true size: if the two ever disagree, every frame is misread at the wrong
    # stride and the output silently becomes garbage rather than failing.
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or pose.metadata.width
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or pose.metadata.height
    fps = pose.metadata.fps
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Which rep, if any, each frame belongs to — precomputed so the per-frame
    # loop stays O(1) rather than scanning the rep list for every frame.
    frame_to_rep = _build_frame_index(reps, len(pose.frames))

    writer = _open_writer(output_path, width, height, fps)
    try:
        for index in range(len(pose.frames)):
            ok, frame = capture.read()
            if not ok:
                break

            # The raw pipe has no framing, so a single wrong-sized frame would
            # desynchronise every frame after it. Cheap insurance.
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            _draw_frame(frame, index, pose, angles, reps, frame_to_rep, settings)
            writer.write(frame)
    finally:
        capture.release()
        writer.close()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoProcessingError("Overlay rendering produced no output.")

    logger.info(
        "Overlay written to %s (%.1f MB)",
        output_path.name,
        output_path.stat().st_size / 1024 / 1024,
    )
    return output_path


def _build_frame_index(reps: list[Rep], frame_count: int) -> list[int | None]:
    """Map each frame index to the 1-based rep it belongs to, or None."""
    mapping: list[int | None] = [None] * frame_count
    for rep in reps:
        for frame in range(rep.start_frame, min(rep.end_frame + 1, frame_count)):
            mapping[frame] = rep.index
    return mapping


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _draw_frame(
    frame: np.ndarray,
    index: int,
    pose: PoseSeries,
    angles: AngleSeries,
    reps: list[Rep],
    frame_to_rep: list[int | None],
    settings: Settings,
) -> None:
    """Annotate one frame in place."""
    height, width = frame.shape[:2]
    frame_pose = pose.frames[index]

    if frame_pose.detected and frame_pose.landmarks:
        _draw_skeleton(frame, frame_pose, width, height, settings)
        _draw_joint_angles(frame, frame_pose, angles, index, width, height)

    completed = sum(1 for rep in reps if rep.end_frame <= index)
    _draw_hud(
        frame,
        width,
        height,
        rep_number=frame_to_rep[index] if index < len(frame_to_rep) else None,
        completed=completed,
        total=len(reps),
        depth_percent=_current_depth(angles, index, settings),
    )


def _draw_skeleton(
    frame: np.ndarray,
    frame_pose,
    width: int,
    height: int,
    settings: Settings,
) -> None:
    """Draw bones and joints for one frame.

    Landmarks are normalised to [0, 1], so they scale to the full-resolution
    frame here regardless of the (smaller) resolution inference ran at.
    """
    threshold = settings.landmark_visibility_threshold
    points: dict[int, tuple[int, int]] = {}

    for i, landmark in enumerate(frame_pose.landmarks):
        if landmark.visibility < threshold:
            continue
        points[i] = (int(landmark.x * width), int(landmark.y * height))

    thickness = max(2, int(min(width, height) / 250))
    radius = max(3, int(min(width, height) / 200))

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(
                frame, points[start], points[end], _COLOR_BONE, thickness, cv2.LINE_AA
            )

    for i, point in points.items():
        # The joints that drive the analysis are highlighted; the rest are
        # drawn small so the skeleton reads as a whole without distracting.
        key = i in {
            LM.LEFT_HIP,
            LM.RIGHT_HIP,
            LM.LEFT_KNEE,
            LM.RIGHT_KNEE,
            LM.LEFT_ANKLE,
            LM.RIGHT_ANKLE,
            LM.LEFT_SHOULDER,
            LM.RIGHT_SHOULDER,
        }
        cv2.circle(
            frame,
            point,
            radius if key else max(2, radius - 2),
            _COLOR_ACCENT if key else _COLOR_JOINT,
            -1,
            cv2.LINE_AA,
        )


def _draw_joint_angles(
    frame: np.ndarray,
    frame_pose,
    angles: AngleSeries,
    index: int,
    width: int,
    height: int,
) -> None:
    """Print each knee angle next to the knee it describes."""
    scale = max(0.4, min(width, height) / 1400)

    for landmark_index, values in (
        (LM.LEFT_KNEE, angles.left_knee_deg),
        (LM.RIGHT_KNEE, angles.right_knee_deg),
    ):
        value = values[index] if index < len(values) else None
        landmark = frame_pose.get(landmark_index)
        if value is None or landmark is None:
            continue

        position = (int(landmark.x * width) + 12, int(landmark.y * height) - 8)
        _text_with_shadow(frame, f"{value:.0f}", position, scale, _COLOR_TEXT)


def _draw_hud(
    frame: np.ndarray,
    width: int,
    height: int,
    rep_number: int | None,
    completed: int,
    total: int,
    depth_percent: float | None,
) -> None:
    """Draw the rep counter and depth bar."""
    scale = max(0.5, min(width, height) / 1100)
    pad = int(min(width, height) * 0.02)
    panel_w = int(width * 0.34)
    panel_h = int(scale * 92)

    # Semi-transparent backing so text stays readable over any footage.
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), _COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    label = f"REP {rep_number}" if rep_number else "READY"
    _text_with_shadow(
        frame, label, (pad + 12, pad + int(scale * 30)), scale * 0.9, _COLOR_ACCENT
    )
    _text_with_shadow(
        frame,
        f"{completed}/{total} complete",
        (pad + 12, pad + int(scale * 58)),
        scale * 0.62,
        _COLOR_TEXT,
    )

    if depth_percent is None:
        return

    bar_x = pad + 12
    bar_y = pad + int(scale * 72)
    bar_w = panel_w - 24
    bar_h = max(6, int(scale * 10))

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    filled = int(bar_w * min(depth_percent, 100.0) / 100.0)
    colour = _COLOR_ACCENT if depth_percent >= 90 else _COLOR_WARN
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), colour, -1)


def _current_depth(angles: AngleSeries, index: int, settings: Settings) -> float | None:
    """Live depth percentage for the current frame."""
    from app.analysis.geometry import linear_scale

    if index >= len(angles.left_knee_deg):
        return None

    present = [
        value
        for value in (angles.left_knee_deg[index], angles.right_knee_deg[index])
        if value is not None
    ]
    if not present:
        return None

    return (
        linear_scale(
            min(present),
            settings.standing_knee_angle_deg,
            settings.parallel_knee_angle_deg,
        )
        * 100.0
    )


def _text_with_shadow(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    """Draw text with a dark outline so it survives a light background."""
    thickness = max(1, int(scale * 2))
    cv2.putText(
        frame, text, position, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(frame, text, position, _FONT, scale, colour, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


class _FrameWriter:
    """Common interface over the ffmpeg pipe and the OpenCV fallback."""

    def write(self, frame: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _FfmpegWriter(_FrameWriter):
    """Pipes raw BGR frames to ffmpeg for H.264 encoding."""

    def __init__(self, exe: str, path: Path, width: int, height: int, fps: float):
        command = [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            # Browsers cannot decode yuv444p. This is as important as the codec.
            "-pix_fmt",
            "yuv420p",
            # H.264 requires even dimensions; odd-sized input would otherwise
            # fail the encode outright.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process.stdin:
            self._process.stdin.close()
        _, stderr = self._process.communicate()
        if self._process.returncode != 0:
            raise VideoProcessingError(
                "Overlay encoding failed.",
                detail={"ffmpeg": stderr.decode("utf-8", "replace")[:500]},
            )


class _OpenCvWriter(_FrameWriter):
    """Fallback encoder. Produces MPEG-4 Part 2, which browsers may refuse."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise VideoProcessingError("No usable video encoder is available.")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


def _open_writer(path: Path, width: int, height: int, fps: float) -> _FrameWriter:
    """Prefer ffmpeg; fall back to OpenCV with a loud warning."""
    try:
        import imageio_ffmpeg

        return _FfmpegWriter(imageio_ffmpeg.get_ffmpeg_exe(), path, width, height, fps)
    except Exception as exc:
        logger.warning(
            "ffmpeg unavailable (%s); falling back to the OpenCV encoder. The "
            "overlay will use MPEG-4 Part 2, which some browsers cannot play.",
            exc,
        )
        return _OpenCvWriter(path, width, height, fps)
