"""Tests for the pipeline orchestrator and the overlay renderer.

The overlay tests write real video files and inspect them with ffprobe. That is
deliberate: the failure mode this guards against — an overlay that encodes
without error and then will not play in a browser — is invisible to any check
that only asserts the file exists.
"""

from __future__ import annotations

import json
import re
import subprocess

import cv2
import numpy as np
import pytest

from app.analysis.pipeline import run_pipeline
from app.analysis.types import VideoMetadata
from app.config import Settings
from tests.synthetic import SyntheticPoseEstimator, build_squat_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def source_video(tmp_path):
    """Write a real (blank) video for the overlay renderer to read frames from.

    The content does not matter — the synthetic estimator supplies the
    landmarks. What matters is that a genuine decodable file exists, so the
    renderer is exercised on its real code path.
    """
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (480, 640))
    assert writer.isOpened()
    for i in range(200):
        frame = np.full((640, 480, 3), 40, dtype=np.uint8)
        cv2.putText(
            frame, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2
        )
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


def ffprobe(path) -> dict:
    """Inspect a media file's video stream.

    `imageio-ffmpeg` bundles only the ffmpeg binary, not ffprobe, so this parses
    the stream description ffmpeg prints to stderr when asked to open a file
    with no output. Less elegant than ffprobe's JSON, but it needs no extra
    dependency and reports exactly the same facts.

    A typical line looks like:
        Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 480x640, ...
    """
    import imageio_ffmpeg

    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
    )
    # ffmpeg exits non-zero because no output was specified; the info we want is
    # already on stderr, so the return code is irrelevant here.
    text = result.stderr.decode("utf-8", "replace")

    match = re.search(r"Stream #\d+:\d+.*?: Video: (.+)", text)
    assert match, f"No video stream found in ffmpeg output:\n{text}"
    fields = [part.strip() for part in match.group(1).split(",")]

    dimensions = next((f for f in fields if re.fullmatch(r"\d+x\d+", f.split()[0])), None)
    width, height = (
        (int(v) for v in dimensions.split()[0].split("x")) if dimensions else (0, 0)
    )

    return {
        "codec_name": fields[0].split()[0],
        # ffmpeg appends a scan-type suffix, e.g. "yuv420p(progressive)".
        "pix_fmt": next((f.split("(")[0] for f in fields if f.startswith("yuv")), None),
        "width": width,
        "height": height,
        "raw": text,
    }


class TestPipeline:
    def test_runs_end_to_end_without_an_overlay(self, source_video, settings):
        output = run_pipeline(
            source_video,
            overlay_path=None,
            estimator=SyntheticPoseEstimator(build_squat_series(reps=3)),
            settings=settings,
        )
        assert output.result.metrics.total_reps == 3
        assert output.result.feedback
        assert output.overlay_path is None
        assert output.duration_s > 0

    def test_produces_every_stage_of_output(self, source_video, settings):
        output = run_pipeline(
            source_video,
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        result = output.result
        assert len(result.angles) > 0
        assert len(result.reps) == 2
        assert result.metrics.total_reps == 2
        assert result.feedback
        assert result.estimator_name == "synthetic"

    def test_landmark_payload_is_archivable(self, source_video, settings):
        """Must survive a JSON round trip, since it is written to disk."""
        output = run_pipeline(
            source_video,
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        payload = output.landmark_payload
        assert payload["estimator"] == "synthetic"
        assert len(payload["frames"]) == len(output.result.angles)
        json.loads(json.dumps(payload))  # raises if anything is unserialisable

    def test_frame_count_comes_from_decoding_not_the_header(self, source_video, settings):
        """Containers lie about frame counts; timestamps come from real frames.

        If duration used the declared count, the charts would run past the end
        of the video.
        """
        series = build_squat_series(reps=2)
        lying = type(series)(
            frames=series.frames,
            metadata=VideoMetadata(480, 640, 30.0, 99999, 3333.3),
            estimator_name="synthetic",
        )
        output = run_pipeline(
            source_video,
            estimator=SyntheticPoseEstimator(lying),
            settings=settings,
        )
        assert output.result.metadata.frame_count == len(series.frames)
        assert output.result.metadata.duration_s == pytest.approx(
            len(series.frames) / 30.0
        )

    def test_empty_pose_series_is_rejected(self, source_video, settings):
        from app.analysis.types import PoseSeries
        from app.core.exceptions import VideoProcessingError

        empty = PoseSeries((), VideoMetadata(480, 640, 30.0, 0, 0.0), "synthetic")
        with pytest.raises(VideoProcessingError):
            run_pipeline(
                source_video,
                estimator=SyntheticPoseEstimator(empty),
                settings=settings,
            )

    def test_overlay_failure_does_not_lose_the_analysis(self, source_video, settings):
        """The overlay visualises results that already exist.

        Losing it degrades the experience; losing the metrics because rendering
        failed would waste the upload entirely.
        """
        unwritable = source_video.parent / "no_such_dir" / "x" / "\0bad.mp4"
        output = run_pipeline(
            source_video,
            overlay_path=unwritable,
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        assert output.overlay_path is None
        assert output.result.metrics.total_reps == 2


class TestOverlayEncoding:
    def test_overlay_is_written(self, source_video, tmp_path, settings):
        output = run_pipeline(
            source_video,
            overlay_path=tmp_path / "overlay.mp4",
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        assert output.overlay_path is not None
        assert output.overlay_path.stat().st_size > 0

    def test_overlay_is_h264(self, source_video, tmp_path, settings):
        """The whole reason ffmpeg is a dependency.

        OpenCV's default mp4v encoder writes a valid .mp4 that Chrome silently
        refuses to play. Asserting the codec is what makes that failure loud.
        """
        output = run_pipeline(
            source_video,
            overlay_path=tmp_path / "overlay.mp4",
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        stream = ffprobe(output.overlay_path)
        assert stream["codec_name"] == "h264"

    def test_overlay_is_yuv420p(self, source_video, tmp_path, settings):
        """Browsers cannot decode yuv444p. As important as the codec itself."""
        output = run_pipeline(
            source_video,
            overlay_path=tmp_path / "overlay.mp4",
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        assert ffprobe(output.overlay_path)["pix_fmt"] == "yuv420p"

    def test_overlay_matches_source_dimensions(self, source_video, tmp_path, settings):
        output = run_pipeline(
            source_video,
            overlay_path=tmp_path / "overlay.mp4",
            estimator=SyntheticPoseEstimator(
                build_squat_series(reps=2, width=480, height=640)
            ),
            settings=settings,
        )
        stream = ffprobe(output.overlay_path)
        assert (stream["width"], stream["height"]) == (480, 640)

    def test_overlay_is_actually_decodable(self, source_video, tmp_path, settings):
        """Read the rendered file back with OpenCV.

        Proves the output is not merely non-empty but genuinely decodable.
        """
        output = run_pipeline(
            source_video,
            overlay_path=tmp_path / "overlay.mp4",
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        capture = cv2.VideoCapture(str(output.overlay_path))
        assert capture.isOpened()
        ok, frame = capture.read()
        capture.release()
        assert ok
        assert frame is not None

    def test_overlay_draws_something(self, source_video, tmp_path, settings):
        """The source is a flat grey frame, so any variation is drawn content."""
        output = run_pipeline(
            source_video,
            overlay_path=tmp_path / "overlay.mp4",
            estimator=SyntheticPoseEstimator(build_squat_series(reps=2)),
            settings=settings,
        )
        capture = cv2.VideoCapture(str(output.overlay_path))
        # Sample a frame from the middle of the clip, where the figure is mid-rep.
        for _ in range(60):
            ok, frame = capture.read()
        capture.release()
        assert ok
        # A skeleton in warm blue and a green HUD accent must widen the colour
        # spread well beyond the flat source.
        assert frame.std() > 15.0
