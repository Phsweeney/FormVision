"""The pose estimation interface.

This is the most important seam in the codebase. Everything downstream —
angles, rep detection, metrics, coaching, the overlay renderer — consumes a
`PoseSeries` and has no idea which library produced it.

That buys three things:

1. **Testability.** The whole analysis stack can be exercised against a
   hand-built `PoseSeries` with no video file, no model download, and no CV
   dependency, which is why the test suite runs in under a second.
2. **Swappability.** V2's webcam/real-time support or an ML-based estimator is
   a new subclass registered in `registry.py`, not a pipeline rewrite.
3. **Isolation of risk.** MediaPipe is the least stable dependency here. When
   it changes — and it already has, dropping `mp.solutions` in 0.10.35 — the
   blast radius is one file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.analysis.types import PoseSeries


class PoseEstimator(ABC):
    """Extracts per-frame body landmarks from a video."""

    #: Stable identifier recorded in results and used by the registry.
    name: str = "base"

    @abstractmethod
    def estimate(self, video_path: Path) -> PoseSeries:
        """Run pose estimation over an entire video.

        Implementations must return one `FramePose` per decoded frame, in
        order, including frames where no person was found — those are marked
        ``detected=False`` rather than omitted. Callers rely on frame index and
        list position agreeing so that landmarks, angles, and video frames stay
        aligned for the overlay.

        Raises:
            VideoProcessingError: the file could not be decoded.
            PoseEstimationError: the estimator could not be initialised.
        """

    def close(self) -> None:
        """Release any native resources. Safe to call more than once.

        Deliberately concrete and empty rather than abstract: most estimators
        hold nothing that needs releasing, and forcing every subclass to write
        an empty override would be noise.
        """
        return None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
