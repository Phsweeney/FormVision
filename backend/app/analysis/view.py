"""Camera view detection.

Works out whether a clip was filmed side-on or front-on, so the pipeline can
report `None` for the measurements that angle cannot see instead of a confident
number derived from a projection that does not contain the information.

**Why this is necessary.** FormVision measures in the image plane. A camera
placed side-on sees the torso hinge forward and the knee close, but sees both
legs collapsed onto one another — so a left/right comparison is measuring
occlusion noise, not the lifter. A camera placed front-on sees the two legs
separately, but sees a forward lean almost edge-on, so it reads as near zero
whatever the lifter does. Each angle is blind to exactly what the other is good
at, and neither blindness announces itself: both produce plausible-looking
numbers. Front-on footage reported 1.4 degrees of lean and earned a "good torso
position", which was flattery rather than measurement.

**The signal.** Shoulder separation divided by torso length. Dividing by the
subject's own torso makes it scale-free, so it does not depend on camera
distance, resolution, or body size — the same reasoning that governs every other
normalised signal in `angles.py`. Turned side-on the shoulders project almost on
top of each other and the ratio collapses; turned front-on they spread to the
subject's full width.

The separation in real footage is not marginal. Across the sample clips the
ratio measures 0.06 to 0.07 side-on and 0.40 to 1.27 front-on — an order of
magnitude, with nothing in between — which is why a simple threshold pair is
enough and no smarter classifier is warranted.
"""

from __future__ import annotations

import numpy as np

from app.analysis.types import (
    CORE_LANDMARKS,
    PoseSeries,
    ViewOrientation,
)
from app.analysis.types import (
    PoseLandmarkIndex as LM,
)
from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)

#: A torso shorter than this in normalised units means the subject occupies
#: almost no pixels; the ratio would be dominated by division noise.
_MIN_TORSO = 1e-6


def shoulder_ratio(series: PoseSeries, threshold: float) -> float | None:
    """Median shoulder separation, in torso lengths, across the clip.

    The median rather than a single frame or a mean: the torso foreshortens at
    the bottom of a rep, which inflates the ratio for those frames, and one
    badly-tracked moment should not decide the camera angle for the whole video.

    Returns None when the subject was never tracked well enough to measure.
    """
    ratios: list[float] = []

    for frame in series.frames:
        landmarks = [frame.get(index) for index in CORE_LANDMARKS]
        if any(
            landmark is None or landmark.visibility < threshold for landmark in landmarks
        ):
            continue

        left_shoulder = frame.get(LM.LEFT_SHOULDER)
        right_shoulder = frame.get(LM.RIGHT_SHOULDER)
        left_hip = frame.get(LM.LEFT_HIP)
        right_hip = frame.get(LM.RIGHT_HIP)

        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2.0
        hip_mid_y = (left_hip.y + right_hip.y) / 2.0
        hip_mid_x = (left_hip.x + right_hip.x) / 2.0

        torso = float(np.hypot(shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y))
        if torso < _MIN_TORSO:
            continue

        ratios.append(abs(left_shoulder.x - right_shoulder.x) / torso)

    return float(np.median(ratios)) if ratios else None


def detect_view(series: PoseSeries, settings: Settings) -> ViewOrientation:
    """Classify the camera angle a clip was filmed from."""
    ratio = shoulder_ratio(series, settings.landmark_visibility_threshold)

    if ratio is None:
        logger.info("Camera view undetermined: the subject was never tracked")
        return ViewOrientation.UNKNOWN

    if ratio <= settings.view_side_max_shoulder_ratio:
        view = ViewOrientation.SIDE
    elif ratio >= settings.view_front_min_shoulder_ratio:
        view = ViewOrientation.FRONT
    else:
        view = ViewOrientation.OBLIQUE

    logger.info(
        "Camera view detected as %s (shoulder separation %.3f torso lengths)",
        view.value,
        ratio,
    )
    return view
