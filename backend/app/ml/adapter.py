"""Turn FormVision's `AngleSeries` into canonical `FrameSample`s.

This is one half of the bridge described in `features.py`; the other half lives
in the training package and reads the CSV corpus. Both produce the same shape,
which is the only reason a model fitted on one can be applied to the other.

**Every sign flip in this file is load-bearing.** Get one wrong and the model
still runs, still returns confident probabilities, and is silently backwards for
that signal. There is no runtime error to catch it, which is why each one is
justified individually below and pinned by a test.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.analysis.types import AngleSeries, Rep
from app.ml.features import ClipReference, FrameSample


def _negate(value: float | None) -> float | None:
    """Flip a signal that is quoted in the opposite sense to canonical."""
    return None if value is None else -value


def frame_sample(angles: AngleSeries, index: int) -> FrameSample:
    """Build the canonical sample for one frame of an `AngleSeries`.

    The four negations here are the whole point of the function:

    - Knee and hip angles *fall* as the joint closes, so flexion is their
      negative. Left as-is, "more bent" would rank as "less".
    - Hip height is measured upward from the ankles, so it *falls* as the lifter
      descends, whereas `depth_phase` must grow. The training corpus's own depth
      column runs the other way, which is exactly the trap: a rank transform
      carries an increasing reparameterisation through unchanged and inverts a
      decreasing one without complaint.

    Ankle openness and valgus need no flip. `angles.py` already defines the
    ankle angle as opening when the heel lifts, and knee lateral offset as
    positive when the knee travels medially.
    """
    return FrameSample(
        knee_flexion_left=_negate(angles.left_knee_deg[index]),
        knee_flexion_right=_negate(angles.right_knee_deg[index]),
        hip_flexion_left=_negate(angles.left_hip_deg[index]),
        hip_flexion_right=_negate(angles.right_hip_deg[index]),
        ankle_openness_left=angles.left_ankle_deg[index],
        ankle_openness_right=angles.right_ankle_deg[index],
        valgus_left=angles.left_knee_lateral[index],
        valgus_right=angles.right_knee_lateral[index],
        torso_lean=angles.torso_lean_deg[index],
        depth_phase=_negate(angles.hip_height[index]),
    )


def clip_samples(angles: AngleSeries) -> list[FrameSample]:
    """Canonical samples for every frame in the clip, in order."""
    return [frame_sample(angles, index) for index in range(len(angles))]


def clip_reference(samples: Sequence[FrameSample]) -> ClipReference:
    """The clip's distribution, for ranking frames against their own clip."""
    return ClipReference.from_samples(samples)


def rep_frame_range(rep: Rep, frame_count: int) -> range:
    """Frame indices belonging to one rep, clamped to the series length.

    Inclusive of the end frame: `Rep` marks lockout as a frame the lifter
    reached, not one they had passed.
    """
    start = max(0, min(rep.start_frame, frame_count))
    end = max(start, min(rep.end_frame + 1, frame_count))
    return range(start, end)
