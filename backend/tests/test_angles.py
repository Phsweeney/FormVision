"""Tests for the angle computation stage.

These run against the synthetic stick figure, so the expected relationships are
consequences of geometry rather than magic numbers: a deeper hip *must* give a
smaller knee angle, a scaled-up figure *must* give identical normalised output.
"""

from __future__ import annotations

import pytest

from app.analysis.angles import compute_angles
from app.analysis.types import PoseSeries, VideoMetadata, ViewOrientation
from app.config import Settings
from tests.synthetic import build_frame, build_squat_series, build_standing_series


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestOneSidedTracking:
    """What the analysis does when it can only see one side of the body."""

    def test_a_single_visible_side_still_yields_every_signal(self, settings):
        """Side-on, the visible landmark *is* the midpoint to within the noise.

        Both sides of a pair project to nearly the same image point, so falling
        back to whichever one is tracked costs nothing — and it is what keeps
        hip height, hip angle and depth working on footage with one leg hidden.
        """
        from dataclasses import replace

        from app.analysis.types import PoseLandmarkIndex as LM

        both = build_squat_series(reps=2, view=ViewOrientation.SIDE)
        one_sided = replace(
            both,
            frames=tuple(
                replace(
                    frame,
                    landmarks=tuple(
                        replace(landmark, visibility=0.1)
                        if index in (LM.LEFT_KNEE, LM.LEFT_ANKLE)
                        else landmark
                        for index, landmark in enumerate(frame.landmarks)
                    ),
                )
                for frame in both.frames
            ),
        )

        full = compute_angles(both, settings)
        partial = compute_angles(one_sided, settings)

        assert partial.valid_fraction == pytest.approx(1.0)
        assert partial.torso_scale == pytest.approx(full.torso_scale, rel=0.05)

        # The signals that matter track the two-legged answer closely, because
        # side-on the discarded leg was sitting on top of the kept one anyway.
        for left, right in zip(full.hip_height, partial.hip_height, strict=True):
            if left is not None and right is not None:
                assert right == pytest.approx(left, abs=0.05)


class TestScaleNormalisation:
    def test_torso_scale_is_measured(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        assert angles.torso_scale is not None
        assert angles.torso_scale > 0
        assert angles.thigh_scale is not None

    def test_camera_distance_does_not_change_normalised_output(self, settings):
        """The whole point of scale normalisation.

        The same squat filmed from further away produces smaller raw
        coordinates. Normalised signals must be unaffected, or every threshold
        in the app would mean something different per video.
        """
        near = build_squat_series(reps=2)

        # Shrink the figure about the frame centre: same movement, further away.
        def shrink(series: PoseSeries, factor: float) -> PoseSeries:
            from app.analysis.types import FramePose, Landmark

            frames = tuple(
                FramePose(
                    frame.frame_index,
                    frame.timestamp_s,
                    tuple(
                        Landmark(
                            0.5 + (point.x - 0.5) * factor,
                            0.5 + (point.y - 0.5) * factor,
                            point.z,
                            point.visibility,
                        )
                        for point in frame.landmarks
                    ),
                    frame.detected,
                )
                for frame in series.frames
            )
            return PoseSeries(frames, series.metadata, series.estimator_name)

        far = shrink(near, 0.5)

        near_angles = compute_angles(near, settings)
        far_angles = compute_angles(far, settings)

        # Raw scale genuinely differs...
        assert far_angles.torso_scale == pytest.approx(
            near_angles.torso_scale * 0.5, rel=1e-3
        )

        # ...but the normalised hip-height signal does not.
        near_hip = [v for v in near_angles.hip_height if v is not None]
        far_hip = [v for v in far_angles.hip_height if v is not None]
        assert far_hip == pytest.approx(near_hip, rel=1e-6)

        # Joint angles are scale-invariant by construction.
        near_knee = [v for v in near_angles.left_knee_deg if v is not None]
        far_knee = [v for v in far_angles.left_knee_deg if v is not None]
        assert far_knee == pytest.approx(near_knee, rel=1e-6)


class TestKneeAngles:
    def test_standing_knees_are_near_extension(self, settings):
        angles = compute_angles(build_standing_series(seconds=2.0), settings)
        knees = [v for v in angles.left_knee_deg if v is not None]
        assert knees
        assert min(knees) > 150.0

    def test_squatting_closes_the_knee(self, settings):
        angles = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)
        knees = [v for v in angles.left_knee_deg if v is not None]
        assert min(knees) < 110.0
        assert max(knees) > 150.0

    def test_deeper_squat_gives_smaller_minimum(self, settings):
        """Direction-of-effect check on the most important signal in the app."""
        shallow = compute_angles(build_squat_series(reps=1, depth_fraction=0.4), settings)
        deep = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)

        shallow_min = min(v for v in shallow.left_knee_deg if v is not None)
        deep_min = min(v for v in deep.left_knee_deg if v is not None)
        assert deep_min < shallow_min

    def test_symmetric_figure_gives_matching_knees(self, settings):
        angles = compute_angles(build_squat_series(reps=1, left_right_bias=0.0), settings)
        for left, right in zip(angles.left_knee_deg, angles.right_knee_deg, strict=True):
            if left is not None and right is not None:
                assert left == pytest.approx(right, abs=1.0)

    def test_biased_figure_gives_diverging_knees(self, settings):
        angles = compute_angles(build_squat_series(reps=1, left_right_bias=0.6), settings)
        differences = [
            abs(left - right)
            for left, right in zip(
                angles.left_knee_deg, angles.right_knee_deg, strict=True
            )
            if left is not None and right is not None
        ]
        assert max(differences) > 3.0


class TestTorsoLean:
    """Lean is a sagittal-plane measurement, so every case here films side-on.

    A front-on camera sees the torso hinge almost directly toward the lens, and
    the last test in this class pins down that we report nothing rather than the
    flattering near-zero that projection produces.
    """

    def test_upright_figure_reads_near_zero(self, settings):
        series = build_squat_series(
            reps=1,
            torso_lean_deg=0.0,
            bottom_lean_deg=0.0,
            view=ViewOrientation.SIDE,
        )
        angles = compute_angles(series, settings)
        leans = [v for v in angles.torso_lean_deg if v is not None]
        assert max(leans) < 2.0

    def test_lean_is_recovered(self, settings):
        series = build_squat_series(
            reps=1,
            torso_lean_deg=30.0,
            bottom_lean_deg=30.0,
            view=ViewOrientation.SIDE,
        )
        angles = compute_angles(series, settings)
        leans = [v for v in angles.torso_lean_deg if v is not None]
        assert max(leans) == pytest.approx(30.0, abs=2.0)

    def test_increasing_lean_at_the_bottom_is_detected(self, settings):
        series = build_squat_series(
            reps=1,
            torso_lean_deg=10.0,
            bottom_lean_deg=55.0,
            view=ViewOrientation.SIDE,
        )
        angles = compute_angles(series, settings)
        leans = [v for v in angles.torso_lean_deg if v is not None]
        assert max(leans) > 45.0

    def test_front_on_footage_reports_no_lean_at_all(self, settings):
        """A front-on camera cannot see a forward hinge, so it must not claim to.

        Left unguarded this reads about 1 degree however far the lifter folds,
        which the coaching layer would report as an excellent torso position.
        Absent is the honest answer; near-zero is a lie.
        """
        series = build_squat_series(
            reps=1,
            torso_lean_deg=50.0,
            bottom_lean_deg=60.0,
            view=ViewOrientation.FRONT,
        )
        angles = compute_angles(series, settings)
        assert angles.view is ViewOrientation.FRONT
        assert all(value is None for value in angles.torso_lean_deg)


class TestHipHeight:
    def test_falls_during_the_descent(self, settings):
        angles = compute_angles(build_squat_series(reps=1), settings)
        heights = [v for v in angles.hip_height if v is not None]
        assert max(heights) > min(heights)

    def test_deeper_squat_travels_further(self, settings):
        shallow = compute_angles(build_squat_series(reps=1, depth_fraction=0.4), settings)
        deep = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)

        def travel(series):
            values = [v for v in series.hip_height if v is not None]
            return max(values) - min(values)

        assert travel(deep) > travel(shallow)

    def test_standing_still_barely_moves(self, settings):
        angles = compute_angles(build_standing_series(seconds=3.0), settings)
        heights = [v for v in angles.hip_height if v is not None]
        assert max(heights) - min(heights) < 0.05

    def test_hip_drops_below_knee_at_full_depth(self, settings):
        """hip_knee_offset >= 0 is the at-or-below-parallel signal."""
        angles = compute_angles(build_squat_series(reps=1, depth_fraction=1.0), settings)
        offsets = [v for v in angles.hip_knee_offset if v is not None]
        assert max(offsets) > -0.1

    def test_shallow_squat_keeps_hip_above_knee(self, settings):
        angles = compute_angles(build_squat_series(reps=1, depth_fraction=0.3), settings)
        offsets = [v for v in angles.hip_knee_offset if v is not None]
        assert max(offsets) < 0.0


class TestTrackingQuality:
    def test_clean_series_is_fully_valid(self, settings):
        angles = compute_angles(build_squat_series(reps=2), settings)
        assert angles.valid_fraction == pytest.approx(1.0)

    def test_undetected_frames_are_marked_invalid(self, settings):
        series = build_squat_series(reps=2, undetected_frames=tuple(range(10, 30)))
        angles = compute_angles(series, settings)
        assert angles.valid_fraction < 1.0
        assert angles.valid[15] is False

    def test_low_visibility_landmarks_are_rejected(self, settings):
        """Visibility below threshold is as good as not detected.

        MediaPipe reports a position for occluded joints; using them regardless
        would produce confident-looking angles from guessed coordinates.
        """
        from app.analysis.types import VideoMetadata

        frames = tuple(build_frame(i, i / 30.0, 0.55, visibility=0.1) for i in range(30))
        series = PoseSeries(frames, VideoMetadata(720, 1280, 30.0, 30, 1.0), "synthetic")
        angles = compute_angles(series, settings)
        assert angles.valid_fraction == 0.0
        assert all(value is None for value in angles.left_knee_deg)

    def test_side_on_footage_with_an_occluded_leg_stays_fully_usable(self, settings):
        """The regression this whole stage was rebuilt around.

        Filmed side-on, the near leg is tracked confidently and the far one is
        hidden behind it — MediaPipe measures 0.93 for the near knee and 0.40
        for the far one on real footage, while returning good coordinates for
        both. Requiring every landmark meant a clean, well-lit side-on clip
        scored 17% tracking quality and lost most of its reps.
        """
        series = build_squat_series(
            reps=3,
            view=ViewOrientation.SIDE,
            far_side_visibility=0.4,
        )
        angles = compute_angles(series, settings)

        assert angles.valid_fraction == pytest.approx(1.0)
        # One leg carries the analysis; the other is honestly reported missing.
        assert angles.right_leg_coverage == pytest.approx(1.0)
        assert angles.left_leg_coverage == pytest.approx(0.0)
        assert all(value is None for value in angles.left_knee_deg)
        assert all(value is not None for value in angles.right_knee_deg)
        # Depth and hip height only ever needed one leg.
        assert all(value is not None for value in angles.hip_height)

    def test_both_legs_lost_makes_the_frame_unusable(self, settings):
        """One leg is enough. Zero is not — the torso alone cannot show a squat."""
        frames = tuple(
            build_frame(i, i / 30.0, 0.55, far_side_visibility=0.1) for i in range(30)
        )
        # Occlude the near side too, leaving only the torso.
        from dataclasses import replace

        from app.analysis.types import PoseLandmarkIndex as LM

        stripped = []
        for frame in frames:
            landmarks = list(frame.landmarks)
            for index in (LM.RIGHT_KNEE, LM.RIGHT_ANKLE):
                landmarks[index] = replace(landmarks[index], visibility=0.1)
            stripped.append(replace(frame, landmarks=tuple(landmarks)))

        series = PoseSeries(
            tuple(stripped), VideoMetadata(720, 1280, 30.0, 30, 1.0), "synthetic"
        )
        angles = compute_angles(series, settings)
        assert angles.valid_fraction == 0.0

    def test_series_lengths_stay_aligned(self, settings):
        """Every parallel list must match the frame count, or the overlay and
        charts desynchronise from the video."""
        series = build_squat_series(reps=3, undetected_frames=(5, 6, 7))
        angles = compute_angles(series, settings)
        n = len(series.frames)
        assert len(angles.timestamps_s) == n
        assert len(angles.left_knee_deg) == n
        assert len(angles.right_knee_deg) == n
        assert len(angles.hip_deg) == n
        assert len(angles.torso_lean_deg) == n
        assert len(angles.hip_height) == n
        assert len(angles.hip_knee_offset) == n
        assert len(angles.valid) == n


class TestEmptyAndDegenerate:
    def test_empty_series_does_not_crash(self, settings):
        series = PoseSeries((), VideoMetadata(720, 1280, 30.0, 0, 0.0), "synthetic")
        angles = compute_angles(series, settings)
        assert len(angles) == 0
        assert angles.valid_fraction == 0.0

    def test_never_detected_yields_no_scale(self, settings):
        frames = tuple(build_frame(i, i / 30.0, 0.5, detected=False) for i in range(30))
        series = PoseSeries(frames, VideoMetadata(720, 1280, 30.0, 30, 1.0), "synthetic")
        angles = compute_angles(series, settings)
        assert angles.torso_scale is None
        assert all(value is None for value in angles.hip_height)

    def test_mean_knee_handles_one_missing_side(self, settings):
        angles = compute_angles(build_squat_series(reps=1), settings)
        angles.right_knee_deg[0] = None
        mean = angles.mean_knee_deg
        assert mean[0] == pytest.approx(angles.left_knee_deg[0])


#: The signals added for the ML layer. Grouped so the length and gating tests
#: below cannot drift out of sync with the dataclass.
_PER_SIDE_SIGNALS = (
    "left_hip_deg",
    "right_hip_deg",
    "left_ankle_deg",
    "right_ankle_deg",
    "left_knee_lateral",
    "right_knee_lateral",
)


class TestPerSideSignals:
    """Per-side hip and ankle angles, and the knee's medial offset."""

    def test_every_signal_is_frame_aligned(self, settings):
        series = build_squat_series(reps=2)
        angles = compute_angles(series, settings)

        for name in _PER_SIDE_SIGNALS:
            assert len(getattr(angles, name)) == len(series.frames), name

    def test_undetected_frames_are_missing_not_zero(self, settings):
        """A frame with no subject must not report a zero-degree joint.

        Zero is a real reading — a fully folded joint, or a knee exactly on its
        hip-ankle line — so it cannot double as "no measurement".
        """
        series = build_squat_series(reps=1, undetected_frames=tuple(range(20, 30)))
        angles = compute_angles(series, settings)

        for name in _PER_SIDE_SIGNALS:
            assert angles.__getattribute__(name)[25] is None, name

    def test_per_side_hip_angles_bracket_the_midpoint_angle(self, settings):
        """Each side's hip angle should sit near the midpoint-derived one.

        They are not identical and should not be: the shoulders are wider than
        the hips, so each side's torso segment is tilted a couple of degrees
        relative to the body's centre line. Agreement to within ten degrees is
        the real geometric relationship.
        """
        angles = compute_angles(build_squat_series(reps=2), settings)

        compared = 0
        for mid, left, right in zip(
            angles.hip_deg, angles.left_hip_deg, angles.right_hip_deg, strict=True
        ):
            if None in (mid, left, right):
                continue
            assert left == pytest.approx(mid, abs=10.0)
            assert right == pytest.approx(mid, abs=10.0)
            compared += 1

        assert compared > 0


class TestViewGating:
    """Each new signal is silent from the camera angle that cannot see it."""

    def test_valgus_is_measured_front_on_and_withheld_side_on(self, settings):
        front = compute_angles(
            build_squat_series(reps=1, view=ViewOrientation.FRONT), settings
        )
        side = compute_angles(
            build_squat_series(reps=1, view=ViewOrientation.SIDE), settings
        )

        assert any(value is not None for value in front.left_knee_lateral)
        assert any(value is not None for value in front.right_knee_lateral)

        # Side-on, a knee projects onto its own hip-to-ankle line however far it
        # has actually collapsed inward, so the number would be a confident zero
        # for a lifter whose knees are caving badly.
        assert all(value is None for value in side.left_knee_lateral)
        assert all(value is None for value in side.right_knee_lateral)

    def test_ankle_angle_is_measured_side_on_and_withheld_front_on(self, settings):
        front = compute_angles(
            build_squat_series(reps=1, view=ViewOrientation.FRONT), settings
        )
        side = compute_angles(
            build_squat_series(reps=1, view=ViewOrientation.SIDE), settings
        )

        assert any(value is not None for value in side.left_ankle_deg)
        assert any(value is not None for value in side.right_ankle_deg)

        # Front-on the foot points at the lens, so shin-over-foot travel barely
        # projects into the image at all. Same reasoning as torso lean.
        assert all(value is None for value in front.left_ankle_deg)
        assert all(value is None for value in front.right_ankle_deg)


class TestValgusSign:
    """Knees caving inward must read positive, on both sides."""

    def test_medial_knee_travel_raises_both_sides(self, settings):
        """Adding valgus to the figure increases the measured offset on both legs.

        Measured as a change against the same clip without it rather than as an
        absolute value, because the synthetic figure bends both knees toward +x
        in the image, which lands medially on one leg and laterally on the other.
        The *difference* isolates the medial travel, which is the quantity the
        sign convention is about.
        """
        neutral = compute_angles(
            build_squat_series(reps=2, view=ViewOrientation.FRONT), settings
        )
        caving = compute_angles(
            build_squat_series(reps=2, view=ViewOrientation.FRONT, knee_valgus=0.35),
            settings,
        )

        for name in ("left_knee_lateral", "right_knee_lateral"):
            before = [v for v in getattr(neutral, name) if v is not None]
            after = [v for v in getattr(caving, name) if v is not None]
            assert before and after

            shift = sum(after) / len(after) - sum(before) / len(before)
            # Both knees moved toward the midline, so both must read *more*
            # medial. A sign error on either side would flip this negative.
            assert shift > 0.1, f"{name} shifted by {shift:.3f}"

    def test_offset_scales_with_how_far_the_knees_cave(self, settings):
        """Twice the medial travel must read as a larger offset, not merely nonzero."""
        mild = compute_angles(build_squat_series(reps=1, knee_valgus=0.2), settings)
        severe = compute_angles(build_squat_series(reps=1, knee_valgus=0.5), settings)

        def mean_offset(angles) -> float:
            values = [v for v in angles.left_knee_lateral if v is not None]
            return sum(values) / len(values)

        assert mean_offset(severe) > mean_offset(mild)
