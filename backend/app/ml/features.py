"""The feature vector shared by training and inference.

This module is the seam that makes the ML layer trustworthy, and it is the one
place in the ML code worth reading carefully. It is imported by *both* the
offline training scripts and the live pipeline, so there is exactly one
definition of what a feature means. Train/serve skew cannot be introduced by
editing one side of the system, because there is only one side.

Like `analysis/types.py`, nothing here imports scikit-learn, numpy, or anything
else heavyweight. A `FrameSample` is plain floats, so the feature builder can be
tested by hand.

**The problem this module solves.** The training corpus was extracted by a
different tool, from videos we do not have, using angle conventions that do not
match FormVision's: its knee angle has a median of 54 degrees and a floor of 5,
where FormVision's is ~175 standing and ~80 at depth. The two are not the same
function of the same joint, so a model fitted on raw values from one would be
meaningless applied to the other.

Two devices bridge that gap:

1. **Canonical orientation.** Every signal is restated so that *larger always
   means more of the thing the fault is about* (see `FrameSample`). Each adapter
   is responsible for flipping its own source's sign. This has to be explicit:
   the training corpus's depth signal grows as the lifter descends while
   FormVision's shrinks, and a rank transform preserves an increasing
   reparameterisation but silently inverts a decreasing one.

2. **Within-clip ranking.** Each value is expressed as its percentile within the
   clip's own distribution of that signal. A rank is invariant to *any* monotone
   reparameterisation, where a z-score would only survive an affine one, which is
   why ranking is the correct tool here rather than merely the convenient one.

Absolute features are kept alongside the ranks, because ranking has a real
weakness: a lifter whose knees cave on every single rep has a flat within-clip
distribution, and the fault disappears into it. Which family actually earns its
place is decided per fault by cross-validation, not by argument here.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields

#: Signals measured on both sides of the body.
PAIRED_SIGNALS: tuple[str, ...] = (
    "knee_flexion",
    "hip_flexion",
    "ankle_openness",
    "valgus",
)

#: Signals with a single value for the whole body.
SINGLE_SIGNALS: tuple[str, ...] = (
    "torso_lean",
    "depth_phase",
)


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One frame's signals, in canonical orientation.

    Every field is `None`-able and `None` means *not measured*, never zero. A
    front-on clip has no ankle reading at all, and a zero there would describe a
    perfectly neutral ankle rather than the absence of one.

    Canonical orientation, which every adapter must honour:

    ==================  ==========================================
    Field               Larger means
    ==================  ==========================================
    knee_flexion        knee more bent (deeper)
    hip_flexion         hip more closed (more folded)
    ankle_openness      shin further over the foot; heel lifting
    valgus              knee further inward, toward the midline
    torso_lean          torso further from upright
    depth_phase         lifter lower in the movement
    ==================  ==========================================

    Note that four of these are negations of how the underlying angle is
    normally quoted. A knee angle *falls* as the knee bends, so `knee_flexion`
    is its negative. Adapters do that flip; this module never guesses.
    """

    knee_flexion_left: float | None = None
    knee_flexion_right: float | None = None
    hip_flexion_left: float | None = None
    hip_flexion_right: float | None = None
    ankle_openness_left: float | None = None
    ankle_openness_right: float | None = None
    valgus_left: float | None = None
    valgus_right: float | None = None
    torso_lean: float | None = None
    depth_phase: float | None = None

    def value(self, name: str) -> float | None:
        """Look a signal up by name, for the generic feature loops below."""
        return getattr(self, name)

    def paired(self, signal: str) -> tuple[float | None, float | None]:
        """The left and right readings of a paired signal."""
        return (getattr(self, f"{signal}_left"), getattr(self, f"{signal}_right"))

    def mean_of(self, signal: str) -> float | None:
        """Average of a paired signal, tolerating one missing side.

        Side-on footage genuinely only tracks one leg, so requiring both would
        discard most of a usable clip. See `CORE_LANDMARKS` in `analysis/types`
        for the same reasoning applied to landmarks.
        """
        present = [value for value in self.paired(signal) if value is not None]
        if not present:
            return None
        return sum(present) / len(present)

    def gap_of(self, signal: str) -> float | None:
        """Absolute left/right difference, or None if either side is missing.

        Unlike `mean_of`, this cannot fall back to one side: a difference needs
        both terms, and inventing the missing one would manufacture symmetry
        that was never observed.
        """
        left, right = self.paired(signal)
        if left is None or right is None:
            return None
        return abs(left - right)


#: Derived quantities computed from a `FrameSample` before ranking. Named here
#: so `ClipReference` and the feature builder agree on the vocabulary without
#: either having to know the other's internals.
def derived_values(sample: FrameSample) -> dict[str, float | None]:
    """Flatten a sample into the full set of rankable quantities."""
    values: dict[str, float | None] = {}

    for signal in PAIRED_SIGNALS:
        left, right = sample.paired(signal)
        values[f"{signal}_left"] = left
        values[f"{signal}_right"] = right
        values[f"{signal}_mean"] = sample.mean_of(signal)
        values[f"{signal}_gap"] = sample.gap_of(signal)

    for signal in SINGLE_SIGNALS:
        values[signal] = sample.value(signal)

    return values


#: Every quantity that gets both an absolute and a ranked feature, in a fixed
#: order. Feature vectors are positional once they reach scikit-learn, so this
#: ordering is part of the model contract and must not be reshuffled without
#: retraining.
QUANTITY_NAMES: tuple[str, ...] = tuple(
    [
        f"{signal}_{suffix}"
        for signal in PAIRED_SIGNALS
        for suffix in ("left", "right", "mean", "gap")
    ]
    + list(SINGLE_SIGNALS)
)


@dataclass(frozen=True, slots=True)
class ClipReference:
    """The clip's own distribution of each quantity, for ranking against.

    Built once per clip and reused for every frame in it. Values are stored
    sorted so a rank is a binary search rather than a scan, which matters when a
    two-minute clip at 30fps asks for eighteen ranks on each of 3,600 frames.
    """

    distributions: Mapping[str, tuple[float, ...]]

    @classmethod
    def from_samples(cls, samples: Iterable[FrameSample]) -> ClipReference:
        """Collect each quantity's observed values across a clip.

        Missing readings are skipped rather than filled. A quantity that was
        never measurable ends up with an empty distribution, and `rank` then
        returns None for it, which is what carries "this camera angle could not
        see it" all the way through to the predictor abstaining.
        """
        collected: dict[str, list[float]] = {name: [] for name in QUANTITY_NAMES}

        for sample in samples:
            for name, value in derived_values(sample).items():
                if value is not None:
                    collected[name].append(value)

        return cls(
            distributions={
                name: tuple(sorted(values)) for name, values in collected.items()
            }
        )

    def rank(self, name: str, value: float | None) -> float | None:
        """Where `value` falls in the clip's distribution, as a 0-1 fraction.

        Ties resolve to the *midpoint* of the tied run rather than its end, and
        that detail is load-bearing. Taking the upper edge would rank a signal
        that never varies at 1.0 in every frame, i.e. maximally extreme, when
        the truth is the exact opposite: a quantity that never moves has nothing
        remarkable about it anywhere. That bug made a perfectly symmetric lifter
        register as asymmetric on every repetition, because a left/right gap
        pinned at zero ranked at the top of its own distribution.

        With the midpoint convention a constant signal ranks 0.5 throughout,
        which reads as "unremarkable" and is what the model was trained to
        expect from ordinary movement.

        Returns None when the value is missing or the clip never measured this
        quantity at all.
        """
        if value is None:
            return None

        ordered = self.distributions.get(name)
        if not ordered:
            return None

        low = bisect_left(ordered, value)
        high = bisect_right(ordered, value)
        return (low + high) / (2 * len(ordered))


#: The full ordered feature vector. Absolute features first, then ranks, so a
#: model card listing the names reads in a predictable order.
FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"abs_{name}" for name in QUANTITY_NAMES]
    + [f"rank_{name}" for name in QUANTITY_NAMES]
)


def build_frame_features(
    sample: FrameSample, reference: ClipReference
) -> dict[str, float | None]:
    """Build one frame's complete feature vector.

    The single function called by both the training pipeline and the live
    predictor. Keys are exactly `FEATURE_NAMES`; values are `None` wherever the
    underlying signal was not measurable.
    """
    values = derived_values(sample)

    features: dict[str, float | None] = {}
    for name in QUANTITY_NAMES:
        features[f"abs_{name}"] = values[name]
        features[f"rank_{name}"] = reference.rank(name, values[name])

    return features


def feature_row(
    features: Mapping[str, float | None], names: Sequence[str]
) -> tuple[list[float | None], float]:
    """Project a feature mapping onto an ordered subset, with a completeness score.

    Returns the values in `names` order alongside the fraction of them that were
    actually measured. The predictor uses that fraction to decide whether it
    knows enough to speak: a valgus verdict resting on two of six features is
    not a verdict, and the honest output is silence.
    """
    row = [features.get(name) for name in names]
    if not row:
        return row, 0.0

    present = sum(1 for value in row if value is not None)
    return row, present / len(row)


#: Which features each fault detector is allowed to see.
#:
#: **Ranks only, deliberately.** The absolute features above are computed and
#: available, but none of them appears here, because an absolute value cannot
#: cross the gap between the training corpus and FormVision. The corpus quotes
#: its knee lateral offset in raw normalised-coordinate units; FormVision
#: divides its own by torso length. A decision boundary learned at 0.05 in one
#: set of units is not the same boundary in the other, and nothing at runtime
#: would reveal the mismatch. Ranks are unitless by construction, so they are
#: the only family that genuinely transfers.
#:
#: The cost is a real and documented blind spot: a lifter whose knees cave on
#: *every* rep has a flat within-clip distribution, so their valgus ranks
#: mid-range and the detector stays quiet. Ranking finds the fault that stands
#: out from a lifter's own movement, not the one baked into all of it. Catching
#: the persistent case needs an absolute threshold in FormVision's own units,
#: which is a rule, not a model.
#:
#: Restricting inputs per fault is also what keeps each detector interpretable
#: and makes a fault the camera cannot see fail cleanly on completeness rather
#: than quietly substituting a correlated signal.
#:
#: **Each shipped detector sees its own signal and nothing else that moves with
#: the label.** This is not tidiness, it is the result of an ablation. The
#: corpus perturbs knee *angle* at the same time as knee lateral offset when it
#: manufactures a caving-knees example, so a valgus detector given the mean knee
#: flexion reached 0.994 AUC while scoring 0.910 from that knee flexion alone.
#: It was mostly detecting the corpus's bookkeeping. Real caving knees do not
#: come with extra knee bend attached, so that detector would have collapsed on
#: real footage while looking excellent in cross-validation. The same held for
#: asymmetry, where perturbing each side independently shifts the mean as well
#: as the gap. Both mean terms are gone.
#:
#: The depth-phase rank stays in all of them, and is the one shared input.
#: The corpus is per-frame with no notion of where in a rep a frame sits, so
#: without it a correct standing frame and a faulty bottom frame are the same
#: point. It is safe to share precisely because it carries no label signal: the
#: corpus never recomputes its depth column after perturbing anything, so it is
#: bit-identical across all six labels of a frame and scores exactly 0.500 AUC
#: on its own against every fault. It contributes context, never evidence.
FAULT_FEATURES: Mapping[str, tuple[str, ...]] = {
    "knee_valgus": (
        "rank_valgus_left",
        "rank_valgus_right",
        "rank_valgus_mean",
        "rank_valgus_gap",
        "rank_depth_phase",
    ),
    "heel_lift": (
        "rank_ankle_openness_left",
        "rank_ankle_openness_right",
        "rank_ankle_openness_mean",
        "rank_depth_phase",
    ),
    "asymmetry": (
        "rank_knee_flexion_gap",
        "rank_hip_flexion_gap",
        "rank_depth_phase",
    ),
    # Trained and reported for the model card, but not wired to feedback:
    # FormVision measures depth and lean directly and correctly, and a direct
    # measurement beats a model fitted on synthetic offsets.
    "shallow": (
        "rank_knee_flexion_mean",
        "rank_hip_flexion_mean",
        "rank_depth_phase",
    ),
    "forward_lean": (
        "rank_torso_lean",
        "rank_depth_phase",
    ),
}

#: The faults whose predictions become coaching feedback. The rest are trained
#: for evaluation only.
SHIPPED_FAULTS: tuple[str, ...] = ("knee_valgus", "heel_lift", "asymmetry")


def sample_field_names() -> tuple[str, ...]:
    """Field names of `FrameSample`, for adapters that build one generically."""
    return tuple(field.name for field in fields(FrameSample))
