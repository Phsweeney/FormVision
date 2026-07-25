"""Read the squat corpus and restate it in canonical feature space.

The other half of the bridge described in `app/ml/features.py`. This module
knows everything peculiar about the CSV; nothing downstream of it does.

**What the corpus actually is**, established by profiling it rather than by
reading its README. Fifteen videos, every one of them a *correct* squat, 7,907
frames. Each frame appears six times: once unmodified as label 0, and five more
times with a uniform random offset added to one group of columns. So every
"fault" example is a correct frame with arithmetic done to it, and the labels
describe which arithmetic.

Three consequences, all of which this module is built around.

**`spine_angle` is dropped.** It is a byte-for-byte duplicate of
`left_hip_angle`, and the augmentation perturbs one without the other. Their
difference is therefore exactly 0.0 for labels {0, 3, 4} and non-zero for
{1, 2, 5}: a single subtraction splits the six classes into two certain groups
of three. Any model given both columns finds this immediately and learns
nothing about squats.

**`symmetry_score` is dropped.** It is never recomputed after perturbation, so
the "asymmetric" class carries the symmetry score of the *correct* frame it was
derived from. It is not merely uninformative about the label it should predict,
it is actively anti-correlated with it.

**Splits must be grouped by video.** Those two columns are float-identical
across all six rows derived from one frame, so a random split lets a model
recognise rows whose siblings it has already seen. `video_file` is the group.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.ml.features import (
    ClipReference,
    FrameSample,
    build_frame_features,
)

#: Columns excluded from every model, and why. Enforced by a test.
EXCLUDED_COLUMNS: Mapping[str, str] = {
    "spine_angle": (
        "Exact duplicate of left_hip_angle; the augmentation perturbs one but "
        "not the other, so their difference identifies the label group exactly."
    ),
    "symmetry_score": (
        "Never recomputed after perturbation, so the asymmetric class carries "
        "the correct frame's symmetry score."
    ),
}

#: Corpus label integers, from its README.
LABEL_CORRECT = 0
FAULT_LABELS: Mapping[str, int] = {
    "shallow": 1,
    "forward_lean": 2,
    "knee_valgus": 3,
    "heel_lift": 4,
    "asymmetry": 5,
}


@dataclass(frozen=True, slots=True)
class CorpusRow:
    """One CSV row, reduced to what training needs."""

    video: str
    frame: int
    label: int
    sample: FrameSample


def dataset_digest(path: Path) -> str:
    """SHA-256 of the corpus file, recorded in the model card.

    Pins exactly which revision of the data produced a given artifact. Without
    it, "retrained on the squat dataset" is not a reproducible statement.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_from_row(row: Mapping[str, float]) -> FrameSample:
    """Restate one corpus row in canonical orientation.

    Every sign here is chosen to match `app/ml/adapter.py`, which does the same
    job for FormVision's own data. The two must agree or the model is trained on
    one convention and applied to its mirror image.

    Knee and hip angles fall as the joint closes, so flexion is their negative,
    exactly as in the adapter. The corpus's `hip_depth` needs *no* flip, which is
    the one place the two sources genuinely differ: it grows as the lifter
    descends, where FormVision's hip height shrinks. The adapter negates its
    own; this does not negate this one. Both arrive as `depth_phase` growing
    with depth.
    """
    return FrameSample(
        knee_flexion_left=-row["left_knee_angle"],
        knee_flexion_right=-row["right_knee_angle"],
        hip_flexion_left=-row["left_hip_angle"],
        hip_flexion_right=-row["right_hip_angle"],
        ankle_openness_left=row["left_ankle_angle"],
        ankle_openness_right=row["right_ankle_angle"],
        valgus_left=row["left_knee_lateral"],
        valgus_right=row["right_knee_lateral"],
        torso_lean=row["torso_lean"],
        depth_phase=row["hip_depth"],
    )


def load_rows(path: Path) -> list[CorpusRow]:
    """Read the CSV into canonical samples, dropping the leak columns."""
    frame = pd.read_csv(path)

    missing = set(EXCLUDED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"corpus is missing expected columns: {sorted(missing)}")
    frame = frame.drop(columns=list(EXCLUDED_COLUMNS))

    return [
        CorpusRow(
            video=str(record["video_file"]),
            frame=int(record["frame"]),
            label=int(record["label"]),
            sample=_sample_from_row(record),
        )
        for record in frame.to_dict("records")
    ]


def clip_references(rows: Sequence[CorpusRow]) -> dict[str, ClipReference]:
    """Build each video's reference distribution from its *unperturbed* frames.

    This is the one modelling choice in the corpus layer worth arguing about.

    At inference the reference is the whole clip, because there is no label to
    filter on. Here it is label 0 only. Using every row instead would build the
    reference from a population that is five-sixths synthetic faults, which no
    real clip ever resembles, and would let the perturbations shift the very
    scale used to judge them.

    Label 0 rows are therefore both the reference *and* the negative class. That
    is intentional, not a leak: it makes a normal frame rank roughly uniformly,
    and a perturbed frame rank high against normal movement, which is exactly
    the comparison a real clip reproduces. It is also the origin of the
    documented blind spot, that a fault present in every frame of a clip raises
    the reference along with itself and stops standing out.
    """
    by_video: dict[str, list[FrameSample]] = {}
    for row in rows:
        if row.label == LABEL_CORRECT:
            by_video.setdefault(row.video, []).append(row.sample)

    return {
        video: ClipReference.from_samples(samples) for video, samples in by_video.items()
    }


def build_matrix(
    rows: Sequence[CorpusRow],
    references: Mapping[str, ClipReference],
    feature_names: Sequence[str],
) -> tuple[list[list[float | None]], list[int], list[str]]:
    """Project rows onto one fault's feature list.

    Returns the feature matrix, the raw labels, and the video of each row so the
    caller can group its splits. Feature construction goes through
    `build_frame_features`, the same call the live predictor makes.
    """
    matrix: list[list[float | None]] = []
    labels: list[int] = []
    groups: list[str] = []

    for row in rows:
        reference = references.get(row.video)
        if reference is None:
            continue

        features = build_frame_features(row.sample, reference)
        matrix.append([features[name] for name in feature_names])
        labels.append(row.label)
        groups.append(row.video)

    return matrix, labels, groups
