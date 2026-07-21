"""VA-space distances used as the primary L4 disagreement signal."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.types import MODALITIES, VAConfidence

MAX_VA_DISTANCE = 2.0 * math.sqrt(2.0)


@dataclass(frozen=True)
class VADistanceResult:
    """Pairwise distances and their normalized disagreement summary.

    ``distance_matrix`` follows the exact order in ``modalities``. A result is
    only created when at least two requested modalities have VA predictions.
    """

    modalities: tuple[str, ...]
    distance_matrix: np.ndarray
    max_distance: float
    max_pair: tuple[str, str]
    disagreement_score: float


def _coerce_va_point(value: Any, modality: str) -> tuple[float, float]:
    if isinstance(value, VAConfidence):
        valence, arousal = value.valence, value.arousal
    elif isinstance(value, Mapping):
        try:
            valence, arousal = value["valence"], value["arousal"]
        except KeyError as exc:
            raise ValueError(
                f"VA value for {modality!r} must contain valence and arousal"
            ) from exc
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                f"VA value for {modality!r} must be VAConfidence or a (v, a) pair"
            )
        if len(value) != 2:
            raise ValueError(f"VA value for {modality!r} must contain exactly 2 values")
        valence, arousal = value

    try:
        point = (float(valence), float(arousal))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"VA value for {modality!r} must be numeric") from exc

    if not all(
        math.isfinite(coordinate) and -1.0 <= coordinate <= 1.0
        for coordinate in point
    ):
        raise ValueError(
            f"VA value for {modality!r} must contain finite values in [-1, 1]"
        )
    return point


def calculate_va_distances(
    va_inter: Mapping[str, Any],
    active_modalities: Iterable[str],
) -> VADistanceResult | None:
    """Calculate pairwise VA distances for available active modalities.

    Missing predictions are ignored. When fewer than two requested modalities
    remain, ``None`` signals that L4 should be skipped. The disagreement score
    is normalized by the theoretical maximum distance in the ``[-1, 1]`` VA
    square: ``2 * sqrt(2)``.
    """

    requested_modalities = tuple(dict.fromkeys(str(item) for item in active_modalities))
    unknown_modalities = [
        modality for modality in requested_modalities if modality not in MODALITIES
    ]
    if unknown_modalities:
        raise ValueError(f"unknown active modalities: {unknown_modalities}")

    modalities = tuple(
        modality for modality in requested_modalities if modality in va_inter
    )
    if len(modalities) < 2:
        return None

    points = np.asarray(
        [_coerce_va_point(va_inter[modality], modality) for modality in modalities],
        dtype=float,
    )
    deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distance_matrix, 0.0)

    upper_rows, upper_columns = np.triu_indices(len(modalities), k=1)
    upper_distances = distance_matrix[upper_rows, upper_columns]
    maximum_index = int(np.argmax(upper_distances))
    left_index = int(upper_rows[maximum_index])
    right_index = int(upper_columns[maximum_index])
    max_distance = float(upper_distances[maximum_index])
    disagreement_score = min(1.0, max(0.0, max_distance / MAX_VA_DISTANCE))

    return VADistanceResult(
        modalities=modalities,
        distance_matrix=distance_matrix,
        max_distance=max_distance,
        max_pair=(modalities[left_index], modalities[right_index]),
        disagreement_score=disagreement_score,
    )


__all__ = [
    "MAX_VA_DISTANCE",
    "VADistanceResult",
    "calculate_va_distances",
]
