"""Quadrant-based threshold contradiction detection (QBTD).

Task 4.2 selects a distance threshold from the current fragment's combined
VA quadrant, then compares the strongest cross-modality VA distance against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.core.types import MODALITIES, VAConfidence
from src.utils.config_loader import load_config

DEFAULT_QUADRANT_THRESHOLDS: tuple[float, float, float, float] = (
    0.50,
    0.65,
    0.70,
    0.55,
)
QUADRANTS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")


@dataclass(frozen=True)
class QuadrantThresholdConfig:
    """Configuration for QBTD threshold selection.

    ``quadrant_thresholds`` are ordered as Q1, Q2, Q3, Q4.
    """

    enabled: bool = True
    quadrant_thresholds: tuple[float, float, float, float] = (
        DEFAULT_QUADRANT_THRESHOLDS
    )
    use_confidence_weights: bool = True

    @classmethod
    def from_pipeline(
        cls,
        pipeline_config: Mapping[str, Any] | None = None,
    ) -> QuadrantThresholdConfig:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        l4 = (
            pipeline_config.get("pipeline", {})
            .get("stages", {})
            .get("L4", {})
        )
        return cls(
            enabled=bool(l4.get("quadrant_threshold_enabled", True)),
            quadrant_thresholds=_validate_thresholds(
                l4.get("quadrant_thresholds", DEFAULT_QUADRANT_THRESHOLDS)
            ),
            use_confidence_weights=bool(l4.get("use_confidence_weights", True)),
        )


@dataclass(frozen=True)
class QBTDResult:
    """Auxiliary QBTD signal for downstream L4 routing.

    ``exceeds_threshold`` indicates whether the current VA distance crosses the
    selected quadrant threshold. It is intentionally only a reference signal;
    downstream routing can combine it with the continuous L4.1 disagreement
    score.
    """

    enabled: bool
    exceeds_threshold: bool
    intensity: float
    strength_reference: float
    max_distance: float
    threshold: float
    quadrant: str
    combined_valence: float
    combined_arousal: float

    @property
    def has_contradiction(self) -> bool:
        """Backward-compatible alias for the threshold crossing flag."""
        return self.exceeds_threshold

    @property
    def combined_va(self) -> tuple[float, float]:
        return self.combined_valence, self.combined_arousal

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exceeds_threshold": self.exceeds_threshold,
            "has_contradiction": self.exceeds_threshold,
            "intensity": float(self.intensity),
            "strength_reference": float(self.strength_reference),
            "max_distance": float(self.max_distance),
            "threshold": float(self.threshold),
            "quadrant": self.quadrant,
            "combined_va": [float(self.combined_valence), float(self.combined_arousal)],
        }


def determine_quadrant(valence: float, arousal: float) -> str:
    """Return Q1-Q4 for a VA point.

    Boundary values are assigned to the non-negative side so every valid VA
    point maps to exactly one quadrant.
    """
    v = _finite_float(valence, "valence")
    a = _finite_float(arousal, "arousal")
    if v >= 0.0 and a >= 0.0:
        return "Q1"
    if v < 0.0 and a >= 0.0:
        return "Q2"
    if v < 0.0 and a < 0.0:
        return "Q3"
    return "Q4"


def quadrant_index(quadrant: str) -> int:
    normalized = str(quadrant).strip().upper()
    if normalized not in QUADRANTS:
        raise ValueError(f"quadrant must be one of {QUADRANTS}, got {quadrant!r}")
    return QUADRANTS.index(normalized)


def threshold_for_quadrant(
    quadrant: str,
    config: QuadrantThresholdConfig | None = None,
) -> float:
    config = config or QuadrantThresholdConfig.from_pipeline()
    return float(config.quadrant_thresholds[quadrant_index(quadrant)])


def combined_va(
    va_inter: Mapping[str, Any],
    *,
    use_confidence_weights: bool = True,
) -> tuple[float, float]:
    """Compute a fragment-level VA point from modality ``VA_inter`` values."""
    entries = [_coerce_va(value) for _, value in _ordered_items(va_inter)]
    if not entries:
        raise ValueError("va_inter must contain at least one modality")

    if use_confidence_weights:
        total_confidence = sum(max(confidence, 0.0) for _, _, confidence in entries)
        if total_confidence > 0.0:
            return (
                sum(v * max(c, 0.0) for v, _, c in entries) / total_confidence,
                sum(a * max(c, 0.0) for _, a, c in entries) / total_confidence,
            )

    count = len(entries)
    return (
        sum(v for v, _, _ in entries) / count,
        sum(a for _, a, _ in entries) / count,
    )


def max_distance_from_matrix(distance_matrix: Any) -> float:
    """Return the maximum finite value from a pairwise distance matrix."""
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return 0.0
    return float(np.max(finite_values))


def evaluate_qbtd(
    va_inter: Mapping[str, Any],
    max_distance: float | None = None,
    *,
    distance_matrix: Any | None = None,
    config: QuadrantThresholdConfig | None = None,
) -> QBTDResult:
    """Evaluate QBTD for a fragment.

    Provide either ``max_distance`` from L4.1 or ``distance_matrix``. If both are
    supplied, ``max_distance`` takes precedence.
    """
    if max_distance is None:
        if distance_matrix is None:
            raise ValueError("either max_distance or distance_matrix is required")
        max_distance = max_distance_from_matrix(distance_matrix)

    config = config or QuadrantThresholdConfig.from_pipeline()
    combined_valence, combined_arousal = combined_va(
        va_inter,
        use_confidence_weights=config.use_confidence_weights,
    )
    quadrant = determine_quadrant(combined_valence, combined_arousal)
    threshold = threshold_for_quadrant(quadrant, config)
    distance = float(max_distance)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("max_distance must be finite and non-negative")

    exceeds_threshold = config.enabled and distance > threshold
    return QBTDResult(
        enabled=config.enabled,
        exceeds_threshold=exceeds_threshold,
        intensity=distance if exceeds_threshold else 0.0,
        strength_reference=_strength_reference(distance, threshold, config.enabled),
        max_distance=distance,
        threshold=threshold,
        quadrant=quadrant,
        combined_valence=combined_valence,
        combined_arousal=combined_arousal,
    )


def evaluate_qbtd_from_distance_result(
    va_inter: Mapping[str, Any],
    distance_result: Any,
    *,
    config: QuadrantThresholdConfig | None = None,
) -> QBTDResult:
    """Evaluate QBTD from the L4.1 ``VADistanceResult`` object.

    The import is intentionally duck-typed so this module remains usable while
    task 4.1 lands on a separate branch/PR.
    """
    if distance_result is None:
        raise ValueError("distance_result must not be None")
    if not hasattr(distance_result, "max_distance"):
        raise TypeError("distance_result must expose max_distance")
    return evaluate_qbtd(
        va_inter,
        float(distance_result.max_distance),
        config=config,
    )


def has_quadrant_contradiction(
    va_inter: Mapping[str, Any],
    max_distance: float | None = None,
    *,
    distance_matrix: Any | None = None,
    config: QuadrantThresholdConfig | None = None,
) -> tuple[bool, float]:
    """Compatibility helper returning ``(has_contradiction, intensity)``."""
    result = evaluate_qbtd(
        va_inter,
        max_distance,
        distance_matrix=distance_matrix,
        config=config,
    )
    return result.has_contradiction, result.intensity


def _validate_thresholds(values: Sequence[Any]) -> tuple[float, float, float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("quadrant_thresholds must be a sequence of four numbers")
    if len(values) != 4:
        raise ValueError("quadrant_thresholds must contain exactly four values")

    thresholds = tuple(_finite_float(value, "quadrant_threshold") for value in values)
    if any(value < 0.0 for value in thresholds):
        raise ValueError("quadrant_thresholds must be finite and non-negative")
    return thresholds  # type: ignore[return-value]


def _strength_reference(distance: float, threshold: float, enabled: bool) -> float:
    if not enabled:
        return 0.0
    if threshold <= 0.0:
        return 1.0 if distance > 0.0 else 0.0
    return min(1.0, max(0.0, distance / threshold))


def _ordered_items(va_inter: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    yielded = set()
    for modality in MODALITIES:
        if modality in va_inter:
            yielded.add(modality)
            yield modality, va_inter[modality]
    for modality, value in va_inter.items():
        if modality not in yielded:
            yield modality, value


def _coerce_va(value: Any) -> tuple[float, float, float]:
    if isinstance(value, VAConfidence):
        return _validated_va(value.valence, value.arousal, value.confidence)
    if isinstance(value, Mapping):
        try:
            valence = value["valence"]
            arousal = value["arousal"]
        except KeyError as exc:
            raise ValueError(
                "VA mapping values must contain valence and arousal"
            ) from exc
        return _validated_va(valence, arousal, value.get("confidence", 1.0))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            raise ValueError("VA sequence values must contain valence and arousal")
        confidence = value[2] if len(value) > 2 else 1.0
        return _validated_va(value[0], value[1], confidence)
    raise TypeError(
        "VA values must be VAConfidence, mapping, or sequence of valence/arousal"
    )


def _validated_va(
    valence: Any,
    arousal: Any,
    confidence: Any,
) -> tuple[float, float, float]:
    v = _finite_float(valence, "valence")
    a = _finite_float(arousal, "arousal")
    c = _finite_float(confidence, "confidence")
    if not -1.0 <= v <= 1.0 or not -1.0 <= a <= 1.0:
        raise ValueError("VA valence and arousal must be in [-1, 1]")
    return v, a, c


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "DEFAULT_QUADRANT_THRESHOLDS",
    "QBTDResult",
    "QUADRANTS",
    "QuadrantThresholdConfig",
    "combined_va",
    "determine_quadrant",
    "evaluate_qbtd",
    "evaluate_qbtd_from_distance_result",
    "has_quadrant_contradiction",
    "max_distance_from_matrix",
    "quadrant_index",
    "threshold_for_quadrant",
]
