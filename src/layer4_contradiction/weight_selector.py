"""Rule-table fusion weight selector for L4.4.

This module provides the cold-start/fallback fusion prior. It maps the
interpreted contradiction type to four modality weights ordered as
``[text, speech, macro, micro]`` and returns an initial routing confidence.
Final routing remains the responsibility of the later selective router.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.core.types import MODALITIES, ContradictionType
from src.utils.config_loader import load_config

DEFAULT_ROUTING_CONFIDENCE_DIVISOR = 1.2
DEFAULT_CONSISTENT_WEIGHTS: tuple[float, float, float, float] = (
    0.25,
    0.25,
    0.25,
    0.25,
)
WEIGHT_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class WeightSelectorConfig:
    """Configuration for the L4.4 rule-table fallback selector."""

    enabled: bool = True
    weight_strategy: str = "rule_table"
    routing_confidence_divisor: float = DEFAULT_ROUTING_CONFIDENCE_DIVISOR
    low_confidence_threshold: float = 0.0

    @classmethod
    def from_pipeline(
        cls,
        pipeline_config: Mapping[str, Any] | None = None,
    ) -> WeightSelectorConfig:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        l4 = (
            pipeline_config.get("pipeline", {})
            .get("stages", {})
            .get("L4", {})
        )
        return cls(
            enabled=bool(l4.get("enabled", True)),
            weight_strategy=str(l4.get("weight_strategy", "rule_table")),
            routing_confidence_divisor=_positive_float(
                l4.get(
                    "routing_confidence_divisor",
                    DEFAULT_ROUTING_CONFIDENCE_DIVISOR,
                ),
                "routing_confidence_divisor",
            ),
            low_confidence_threshold=_confidence_value(
                l4.get("low_confidence_threshold", 0.0),
                "low_confidence_threshold",
            ),
        )


@dataclass(frozen=True)
class WeightSelectionResult:
    """Fusion weights and the initial confidence for a candidate route."""

    weights: list[float]
    routing_confidence: float
    contradiction_type: ContradictionType
    used_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": [float(weight) for weight in self.weights],
            "routing_confidence": float(self.routing_confidence),
            "contradiction_type": self.contradiction_type.value,
            "used_fallback": self.used_fallback,
        }


def get_weights(
    contradiction_type: ContradictionType | str,
    intensity: float,
    confidences: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    config: WeightSelectorConfig | None = None,
    weight_table: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[list[float], float]:
    """Return ``(weights, routing_confidence)`` for a contradiction type.

    Unknown types, disabled strategies, and low-confidence routes fall back to
    ``consistent`` equal weights. When per-modality confidences are supplied,
    their mean caps the normalized intensity confidence.
    """
    result = select_weights(
        contradiction_type,
        intensity,
        confidences,
        config=config,
        weight_table=weight_table,
    )
    return result.weights, result.routing_confidence


def select_weights(
    contradiction_type: ContradictionType | str,
    intensity: float,
    confidences: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    config: WeightSelectorConfig | None = None,
    weight_table: Mapping[str, Sequence[Any]] | None = None,
) -> WeightSelectionResult:
    """Return a detailed L4.4 weight selection result."""
    selector_config = config or WeightSelectorConfig.from_pipeline()
    table = _validated_table(
        weight_table if weight_table is not None else load_config("weight_table")
    )
    route_confidence = _routing_confidence(
        intensity,
        selector_config.routing_confidence_divisor,
        confidences,
    )

    normalized_type = _normalize_type(contradiction_type)
    low_confidence_fallback = (
        selector_config.low_confidence_threshold > 0.0
        and route_confidence < selector_config.low_confidence_threshold
    )
    use_fallback = (
        normalized_type is None
        or not selector_config.enabled
        or selector_config.weight_strategy != "rule_table"
        or normalized_type.value not in table
        or low_confidence_fallback
    )
    selected_type = (
        ContradictionType.CONSISTENT if use_fallback else normalized_type
    )
    weights = list(table.get(selected_type.value, DEFAULT_CONSISTENT_WEIGHTS))

    return WeightSelectionResult(
        weights=weights,
        routing_confidence=route_confidence,
        contradiction_type=selected_type,
        used_fallback=use_fallback,
    )


def _routing_confidence(
    intensity: float,
    divisor: float,
    confidences: Mapping[str, Any] | Sequence[Any] | None,
) -> float:
    value = _non_negative_float(intensity, "intensity")
    base = min(1.0, value / divisor)
    if confidences is None:
        return base
    return base * _mean_confidence(confidences)


def _mean_confidence(confidences: Mapping[str, Any] | Sequence[Any]) -> float:
    if isinstance(confidences, Mapping):
        unknown = [name for name in confidences if name not in MODALITIES]
        if unknown:
            raise ValueError(f"unknown confidence modalities: {unknown}")
        values = [
            _confidence_value(value, f"confidences.{name}")
            for name, value in confidences.items()
        ]
    elif isinstance(confidences, Sequence) and not isinstance(confidences, (str, bytes)):
        values = [
            _confidence_value(value, f"confidences[{index}]")
            for index, value in enumerate(confidences)
        ]
    else:
        raise TypeError("confidences must be a mapping, sequence, or None")

    if not values:
        return 1.0
    return sum(values) / len(values)


def _normalize_type(value: ContradictionType | str) -> ContradictionType | None:
    if isinstance(value, ContradictionType):
        return value
    try:
        return ContradictionType(str(value).strip().lower())
    except ValueError:
        return None


def _validated_table(table: Mapping[str, Sequence[Any]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key, values in table.items():
        normalized_type = _normalize_type(str(key))
        if normalized_type is None:
            continue
        result[normalized_type.value] = _validated_weights(
            values,
            f"weight_table.{normalized_type.value}",
        )

    if ContradictionType.CONSISTENT.value not in result:
        result[ContradictionType.CONSISTENT.value] = list(DEFAULT_CONSISTENT_WEIGHTS)
    return result


def _validated_weights(values: Sequence[Any], name: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of {len(MODALITIES)} weights")
    if len(values) != len(MODALITIES):
        raise ValueError(f"{name} must contain {len(MODALITIES)} weights")

    weights = [
        _non_negative_float(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    ]
    total = sum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1.0 (±{WEIGHT_SUM_TOLERANCE})")
    return weights


def _confidence_value(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "DEFAULT_CONSISTENT_WEIGHTS",
    "DEFAULT_ROUTING_CONFIDENCE_DIVISOR",
    "WeightSelectionResult",
    "WeightSelectorConfig",
    "get_weights",
    "select_weights",
]
