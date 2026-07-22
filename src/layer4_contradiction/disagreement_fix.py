"""High-disagreement local correction for L4.5.

Only fragments that exceed configured disagreement thresholds are adjusted.
Correction changes fusion weights within ``max_va_adjustment``; expert VA
predictions are never rewritten. MVP ignores ``reason_text`` unless
``reason_guided`` is enabled after AffectGPT integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.core.types import MODALITIES, VAConfidence
from src.utils.config_loader import load_config

DEFAULT_FUSION = "default_fusion"
DISAGREEMENT_FIX = "disagreement_fix"


@dataclass(frozen=True)
class DisagreementFixConfig:
    """Configuration loaded from ``fusion_policy.yaml`` DTRB section."""

    enabled: bool = True
    min_va_distance: float = 0.6
    min_disagreement_score: float = 0.35
    max_va_adjustment: float = 0.15
    reason_guided: bool = False

    @classmethod
    def from_fusion_policy(
        cls,
        fusion_policy: Mapping[str, Any] | None = None,
    ) -> DisagreementFixConfig:
        if fusion_policy is None:
            fusion_policy = load_config("fusion_policy")
        dtrb = fusion_policy.get("dtrb", {})
        trigger = dtrb.get("trigger", {}) if isinstance(dtrb, Mapping) else {}
        return cls(
            enabled=bool(dtrb.get("enabled", True)),
            min_va_distance=float(trigger.get("min_va_distance", 0.6)),
            min_disagreement_score=float(trigger.get("min_disagreement_score", 0.35)),
            max_va_adjustment=float(dtrb.get("max_va_adjustment", 0.15)),
            reason_guided=bool(dtrb.get("reason_guided", False)),
        )


@dataclass(frozen=True)
class DisagreementFixResult:
    """Bounded weight correction result with optional audit trail."""

    weights: list[float]
    adjusted: bool
    routing_decision: str
    audit_entries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": [float(weight) for weight in self.weights],
            "adjusted": self.adjusted,
            "routing_decision": self.routing_decision,
            "audit_entries": [dict(entry) for entry in self.audit_entries],
        }


def apply_disagreement_fix(
    weights: Sequence[float],
    modality_va: Mapping[str, Any],
    *,
    disagreement_score: float,
    max_distance: float,
    max_pair: tuple[str, str] | None = None,
    confidences: Mapping[str, float] | None = None,
    reason_text: str | None = None,
    prior_decision: str = DEFAULT_FUSION,
    config: DisagreementFixConfig | None = None,
) -> DisagreementFixResult:
    """Apply bounded local correction when disagreement thresholds are met.

    Below threshold (or when disabled), returns normalized input weights
    unchanged with an empty audit trail. When triggered, lowers weight on the
    low-confidence outlier modality, renormalizes, and caps fused-VA shift by
    ``max_va_adjustment``.
    """
    fix_config = config if config is not None else DisagreementFixConfig.from_fusion_policy()
    current_weights = _normalize_weights(weights)
    prior = str(prior_decision)

    if not fix_config.enabled:
        return DisagreementFixResult(
            weights=current_weights,
            adjusted=False,
            routing_decision=prior,
            audit_entries=[],
        )

    if (
        float(disagreement_score) < fix_config.min_disagreement_score
        or float(max_distance) < fix_config.min_va_distance
    ):
        return DisagreementFixResult(
            weights=current_weights,
            adjusted=False,
            routing_decision=prior,
            audit_entries=[],
        )

    # Reserved for AffectGPT evidence gating; MVP must not depend on reason.
    if fix_config.reason_guided and reason_text:
        pass

    resolved_confidences = {
        modality: (
            float(confidences[modality])
            if confidences and modality in confidences
            else _coerce_confidence(modality_va[modality], modality)
        )
        for modality in MODALITIES
        if modality in modality_va
    }

    fused_before = _weighted_fusion_va(current_weights, modality_va)
    outlier_modalities = _select_outliers(
        resolved_confidences,
        modality_va,
        fused_before,
        max_pair=max_pair,
    )
    if not outlier_modalities:
        return DisagreementFixResult(
            weights=current_weights,
            adjusted=False,
            routing_decision=prior,
            audit_entries=[],
        )

    adjusted_weights = list(current_weights)
    reduction_budget = fix_config.max_va_adjustment
    for modality in outlier_modalities:
        index = MODALITIES.index(modality)
        original = adjusted_weights[index]
        reduction = min(original * 0.5, reduction_budget)
        adjusted_weights[index] = max(0.0, original - reduction)

    candidate_weights = _normalize_weights(adjusted_weights)
    fused_after = _weighted_fusion_va(candidate_weights, modality_va)
    shift = _va_distance(fused_before, fused_after)
    if shift > fix_config.max_va_adjustment and shift > 0.0:
        scale = fix_config.max_va_adjustment / shift
        candidate_weights = _normalize_weights(
            [
                current_weights[index]
                + scale * (candidate_weights[index] - current_weights[index])
                for index in range(len(MODALITIES))
            ]
        )
        fused_after = _weighted_fusion_va(candidate_weights, modality_va)
        shift = _va_distance(fused_before, fused_after)

    if candidate_weights == current_weights:
        return DisagreementFixResult(
            weights=current_weights,
            adjusted=False,
            routing_decision=prior,
            audit_entries=[],
        )

    audit_entry = {
        "action": "disagreement_fix",
        "trigger": {
            "disagreement_score": float(disagreement_score),
            "max_distance": float(max_distance),
            "min_disagreement_score": fix_config.min_disagreement_score,
            "min_va_distance": fix_config.min_va_distance,
        },
        "outlier_modalities": list(outlier_modalities),
        "weights_before": current_weights,
        "weights_after": candidate_weights,
        "fused_va_before": list(fused_before),
        "fused_va_after": list(fused_after),
        "va_shift": float(shift),
        "max_va_adjustment": fix_config.max_va_adjustment,
    }
    return DisagreementFixResult(
        weights=candidate_weights,
        adjusted=True,
        routing_decision=DISAGREEMENT_FIX,
        audit_entries=[audit_entry],
    )


def _select_outliers(
    confidences: Mapping[str, float],
    modality_va: Mapping[str, Any],
    fused_va: tuple[float, float],
    *,
    max_pair: tuple[str, str] | None,
) -> list[str]:
    if max_pair is not None:
        left, right = max_pair
        if left in confidences and right in confidences:
            if confidences[left] <= confidences[right]:
                return [left]
            return [right]

    distances: list[tuple[float, float, str]] = []
    for modality in confidences:
        if modality not in modality_va:
            continue
        point = _coerce_va(modality_va[modality], modality)
        distances.append(
            (
                _va_distance(point, fused_va),
                confidences[modality],
                modality,
            )
        )
    distances.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not distances:
        return []
    return [distances[0][2]]


def _coerce_confidence(value: Any, modality: str) -> float:
    if isinstance(value, VAConfidence):
        confidence = value.confidence
    elif isinstance(value, Mapping):
        confidence = value.get("confidence", 1.0)
    else:
        confidence = 1.0
    try:
        number = float(confidence)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"confidence for {modality!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"confidence for {modality!r} must be finite")
    return max(0.0, min(1.0, number))


def _coerce_va(value: Any, modality: str) -> tuple[float, float]:
    if isinstance(value, VAConfidence):
        point = (value.valence, value.arousal)
    elif isinstance(value, Mapping):
        point = (value["valence"], value["arousal"])
    else:
        point = (value[0], value[1])
    return float(point[0]), float(point[1])


def _weighted_fusion_va(
    weights: Sequence[float],
    modality_va: Mapping[str, Any],
) -> tuple[float, float]:
    total_weight = 0.0
    valence = 0.0
    arousal = 0.0
    for index, modality in enumerate(MODALITIES):
        if modality not in modality_va:
            continue
        weight = float(weights[index])
        if weight <= 0.0:
            continue
        v_point, a_point = _coerce_va(modality_va[modality], modality)
        valence += weight * v_point
        arousal += weight * a_point
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0, 0.0
    return valence / total_weight, arousal / total_weight


def _va_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        return [1.0 / len(MODALITIES)] * len(MODALITIES)
    return [float(weight) / total for weight in weights]


__all__ = [
    "DEFAULT_FUSION",
    "DISAGREEMENT_FIX",
    "DisagreementFixConfig",
    "DisagreementFixResult",
    "apply_disagreement_fix",
]
