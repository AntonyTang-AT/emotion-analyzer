"""Selective gating router for L4.6 (SER).

Chooses between default equal-weight fusion and type-table candidate weights
under disagreement-score and confidence gates. Session-level switch rate is
capped by ``max_switch_rate`` (cold-start default 0.10; recalibrate on D9-min).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.types import MODALITIES, ContradictionType
from src.utils.config_loader import load_config

DEFAULT_FUSION = "default_fusion"
TYPED_FUSION = "typed_fusion"

DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_MAX_SWITCH_RATE = 0.10
DEFAULT_STRATEGY = "weighted_fusion"
DEFAULT_MIN_DISAGREEMENT_SCORE = 0.35
WEIGHT_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class SelectiveRouterConfig:
    """Configuration from ``fusion_policy.yaml`` SER (+ DTRB disagreement floor)."""

    enabled: bool = True
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    max_switch_rate: float = DEFAULT_MAX_SWITCH_RATE
    default_strategy: str = DEFAULT_STRATEGY
    min_disagreement_score: float = DEFAULT_MIN_DISAGREEMENT_SCORE

    @classmethod
    def from_fusion_policy(
        cls,
        fusion_policy: Mapping[str, Any] | None = None,
    ) -> SelectiveRouterConfig:
        if fusion_policy is None:
            fusion_policy = load_config("fusion_policy")
        ser = fusion_policy.get("ser", {})
        dtrb = fusion_policy.get("dtrb", {})
        trigger = dtrb.get("trigger", {}) if isinstance(dtrb, Mapping) else {}
        return cls(
            enabled=bool(ser.get("enabled", True)),
            confidence_threshold=_confidence_value(
                ser.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
                "ser.confidence_threshold",
            ),
            max_switch_rate=_rate_value(
                ser.get("max_switch_rate", DEFAULT_MAX_SWITCH_RATE),
                "ser.max_switch_rate",
            ),
            default_strategy=str(ser.get("default_strategy", DEFAULT_STRATEGY)),
            min_disagreement_score=_non_negative_float(
                trigger.get("min_disagreement_score", DEFAULT_MIN_DISAGREEMENT_SCORE),
                "dtrb.trigger.min_disagreement_score",
            ),
        )


@dataclass(frozen=True)
class RoutingResult:
    """Per-segment routing decision after session-level switch-rate capping."""

    routing_decision: str
    final_weights: list[float]
    audit_entries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_decision": self.routing_decision,
            "final_weights": [float(weight) for weight in self.final_weights],
            "audit_entries": [dict(entry) for entry in self.audit_entries],
        }


@dataclass
class _SegmentCandidate:
    index: int
    candidate_weights: list[float]
    routing_confidence: float
    disagreement_score: float
    contradiction_type: ContradictionType
    is_candidate: bool
    reject_reason: str | None = None


@dataclass
class RoutingSession:
    """Collect segment evaluations, then finalize under ``max_switch_rate``."""

    total_segments: int
    config: SelectiveRouterConfig | None = None
    _records: dict[int, _SegmentCandidate] = field(default_factory=dict, init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.total_segments < 0:
            raise ValueError("total_segments must be non-negative")
        if self.config is None:
            self.config = SelectiveRouterConfig.from_fusion_policy()
        if not isinstance(self.config, SelectiveRouterConfig):
            raise TypeError("config must be SelectiveRouterConfig")

    def consider(
        self,
        *,
        index: int,
        candidate_weights: Sequence[float],
        routing_confidence: float,
        disagreement_score: float,
        contradiction_type: ContradictionType | str,
    ) -> None:
        if self._finalized:
            raise RuntimeError("RoutingSession already finalized")
        if not 0 <= index < self.total_segments:
            raise ValueError(
                f"index must be in [0, {self.total_segments}), got {index}"
            )
        if index in self._records:
            raise ValueError(f"segment index {index} already considered")

        weights = _validated_weights(candidate_weights, "candidate_weights")
        confidence = _confidence_value(routing_confidence, "routing_confidence")
        score = _non_negative_float(disagreement_score, "disagreement_score")
        if score > 1.0:
            raise ValueError("disagreement_score must be in [0, 1]")
        ctype = _normalize_type(contradiction_type)

        eligible, reason = is_switch_candidate(
            disagreement_score=score,
            routing_confidence=confidence,
            contradiction_type=ctype,
            config=self.config,
        )
        self._records[index] = _SegmentCandidate(
            index=index,
            candidate_weights=weights,
            routing_confidence=confidence,
            disagreement_score=score,
            contradiction_type=ctype,
            is_candidate=eligible,
            reject_reason=None if eligible else reason,
        )

    def finalize(self) -> list[RoutingResult]:
        if self._finalized:
            raise RuntimeError("RoutingSession already finalized")
        if len(self._records) != self.total_segments:
            missing = sorted(
                set(range(self.total_segments)) - set(self._records)
            )
            raise ValueError(
                f"missing segment indices before finalize: {missing}"
            )

        max_switches = int(self.total_segments * self.config.max_switch_rate)
        eligible = [
            record
            for record in self._records.values()
            if record.is_candidate
        ]
        eligible.sort(
            key=lambda item: (-item.routing_confidence, item.index)
        )
        kept_indices = {item.index for item in eligible[:max_switches]}
        rejected_for_rate = {
            item.index for item in eligible if item.index not in kept_indices
        }

        results: list[RoutingResult] = []
        for index in range(self.total_segments):
            record = self._records[index]
            if index in kept_indices:
                results.append(
                    RoutingResult(
                        routing_decision=TYPED_FUSION,
                        final_weights=list(record.candidate_weights),
                        audit_entries=[
                            {
                                "action": "typed_fusion",
                                "index": index,
                                "routing_confidence": record.routing_confidence,
                                "disagreement_score": record.disagreement_score,
                                "contradiction_type": record.contradiction_type.value,
                                "confidence_threshold": self.config.confidence_threshold,
                                "min_disagreement_score": self.config.min_disagreement_score,
                                "max_switch_rate": self.config.max_switch_rate,
                                "max_switches": max_switches,
                            }
                        ],
                    )
                )
            else:
                audit: list[dict[str, Any]] = []
                if index in rejected_for_rate:
                    audit.append(
                        {
                            "action": "rejected_by_switch_rate",
                            "index": index,
                            "routing_confidence": record.routing_confidence,
                            "disagreement_score": record.disagreement_score,
                            "contradiction_type": record.contradiction_type.value,
                            "max_switch_rate": self.config.max_switch_rate,
                            "max_switches": max_switches,
                        }
                    )
                results.append(
                    RoutingResult(
                        routing_decision=DEFAULT_FUSION,
                        final_weights=default_fusion_weights(),
                        audit_entries=audit,
                    )
                )

        self._finalized = True
        return results


def default_fusion_weights() -> list[float]:
    """Equal weights for the four modalities."""
    return [1.0 / len(MODALITIES)] * len(MODALITIES)


def is_switch_candidate(
    *,
    disagreement_score: float,
    routing_confidence: float,
    contradiction_type: ContradictionType | str,
    config: SelectiveRouterConfig | None = None,
) -> tuple[bool, str | None]:
    """Return whether a segment passes SER gates (ignores switch-rate cap)."""
    router_config = config if config is not None else SelectiveRouterConfig()
    ctype = _normalize_type(contradiction_type)
    score = float(disagreement_score)
    confidence = float(routing_confidence)

    if not router_config.enabled:
        return False, "ser_disabled"
    if ctype == ContradictionType.CONSISTENT:
        return False, "consistent_type"
    if score < router_config.min_disagreement_score:
        return False, "low_disagreement"
    if confidence < router_config.confidence_threshold:
        return False, "low_confidence"
    return True, None


def _normalize_type(value: ContradictionType | str) -> ContradictionType:
    if isinstance(value, ContradictionType):
        return value
    return ContradictionType(str(value).strip().lower())


def _validated_weights(values: Sequence[Any], name: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of {len(MODALITIES)} weights")
    if len(values) != len(MODALITIES):
        raise ValueError(f"{name} must contain {len(MODALITIES)} weights")
    weights = [
        _non_negative_float(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    ]
    total = sum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"{name} must sum to 1.0 (±{WEIGHT_SUM_TOLERANCE}), got {total:.4f}"
        )
    return weights


def _confidence_value(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _rate_value(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
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
    "DEFAULT_FUSION",
    "TYPED_FUSION",
    "RoutingResult",
    "RoutingSession",
    "SelectiveRouterConfig",
    "default_fusion_weights",
    "is_switch_candidate",
]
