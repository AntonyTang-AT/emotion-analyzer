"""L4 contradiction detection, routing, and fusion prior orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.core.context import DataContext
from src.core.types import (
    MODALITIES,
    ContradictionResult,
    ContradictionType,
    Fragment,
    RoutingDecision,
    VAConfidence,
)
from src.layer4_contradiction.disagreement_fix import (
    DEFAULT_FUSION,
    DISAGREEMENT_FIX,
    DisagreementFixConfig,
    DisagreementFixResult,
    apply_disagreement_fix,
)
from src.layer4_contradiction.expert_rules import (
    ExpertRuleResult,
    ExpertRulesConfig,
    classify_contradiction,
    same_sign,
)
from src.layer4_contradiction.quadrant_threshold import (
    QBTDResult,
    QuadrantThresholdConfig,
    evaluate_qbtd,
)
from src.layer4_contradiction.selective_router import (
    TYPED_FUSION,
    RoutingResult,
    RoutingSession,
    SelectiveRouterConfig,
    default_fusion_weights,
    is_switch_candidate,
)
from src.layer4_contradiction.va_distance import (
    MAX_VA_DISTANCE,
    VADistanceResult,
    calculate_va_distances,
)
from src.layer4_contradiction.weight_selector import (
    WeightSelectionResult,
    WeightSelectorConfig,
    get_weights,
    select_weights,
)
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedSegment:
    fragment: Fragment
    distance_result: VADistanceResult
    rule_result: ExpertRuleResult
    weight_result: WeightSelectionResult


def run_l4(context: DataContext) -> DataContext:
    """Run L4 analysis for each segment and store a session-level summary.

    Pipeline per processable fragment:
    VA distance → expert-rule type → weight-table candidate → selective
    routing → disagreement fix. Fragments with fewer than two active
    modalities are skipped (single-modality profiles).
    """
    if not context.segments:
        context.mark_stage_failed("L4", "no segments available")
        return context

    active_modalities = context.active_modalities or list(MODALITIES)
    fusion_policy = load_config("fusion_policy")
    router_config = SelectiveRouterConfig.from_fusion_policy(fusion_policy)
    fix_config = DisagreementFixConfig.from_fusion_policy(fusion_policy)
    selector_config = WeightSelectorConfig.from_pipeline()

    prepared: list[_PreparedSegment] = []
    for fragment in context.segments:
        if not fragment.va_inter:
            logger.debug("Skipping L4 for fragment %s: no va_inter", fragment.id)
            continue

        distance_result = calculate_va_distances(fragment.va_inter, active_modalities)
        if distance_result is None:
            logger.debug(
                "Skipping L4 for fragment %s: insufficient modalities",
                fragment.id,
            )
            continue

        rule_result = classify_contradiction(fragment.va_inter)
        weight_result = select_weights(
            rule_result.contradiction_type,
            distance_result.max_distance,
            _extract_confidences(fragment.va_inter),
            config=selector_config,
        )
        prepared.append(
            _PreparedSegment(
                fragment=fragment,
                distance_result=distance_result,
                rule_result=rule_result,
                weight_result=weight_result,
            )
        )

    if not prepared:
        summary = _empty_session_summary()
        context.metadata["contradiction_summary"] = summary.to_dict()
        context.set_stage("L4", {"contradiction": summary})
        return context

    session = RoutingSession(
        total_segments=len(prepared),
        config=router_config,
    )
    for index, item in enumerate(prepared):
        session.consider(
            index=index,
            candidate_weights=item.weight_result.weights,
            routing_confidence=item.weight_result.routing_confidence,
            disagreement_score=item.distance_result.disagreement_score,
            contradiction_type=item.rule_result.contradiction_type,
        )
    routing_results = session.finalize()

    summary_events: list[dict[str, Any]] = []
    for item, routing in zip(prepared, routing_results, strict=True):
        fix_result = apply_disagreement_fix(
            routing.final_weights,
            item.fragment.va_inter,
            disagreement_score=item.distance_result.disagreement_score,
            max_distance=item.distance_result.max_distance,
            max_pair=item.distance_result.max_pair,
            confidences=_extract_confidences(item.fragment.va_inter),
            prior_decision=routing.routing_decision,
            config=fix_config,
        )
        audit_trail = [*routing.audit_entries, *fix_result.audit_entries]
        decision = RoutingDecision(fix_result.routing_decision)
        contradiction = ContradictionResult(
            contradiction_type=item.rule_result.contradiction_type,
            contradiction_intensity=item.distance_result.max_distance,
            involved_modalities=list(item.rule_result.involved_modalities),
            suggested_fusion_weights=fix_result.weights,
            routing_confidence=item.weight_result.routing_confidence,
            disagreement_score=item.distance_result.disagreement_score,
            routing_decision=decision,
            fusion_audit_trail=audit_trail,
        )
        item.fragment.contradiction = contradiction

        if decision != RoutingDecision.DEFAULT_FUSION:
            summary_events.append(
                {
                    "fragment_id": item.fragment.id,
                    "contradiction_type": item.rule_result.contradiction_type.value,
                    "routing_decision": decision.value,
                    "disagreement_score": item.distance_result.disagreement_score,
                }
            )

    summary = _build_session_summary(context.segments, summary_events)
    context.metadata["contradiction_summary"] = summary.to_dict()
    context.set_stage("L4", {"contradiction": summary})
    return context


def _extract_confidences(modality_va: dict[str, VAConfidence]) -> dict[str, float]:
    return {
        modality: float(value.confidence)
        for modality, value in modality_va.items()
        if isinstance(value, VAConfidence)
    }


def _empty_session_summary() -> ContradictionResult:
    return ContradictionResult(
        contradiction_type=ContradictionType.CONSISTENT,
        contradiction_intensity=0.0,
        involved_modalities=[],
        suggested_fusion_weights=default_fusion_weights(),
        routing_confidence=0.0,
        disagreement_score=0.0,
        routing_decision=RoutingDecision.DEFAULT_FUSION,
        fusion_audit_trail=[],
    )


def _build_session_summary(
    segments: list[Fragment],
    events: list[dict[str, Any]],
) -> ContradictionResult:
    fragments_with_results = [
        segment for segment in segments if segment.contradiction is not None
    ]
    if not fragments_with_results:
        return _empty_session_summary()

    dominant = max(
        fragments_with_results,
        key=lambda segment: segment.contradiction.disagreement_score,  # type: ignore[union-attr]
    )
    dominant_result = dominant.contradiction
    assert dominant_result is not None

    return ContradictionResult(
        contradiction_type=dominant_result.contradiction_type,
        contradiction_intensity=dominant_result.contradiction_intensity,
        involved_modalities=list(dominant_result.involved_modalities),
        suggested_fusion_weights=list(dominant_result.suggested_fusion_weights),
        routing_confidence=dominant_result.routing_confidence,
        disagreement_score=dominant_result.disagreement_score,
        routing_decision=dominant_result.routing_decision,
        fusion_audit_trail=[
            {
                "action": "session_summary",
                "processed_segments": len(fragments_with_results),
                "events": events,
            }
        ],
    )


__all__ = [
    "DEFAULT_FUSION",
    "DISAGREEMENT_FIX",
    "DisagreementFixConfig",
    "DisagreementFixResult",
    "ExpertRuleResult",
    "ExpertRulesConfig",
    "MAX_VA_DISTANCE",
    "QBTDResult",
    "QuadrantThresholdConfig",
    "RoutingResult",
    "RoutingSession",
    "SelectiveRouterConfig",
    "TYPED_FUSION",
    "VADistanceResult",
    "WeightSelectionResult",
    "WeightSelectorConfig",
    "apply_disagreement_fix",
    "calculate_va_distances",
    "classify_contradiction",
    "default_fusion_weights",
    "evaluate_qbtd",
    "get_weights",
    "is_switch_candidate",
    "run_l4",
    "same_sign",
    "select_weights",
]
