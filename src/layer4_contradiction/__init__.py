"""L4 contradiction detection and fusion prior helpers."""

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
    "same_sign",
    "select_weights",
]
