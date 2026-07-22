"""L4 contradiction detection and fusion prior helpers."""

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
    "ExpertRuleResult",
    "ExpertRulesConfig",
    "MAX_VA_DISTANCE",
    "QBTDResult",
    "QuadrantThresholdConfig",
    "VADistanceResult",
    "WeightSelectionResult",
    "WeightSelectorConfig",
    "calculate_va_distances",
    "classify_contradiction",
    "evaluate_qbtd",
    "get_weights",
    "same_sign",
    "select_weights",
]
