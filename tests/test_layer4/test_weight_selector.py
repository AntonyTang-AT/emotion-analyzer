"""Tests for L4.4 rule-table fusion weight selection."""

from __future__ import annotations

import pytest

from src.core import ContradictionType
from src.layer4_contradiction.weight_selector import (
    DEFAULT_CONSISTENT_WEIGHTS,
    WeightSelectorConfig,
    get_weights,
    select_weights,
)


def test_get_weights_reads_rule_table_for_known_type(config_manager):
    table = config_manager.load("weight_table")

    weights, confidence = get_weights(ContradictionType.MASKING, 0.6)

    assert weights == pytest.approx(table["masking"])
    assert confidence == pytest.approx(0.5)


@pytest.mark.parametrize(
    "contradiction_type",
    [
        ContradictionType.MASKING,
        ContradictionType.SARCASM,
        ContradictionType.HIDDEN_EMOTION,
        ContradictionType.INTENSITY_MISMATCH,
        ContradictionType.CONSISTENT,
    ],
)
def test_all_table_types_sum_to_one(config_manager, contradiction_type):
    table = config_manager.load("weight_table")
    weights, _ = get_weights(contradiction_type, 0.6, weight_table=table)
    assert weights == pytest.approx(table[contradiction_type.value])
    assert abs(sum(weights) - 1.0) < 0.01


def test_get_weights_accepts_string_type_and_caps_confidence():
    weights, confidence = get_weights("hidden_emotion", 2.4)

    assert weights == pytest.approx([0.1, 0.1, 0.1, 0.7])
    assert confidence == pytest.approx(1.0)


def test_confidences_scale_routing_confidence():
    weights, confidence = get_weights(
        "sarcasm",
        0.6,
        confidences={"text": 0.8, "speech": 0.6, "macro": 1.0, "micro": 0.6},
    )

    assert weights == pytest.approx([0.4, 0.4, 0.1, 0.1])
    assert confidence == pytest.approx(0.5 * 0.75)


def test_unknown_type_falls_back_to_consistent_weights():
    result = select_weights("unknown_type", 0.6)

    assert result.weights == pytest.approx(DEFAULT_CONSISTENT_WEIGHTS)
    assert result.contradiction_type == ContradictionType.CONSISTENT
    assert result.routing_confidence == pytest.approx(0.5)
    assert result.used_fallback is True


def test_disabled_or_low_confidence_strategy_falls_back():
    disabled = WeightSelectorConfig(enabled=False)
    result = select_weights("masking", 0.6, config=disabled)

    assert result.weights == pytest.approx(DEFAULT_CONSISTENT_WEIGHTS)
    assert result.used_fallback is True

    thresholded = WeightSelectorConfig(low_confidence_threshold=0.6)
    result = select_weights("masking", 0.6, config=thresholded)

    assert result.weights == pytest.approx(DEFAULT_CONSISTENT_WEIGHTS)
    assert result.routing_confidence == pytest.approx(0.5)
    assert result.used_fallback is True


def test_select_weights_allows_injected_table():
    table = {
        "masking": [0.7, 0.1, 0.1, 0.1],
        "consistent": [0.25, 0.25, 0.25, 0.25],
    }

    result = select_weights("masking", 0.3, weight_table=table)

    assert result.weights == pytest.approx([0.7, 0.1, 0.1, 0.1])
    assert result.routing_confidence == pytest.approx(0.25)


def test_result_to_dict_uses_plain_values():
    result = select_weights("consistent", 0.0)

    assert result.to_dict() == {
        "weights": [0.25, 0.25, 0.25, 0.25],
        "routing_confidence": 0.0,
        "contradiction_type": "consistent",
        "used_fallback": False,
    }


def test_rejects_invalid_intensity_confidence_and_weight_table():
    with pytest.raises(ValueError, match="intensity must be non-negative"):
        get_weights("masking", -0.1)

    with pytest.raises(ValueError, match=r"confidences\.text must be in \[0, 1\]"):
        get_weights("masking", 0.6, confidences={"text": 1.5})

    with pytest.raises(ValueError, match="unknown confidence modalities"):
        get_weights("masking", 0.6, confidences={"face": 0.8})

    with pytest.raises(ValueError, match="must sum to 1.0"):
        select_weights(
            "masking",
            0.6,
            weight_table={
                "masking": [0.5, 0.5, 0.5, 0.5],
                "consistent": [0.25, 0.25, 0.25, 0.25],
            },
        )
