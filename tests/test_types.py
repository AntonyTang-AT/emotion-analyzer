"""Tests for core data types."""

from __future__ import annotations

import pytest

from src.core import (
    MODALITIES,
    ContradictionResult,
    ContradictionType,
    Fragment,
    PersonalityResult,
    VAConfidence,
)


def test_modalities_match_weight_table_order(config_manager):
    table = config_manager.load("weight_table")
    assert list(table.keys())[:1]  # sanity
    assert MODALITIES == ("text", "speech", "macro", "micro")
    for weights in table.values():
        assert len(weights) == len(MODALITIES)


def test_contradiction_result_rejects_bad_weight_sum():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ContradictionResult(
            contradiction_type=ContradictionType.CONSISTENT,
            contradiction_intensity=0.5,
            involved_modalities=["text", "speech"],
            suggested_fusion_weights=[0.5, 0.5, 0.5, 0.5],
            routing_confidence=0.5,
        )


def test_contradiction_result_rejects_bad_routing_confidence():
    with pytest.raises(ValueError, match="routing_confidence"):
        ContradictionResult(
            contradiction_type=ContradictionType.CONSISTENT,
            contradiction_intensity=0.5,
            involved_modalities=["text"],
            suggested_fusion_weights=[0.25, 0.25, 0.25, 0.25],
            routing_confidence=1.5,
        )


def test_fragment_rejects_invalid_time_range():
    with pytest.raises(ValueError, match="end_time"):
        Fragment(id="bad", start_time=5.0, end_time=1.0)


def test_va_confidence_roundtrip():
    original = VAConfidence(valence=0.2, arousal=-0.3, confidence=0.88)
    restored = VAConfidence.from_dict(original.to_dict())
    assert restored.valence == pytest.approx(0.2)
    assert restored.confidence == pytest.approx(0.88)


def test_personality_result_roundtrip():
    original = PersonalityResult(
        O=7.2,
        C=6.8,
        E=5.1,
        A=8.0,
        N=3.2,
        confidence_interval=[0.7, 0.6, 0.8, 0.7, 0.9],
        behavioral_evidence="sample",
    )
    restored = PersonalityResult.from_dict(original.to_dict())
    assert restored.A == pytest.approx(8.0)
    assert restored.behavioral_evidence == "sample"


def test_contradiction_result_to_dict_matches_config_weights(config_manager):
    table = config_manager.load("weight_table")
    result = ContradictionResult(
        contradiction_type="masking",
        contradiction_intensity=0.7,
        involved_modalities=["text", "micro"],
        suggested_fusion_weights=table["masking"],
        routing_confidence=0.8,
        disagreement_score=0.55,
        routing_decision="typed_fusion",
        fusion_audit_trail=[{"action": "typed_fusion"}],
    )
    payload = result.to_dict()
    assert payload["suggested_fusion_weights"] == table["masking"]
    assert payload["disagreement_score"] == pytest.approx(0.55)
    assert payload["routing_decision"] == "typed_fusion"
    assert payload["fusion_audit_trail"] == [{"action": "typed_fusion"}]

def test_contradiction_result_from_dict_compatible_with_legacy_payload():
    restored = ContradictionResult.from_dict(
        {
            "contradiction_type": "consistent",
            "contradiction_intensity": 0.2,
            "involved_modalities": [],
            "suggested_fusion_weights": [0.25, 0.25, 0.25, 0.25],
            "routing_confidence": 0.3,
        }
    )
    assert restored.disagreement_score == 0.0
    assert restored.routing_decision.value == "default_fusion"
    assert restored.fusion_audit_trail == []
