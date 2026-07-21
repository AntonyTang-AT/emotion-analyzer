"""Tests for L4.3 expert-rule contradiction type interpreter."""

from __future__ import annotations

import pytest

from src.core import ContradictionType, VAConfidence
from src.layer4_contradiction.expert_rules import (
    ExpertRulesConfig,
    classify_contradiction,
    same_sign,
)


def _va(v: float, a: float = 0.0, c: float = 1.0) -> VAConfidence:
    return VAConfidence(v, a, c)


def test_same_sign_includes_zero_boundary():
    assert same_sign(0.5, 0.2) is True
    assert same_sign(-0.5, -0.2) is True
    assert same_sign(0.5, -0.2) is False
    assert same_sign(0.0, 0.5) is True
    assert same_sign(0.0, -0.5) is True
    assert same_sign(0.0, 0.0) is True


def test_classify_masking():
    result = classify_contradiction(
        {
            "text": _va(0.7),
            "speech": _va(0.1),
            "macro": _va(0.1),
            "micro": _va(-0.5),
        }
    )
    assert result.contradiction_type == ContradictionType.MASKING
    assert result.involved_modalities == ["text", "micro"]
    assert result.matched_rule == "masking"


def test_classify_sarcasm_from_speech_or_macro():
    speech_hit = classify_contradiction(
        {
            "text": _va(0.4),
            "speech": _va(-0.5),
            "macro": _va(0.1),
            "micro": _va(0.1),
        }
    )
    assert speech_hit.contradiction_type == ContradictionType.SARCASM
    assert speech_hit.involved_modalities == ["text", "speech"]

    macro_hit = classify_contradiction(
        {
            "text": _va(0.4),
            "speech": _va(0.1),
            "macro": _va(-0.5),
            "micro": _va(0.1),
        }
    )
    assert macro_hit.contradiction_type == ContradictionType.SARCASM
    assert macro_hit.involved_modalities == ["text", "macro"]


def test_classify_hidden_emotion():
    result = classify_contradiction(
        {
            "text": _va(0.1),
            "speech": _va(0.1),
            "macro": _va(0.1),
            "micro": _va(0.6),
        }
    )
    assert result.contradiction_type == ContradictionType.HIDDEN_EMOTION
    assert result.involved_modalities == ["macro", "micro"]


def test_classify_intensity_mismatch():
    result = classify_contradiction(
        {
            "text": _va(0.8),
            "speech": _va(0.1),
            "macro": _va(0.1),
            "micro": _va(0.1),
        }
    )
    assert result.contradiction_type == ContradictionType.INTENSITY_MISMATCH
    assert result.involved_modalities == ["text", "micro"]


def test_classify_consistent():
    result = classify_contradiction(
        {
            "text": _va(0.2),
            "speech": _va(0.1),
            "macro": _va(0.1),
            "micro": _va(0.15),
        }
    )
    assert result.contradiction_type == ContradictionType.CONSISTENT
    assert result.involved_modalities == []
    assert result.matched_rule is None


def test_masking_has_priority_over_sarcasm():
    result = classify_contradiction(
        {
            "text": _va(0.7),
            "speech": _va(-0.5),
            "macro": _va(-0.5),
            "micro": _va(-0.5),
        }
    )
    assert result.contradiction_type == ContradictionType.MASKING
    assert result.involved_modalities == ["text", "micro"]


def test_masking_boundary_is_strict_greater_than():
    at_boundary = classify_contradiction(
        {
            "text": _va(0.6),
            "micro": _va(-0.5),
        }
    )
    assert at_boundary.contradiction_type != ContradictionType.MASKING

    above_boundary = classify_contradiction(
        {
            "text": _va(0.6001),
            "micro": _va(-0.5),
        }
    )
    assert above_boundary.contradiction_type == ContradictionType.MASKING


def test_intensity_mismatch_boundary_and_opposite_signs():
    equal_diff = classify_contradiction(
        {
            "text": _va(0.7),
            "micro": _va(0.1),
        }
    )
    assert equal_diff.contradiction_type == ContradictionType.CONSISTENT

    opposite = classify_contradiction(
        {
            "text": _va(0.8),
            "micro": _va(-0.1),
        }
    )
    assert opposite.contradiction_type == ContradictionType.CONSISTENT


def test_missing_micro_skips_masking_and_intensity():
    result = classify_contradiction(
        {
            "text": _va(0.7),
            "speech": _va(-0.5),
            "macro": _va(0.1),
        }
    )
    assert result.contradiction_type == ContradictionType.SARCASM
    assert result.involved_modalities == ["text", "speech"]


def test_missing_speech_and_macro_skips_sarcasm():
    result = classify_contradiction(
        {
            "text": _va(0.4),
            "micro": _va(0.1),
        }
    )
    assert result.contradiction_type == ContradictionType.CONSISTENT


def test_sarcasm_works_with_only_one_negative_modality_present():
    result = classify_contradiction(
        {
            "text": _va(0.4),
            "speech": _va(-0.5),
            "micro": _va(0.1),
        }
    )
    assert result.contradiction_type == ContradictionType.SARCASM
    assert result.involved_modalities == ["text", "speech"]


def test_injected_config_overrides_defaults():
    config = ExpertRulesConfig(
        masking_text_v_min=0.2,
        masking_micro_v_max=-0.1,
    )
    result = classify_contradiction(
        {
            "text": _va(0.3),
            "micro": _va(-0.2),
        },
        config=config,
    )
    assert result.contradiction_type == ContradictionType.MASKING


def test_accepts_mapping_and_pair_inputs():
    result = classify_contradiction(
        {
            "text": {"valence": 0.7, "arousal": 0.1, "confidence": 0.9},
            "micro": ( -0.5, 0.0),
        }
    )
    assert result.contradiction_type == ContradictionType.MASKING


def test_rejects_unknown_modality_and_invalid_valence():
    with pytest.raises(ValueError, match="unknown modalities"):
        classify_contradiction({"face": _va(0.1)})

    with pytest.raises(ValueError, match="finite value in \\[-1, 1\\]"):
        classify_contradiction({"text": _va(1.5)})
