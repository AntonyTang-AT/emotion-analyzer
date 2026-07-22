"""Tests for L4.5 high-disagreement local correction."""

from __future__ import annotations

import pytest

from src.core import VAConfidence
from src.layer4_contradiction.disagreement_fix import (
    DEFAULT_FUSION,
    DISAGREEMENT_FIX,
    DisagreementFixConfig,
    apply_disagreement_fix,
)


def _va(v: float, a: float, c: float = 0.9) -> VAConfidence:
    return VAConfidence(v, a, c)


def test_disagreement_fix_returns_unchanged_when_below_threshold():
    weights = [0.25, 0.25, 0.25, 0.25]
    modality_va = {
        "text": _va(0.7, 0.2, 0.9),
        "micro": _va(-0.5, 0.1, 0.9),
    }
    result = apply_disagreement_fix(
        weights,
        modality_va,
        disagreement_score=0.2,
        max_distance=0.4,
        max_pair=("text", "micro"),
        config=DisagreementFixConfig(
            min_va_distance=0.6,
            min_disagreement_score=0.35,
        ),
    )

    assert result.adjusted is False
    assert result.weights == weights
    assert result.routing_decision == DEFAULT_FUSION
    assert result.audit_entries == []


def test_disagreement_fix_triggers_with_bounded_adjustment_and_audit():
    weights = [0.1, 0.2, 0.2, 0.5]
    modality_va = {
        "text": _va(0.8, 0.6, 0.95),
        "speech": _va(0.1, 0.1, 0.9),
        "macro": _va(0.0, 0.0, 0.9),
        "micro": _va(-0.7, -0.5, 0.2),
    }
    result = apply_disagreement_fix(
        weights,
        modality_va,
        disagreement_score=0.8,
        max_distance=1.5,
        max_pair=("text", "micro"),
        prior_decision="typed_fusion",
        config=DisagreementFixConfig(
            min_va_distance=0.6,
            min_disagreement_score=0.35,
            max_va_adjustment=0.15,
        ),
    )

    assert result.adjusted is True
    assert result.routing_decision == DISAGREEMENT_FIX
    assert len(result.audit_entries) == 1
    assert result.audit_entries[0]["action"] == "disagreement_fix"
    assert result.audit_entries[0]["va_shift"] <= 0.15 + 1e-9
    assert abs(sum(result.weights) - 1.0) < 0.01
    assert result.weights != weights
    assert result.audit_entries[0]["outlier_modalities"] == ["micro"]


def test_disagreement_fix_ignores_reason_text_when_reason_guided_disabled():
    weights = [0.25, 0.25, 0.25, 0.25]
    modality_va = {
        "text": _va(0.8, 0.6, 0.95),
        "micro": _va(-0.7, -0.5, 0.2),
    }
    without_reason = apply_disagreement_fix(
        weights,
        modality_va,
        disagreement_score=0.8,
        max_distance=1.5,
        max_pair=("text", "micro"),
        config=DisagreementFixConfig(reason_guided=False),
    )
    with_reason = apply_disagreement_fix(
        weights,
        modality_va,
        disagreement_score=0.8,
        max_distance=1.5,
        max_pair=("text", "micro"),
        reason_text="The speaker seems sarcastic.",
        config=DisagreementFixConfig(reason_guided=False),
    )

    assert without_reason.weights == with_reason.weights
    assert without_reason.adjusted == with_reason.adjusted


@pytest.mark.parametrize(
    ("disagreement_score", "max_distance"),
    [(0.34, 1.0), (0.5, 0.59)],
)
def test_disagreement_fix_does_not_trigger_on_partial_threshold_miss(
    disagreement_score,
    max_distance,
):
    result = apply_disagreement_fix(
        [0.25, 0.25, 0.25, 0.25],
        {
            "text": _va(0.8, 0.6, 0.95),
            "micro": _va(-0.7, -0.5, 0.2),
        },
        disagreement_score=disagreement_score,
        max_distance=max_distance,
        max_pair=("text", "micro"),
    )
    assert result.adjusted is False
    assert result.audit_entries == []


def test_disagreement_fix_handles_missing_modalities_safely():
    result = apply_disagreement_fix(
        [0.5, 0.0, 0.0, 0.5],
        {
            "text": _va(0.9, 0.8, 0.95),
            "micro": _va(-0.8, -0.6, 0.15),
        },
        disagreement_score=0.9,
        max_distance=1.8,
        max_pair=("text", "micro"),
        config=DisagreementFixConfig(max_va_adjustment=0.15),
    )

    assert abs(sum(result.weights) - 1.0) < 0.01
    assert result.weights[1] == pytest.approx(0.0)
    assert result.weights[2] == pytest.approx(0.0)
    if result.adjusted:
        assert result.audit_entries
        assert result.audit_entries[0]["va_shift"] <= 0.15 + 1e-9
