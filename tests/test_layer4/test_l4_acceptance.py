"""Acceptance-focused L4.8 coverage for run_l4 and end-to-end routing."""

from __future__ import annotations

import pytest

from src.core import (
    ContradictionType,
    Fragment,
    RoutingDecision,
    VAConfidence,
)
from src.layer4_contradiction import classify_contradiction, run_l4
from src.layer4_contradiction.selective_router import default_fusion_weights
from tests.test_layer4.conftest import make_l4_context


def _va(v: float, a: float = 0.0, c: float = 0.9) -> VAConfidence:
    return VAConfidence(v, a, c)


@pytest.mark.parametrize(
    ("modality_va", "expected"),
    [
        (
            {
                "text": _va(0.7),
                "speech": _va(0.1),
                "macro": _va(0.1),
                "micro": _va(-0.5),
            },
            ContradictionType.MASKING,
        ),
        (
            {
                "text": _va(0.4),
                "speech": _va(-0.5),
                "macro": _va(0.1),
                "micro": _va(0.1),
            },
            ContradictionType.SARCASM,
        ),
        (
            {
                "text": _va(0.1),
                "speech": _va(0.1),
                "macro": _va(0.1),
                "micro": _va(0.6),
            },
            ContradictionType.HIDDEN_EMOTION,
        ),
        (
            {
                "text": _va(0.8),
                "speech": _va(0.1),
                "macro": _va(0.1),
                "micro": _va(0.1),
            },
            ContradictionType.INTENSITY_MISMATCH,
        ),
        (
            {
                "text": _va(0.2),
                "speech": _va(0.1),
                "macro": _va(0.1),
                "micro": _va(0.15),
            },
            ContradictionType.CONSISTENT,
        ),
    ],
)
def test_five_contradiction_types_via_expert_rules(modality_va, expected):
    result = classify_contradiction(modality_va)
    assert result.contradiction_type == expected


def test_low_disagreement_keeps_default_fusion_and_empty_switch_audit():
    fragment = Fragment(
        id="consistent-low",
        start_time=0.0,
        end_time=1.0,
        va_inter={
            "text": _va(0.2, 0.1, 0.9),
            "speech": _va(0.15, 0.1, 0.9),
            "macro": _va(0.1, 0.0, 0.9),
            "micro": _va(0.12, 0.05, 0.9),
        },
    )
    context = make_l4_context(fragment)
    run_l4(context)

    result = context.segments[0].contradiction
    assert result is not None
    assert result.contradiction_type == ContradictionType.CONSISTENT
    assert result.routing_decision == RoutingDecision.DEFAULT_FUSION
    assert result.suggested_fusion_weights == pytest.approx(default_fusion_weights())
    assert result.disagreement_score < 0.35
    assert not any(
        entry.get("action") in {"typed_fusion", "disagreement_fix"}
        for entry in result.fusion_audit_trail
    )


def test_triggered_route_writes_non_empty_fusion_audit_trail():
    fragment = Fragment(
        id="masking-high",
        start_time=0.0,
        end_time=1.0,
        va_inter={
            "text": _va(0.95, 0.9, 0.99),
            "speech": _va(-0.9, -0.8, 0.95),
            "macro": _va(-0.85, -0.7, 0.95),
            "micro": _va(-0.95, -0.9, 0.15),
        },
    )
    context = make_l4_context(fragment)
    run_l4(context)

    result = context.segments[0].contradiction
    assert result is not None
    assert result.contradiction_type == ContradictionType.MASKING
    assert result.disagreement_score >= 0.35
    # With high confidence + high disagreement, SER/DTRB should leave a trail.
    assert result.fusion_audit_trail
    assert any(
        entry.get("action") in {"typed_fusion", "disagreement_fix"}
        for entry in result.fusion_audit_trail
    )
    assert abs(sum(result.suggested_fusion_weights) - 1.0) < 0.01


def test_audio_profile_two_modalities_runs_l4():
    fragment = Fragment(
        id="audio-seg",
        start_time=0.0,
        end_time=2.0,
        va_inter={
            "text": _va(0.4, 0.2, 0.9),
            "speech": _va(-0.5, 0.1, 0.9),
        },
    )
    context = make_l4_context(fragment, modalities=["text", "speech"])
    run_l4(context)

    result = context.segments[0].contradiction
    assert result is not None
    assert result.contradiction_type == ContradictionType.SARCASM
    assert abs(sum(result.suggested_fusion_weights) - 1.0) < 0.01
    assert context.metadata["stage_status"]["L4"] == "completed"


def test_multi_segment_serialization_roundtrip():
    fragments = [
        Fragment(
            id="a",
            start_time=0.0,
            end_time=1.0,
            va_inter={
                "text": _va(0.7, 0.2, 0.95),
                "micro": _va(-0.5, 0.0, 0.2),
            },
        ),
        Fragment(
            id="b",
            start_time=1.0,
            end_time=2.0,
            va_inter={
                "text": _va(0.2, 0.1, 0.9),
                "speech": _va(0.15, 0.1, 0.9),
                "macro": _va(0.1, 0.0, 0.9),
                "micro": _va(0.12, 0.05, 0.9),
            },
        ),
    ]
    context = make_l4_context(*fragments, modalities=["text", "speech", "macro", "micro"])
    run_l4(context)

    restored = type(context).from_dict(context.to_dict())
    assert restored.segments[0].contradiction is not None
    assert restored.segments[1].contradiction is not None
    assert restored.segments[0].contradiction.contradiction_type == ContradictionType.MASKING
    assert restored.segments[1].contradiction.routing_decision == RoutingDecision.DEFAULT_FUSION
