"""Tests for L4 unified entrypoint run_l4 (task 4.7 / Issue #66)."""

from __future__ import annotations

import pytest

from src.core import (
    ContradictionType,
    DataContext,
    Fragment,
    RoutingDecision,
    VAConfidence,
)
from src.layer4_contradiction import run_l4


def _va(v: float, a: float, c: float = 0.9) -> VAConfidence:
    return VAConfidence(v, a, c)


def _make_context(*fragments: Fragment, modalities: list[str] | None = None) -> DataContext:
    context = DataContext.create(user_id="l4-user", video_path="data/raw/test.mp4")
    context.metadata["active_modalities"] = modalities or [
        "text",
        "speech",
        "macro",
        "micro",
    ]
    context.set_stage("L3", {"segments": list(fragments), "memory_retrieved": []})
    return context


def test_run_l4_populates_fragment_fields_and_session_summary():
    fragments = [
        Fragment(
            id="seg-1",
            start_time=0.0,
            end_time=2.0,
            va_inter={
                "text": _va(0.7, 0.2, 0.95),
                "speech": _va(0.1, 0.1, 0.9),
                "macro": _va(0.0, 0.0, 0.9),
                "micro": _va(-0.5, 0.0, 0.2),
            },
        ),
        Fragment(
            id="seg-2",
            start_time=2.0,
            end_time=4.0,
            va_inter={
                "text": _va(0.2, 0.1, 0.9),
                "speech": _va(0.15, 0.1, 0.9),
                "macro": _va(0.1, 0.0, 0.9),
                "micro": _va(0.12, 0.05, 0.9),
            },
        ),
    ]
    context = _make_context(*fragments)

    result = run_l4(context)

    assert result.contradiction is not None
    assert result.segments[0].contradiction is not None
    assert result.segments[1].contradiction is not None
    assert result.segments[0].contradiction.contradiction_type == ContradictionType.MASKING
    assert result.segments[1].contradiction.contradiction_type == ContradictionType.CONSISTENT
    assert 0.0 <= result.segments[0].contradiction.disagreement_score <= 1.0
    assert isinstance(result.segments[0].contradiction.routing_decision, RoutingDecision)
    assert abs(sum(result.segments[0].contradiction.suggested_fusion_weights) - 1.0) < 0.01
    assert result.metadata["stage_status"]["L4"] == "completed"
    assert "contradiction_summary" in result.metadata


def test_run_l4_result_is_serializable_and_old_payload_compatible():
    fragment = Fragment(
        id="seg-serial",
        start_time=0.0,
        end_time=1.0,
        va_inter={
            "text": _va(0.7, 0.2, 0.95),
            "micro": _va(-0.5, 0.0, 0.2),
        },
    )
    context = _make_context(fragment, modalities=["text", "micro"])
    run_l4(context)

    payload = context.to_dict()
    assert payload["segments"][0]["contradiction"]["contradiction_type"] == "masking"
    assert "disagreement_score" in payload["segments"][0]["contradiction"]
    assert "routing_decision" in payload["segments"][0]["contradiction"]
    assert "fusion_audit_trail" in payload["segments"][0]["contradiction"]

    # Old payloads without new fields still load with defaults.
    from src.core import ContradictionResult

    legacy = ContradictionResult.from_dict(
        {
            "contradiction_type": "consistent",
            "contradiction_intensity": 0.1,
            "involved_modalities": [],
            "suggested_fusion_weights": [0.25, 0.25, 0.25, 0.25],
            "routing_confidence": 0.2,
        }
    )
    assert legacy.disagreement_score == 0.0
    assert legacy.routing_decision == RoutingDecision.DEFAULT_FUSION
    assert legacy.fusion_audit_trail == []


def test_run_l4_marks_failed_when_no_segments():
    context = DataContext.create(user_id="empty", video_path="data/raw/test.mp4")
    context.set_stage("L3", {"segments": [], "memory_retrieved": []})

    run_l4(context)

    assert context.metadata["stage_status"]["L4"] == "failed"
    assert context.metadata["errors"]["L4"] == "no segments available"


def test_run_l4_skips_single_modality_fragment():
    fragment = Fragment(
        id="solo",
        start_time=0.0,
        end_time=1.0,
        va_inter={"text": _va(0.2, 0.1, 0.9)},
    )
    context = _make_context(fragment, modalities=["text", "speech"])

    run_l4(context)

    assert context.segments[0].contradiction is None
    assert context.contradiction is not None
    assert context.contradiction.contradiction_type == ContradictionType.CONSISTENT
    assert context.metadata["stage_status"]["L4"] == "completed"


def test_non_default_routing_writes_audit_trail():
    fragment = Fragment(
        id="masking-high",
        start_time=0.0,
        end_time=1.0,
        va_inter={
            "text": _va(0.9, 0.8, 0.95),
            "speech": _va(-0.8, -0.7, 0.9),
            "macro": _va(-0.7, -0.6, 0.9),
            "micro": _va(-0.9, -0.8, 0.2),
        },
    )
    context = _make_context(fragment)
    run_l4(context)

    result = context.segments[0].contradiction
    assert result is not None
    assert result.routing_decision != RoutingDecision.DEFAULT_FUSION
    assert result.fusion_audit_trail
    assert any(
        entry.get("action") in {"typed_fusion", "disagreement_fix"}
        for entry in result.fusion_audit_trail
    )
