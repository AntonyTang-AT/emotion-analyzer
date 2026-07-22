"""Tests for L4.6 selective gating router."""

from __future__ import annotations

import pytest

from src.core import ContradictionType
from src.layer4_contradiction.selective_router import (
    DEFAULT_FUSION,
    TYPED_FUSION,
    RoutingSession,
    SelectiveRouterConfig,
    default_fusion_weights,
    is_switch_candidate,
)


MASKING_WEIGHTS = [0.1, 0.2, 0.2, 0.5]


def _session_config(**overrides: float | bool | str) -> SelectiveRouterConfig:
    defaults = dict(
        enabled=True,
        confidence_threshold=0.65,
        max_switch_rate=0.10,
        min_disagreement_score=0.35,
    )
    defaults.update(overrides)
    return SelectiveRouterConfig(**defaults)


def test_default_fusion_weights_are_equal():
    assert default_fusion_weights() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_low_disagreement_or_low_confidence_does_not_switch():
    config = _session_config()
    assert is_switch_candidate(
        disagreement_score=0.2,
        routing_confidence=0.9,
        contradiction_type="masking",
        config=config,
    ) == (False, "low_disagreement")
    assert is_switch_candidate(
        disagreement_score=0.8,
        routing_confidence=0.5,
        contradiction_type="masking",
        config=config,
    ) == (False, "low_confidence")


def test_high_disagreement_high_confidence_is_switch_candidate():
    ok, reason = is_switch_candidate(
        disagreement_score=0.8,
        routing_confidence=0.9,
        contradiction_type=ContradictionType.MASKING,
        config=_session_config(),
    )
    assert ok is True
    assert reason is None


def test_consistent_keeps_default_equal_weights():
    session = RoutingSession(1, config=_session_config())
    session.consider(
        index=0,
        candidate_weights=[0.25, 0.25, 0.25, 0.25],
        routing_confidence=0.95,
        disagreement_score=0.1,
        contradiction_type="consistent",
    )
    results = session.finalize()

    assert results[0].routing_decision == DEFAULT_FUSION
    assert results[0].final_weights == pytest.approx(default_fusion_weights())
    assert results[0].audit_entries == []


def test_typed_fusion_when_gates_pass():
    session = RoutingSession(1, config=_session_config(max_switch_rate=1.0))
    session.consider(
        index=0,
        candidate_weights=MASKING_WEIGHTS,
        routing_confidence=0.9,
        disagreement_score=0.8,
        contradiction_type="masking",
    )
    result = session.finalize()[0]

    assert result.routing_decision == TYPED_FUSION
    assert result.final_weights == pytest.approx(MASKING_WEIGHTS)
    assert len(result.audit_entries) == 1
    assert result.audit_entries[0]["action"] == "typed_fusion"


def test_switch_rate_keeps_only_top_confidence_candidates():
    session = RoutingSession(10, config=_session_config(max_switch_rate=0.1))
    for index in range(10):
        session.consider(
            index=index,
            candidate_weights=MASKING_WEIGHTS,
            routing_confidence=0.70 + index * 0.01,
            disagreement_score=0.8,
            contradiction_type="masking",
        )
    results = session.finalize()

    typed = [
        (index, item)
        for index, item in enumerate(results)
        if item.routing_decision == TYPED_FUSION
    ]
    assert len(typed) == 1
    assert typed[0][0] == 9  # highest confidence 0.79
    assert typed[0][1].audit_entries[0]["action"] == "typed_fusion"

    rejected = [
        item
        for item in results
        if item.audit_entries
        and item.audit_entries[0].get("action") == "rejected_by_switch_rate"
    ]
    assert len(rejected) == 9
    assert all(item.routing_decision == DEFAULT_FUSION for item in rejected)
    assert all(
        item.final_weights == pytest.approx(default_fusion_weights())
        for item in rejected
    )


def test_ser_disabled_forces_default_fusion():
    session = RoutingSession(2, config=_session_config(enabled=False, max_switch_rate=1.0))
    for index in range(2):
        session.consider(
            index=index,
            candidate_weights=MASKING_WEIGHTS,
            routing_confidence=0.99,
            disagreement_score=0.99,
            contradiction_type="sarcasm",
        )
    results = session.finalize()

    assert all(item.routing_decision == DEFAULT_FUSION for item in results)
    assert all(
        item.final_weights == pytest.approx(default_fusion_weights())
        for item in results
    )


def test_result_to_dict_and_input_validation():
    session = RoutingSession(1, config=_session_config(max_switch_rate=1.0))
    session.consider(
        index=0,
        candidate_weights=MASKING_WEIGHTS,
        routing_confidence=0.9,
        disagreement_score=0.8,
        contradiction_type="masking",
    )
    payload = session.finalize()[0].to_dict()
    assert payload["routing_decision"] == TYPED_FUSION
    assert payload["final_weights"] == pytest.approx(MASKING_WEIGHTS)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        bad = RoutingSession(1, config=_session_config())
        bad.consider(
            index=0,
            candidate_weights=[0.5, 0.5, 0.5, 0.5],
            routing_confidence=0.9,
            disagreement_score=0.8,
            contradiction_type="masking",
        )
