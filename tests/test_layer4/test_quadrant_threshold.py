"""Tests for L4.2 quadrant threshold contradiction detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import VAConfidence
from src.layer4_contradiction.quadrant_threshold import (
    QuadrantThresholdConfig,
    combined_va,
    determine_quadrant,
    evaluate_qbtd,
    evaluate_qbtd_from_distance_result,
    has_quadrant_contradiction,
    max_distance_from_matrix,
    threshold_for_quadrant,
)


def _va(v: float, a: float, c: float = 1.0) -> VAConfidence:
    return VAConfidence(v, a, c)


def test_determine_quadrant_for_all_signs_and_boundaries():
    assert determine_quadrant(0.2, 0.3) == "Q1"
    assert determine_quadrant(-0.2, 0.3) == "Q2"
    assert determine_quadrant(-0.2, -0.3) == "Q3"
    assert determine_quadrant(0.2, -0.3) == "Q4"
    assert determine_quadrant(0.0, 0.0) == "Q1"


def test_combined_va_uses_confidence_weights_by_default():
    va_inter = {
        "text": _va(1.0, 1.0, 0.9),
        "speech": _va(-1.0, -1.0, 0.1),
    }

    assert combined_va(va_inter) == pytest.approx((0.8, 0.8))


def test_combined_va_falls_back_to_equal_weights_when_confidence_is_zero():
    va_inter = {
        "text": _va(1.0, 0.5, 0.0),
        "speech": _va(-1.0, -0.5, 0.0),
    }

    assert combined_va(va_inter) == pytest.approx((0.0, 0.0))


def test_threshold_for_quadrant_uses_q1_to_q4_order():
    config = QuadrantThresholdConfig(quadrant_thresholds=(0.1, 0.2, 0.3, 0.4))

    assert threshold_for_quadrant("Q1", config) == pytest.approx(0.1)
    assert threshold_for_quadrant("q4", config) == pytest.approx(0.4)


def test_evaluate_qbtd_flags_distance_above_quadrant_threshold():
    config = QuadrantThresholdConfig(quadrant_thresholds=(0.5, 0.65, 0.7, 0.55))
    va_inter = {
        "text": _va(0.4, 0.6, 0.8),
        "speech": _va(0.2, 0.4, 0.8),
    }

    result = evaluate_qbtd(va_inter, 0.51, config=config)

    assert result.has_contradiction is True
    assert result.exceeds_threshold is True
    assert result.intensity == pytest.approx(0.51)
    assert result.strength_reference == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.5)
    assert result.quadrant == "Q1"
    result_dict = result.to_dict()
    assert result_dict["combined_va"] == pytest.approx([0.3, 0.5])
    assert result_dict["exceeds_threshold"] is True


def test_evaluate_qbtd_returns_zero_intensity_when_below_threshold():
    config = QuadrantThresholdConfig(quadrant_thresholds=(0.5, 0.65, 0.7, 0.55))
    va_inter = {
        "text": _va(-0.4, -0.6),
        "speech": _va(-0.2, -0.4),
    }

    result = evaluate_qbtd(va_inter, 0.69, config=config)

    assert result.has_contradiction is False
    assert result.exceeds_threshold is False
    assert result.intensity == pytest.approx(0.0)
    assert result.strength_reference == pytest.approx(0.69 / 0.7)
    assert result.max_distance == pytest.approx(0.69)
    assert result.threshold == pytest.approx(0.7)
    assert result.quadrant == "Q3"


def test_evaluate_qbtd_can_read_max_distance_from_matrix():
    va_inter = {"text": _va(0.1, -0.2), "speech": _va(0.2, -0.3)}
    matrix = np.array([[0.0, 0.56], [0.56, 0.0]])
    config = QuadrantThresholdConfig(quadrant_thresholds=(0.5, 0.65, 0.7, 0.55))

    result = evaluate_qbtd(va_inter, distance_matrix=matrix, config=config)

    assert result.has_contradiction is True
    assert result.quadrant == "Q4"
    assert max_distance_from_matrix(matrix) == pytest.approx(0.56)


def test_evaluate_qbtd_accepts_l41_distance_result_shape():
    class DistanceResult:
        max_distance = 0.56

    va_inter = {"text": _va(0.1, -0.2), "speech": _va(0.2, -0.3)}
    config = QuadrantThresholdConfig(quadrant_thresholds=(0.5, 0.65, 0.7, 0.55))

    result = evaluate_qbtd_from_distance_result(
        va_inter,
        DistanceResult(),
        config=config,
    )

    assert result.has_contradiction is True
    assert result.max_distance == pytest.approx(0.56)


def test_evaluate_qbtd_can_be_disabled_without_overriding_distance_signal():
    config = QuadrantThresholdConfig(
        enabled=False,
        quadrant_thresholds=(0.5, 0.65, 0.7, 0.55),
    )
    va_inter = {"text": _va(0.1, 0.2), "speech": _va(0.2, 0.3)}

    result = evaluate_qbtd(va_inter, 0.99, config=config)

    assert result.enabled is False
    assert result.exceeds_threshold is False
    assert result.has_contradiction is False
    assert result.intensity == pytest.approx(0.0)
    assert result.strength_reference == pytest.approx(0.0)
    assert result.max_distance == pytest.approx(0.99)


def test_has_quadrant_contradiction_returns_boolean_and_intensity():
    config = QuadrantThresholdConfig(quadrant_thresholds=(0.5, 0.65, 0.7, 0.55))
    va_inter = {"text": _va(-0.1, 0.4), "speech": _va(-0.2, 0.2)}

    assert has_quadrant_contradiction(va_inter, 0.66, config=config) == (
        True,
        pytest.approx(0.66),
    )


def test_config_from_pipeline_validates_threshold_count():
    with pytest.raises(ValueError, match="exactly four"):
        QuadrantThresholdConfig.from_pipeline(
            {"pipeline": {"stages": {"L4": {"quadrant_thresholds": [0.5, 0.6]}}}}
        )


def test_evaluate_qbtd_rejects_missing_distance():
    with pytest.raises(ValueError, match="max_distance or distance_matrix"):
        evaluate_qbtd({"text": _va(0.1, 0.2)})
