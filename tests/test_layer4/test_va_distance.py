"""Tests for L4 VA-space distance and disagreement scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core import VAConfidence
from src.layer4_contradiction.va_distance import calculate_va_distances


def _va(valence: float, arousal: float) -> VAConfidence:
    return VAConfidence(valence=valence, arousal=arousal, confidence=0.9)


def test_calculate_va_distances_matches_hand_calculation():
    result = calculate_va_distances(
        {
            "text": _va(0.0, 0.0),
            "speech": _va(0.6, 0.8),
            "micro": _va(0.6, 0.0),
        },
        ["text", "speech", "micro"],
    )

    assert result is not None
    assert result.modalities == ("text", "speech", "micro")
    np.testing.assert_allclose(
        result.distance_matrix,
        np.array(
            [
                [0.0, 1.0, 0.6],
                [1.0, 0.0, 0.8],
                [0.6, 0.8, 0.0],
            ]
        ),
    )
    assert result.max_distance == pytest.approx(1.0)
    assert result.max_pair == ("text", "speech")
    assert result.disagreement_score == pytest.approx(1.0 / (2.0 * math.sqrt(2.0)))


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    result = calculate_va_distances(
        {
            "text": (0.2, -0.3),
            "speech": (-0.4, 0.5),
            "macro": (0.8, 0.1),
            "micro": (-0.7, -0.6),
        },
        ["text", "speech", "macro", "micro"],
    )

    assert result is not None
    np.testing.assert_allclose(result.distance_matrix, result.distance_matrix.T)
    np.testing.assert_allclose(np.diag(result.distance_matrix), np.zeros(4))


def test_theoretical_va_extremes_normalize_to_one():
    result = calculate_va_distances(
        {"speech": (-1.0, -1.0), "text": (1.0, 1.0)},
        ["speech", "text"],
    )

    assert result is not None
    assert result.distance_matrix.shape == (2, 2)
    assert result.max_distance == pytest.approx(2.0 * math.sqrt(2.0))
    assert result.max_pair == ("speech", "text")
    assert result.disagreement_score == pytest.approx(1.0)


def test_identical_modalities_have_zero_disagreement():
    result = calculate_va_distances(
        {"text": (0.25, -0.5), "speech": (0.25, -0.5)},
        ["text", "speech"],
    )

    assert result is not None
    assert result.max_distance == pytest.approx(0.0)
    assert result.disagreement_score == pytest.approx(0.0)


def test_missing_active_modality_is_ignored():
    result = calculate_va_distances(
        {"text": (0.0, 0.0), "speech": (0.0, 1.0)},
        ["text", "macro", "speech"],
    )

    assert result is not None
    assert result.modalities == ("text", "speech")
    assert result.distance_matrix.shape == (2, 2)
    assert result.max_pair == ("text", "speech")


@pytest.mark.parametrize(
    ("va_inter", "active_modalities"),
    [
        ({}, []),
        ({"text": (0.1, 0.2)}, ["text"]),
        ({"text": (0.1, 0.2)}, ["text", "speech"]),
    ],
)
def test_fewer_than_two_available_modalities_skips_l4(va_inter, active_modalities):
    assert calculate_va_distances(va_inter, active_modalities) is None


def test_first_active_pair_wins_when_maximum_distances_are_tied():
    result = calculate_va_distances(
        {
            "text": (-1.0, -1.0),
            "speech": (1.0, 1.0),
            "macro": (-1.0, 1.0),
            "micro": (1.0, -1.0),
        },
        ["text", "speech", "macro", "micro"],
    )

    assert result is not None
    assert result.max_pair == ("text", "speech")


@pytest.mark.parametrize(
    "point",
    [
        (1.1, 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
    ],
)
def test_invalid_va_coordinates_are_rejected(point):
    with pytest.raises(ValueError, match=r"finite values in \[-1, 1\]"):
        calculate_va_distances(
            {"text": point, "speech": (0.0, 0.0)},
            ["text", "speech"],
        )
