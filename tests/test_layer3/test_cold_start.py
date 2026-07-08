"""Tests for L3 collaborative cold-start (task 3.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core import VAConfidence
from src.layer3_segment.cold_start import (
    ColdStartConfig,
    ColdStartUserRecord,
    apply_cold_start,
    blend_delta_va,
    cosine_similarity,
    find_similar_users,
    get_cold_start_delta,
)


def _va(v: float, a: float, c: float = 0.9) -> VAConfidence:
    return VAConfidence(v, a, c)


def _library_fixture() -> list[ColdStartUserRecord]:
    return [
        ColdStartUserRecord(
            user_id="near",
            first_session_avg_va_self=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            delta_va=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ColdStartUserRecord(
            user_id="mid",
            first_session_avg_va_self=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            delta_va=[0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ColdStartUserRecord(
            user_id="far",
            first_session_avg_va_self=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            delta_va=[0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]


def test_cosine_similarity():
    identical = [0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    orthogonal_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    orthogonal_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert cosine_similarity(identical, identical) == pytest.approx(1.0)
    assert cosine_similarity(orthogonal_a, orthogonal_b) == pytest.approx(0.0)
    assert cosine_similarity([0.0] * 8, identical) == pytest.approx(0.0)


def test_find_similar_users_top_k_order():
    library = _library_fixture()
    query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    neighbors = find_similar_users(query, library, top_k=2)

    assert [record.user_id for record, _ in neighbors] == ["near", "mid"]
    assert neighbors[0][1] > neighbors[1][1]


def test_blend_delta_va_weighted_average():
    neighbors = [
        (
            ColdStartUserRecord("a", [0.0] * 8, [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1.0,
        ),
        (
            ColdStartUserRecord("b", [0.0] * 8, [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            2.0,
        ),
    ]

    blended = blend_delta_va(neighbors)

    assert blended is not None
    assert blended[0] == pytest.approx((0.2 * 1.0 + 0.8 * 2.0) / 3.0)
    assert blended[1] == pytest.approx(0.0)


def test_get_cold_start_delta_reproducible(tmp_path: Path):
    library_path = tmp_path / "cold_start_users.json"
    library_path.write_text(
        json.dumps(
            {
                "users": [
                    {
                        "user_id": "seed",
                        "first_session_avg_va_self": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "delta_va": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = ColdStartConfig(user_library_path=library_path)
    first_fragment = {"text": _va(1.0, 0.0)}

    delta = get_cold_start_delta("new_user", first_fragment, config=cfg)

    assert delta == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert get_cold_start_delta("new_user", first_fragment, config=cfg) == delta


def test_empty_library_returns_none():
    cfg = ColdStartConfig(enabled=True)
    first_fragment = {"text": _va(0.5, 0.3)}

    assert get_cold_start_delta("new_user", first_fragment, config=cfg, library=[]) is None


def test_single_user_library():
    library = [
        ColdStartUserRecord(
            user_id="only",
            first_session_avg_va_self=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            delta_va=[0.12, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    ]
    first_fragment = {"text": _va(0.5, 0.5)}

    delta = get_cold_start_delta("new_user", first_fragment, library=library)

    assert delta == pytest.approx([0.12, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_apply_cold_start_end_to_end():
    library = [
        ColdStartUserRecord(
            user_id="seed",
            first_session_avg_va_self=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            delta_va=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    ]
    first_fragment = {"text": _va(1.0, 0.0)}
    va_self = {"text": _va(0.9, 0.5, 0.88)}

    calibrated = apply_cold_start(
        va_self,
        "new_user",
        first_fragment,
        library=library,
    )

    assert calibrated["text"].valence == pytest.approx(0.7)
    assert calibrated["text"].arousal == pytest.approx(0.4)
    assert calibrated["text"].confidence == pytest.approx(0.88)


def test_skips_when_static_baseline_exists(monkeypatch: pytest.MonkeyPatch):
    library = [
        ColdStartUserRecord(
            user_id="seed",
            first_session_avg_va_self=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            delta_va=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    ]
    monkeypatch.setattr(
        "src.layer3_segment.cold_start.load_baseline",
        lambda user_id: {"text": (0.5, 0.5)} if user_id == "existing_user" else None,
    )

    delta = get_cold_start_delta(
        "existing_user",
        {"text": _va(1.0, 0.0)},
        library=library,
    )

    assert delta is None
