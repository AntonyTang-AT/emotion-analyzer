"""Tests for L3 static baseline calibrator (task 3.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core import VAConfidence
from src.core.types import Fragment
from src.layer3_segment.baseline_calibrator import (
    BaselineConfig,
    apply_baseline,
    apply_baseline_to_fragment,
    calibrate_from_responses,
    compute_delta_va,
    load_baseline,
    load_population_baseline,
    save_baseline,
    va_self_to_vector,
)


def _va(v: float, a: float, c: float = 0.9) -> VAConfidence:
    return VAConfidence(v, a, c)


def _population_fixture(tmp_path: Path) -> Path:
    payload = {
        "population_average": {
            "text": {"valence": 0.2, "arousal": 0.1},
            "speech": {"valence": 0.0, "arousal": 0.0},
            "macro": {"valence": -0.1, "arousal": 0.2},
            "micro": {"valence": 0.3, "arousal": -0.2},
        }
    }
    path = tmp_path / "population_baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compute_delta_va():
    user_avg = {
        "text": _va(0.5, 0.3),
        "speech": _va(-0.2, 0.4),
        "macro": _va(0.0, 0.5),
    }
    population_avg = {
        "text": _va(0.2, 0.1),
        "speech": _va(0.0, 0.0),
        "macro": _va(-0.1, 0.2),
        "micro": _va(0.3, -0.2),
    }

    delta = compute_delta_va(user_avg, population_avg)

    assert delta["text"] == pytest.approx((0.3, 0.2))
    assert delta["speech"] == pytest.approx((-0.2, 0.4))
    assert delta["macro"] == pytest.approx((0.1, 0.3))
    assert "micro" not in delta


def test_apply_baseline_subtracts_and_preserves_confidence():
    va_self = {
        "text": _va(0.8, 0.6, 0.75),
        "speech": _va(0.1, -0.2, 0.85),
    }
    delta = {"text": (0.3, 0.2), "speech": (-0.1, 0.1)}

    calibrated = apply_baseline(va_self, "user-1", delta=delta)

    assert calibrated["text"].valence == pytest.approx(0.5)
    assert calibrated["text"].arousal == pytest.approx(0.4)
    assert calibrated["text"].confidence == pytest.approx(0.75)
    assert calibrated["speech"].valence == pytest.approx(0.2)
    assert calibrated["speech"].arousal == pytest.approx(-0.3)
    assert calibrated["speech"].confidence == pytest.approx(0.85)


def test_apply_baseline_no_baseline_returns_unchanged():
    va_self = {"text": _va(0.4, -0.1, 0.9)}

    result = apply_baseline(va_self, "missing-user", delta=None)

    assert result["text"].valence == pytest.approx(0.4)
    assert result["text"].arousal == pytest.approx(-0.1)
    assert result["text"].confidence == pytest.approx(0.9)
    assert result is not va_self


def test_save_load_roundtrip(tmp_path: Path):
    storage = tmp_path / "delta_va"
    delta = {"text": (0.1, -0.05), "macro": (0.2, 0.0)}

    saved = save_baseline("alice", delta, storage_dir=storage)
    assert saved.is_file()

    loaded = load_baseline("alice", storage_dir=storage)
    assert loaded == delta


def test_calibrate_from_responses(tmp_path: Path):
    pop_path = _population_fixture(tmp_path)
    storage = tmp_path / "delta_va"
    cfg = BaselineConfig(
        delta_va_storage=storage,
        population_baseline_path=pop_path,
    )
    user_avg = {
        "text": _va(0.5, 0.3),
        "speech": _va(-0.2, 0.4),
        "macro": _va(0.0, 0.5),
        "micro": _va(0.1, 0.0),
    }
    population = load_population_baseline(pop_path)

    delta = calibrate_from_responses(
        "bob",
        user_avg,
        population_avg=population,
        config=cfg,
    )

    assert delta["text"] == pytest.approx((0.3, 0.2))
    assert delta["speech"] == pytest.approx((-0.2, 0.4))
    assert delta["macro"] == pytest.approx((0.1, 0.3))
    assert delta["micro"] == pytest.approx((-0.2, 0.2))
    assert load_baseline("bob", storage_dir=storage) == delta


def test_va_self_to_vector_order():
    va_dict = {
        "text": _va(0.1, 0.2),
        "speech": _va(0.3, 0.4),
        "macro": _va(0.5, 0.6),
        "micro": _va(0.7, 0.8),
    }

    vector = va_self_to_vector(va_dict)

    assert vector == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def test_apply_baseline_to_fragment(tmp_path: Path):
    storage = tmp_path / "delta_va"
    cfg = BaselineConfig(delta_va_storage=storage)
    save_baseline("user-1", {"text": (0.2, 0.1)}, storage_dir=storage)

    fragment = Fragment(
        id="f1",
        start_time=0.0,
        end_time=1.0,
        va_self={"text": _va(0.9, 0.5, 0.88)},
    )

    calibrated = apply_baseline_to_fragment(fragment, "user-1", config=cfg)

    assert calibrated.va_self["text"].valence == pytest.approx(0.7)
    assert calibrated.va_self["text"].arousal == pytest.approx(0.4)
    assert calibrated.va_self["text"].confidence == pytest.approx(0.88)
    assert fragment.va_self["text"].valence == pytest.approx(0.9)


def test_baseline_user_id_paths_do_not_collide(tmp_path: Path):
    storage = tmp_path / "delta_va"
    save_baseline("a/b", {"text": (0.1, 0.0)}, storage_dir=storage)
    save_baseline("a_b", {"text": (0.2, 0.0)}, storage_dir=storage)

    assert load_baseline("a/b", storage_dir=storage) == {"text": (0.1, 0.0)}
    assert load_baseline("a_b", storage_dir=storage) == {"text": (0.2, 0.0)}


def test_load_baseline_empty_delta_returns_none(tmp_path: Path):
    storage = tmp_path / "delta_va"
    path = storage / "empty.json"
    storage.mkdir(parents=True)
    path.write_text(json.dumps({"delta_va": {}, "updated_at": "2026-01-01T00:00:00Z"}), encoding="utf-8")

    assert load_baseline("empty", storage_dir=storage) is None
