"""Tests for the L3 persistent fragment memory (task 3.4)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from src.core import MemoryHit
from src.layer3_segment.memory_store import MemoryConfig, MemoryStore


def _vector(index: int, value: float = 1.0) -> list[float]:
    vector = [0.0] * 8
    vector[index] = value
    return vector


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        MemoryConfig(db_path=tmp_path / "chroma", decay_alpha=0.1, top_k=5)
    )


def test_add_and_query_roundtrip(store: MemoryStore):
    fragment_id = store.add_fragment(
        "alice",
        _vector(0),
        _vector(1),
        {
            "fragment_id": "fragment-1",
            "timestamp": "2026-07-01T00:00:00Z",
            "emotion_label": "happy",
            "au_pattern": [6, 12],
        },
    )

    hits = store.query_similar(
        "alice",
        _vector(1),
        time_decay=False,
        embedding_type="inter",
    )

    assert fragment_id == "fragment-1"
    assert len(hits) == 1
    assert isinstance(hits[0], MemoryHit)
    assert hits[0].fragment_id == "fragment-1"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].metadata["emotion_label"] == "happy"
    assert hits[0].metadata["au_pattern"] == [6, 12]
    assert hits[0].metadata["user_id"] == "alice"


def test_embedding_type_switches_collections(store: MemoryStore):
    store.add_fragment(
        "alice",
        _vector(0),
        _vector(1),
        {"fragment_id": "self-match", "timestamp": "2026-07-01T00:00:00Z"},
    )
    store.add_fragment(
        "alice",
        _vector(1),
        _vector(0),
        {"fragment_id": "inter-match", "timestamp": "2026-07-01T00:00:00Z"},
    )

    self_hits = store.query_similar(
        "alice", _vector(0), top_k=1, time_decay=False, embedding_type="self"
    )
    inter_hits = store.query_similar(
        "alice", _vector(0), top_k=1, time_decay=False, embedding_type="inter"
    )

    assert self_hits[0].fragment_id == "self-match"
    assert self_hits[0].embedding_type == "self"
    assert inter_hits[0].fragment_id == "inter-match"
    assert inter_hits[0].embedding_type == "inter"


def test_query_is_scoped_to_user(store: MemoryStore):
    for user_id in ("alice", "bob"):
        store.add_fragment(
            user_id,
            _vector(0),
            _vector(0),
            {
                "fragment_id": f"{user_id}-fragment",
                "timestamp": "2026-07-01T00:00:00Z",
            },
        )

    hits = store.query_similar("alice", _vector(0), time_decay=False)

    assert [hit.fragment_id for hit in hits] == ["alice-fragment"]


def test_time_decay_changes_ranking(store: MemoryStore):
    # The old item has perfect similarity; the recent item is only slightly worse.
    store.add_fragment(
        "alice",
        _vector(0),
        _vector(0),
        {"fragment_id": "old", "timestamp": "2026-05-01T00:00:00Z"},
    )
    recent = _vector(0)
    recent[1] = 0.1
    store.add_fragment(
        "alice",
        recent,
        recent,
        {"fragment_id": "recent", "timestamp": "2026-07-09T00:00:00Z"},
    )
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)

    without_decay = store.query_similar(
        "alice", _vector(0), top_k=2, time_decay=False, now=now
    )
    with_decay = store.query_similar(
        "alice", _vector(0), top_k=2, time_decay=True, now=now
    )

    assert [hit.fragment_id for hit in without_decay] == ["old", "recent"]
    assert [hit.fragment_id for hit in with_decay] == ["recent", "old"]
    assert with_decay[0].score > with_decay[1].score


def test_store_persists_across_instances(tmp_path: Path):
    config = MemoryConfig(db_path=tmp_path / "persistent")
    first = MemoryStore(config)
    first.add_fragment(
        "alice",
        _vector(0),
        _vector(0),
        {"fragment_id": "persisted", "timestamp": "2026-07-01T00:00:00Z"},
    )

    second = MemoryStore(config)
    hits = second.query_similar("alice", _vector(0), time_decay=False)

    assert [hit.fragment_id for hit in hits] == ["persisted"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"embedding_type": "unknown"}, "embedding_type"),
        ({"top_k": 0}, "top_k"),
    ],
)
def test_query_validates_options(store: MemoryStore, kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        store.query_similar("alice", _vector(0), **kwargs)


def test_add_validates_embedding_dimension(store: MemoryStore):
    with pytest.raises(ValueError, match="8-d"):
        store.add_fragment("alice", [1.0, 2.0], _vector(0), {})


def test_memory_config_from_pipeline():
    config = MemoryConfig.from_pipeline(
        {
            "pipeline": {
                "stages": {
                    "L3": {
                        "memory": {
                            "enabled": False,
                            "db_path": "custom/chroma",
                            "decay_alpha": 0.25,
                            "top_k": 7,
                            "embedding_type": "self",
                        }
                    }
                }
            }
        }
    )

    assert config.enabled is False
    assert config.db_path == Path("custom/chroma")
    assert config.decay_alpha == pytest.approx(0.25)
    assert config.top_k == 7
    assert config.embedding_type == "self"
