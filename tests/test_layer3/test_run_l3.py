"""Tests for the unified L3 entry point (task 3.6)."""

from __future__ import annotations

from src.core.context import DataContext
from src.core.types import MemoryHit, VAConfidence
from src.layer3_segment.fragment import (
    get_va_inter_embedding,
    get_va_self_embedding,
    run_l3,
)


class FakeMemoryStore:
    def __init__(self) -> None:
        self.queries: list[tuple[str, list[float], str]] = []
        self.added: list[tuple[str, list[float], list[float], dict]] = []

    def query_similar(
        self,
        user_id,
        query_embedding,
        top_k=None,
        time_decay=True,
        embedding_type=None,
        **kwargs,
    ):
        self.queries.append((user_id, list(query_embedding), embedding_type))
        return [MemoryHit("historic", 0.9, {"label": "calm"}, embedding_type)]

    def add_fragment(
        self,
        user_id,
        embedding_self,
        embedding_inter,
        metadata=None,
        *,
        fragment_id=None,
    ):
        self.added.append(
            (user_id, list(embedding_self), list(embedding_inter), dict(metadata or {}))
        )
        return fragment_id


def _context(*, memory_enabled: bool = True) -> DataContext:
    context = DataContext.create(
        user_id="alice",
        input_type="text",
        text_content="hello",
        profile_metadata={"segmentation_mode": "single"},
        config_snapshot={
            "pipeline": {
                "pipeline": {
                    "stages": {
                        "L3": {
                            "segmentation": {"use_va_type": "inter"},
                            "baseline": {"enabled": False},
                            "cold_start": {"enabled": False},
                            "memory": {
                                "enabled": memory_enabled,
                                "embedding_type": "inter",
                                "top_k": 3,
                            },
                        }
                    }
                }
            }
        },
    )
    context.features = {
        "text": [{"start_time": 0.0, "end_time": 1.0}]
    }
    context.va_self_predictions = {
        "text": [VAConfidence(0.2, 0.3, 0.8)]
    }
    context.va_inter_predictions = {
        "text": [VAConfidence(0.4, 0.5, 0.9)]
    }
    return context


def test_embedding_helpers_use_stable_eight_dimensions():
    context = _context(memory_enabled=False)
    result = run_l3(context)
    fragment = result.segments[0]

    assert get_va_self_embedding(fragment) == [0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert get_va_inter_embedding(fragment) == [0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_run_l3_segments_queries_then_persists_memory():
    context = _context()
    store = FakeMemoryStore()

    result = run_l3(context, memory_store=store)

    assert result.metadata["stage_status"]["L3"] == "completed"
    assert len(result.segments) == 1
    assert result.memory_retrieved[0].fragment_id == "historic"
    assert result.memory_retrieved[0].metadata["query_fragment_id"] == "seg-0000"
    assert store.queries[0][2] == "inter"
    assert store.added[0][0] == "alice"
    assert store.added[0][3]["fragment_id"].endswith(":seg-0000")
    assert store.added[0][3]["source_fragment_id"] == "seg-0000"


def test_run_l3_marks_missing_l2_output_failed():
    context = _context(memory_enabled=False)
    context.va_inter_predictions = {}

    result = run_l3(context)

    assert result.metadata["stage_status"]["L3"] == "failed"
    assert result.segments == []


def test_run_l3_memory_disabled_does_not_create_store(monkeypatch):
    context = _context(memory_enabled=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("MemoryStore must not be initialized")

    monkeypatch.setattr("src.layer3_segment.fragment.MemoryStore", fail_if_called)
    result = run_l3(context)

    assert result.metadata["stage_status"]["L3"] == "completed"
    assert result.memory_retrieved == []


def test_run_l3_applies_cold_start_personalization():
    context = DataContext.create(
        user_id="brand-new-cold-start-user",
        input_type="text",
        text_content="hello",
        profile_metadata={"segmentation_mode": "single"},
        config_snapshot={
            "pipeline": {
                "pipeline": {
                    "stages": {
                        "L3": {
                            "segmentation": {"use_va_type": "inter"},
                            "baseline": {"enabled": False},
                            "cold_start": {"enabled": True, "top_k_users": 3},
                            "memory": {"enabled": False},
                        }
                    }
                }
            }
        },
    )
    context.features = {"text": [{"start_time": 0.0, "end_time": 1.0}]}
    context.va_self_predictions = {
        "text": [VAConfidence(0.20, 0.10, 0.9)]
    }
    context.va_inter_predictions = {
        "text": [VAConfidence(0.40, 0.50, 0.9)]
    }

    result = run_l3(context)

    assert result.metadata["stage_status"]["L3"] == "completed"
    assert result.metadata["l3_personalization"] == "cold_start"
    adjusted = result.segments[0].va_self["text"]
    assert adjusted.valence < 0.20
    assert adjusted.valence != 0.20 or adjusted.arousal != 0.10
