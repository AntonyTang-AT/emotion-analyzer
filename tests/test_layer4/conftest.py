"""Shared helpers for L4 unit and integration tests (task 4.8 / Issue #69)."""

from __future__ import annotations

from src.core import DataContext, Fragment


def make_l4_context(
    *fragments: Fragment,
    modalities: list[str] | None = None,
) -> DataContext:
    context = DataContext.create(user_id="l4-test", video_path="data/raw/test.mp4")
    context.metadata["active_modalities"] = modalities or [
        "text",
        "speech",
        "macro",
        "micro",
    ]
    context.set_stage("L3", {"segments": list(fragments), "memory_retrieved": []})
    return context
