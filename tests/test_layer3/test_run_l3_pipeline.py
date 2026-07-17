"""L1 stub → L2 → L3 pipeline integration coverage for Issue #54."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="L2 predictors require the declared torch dependency")


def test_text_pipeline_produces_l3_segments_without_persistent_memory(
    run_l3_text_pipeline,
):
    result = run_l3_text_pipeline()

    assert result["stage_status"]["L1"] == "completed"
    assert result["stage_status"]["L2"] == "completed"
    assert result["stage_status"]["L3"] == "completed"
    assert result["stage_status"]["L4"] == "skipped"
    assert result["stage_status"]["L5"] == "skipped"
    assert result["stage_status"]["L6"] == "skipped"

    context = result["context"]
    assert context["memory_retrieved"] == []
    assert len(context["segments"]) >= 1
    first_segment = context["segments"][0]
    assert first_segment["id"].startswith("seg-")
    assert set(first_segment["va_self"]) == {"text"}
    assert set(first_segment["va_inter"]) == {"text"}
    assert set(first_segment["va_self"]["text"]) == {
        "valence",
        "arousal",
        "confidence",
    }
