"""Ensure layer run() methods align with STAGE_FIELDS."""

from __future__ import annotations

import re

import pytest

from src.core import Fragment, MemoryHit
from src.core.context import STAGE_FIELDS, DataContext
from src.core.interfaces import (
    ContradictionDetector,
    FeatureExtractor,
    MemoryRetriever,
    PersonalityRegressor,
    ReportGenerator,
    SegmentController,
    VAPredictionLayer,
)


def _set_stage_keys(run_source: str) -> set[str]:
    block_match = re.search(
        r"set_stage\s*\(\s*[^,]+,\s*\{(.*?)\}\s*,?\s*\)",
        run_source,
        re.DOTALL,
    )
    if not block_match:
        return set()
    block = block_match.group(1)
    return set(re.findall(r'"(\w+)"\s*:', block)) | set(re.findall(r"'(\w+)'\s*:", block))


@pytest.mark.parametrize(
    ("stage", "interface_cls", "required_keys"),
    [
        ("L1", FeatureExtractor, set(STAGE_FIELDS["L1"])),
        ("L2", VAPredictionLayer, set(STAGE_FIELDS["L2"])),
        ("L3", SegmentController, set(STAGE_FIELDS["L3"])),
        ("L3", MemoryRetriever, set(STAGE_FIELDS["L3"])),
        ("L4", ContradictionDetector, set(STAGE_FIELDS["L4"])),
        ("L5", ReportGenerator, set(STAGE_FIELDS["L5"])),
        ("L6", PersonalityRegressor, set(STAGE_FIELDS["L6"])),
    ],
)
def test_run_set_stage_covers_stage_fields(stage, interface_cls, required_keys):
    import inspect

    source = inspect.getsource(interface_cls.run)
    passed_keys = _set_stage_keys(source)
    assert required_keys.issubset(passed_keys), (
        f"{interface_cls.__name__}.run() must pass {required_keys}, got {passed_keys}"
    )


def test_feature_extractor_run_sets_raw_visual(sample_data_context):
    class VisualExtractor(FeatureExtractor):
        def extract(self, context: DataContext):
            return {"macro": [{"dim": 512}]}

        def extract_raw_visual(self, context: DataContext):
            return {"macro": [1.0, 2.0]}

    ctx = VisualExtractor().run(sample_data_context)
    stage = ctx.get_stage("L1")
    assert "features" in stage
    assert stage["raw_visual_features"] == {"macro": [1.0, 2.0]}


def test_memory_retriever_preserves_segments(sample_data_context):
    sample_data_context.segments = [
        Fragment(id="s1", start_time=0.0, end_time=1.0),
    ]

    class DummyMemory(MemoryRetriever):
        def retrieve_memory(self, context: DataContext):
            return [MemoryHit("hist-1", 0.95)]

    ctx = DummyMemory().run(sample_data_context)
    assert len(ctx.segments) == 1
    assert len(ctx.memory_retrieved) == 1
    assert ctx.metadata["stage_status"]["L3"] == "completed"
