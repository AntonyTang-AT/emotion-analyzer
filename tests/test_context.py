"""Tests for DataContext stage storage and serialization."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import DataContext, Fragment, VAConfidence, VALID_STAGES


def test_create_sets_metadata(sample_data_context):
    ctx = sample_data_context
    assert ctx.metadata["user_id"] == "test-user"
    assert ctx.metadata["session_id"]
    assert set(ctx.metadata["stage_status"]) == set(VALID_STAGES)
    assert ctx.metadata["stage_status"]["L2"] == "completed"
    assert ctx.metadata["stage_status"]["L1"] == "pending"


def test_set_stage_updates_status(sample_data_context):
    ctx = sample_data_context
    ctx.set_stage("L1", {"features": {"speech": []}, "raw_visual_features": {}})
    assert ctx.metadata["stage_status"]["L1"] == "completed"
    assert "speech" in ctx.features


def test_get_stage_l2_roundtrip(sample_data_context):
    stage = sample_data_context.get_stage("L2")
    assert stage["va_self_predictions"]["text"]["valence"] == pytest.approx(0.1)
    assert stage["va_inter_predictions"]["text"]["arousal"] == pytest.approx(0.4)


def test_unknown_stage_raises(sample_data_context):
    with pytest.raises(ValueError, match="Unknown stage"):
        sample_data_context.set_stage("L9", {})
    with pytest.raises(ValueError, match="Unknown stage"):
        sample_data_context.get_stage("L9")


def test_to_dict_from_dict_roundtrip(full_data_context):
    restored = DataContext.from_dict(full_data_context.to_dict())
    assert restored.metadata["user_id"] == full_data_context.metadata["user_id"]
    assert restored.personality.O == pytest.approx(7.0)
    assert restored.segments[0].contradiction.contradiction_type.value == "masking"


def test_json_roundtrip(full_data_context):
    restored = DataContext.from_json(full_data_context.to_json())
    assert restored.reports.overall_report == "overall"


def test_save_and_load_json(full_data_context, tmp_path):
    path = tmp_path / "context.json"
    full_data_context.save_json(path)
    loaded = DataContext.load_json(path)
    assert loaded.segments[0].id == "seg-1"


def test_ndarray_in_raw_visual_features(full_data_context):
    payload = full_data_context.to_dict()
    restored = DataContext.from_dict(payload)
    arr = restored.raw_visual_features["micro"]
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_mark_stage_failed(sample_data_context):
    ctx = sample_data_context
    ctx.mark_stage_failed("L3", "segmentation error")
    assert ctx.metadata["stage_status"]["L3"] == "failed"
    assert ctx.metadata["errors"]["L3"] == "segmentation error"


def test_fragment_contradiction_serialization(sample_contradiction):
    fragment = Fragment(
        id="f1",
        start_time=0.0,
        end_time=1.0,
        contradiction=sample_contradiction,
    )
    restored = Fragment.from_dict(fragment.to_dict())
    assert restored.contradiction.suggested_fusion_weights == [0.1, 0.2, 0.2, 0.5]
