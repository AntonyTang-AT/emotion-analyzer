"""Tests for L2 run_l2 entry point."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import DataContext, MODALITIES, VAConfidence
from src.layer2_predict import PredictorRegistry, run_l2
from src.layer2_predict.registry import PredictorRegistry as RegistryClass

from tests.test_layer2.conftest import (
    MODALITY_DIMS,
    context_with_l1_features,
    init_test_registry,
    make_l1_feature_items,
)


@pytest.fixture(autouse=True)
def _patch_initialize_registry(monkeypatch):
    monkeypatch.setattr(
        "src.layer2_predict.predictor.initialize_registry",
        init_test_registry,
    )


def test_run_l2_all_modalities():
    context = context_with_l1_features(
        list(MODALITIES),
        {
            "text": make_l1_feature_items("text", 2),
            "speech": make_l1_feature_items("speech", 2),
            "macro": make_l1_feature_items("macro", 2),
            "micro": make_l1_feature_items("micro", 2),
        },
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "completed"
    for modality in MODALITIES:
        assert len(result.va_self_predictions[modality]) == 2
        assert len(result.va_inter_predictions[modality]) == 2
        assert all(isinstance(item, VAConfidence) for item in result.va_self_predictions[modality])


def test_run_l2_speech_dim_1040():
    context = context_with_l1_features(
        ["speech"],
        {"speech": make_l1_feature_items("speech", 1)},
    )

    result = run_l2(context)

    assert len(result.va_self_predictions["speech"]) == 1
    assert isinstance(result.va_self_predictions["speech"][0], VAConfidence)


def test_run_l2_macro_tuple_format():
    context = context_with_l1_features(
        ["macro"],
        {"macro": make_l1_feature_items("macro", 2)},
    )

    result = run_l2(context)

    assert len(result.va_self_predictions["macro"]) == 2


def test_run_l2_skips_missing_features():
    context = context_with_l1_features(
        ["text", "speech"],
        {"text": make_l1_feature_items("text", 1)},
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "completed"
    assert "text" in result.va_self_predictions
    assert "speech" not in result.va_self_predictions
    assert result.metadata["l2_partial"] is True
    assert "speech" in result.metadata["l2_failures"]


def test_run_l2_all_fail_marks_failed():
    context = context_with_l1_features(["text"], {})

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "failed"
    assert not result.va_self_predictions


def test_run_l2_empty_active_modalities_marks_failed():
    context = DataContext.create(
        user_id="test-user",
        input_type="text",
        text_content="hello",
        profile_metadata={"active_modalities": []},
    )
    context.set_stage(
        "L1",
        {"features": {"text": make_l1_feature_items("text", 1)}, "raw_visual_features": {}},
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "failed"
    assert not result.va_self_predictions


def test_run_l2_partial_metadata():
    context = context_with_l1_features(
        ["text", "micro"],
        {"text": make_l1_feature_items("text", 1)},
    )

    result = run_l2(context)

    assert result.metadata["l2_partial"] is True
    assert set(result.metadata["l2_failures"]) == {"micro"}


def test_run_l2_normalizes_modality_names():
    context = DataContext.create(
        user_id="test-user",
        input_type="video",
        video_path="data/raw/test.mp4",
        profile_metadata={"active_modalities": ["Text"]},
    )
    context.set_stage(
        "L1",
        {"features": {"text": make_l1_feature_items("text", 1)}, "raw_visual_features": {}},
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "completed"
    assert "text" in result.va_self_predictions
    assert len(result.va_self_predictions["text"]) == 1


def test_extract_feature_vectors_sorts_macro_by_timestamp():
    from src.layer2_predict.predictor import _extract_feature_vectors

    items = make_l1_feature_items("macro", 2)
    items = list(reversed(items))
    vectors = _extract_feature_vectors("macro", items)

    assert len(vectors) == 2
    assert vectors[0].shape == (MODALITY_DIMS["macro"],)


def test_run_l2_uses_isolated_registry(monkeypatch):
    captured: list[RegistryClass] = []

    def _capture_registry(**kwargs):
        registry = kwargs.get("registry")
        if registry is not None:
            captured.append(registry)
        return init_test_registry(**kwargs)

    monkeypatch.setattr(
        "src.layer2_predict.predictor.initialize_registry",
        _capture_registry,
    )
    context = context_with_l1_features(
        ["text"],
        {"text": make_l1_feature_items("text", 1)},
    )

    run_l2(context)

    assert len(captured) == 1
    assert isinstance(captured[0], PredictorRegistry)


@pytest.mark.parametrize("step_count", [1, 3, 5])
def test_run_l2_va_list_length_matches_l1_steps(step_count: int):
    context = context_with_l1_features(
        ["text"],
        {"text": make_l1_feature_items("text", step_count)},
    )

    result = run_l2(context)

    assert len(result.va_self_predictions["text"]) == step_count
    assert len(result.va_inter_predictions["text"]) == step_count


def test_run_l2_self_inter_same_length():
    context = context_with_l1_features(
        list(MODALITIES),
        {
            modality: make_l1_feature_items(modality, 3)
            for modality in MODALITIES
        },
    )

    result = run_l2(context)

    for modality in MODALITIES:
        assert len(result.va_self_predictions[modality]) == len(
            result.va_inter_predictions[modality]
        )


def test_run_l2_va_confidence_in_range():
    context = context_with_l1_features(
        list(MODALITIES),
        {
            modality: make_l1_feature_items(modality, 2)
            for modality in MODALITIES
        },
    )

    result = run_l2(context)

    for modality in MODALITIES:
        for branch in (result.va_self_predictions, result.va_inter_predictions):
            for item in branch[modality]:
                assert -1.0 <= item.valence <= 1.0
                assert -1.0 <= item.arousal <= 1.0
                assert 0.0 <= item.confidence <= 1.0
