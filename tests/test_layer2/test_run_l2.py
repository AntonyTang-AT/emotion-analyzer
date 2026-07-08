"""Tests for L2 run_l2 entry point."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import DataContext, MODALITIES, VAConfidence
from src.layer2_predict import PredictorRegistry, TwoBranchMLP, run_l2
from src.layer2_predict.registry import PredictorRegistry as RegistryClass


MODALITY_DIMS = {
    "text": 768,
    "speech": 1040,
    "macro": 512,
    "micro": 256,
}


def _models_config() -> dict:
    return {
        "layer2": {
            "two_branch_mlp": {
                "shared_dim": 16,
                "branch_hidden": 8,
                "dropout": 0.0,
                "modalities": {
                    modality: {"input_dim": dim, "checkpoint": None}
                    for modality, dim in MODALITY_DIMS.items()
                },
            }
        }
    }


def _pipeline_config() -> dict:
    return {"pipeline": {"stages": {"L2": {"enabled": True}}}}


def _init_test_registry(
    *,
    active_modalities=None,
    registry=None,
    models_config=None,
    pipeline_config=None,
    device="cpu",
):
    target = registry or PredictorRegistry()
    target.clear()
    models = models_config or _models_config()
    modalities = tuple(active_modalities) if active_modalities is not None else MODALITIES
    for modality in modalities:
        target.register(
            modality,
            TwoBranchMLP.from_config(modality, models, device=device),
        )
    return target


@pytest.fixture(autouse=True)
def _patch_initialize_registry(monkeypatch):
    monkeypatch.setattr(
        "src.layer2_predict.predictor.initialize_registry",
        _init_test_registry,
    )


def _text_items(count: int = 2) -> list[dict]:
    return [
        {
            "text_embedding": np.ones(MODALITY_DIMS["text"], dtype=np.float32),
            "start_time": float(index),
            "end_time": float(index + 1),
        }
        for index in range(count)
    ]


def _speech_items(count: int = 2) -> list[dict]:
    return [
        {
            "speech_feature": np.ones(MODALITY_DIMS["speech"], dtype=np.float32),
            "timestamp": float(index),
        }
        for index in range(count)
    ]


def _macro_items(count: int = 2) -> list[tuple[np.ndarray, float]]:
    return [
        (np.ones(MODALITY_DIMS["macro"], dtype=np.float32), float(index))
        for index in range(count)
    ]


def _micro_items(count: int = 2) -> list[dict]:
    return [
        {
            "micro_feature": np.ones(MODALITY_DIMS["micro"], dtype=np.float32),
            "start_time": float(index),
            "end_time": float(index + 1),
        }
        for index in range(count)
    ]


def _context_with_features(
    active_modalities: list[str],
    features: dict,
) -> DataContext:
    ctx = DataContext.create(
        user_id="test-user",
        input_type="video",
        video_path="data/raw/test.mp4",
        profile_metadata={"active_modalities": active_modalities},
    )
    ctx.set_stage(
        "L1",
        {"features": features, "raw_visual_features": {}},
    )
    return ctx


def test_run_l2_all_modalities():
    context = _context_with_features(
        list(MODALITIES),
        {
            "text": _text_items(2),
            "speech": _speech_items(2),
            "macro": _macro_items(2),
            "micro": _micro_items(2),
        },
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "completed"
    for modality in MODALITIES:
        assert len(result.va_self_predictions[modality]) == 2
        assert len(result.va_inter_predictions[modality]) == 2
        assert all(isinstance(item, VAConfidence) for item in result.va_self_predictions[modality])


def test_run_l2_speech_dim_1040():
    context = _context_with_features(["speech"], {"speech": _speech_items(1)})

    result = run_l2(context)

    assert len(result.va_self_predictions["speech"]) == 1
    assert isinstance(result.va_self_predictions["speech"][0], VAConfidence)


def test_run_l2_macro_tuple_format():
    context = _context_with_features(["macro"], {"macro": _macro_items(2)})

    result = run_l2(context)

    assert len(result.va_self_predictions["macro"]) == 2


def test_run_l2_skips_missing_features():
    context = _context_with_features(
        ["text", "speech"],
        {"text": _text_items(1)},
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "completed"
    assert "text" in result.va_self_predictions
    assert "speech" not in result.va_self_predictions
    assert result.metadata["l2_partial"] is True
    assert "speech" in result.metadata["l2_failures"]


def test_run_l2_all_fail_marks_failed():
    context = _context_with_features(["text"], {})

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
    context.set_stage("L1", {"features": {"text": _text_items(1)}, "raw_visual_features": {}})

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "failed"
    assert not result.va_self_predictions


def test_run_l2_partial_metadata():
    context = _context_with_features(
        ["text", "micro"],
        {"text": _text_items(1)},
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
        {"features": {"text": _text_items(1)}, "raw_visual_features": {}},
    )

    result = run_l2(context)

    assert result.metadata["stage_status"]["L2"] == "completed"
    assert "text" in result.va_self_predictions
    assert len(result.va_self_predictions["text"]) == 1


def test_extract_feature_vectors_sorts_macro_by_timestamp():
    from src.layer2_predict.predictor import _extract_feature_vectors

    items = _macro_items(2)
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
        return _init_test_registry(**kwargs)

    monkeypatch.setattr(
        "src.layer2_predict.predictor.initialize_registry",
        _capture_registry,
    )
    context = _context_with_features(["text"], {"text": _text_items(1)})

    run_l2(context)

    assert len(captured) == 1
    assert isinstance(captured[0], PredictorRegistry)
