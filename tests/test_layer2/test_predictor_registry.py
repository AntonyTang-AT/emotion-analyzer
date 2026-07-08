"""Tests for L2 predictor registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.core.types import MODALITIES
from src.layer2_predict import PredictorRegistry, TwoBranchMLP, get_predictor
from src.layer2_predict.registry import default_registry, initialize_registry


def _models_config(*, text_checkpoint: str | None = None) -> dict:
    return {
        "layer2": {
            "two_branch_mlp": {
                "shared_dim": 16,
                "branch_hidden": 8,
                "dropout": 0.0,
                "modalities": {
                    "text": {"input_dim": 768, "checkpoint": text_checkpoint},
                    "speech": {"input_dim": 1040, "checkpoint": None},
                    "macro": {"input_dim": 512, "checkpoint": None},
                    "micro": {"input_dim": 256, "checkpoint": None},
                },
            }
        }
    }


def _pipeline_config(enabled: bool = True) -> dict:
    return {"pipeline": {"stages": {"L2": {"enabled": enabled}}}}


@pytest.fixture(autouse=True)
def _reset_default_registry():
    default_registry.clear()
    yield
    default_registry.clear()


def test_initialize_registry_registers_all_modalities():
    registry = initialize_registry(
        registry=PredictorRegistry(),
        models_config=_models_config(),
        pipeline_config=_pipeline_config(),
    )

    assert registry.names() == MODALITIES
    for modality in MODALITIES:
        predictor = registry.get_predictor(modality)
        assert isinstance(predictor, TwoBranchMLP)
        assert predictor.modality == modality


def test_initialize_registry_uses_expected_input_dims():
    registry = initialize_registry(
        registry=PredictorRegistry(),
        models_config=_models_config(),
        pipeline_config=_pipeline_config(),
    )

    assert registry.get_predictor("text").input_dim == 768
    assert registry.get_predictor("speech").input_dim == 1040
    assert registry.get_predictor("macro").input_dim == 512
    assert registry.get_predictor("micro").input_dim == 256


def test_initialize_registry_respects_active_modalities():
    registry = initialize_registry(
        active_modalities=["text", "speech"],
        registry=PredictorRegistry(),
        models_config=_models_config(),
        pipeline_config=_pipeline_config(),
    )

    assert registry.names() == ("text", "speech")


def test_initialize_registry_skips_when_l2_disabled():
    registry = initialize_registry(
        registry=PredictorRegistry(),
        models_config=_models_config(),
        pipeline_config=_pipeline_config(enabled=False),
    )

    assert len(registry) == 0


def test_get_predictor_reports_missing_modality():
    registry = PredictorRegistry()

    with pytest.raises(KeyError, match="No L2 predictor registered"):
        registry.get_predictor("unknown")


def test_registry_normalizes_modality_names():
    registry = PredictorRegistry()
    predictor = TwoBranchMLP(modality="speech", input_dim=1040, shared_dim=8, hidden_dim=4)
    registry.register("Speech", predictor)

    assert registry.get("SPEECH") is predictor
    assert "Speech" in registry


def test_initialize_registry_rejects_unknown_modality():
    with pytest.raises(ValueError, match="No layer2 config for modality"):
        initialize_registry(
            active_modalities=["unknown"],
            registry=PredictorRegistry(),
            models_config=_models_config(),
            pipeline_config=_pipeline_config(),
        )


def test_initialize_registry_loads_checkpoint(tmp_path: Path):
    model = TwoBranchMLP(
        modality="text",
        input_dim=768,
        shared_dim=16,
        hidden_dim=8,
        dropout=0.0,
    )
    features = np.random.randn(768).astype(np.float32)
    torch.manual_seed(0)
    before = model.predict_self(features)

    checkpoint_path = tmp_path / "text_l2.pt"
    torch.save(model.state_dict(), checkpoint_path)

    registry = initialize_registry(
        active_modalities=["text"],
        registry=PredictorRegistry(),
        models_config=_models_config(text_checkpoint=str(checkpoint_path)),
        pipeline_config=_pipeline_config(),
    )

    after = registry.get_predictor("text").predict_self(features)
    assert before.valence == pytest.approx(after.valence)
    assert before.arousal == pytest.approx(after.arousal)
    assert before.confidence == pytest.approx(after.confidence)


def test_package_get_predictor_uses_default_registry():
    initialize_registry(
        active_modalities=["micro"],
        registry=default_registry,
        models_config=_models_config(),
        pipeline_config=_pipeline_config(),
    )

    assert get_predictor("micro").input_dim == 256
