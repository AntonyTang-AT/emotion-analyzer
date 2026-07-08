"""Tests for L2 predictor registry."""

from __future__ import annotations

import pytest

from src.core.types import MODALITIES
from src.layer2_predict import PredictorRegistry, TwoBranchMLP, get_predictor
from src.layer2_predict.registry import default_registry, initialize_registry


def _models_config() -> dict:
    return {
        "layer2": {
            "two_branch_mlp": {
                "shared_dim": 16,
                "branch_hidden": 8,
                "dropout": 0.0,
                "modalities": {
                    "text": {"input_dim": 768, "checkpoint": None},
                    "speech": {"input_dim": 1040, "checkpoint": None},
                    "macro": {"input_dim": 512, "checkpoint": None},
                    "micro": {"input_dim": 256, "checkpoint": None},
                },
            }
        }
    }


def _pipeline_config(enabled: bool = True) -> dict:
    return {"pipeline": {"stages": {"L2": {"enabled": enabled}}}}


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


def test_package_get_predictor_uses_default_registry():
    initialize_registry(
        active_modalities=["micro"],
        registry=default_registry,
        models_config=_models_config(),
        pipeline_config=_pipeline_config(),
    )

    assert get_predictor("micro").input_dim == 256
    default_registry.clear()
