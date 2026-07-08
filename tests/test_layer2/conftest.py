"""Shared constants and helpers for L2 layer tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import DataContext, MODALITIES
from src.layer2_predict import PredictorRegistry, TwoBranchMLP
from src.layer2_predict.registry import default_registry

MODALITY_DIMS = {
    "text": 768,
    "speech": 1040,
    "macro": 512,
    "micro": 256,
}

FEATURE_KEYS = {
    "text": "text_embedding",
    "speech": "speech_feature",
    "micro": "micro_feature",
}


def tiny_models_config(*, text_checkpoint: str | None = None) -> dict:
    return {
        "layer2": {
            "two_branch_mlp": {
                "shared_dim": 16,
                "branch_hidden": 8,
                "dropout": 0.0,
                "modalities": {
                    "text": {
                        "input_dim": MODALITY_DIMS["text"],
                        "checkpoint": text_checkpoint,
                    },
                    "speech": {
                        "input_dim": MODALITY_DIMS["speech"],
                        "checkpoint": None,
                    },
                    "macro": {
                        "input_dim": MODALITY_DIMS["macro"],
                        "checkpoint": None,
                    },
                    "micro": {
                        "input_dim": MODALITY_DIMS["micro"],
                        "checkpoint": None,
                    },
                },
            }
        }
    }


def pipeline_config_l2(*, enabled: bool = True) -> dict:
    return {"pipeline": {"stages": {"L2": {"enabled": enabled}}}}


def init_test_registry(
    *,
    active_modalities=None,
    registry=None,
    models_config=None,
    pipeline_config=None,
    device="cpu",
):
    target = registry or PredictorRegistry()
    target.clear()
    models = models_config or tiny_models_config()
    modalities = tuple(active_modalities) if active_modalities is not None else MODALITIES
    for modality in modalities:
        target.register(
            modality,
            TwoBranchMLP.from_config(modality, models, device=device),
        )
    return target


def make_l1_feature_items(modality: str, count: int = 2) -> list:
    name = modality.strip().lower()
    if name == "text":
        return [
            {
                FEATURE_KEYS["text"]: np.ones(MODALITY_DIMS["text"], dtype=np.float32),
                "start_time": float(index),
                "end_time": float(index + 1),
            }
            for index in range(count)
        ]
    if name == "speech":
        return [
            {
                FEATURE_KEYS["speech"]: np.ones(
                    MODALITY_DIMS["speech"], dtype=np.float32
                ),
                "timestamp": float(index),
            }
            for index in range(count)
        ]
    if name == "macro":
        return [
            (np.ones(MODALITY_DIMS["macro"], dtype=np.float32), float(index))
            for index in range(count)
        ]
    if name == "micro":
        return [
            {
                FEATURE_KEYS["micro"]: np.ones(MODALITY_DIMS["micro"], dtype=np.float32),
                "start_time": float(index),
                "end_time": float(index + 1),
            }
            for index in range(count)
        ]
    raise ValueError(f"Unsupported modality '{modality}'")


def context_with_l1_features(
    active_modalities: list[str],
    features: dict,
    *,
    input_type: str = "video",
    video_path: str = "data/raw/test.mp4",
) -> DataContext:
    ctx = DataContext.create(
        user_id="test-user",
        input_type=input_type,
        video_path=video_path,
        profile_metadata={"active_modalities": active_modalities},
    )
    ctx.set_stage(
        "L1",
        {"features": features, "raw_visual_features": {}},
    )
    return ctx


@pytest.fixture(autouse=True)
def _reset_default_registry():
    default_registry.clear()
    yield
    default_registry.clear()


@pytest.fixture
def patched_l2_registry(monkeypatch):
    """Patch run_l2 to use tiny in-memory predictors instead of config checkpoints."""

    def _patch(**kwargs):
        return init_test_registry(**kwargs)

    monkeypatch.setattr(
        "src.layer2_predict.predictor.initialize_registry",
        _patch,
    )
    return _patch
