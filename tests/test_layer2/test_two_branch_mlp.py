"""Unit tests for L2 TwoBranchMLP predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.core.interfaces import VAPredictor
from src.core.types import MODALITIES, VAConfidence
from src.layer2_predict import BasePredictor, TwoBranchMLP

MODALITY_DIMS = {
    "text": 768,
    "speech": 1040,
    "macro": 512,
    "micro": 256,
}


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.mark.parametrize("modality", MODALITIES)
def test_instantiate_all_modalities(modality: str):
    model = TwoBranchMLP.from_config(modality)
    assert model.modality == modality
    assert model.input_dim == MODALITY_DIMS[modality]


@pytest.mark.parametrize("modality", MODALITIES)
def test_predict_self_output_range(modality: str):
    model = TwoBranchMLP.from_config(modality)
    features = np.random.randn(MODALITY_DIMS[modality]).astype(np.float32)
    result = model.predict_self(features)

    assert isinstance(result, VAConfidence)
    assert -1.0 <= result.valence <= 1.0
    assert -1.0 <= result.arousal <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.valence, float)
    assert isinstance(result.arousal, float)
    assert isinstance(result.confidence, float)


@pytest.mark.parametrize("modality", MODALITIES)
def test_predict_inter_output_range(modality: str):
    model = TwoBranchMLP.from_config(modality)
    features = np.random.randn(MODALITY_DIMS[modality]).astype(np.float32)
    result = model.predict_inter(features)

    assert isinstance(result, VAConfidence)
    assert -1.0 <= result.valence <= 1.0
    assert -1.0 <= result.arousal <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_branches_are_independent():
    model = TwoBranchMLP.from_config("text")
    features = np.random.randn(768).astype(np.float32)
    self_result = model.predict_self(features)
    inter_result = model.predict_inter(features)

    assert (self_result.valence, self_result.arousal, self_result.confidence) != (
        inter_result.valence,
        inter_result.arousal,
        inter_result.confidence,
    )


def test_invalid_input_dim_raises():
    model = TwoBranchMLP.from_config("micro")
    with pytest.raises(ValueError, match="Expected feature length"):
        model.predict_self(np.zeros(128, dtype=np.float32))


def test_checkpoint_roundtrip(tmp_path: Path):
    model = TwoBranchMLP(modality="text", input_dim=768)
    features = np.random.randn(768).astype(np.float32)
    before = model.predict_self(features)

    checkpoint_path = tmp_path / "text_l2.pt"
    torch.save(model.state_dict(), checkpoint_path)

    reloaded = TwoBranchMLP(
        modality="text",
        input_dim=768,
        checkpoint_path=checkpoint_path,
    )
    after = reloaded.predict_self(features)

    assert before.valence == pytest.approx(after.valence)
    assert before.arousal == pytest.approx(after.arousal)
    assert before.confidence == pytest.approx(after.confidence)


def test_is_base_predictor():
    model = TwoBranchMLP.from_config("speech")
    assert isinstance(model, BasePredictor)
    assert isinstance(model, VAPredictor)


def test_forward_supports_batch():
    model = TwoBranchMLP(modality="macro", input_dim=512)
    batch = torch.randn(4, 512)
    self_out, inter_out = model.forward(batch)

    assert self_out.shape == (4, 3)
    assert inter_out.shape == (4, 3)
