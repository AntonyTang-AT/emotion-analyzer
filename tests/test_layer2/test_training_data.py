"""Tests for L2 training datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.layer2_predict.training_data import (
    MODALITY_DIMS,
    ManifestVATrainingDataset,
    MockVATrainingDataset,
    build_dataloaders,
)


def test_mock_dataset_returns_expected_shapes():
    dataset = MockVATrainingDataset("speech", num_samples=8, seed=0)
    feature, target_self, target_inter = dataset[0]

    assert feature.shape == (MODALITY_DIMS["speech"],)
    assert target_self.shape == (3,)
    assert target_inter.shape == (3,)


def test_build_dataloaders_mock_split():
    train_loader, val_loader = build_dataloaders(
        "text",
        mock=True,
        batch_size=4,
        val_split=0.25,
        mock_samples=20,
        seed=1,
    )

    train_count = sum(batch[0].shape[0] for batch in train_loader)
    val_count = sum(batch[0].shape[0] for batch in val_loader)
    assert train_count + val_count == 20
    assert val_count >= 1


def test_manifest_dataset_loads_jsonl(tmp_path: Path):
    feature_path = tmp_path / "sample.npy"
    np.save(feature_path, np.ones(MODALITY_DIMS["micro"], dtype=np.float32))

    manifest_path = tmp_path / "manifest.jsonl"
    record = {
        "feature_path": str(feature_path),
        "v_self": 0.2,
        "a_self": -0.4,
        "c_self": 0.8,
        "v_inter": 0.1,
        "a_inter": -0.2,
        "c_inter": 0.9,
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = ManifestVATrainingDataset(manifest_path)
    feature, target_self, target_inter = dataset[0]

    assert feature.shape == (MODALITY_DIMS["micro"],)
    assert target_self.tolist() == pytest.approx([0.2, -0.4, 0.8])
    assert target_inter.tolist() == pytest.approx([0.1, -0.2, 0.9])


def test_manifest_dataset_requires_fields(tmp_path: Path):
    manifest_path = tmp_path / "bad.jsonl"
    manifest_path.write_text('{"feature_path": "missing.npy"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        ManifestVATrainingDataset(manifest_path)
