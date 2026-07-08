"""Tests for L2 training loop."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.layer2_predict.trainer import train_modality
from src.layer2_predict.two_branch_mlp import TwoBranchMLP


def _tiny_models_config() -> dict:
    return {
        "layer2": {
            "two_branch_mlp": {
                "shared_dim": 8,
                "branch_hidden": 4,
                "dropout": 0.0,
                "modalities": {
                    "text": {"input_dim": 768, "checkpoint": None},
                    "speech": {"input_dim": 1040, "checkpoint": None},
                    "macro": {"input_dim": 512, "checkpoint": None},
                    "micro": {"input_dim": 256, "checkpoint": None},
                },
            },
            "multitask": {
                "learnable_loss_weights": True,
                "self_loss_weight": 0.3,
                "inter_loss_weight": 0.7,
            },
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.01,
                "val_split": 0.2,
                "mock_samples": 32,
                "seed": 0,
            },
        }
    }


def test_train_modality_mock_saves_checkpoints(tmp_path: Path):
    output_dir = tmp_path / "text"
    result = train_modality(
        "text",
        mock=True,
        models_config=_tiny_models_config(),
        output_dir=output_dir,
        epochs=1,
        batch_size=8,
        mock_samples=32,
        device="cpu",
        shared_dim=8,
        branch_hidden=4,
    )

    assert result.best_checkpoint.is_file()
    assert result.last_checkpoint.is_file()
    assert result.metrics_path.is_file()

    history = json.loads(result.metrics_path.read_text(encoding="utf-8"))["history"]
    assert len(history) == 1
    assert "mae_va" in history[0]

    reloaded = TwoBranchMLP(
        modality="text",
        input_dim=768,
        shared_dim=8,
        hidden_dim=4,
        checkpoint_path=result.best_checkpoint,
    )
    assert isinstance(reloaded, TwoBranchMLP)


def test_train_modality_resume_continues(tmp_path: Path):
    output_dir = tmp_path / "micro"
    config = _tiny_models_config()

    first = train_modality(
        "micro",
        mock=True,
        models_config=config,
        output_dir=output_dir,
        epochs=1,
        batch_size=8,
        mock_samples=32,
        device="cpu",
        shared_dim=8,
        branch_hidden=4,
    )
    first_payload = torch.load(first.last_checkpoint, map_location="cpu", weights_only=False)
    assert first_payload["epoch"] == 1

    second = train_modality(
        "micro",
        mock=True,
        models_config=config,
        output_dir=output_dir,
        epochs=2,
        batch_size=8,
        mock_samples=32,
        device="cpu",
        resume_path=first.last_checkpoint,
        shared_dim=8,
        branch_hidden=4,
    )

    history = json.loads(second.metrics_path.read_text(encoding="utf-8"))["history"]
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2
    assert second.best_checkpoint.is_file()

    second_payload = torch.load(second.last_checkpoint, map_location="cpu", weights_only=False)
    assert second_payload["epoch"] == 2
