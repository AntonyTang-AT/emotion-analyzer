"""Datasets and dataloaders for L2 VA training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.core.types import MODALITIES

MODALITY_DIMS: dict[str, int] = {
    "text": 768,
    "speech": 1040,
    "macro": 512,
    "micro": 256,
}

MANIFEST_FIELDS = (
    "feature_path",
    "v_self",
    "a_self",
    "c_self",
    "v_inter",
    "a_inter",
    "c_inter",
)


def _normalize_modality(modality: str) -> str:
    name = str(modality).strip().lower()
    if name not in MODALITY_DIMS:
        raise ValueError(f"Unsupported modality '{modality}'")
    return name


def _va_tensor(record: dict[str, Any], prefix: str) -> torch.Tensor:
    return torch.tensor(
        [
            float(record[f"v_{prefix}"]),
            float(record[f"a_{prefix}"]),
            float(record[f"c_{prefix}"]),
        ],
        dtype=torch.float32,
    )


class MockVATrainingDataset(Dataset):
    """Synthetic features and VA labels for development training."""

    def __init__(
        self,
        modality: str,
        *,
        num_samples: int = 512,
        seed: int = 42,
    ) -> None:
        self.modality = _normalize_modality(modality)
        self.feature_dim = MODALITY_DIMS[self.modality]
        rng = np.random.default_rng(seed)
        self.features = rng.standard_normal((num_samples, self.feature_dim)).astype(
            np.float32
        )
        self.target_self = rng.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(
            np.float32
        )
        self.target_self = np.concatenate(
            [self.target_self, rng.uniform(0.0, 1.0, size=(num_samples, 1)).astype(np.float32)],
            axis=1,
        )
        self.target_inter = rng.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(
            np.float32
        )
        self.target_inter = np.concatenate(
            [self.target_inter, rng.uniform(0.0, 1.0, size=(num_samples, 1)).astype(np.float32)],
            axis=1,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(self.features[index])
        target_self = torch.from_numpy(self.target_self[index])
        target_inter = torch.from_numpy(self.target_inter[index])
        return feature, target_self, target_inter


class ManifestVATrainingDataset(Dataset):
    """Load features and VA labels from a JSONL manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")

        self.records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [field for field in MANIFEST_FIELDS if field not in record]
            if missing:
                raise ValueError(f"Manifest row missing fields {missing}: {record}")
            self.records.append(record)

        if not self.records:
            raise ValueError(f"Manifest is empty: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        feature_path = Path(record["feature_path"])
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        feature = torch.from_numpy(np.load(feature_path).astype(np.float32))
        if feature.ndim != 1:
            feature = feature.reshape(-1)
        target_self = _va_tensor(record, "self")
        target_inter = _va_tensor(record, "inter")
        return feature, target_self, target_inter


def build_dataloaders(
    modality: str,
    *,
    mock: bool = False,
    manifest_path: str | Path | None = None,
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
    mock_samples: int = 512,
) -> tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders."""
    _normalize_modality(modality)

    if mock:
        dataset: Dataset = MockVATrainingDataset(
            modality,
            num_samples=mock_samples,
            seed=seed,
        )
    elif manifest_path is not None:
        dataset = ManifestVATrainingDataset(manifest_path)
    else:
        raise ValueError("Either mock=True or manifest_path must be provided")

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 samples")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size < 1:
        val_size = 1
        train_size = len(dataset) - 1

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def supported_modalities() -> tuple[str, ...]:
    return MODALITIES
