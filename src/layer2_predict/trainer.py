"""Training loop helpers for L2 two-branch models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.layer2_predict.losses import TwoBranchMultitaskLoss, activate_va_logits
from src.layer2_predict.training_data import build_dataloaders
from src.layer2_predict.two_branch_mlp import TwoBranchMLP
from src.utils.config_loader import get_config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_path: Path
    history: list[dict[str, Any]]
    best_mae_va: float


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _evaluate_batch_metrics(
    self_raw: torch.Tensor,
    inter_raw: torch.Tensor,
    target_self: torch.Tensor,
    target_inter: torch.Tensor,
) -> dict[str, float]:
    self_pred = activate_va_logits(self_raw).detach().cpu().numpy()
    inter_pred = activate_va_logits(inter_raw).detach().cpu().numpy()
    self_true = target_self.detach().cpu().numpy()
    inter_true = target_inter.detach().cpu().numpy()

    return {
        "mae_v_self": float(np.mean(np.abs(self_pred[:, 0] - self_true[:, 0]))),
        "mae_a_self": float(np.mean(np.abs(self_pred[:, 1] - self_true[:, 1]))),
        "mae_v_inter": float(np.mean(np.abs(inter_pred[:, 0] - inter_true[:, 0]))),
        "mae_a_inter": float(np.mean(np.abs(inter_pred[:, 1] - inter_true[:, 1]))),
        "pearson_v_self": _pearson(self_pred[:, 0], self_true[:, 0]),
        "pearson_a_self": _pearson(self_pred[:, 1], self_true[:, 1]),
        "pearson_v_inter": _pearson(inter_pred[:, 0], inter_true[:, 0]),
        "pearson_a_inter": _pearson(inter_pred[:, 1], inter_true[:, 1]),
    }


def _aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    result: dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in items if not np.isnan(item[key])]
        result[key] = float(np.mean(values)) if values else float("nan")
    result["mae_va"] = float(
        np.nanmean(
            [
                result.get("mae_v_self", np.nan),
                result.get("mae_a_self", np.nan),
                result.get("mae_v_inter", np.nan),
                result.get("mae_a_inter", np.nan),
            ]
        )
    )
    return result


@torch.no_grad()
def evaluate(
    model: TwoBranchMLP,
    dataloader: DataLoader,
    criterion: TwoBranchMultitaskLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []

    for features, target_self, target_inter in dataloader:
        features = features.to(device)
        target_self = target_self.to(device)
        target_inter = target_inter.to(device)
        self_raw, inter_raw = model(features)
        loss, _ = criterion(self_raw, inter_raw, target_self, target_inter)
        losses.append(float(loss.item()))
        metric_items.append(
            _evaluate_batch_metrics(self_raw, inter_raw, target_self, target_inter)
        )

    metrics = _aggregate_metrics(metric_items)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def _build_model_config(
    models_config: dict[str, Any],
    *,
    shared_dim: int | None = None,
    branch_hidden: int | None = None,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(models_config))
    l2_cfg = cfg.setdefault("layer2", {}).setdefault("two_branch_mlp", {})
    if shared_dim is not None:
        l2_cfg["shared_dim"] = shared_dim
    if branch_hidden is not None:
        l2_cfg["branch_hidden"] = branch_hidden
    return cfg


def _save_checkpoint(
    path: Path,
    *,
    model: TwoBranchMLP,
    optimizer: torch.optim.Optimizer,
    criterion: TwoBranchMultitaskLoss,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def train_modality(
    modality: str,
    *,
    mock: bool = False,
    manifest_path: str | Path | None = None,
    models_config: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_split: float | None = None,
    mock_samples: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
    resume_path: str | Path | None = None,
    shared_dim: int | None = None,
    branch_hidden: int | None = None,
) -> TrainResult:
    """Train one modality TwoBranchMLP and save checkpoints."""
    models_cfg = models_config or load_config("models")
    pipeline_cfg = pipeline_config or load_config("pipeline")
    train_cfg = models_cfg.get("layer2", {}).get("training", {})

    resolved_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 20))
    resolved_batch_size = int(
        batch_size if batch_size is not None else train_cfg.get("batch_size", 64)
    )
    resolved_lr = float(
        learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 1e-3)
    )
    resolved_val_split = float(
        val_split if val_split is not None else train_cfg.get("val_split", 0.2)
    )
    resolved_mock_samples = int(
        mock_samples if mock_samples is not None else train_cfg.get("mock_samples", 512)
    )
    resolved_seed = int(seed if seed is not None else train_cfg.get("seed", 42))

    checkpoint_root = Path(
        output_dir
        if output_dir is not None
        else train_cfg.get("checkpoint_dir", "models/l2")
    )
    if output_dir is None:
        checkpoint_root = checkpoint_root / modality
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(resolved_seed)
    np.random.seed(resolved_seed)

    train_loader, val_loader = build_dataloaders(
        modality,
        mock=mock,
        manifest_path=manifest_path,
        batch_size=resolved_batch_size,
        val_split=resolved_val_split,
        seed=resolved_seed,
        mock_samples=resolved_mock_samples,
    )

    model_cfg = _build_model_config(
        models_cfg,
        shared_dim=shared_dim,
        branch_hidden=branch_hidden,
    )
    torch_device = torch.device(device)
    model = TwoBranchMLP.from_config(modality, model_cfg, device=torch_device)
    criterion = TwoBranchMultitaskLoss.from_config(model_cfg, pipeline_cfg)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=resolved_lr,
    )

    start_epoch = 0
    best_mae_va = float("inf")
    history: list[dict[str, Any]] = []

    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    metrics_path = checkpoint_root / "metrics.json"

    if resume_path is not None:
        resume_file = Path(resume_path)
        if not resume_file.is_file():
            resolved = get_config().resolve_path(str(resume_path))
            if resolved.is_file():
                resume_file = resolved
        if resume_file.is_file():
            payload = torch.load(resume_file, map_location=torch_device, weights_only=False)
            model.load_state_dict(payload["state_dict"])
            optimizer.load_state_dict(payload["optimizer"])
            criterion.load_state_dict(payload["criterion"])
            start_epoch = int(payload.get("epoch", 0))
            best_mae_va = float(payload.get("metrics", {}).get("mae_va", best_mae_va))
            if metrics_path.is_file():
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                history = list(existing.get("history", []))
                if np.isfinite(existing.get("best_mae_va", float("nan"))):
                    best_mae_va = float(existing["best_mae_va"])
            logger.info("Resumed training from %s at epoch %s", resume_file, start_epoch)

    for epoch in range(start_epoch, resolved_epochs):
        model.train()
        train_losses: list[float] = []
        for features, target_self, target_inter in train_loader:
            features = features.to(torch_device)
            target_self = target_self.to(torch_device)
            target_inter = target_inter.to(torch_device)

            optimizer.zero_grad()
            self_raw, inter_raw = model(features)
            loss, parts = criterion(self_raw, inter_raw, target_self, target_inter)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, criterion, torch_device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **val_metrics,
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %s/%s train_loss=%.4f val_mae_va=%.4f",
            epoch + 1,
            resolved_epochs,
            epoch_record["train_loss"],
            val_metrics.get("mae_va", float("nan")),
        )

        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch + 1,
            metrics=val_metrics,
        )

        mae_va = float(val_metrics.get("mae_va", float("inf")))
        if np.isfinite(mae_va) and mae_va < best_mae_va:
            best_mae_va = mae_va
            torch.save(model.state_dict(), best_path)
        elif not best_path.is_file():
            torch.save(model.state_dict(), best_path)
            if np.isfinite(mae_va):
                best_mae_va = mae_va

    if not best_path.is_file() and last_path.is_file():
        payload = torch.load(last_path, map_location=torch_device, weights_only=False)
        torch.save(payload["state_dict"], best_path)

    metrics_path.write_text(
        json.dumps({"history": history, "best_mae_va": best_mae_va}, indent=2),
        encoding="utf-8",
    )

    return TrainResult(
        output_dir=checkpoint_root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        metrics_path=metrics_path,
        history=history,
        best_mae_va=best_mae_va,
    )
