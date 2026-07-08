"""Multitask losses for L2 two-branch VA training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config_loader import load_config


def activate_va_logits(raw: torch.Tensor) -> torch.Tensor:
    """Apply the same activations used at inference time."""
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"Expected logits shape [B, 3], got {tuple(raw.shape)}")
    return torch.stack(
        [
            torch.tanh(raw[:, 0]),
            torch.tanh(raw[:, 1]),
            torch.sigmoid(raw[:, 2]),
        ],
        dim=1,
    )


def branch_va_loss(pred_raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on valence/arousal and BCE on confidence for one branch."""
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"Expected target shape [B, 3], got {tuple(target.shape)}")
    pred = activate_va_logits(pred_raw)
    mse_v = F.mse_loss(pred[:, 0], target[:, 0])
    mse_a = F.mse_loss(pred[:, 1], target[:, 1])
    bce_c = F.binary_cross_entropy(pred[:, 2], target[:, 2])
    return mse_v + mse_a + bce_c


class TwoBranchMultitaskLoss(nn.Module):
    """Combine self and inter branch losses with fixed or learnable weights."""

    def __init__(
        self,
        *,
        learnable: bool = False,
        self_weight: float = 0.3,
        inter_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.log_var_self = nn.Parameter(torch.zeros(1))
            self.log_var_inter = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("self_weight", torch.tensor(float(self_weight)))
            self.register_buffer("inter_weight", torch.tensor(float(inter_weight)))

    def forward(
        self,
        self_raw: torch.Tensor,
        inter_raw: torch.Tensor,
        target_self: torch.Tensor,
        target_inter: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_self = branch_va_loss(self_raw, target_self)
        loss_inter = branch_va_loss(inter_raw, target_inter)

        if self.learnable:
            total = (
                0.5 * torch.exp(-self.log_var_self) * loss_self
                + 0.5 * self.log_var_self
                + 0.5 * torch.exp(-self.log_var_inter) * loss_inter
                + 0.5 * self.log_var_inter
            )
        else:
            total = self.self_weight * loss_self + self.inter_weight * loss_inter

        return total, {"loss_self": loss_self.detach(), "loss_inter": loss_inter.detach()}

    @classmethod
    def from_config(
        cls,
        models_config: dict[str, Any] | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> TwoBranchMultitaskLoss:
        models_cfg = models_config or load_config("models")
        multitask = models_cfg.get("layer2", {}).get("multitask", {})
        learnable = bool(multitask.get("learnable_loss_weights", False))

        self_weight = float(multitask.get("self_loss_weight", 0.3))
        inter_weight = float(multitask.get("inter_loss_weight", 0.7))

        if pipeline_config is not None:
            l2 = pipeline_config.get("pipeline", {}).get("stages", {}).get("L2", {})
            if "self_loss_weight" in l2:
                self_weight = float(l2["self_loss_weight"])
            if "inter_loss_weight" in l2:
                inter_weight = float(l2["inter_loss_weight"])

        if not learnable:
            total = self_weight + inter_weight
            if total > 0:
                self_weight /= total
                inter_weight /= total

        return cls(
            learnable=learnable,
            self_weight=self_weight,
            inter_weight=inter_weight,
        )
