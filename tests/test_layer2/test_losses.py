"""Tests for L2 multitask training losses."""

from __future__ import annotations

import torch

from src.layer2_predict.losses import (
    TwoBranchMultitaskLoss,
    activate_va_logits,
    branch_va_loss,
)


def test_activate_va_logits_shapes_and_ranges():
    raw = torch.tensor([[2.0, -2.0, 0.0], [0.0, 0.0, 5.0]])
    activated = activate_va_logits(raw)

    assert activated.shape == (2, 3)
    assert torch.all(activated[:, 0] <= 1.0) and torch.all(activated[:, 0] >= -1.0)
    assert torch.all(activated[:, 1] <= 1.0) and torch.all(activated[:, 1] >= -1.0)
    assert torch.all(activated[:, 2] >= 0.0) and torch.all(activated[:, 2] <= 1.0)


def test_branch_va_loss_is_scalar():
    pred_raw = torch.randn(4, 3)
    target = torch.tensor(
        [
            [0.2, -0.3, 0.8],
            [0.1, 0.4, 0.6],
            [-0.5, 0.2, 0.9],
            [0.0, 0.0, 0.5],
        ]
    )
    loss = branch_va_loss(pred_raw, target)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_fixed_weight_multitask_loss():
    criterion = TwoBranchMultitaskLoss(learnable=False, self_weight=0.3, inter_weight=0.7)
    self_raw = torch.zeros(2, 3)
    inter_raw = torch.zeros(2, 3)
    target = torch.tensor([[0.5, -0.2, 0.7], [0.1, 0.3, 0.6]])

    total, parts = criterion(self_raw, inter_raw, target, target)

    assert total.ndim == 0
    assert "loss_self" in parts and "loss_inter" in parts
    expected = 0.3 * parts["loss_self"] + 0.7 * parts["loss_inter"]
    assert torch.allclose(total, expected)


def test_learnable_multitask_loss_supports_backward():
    criterion = TwoBranchMultitaskLoss(learnable=True)
    self_raw = torch.randn(3, 3, requires_grad=True)
    inter_raw = torch.randn(3, 3, requires_grad=True)
    target = torch.rand(3, 3)

    total, _ = criterion(self_raw, inter_raw, target, target)
    total.backward()

    assert criterion.log_var_self.grad is not None
    assert criterion.log_var_inter.grad is not None


def test_from_config_reads_models_yaml(sample_config):
    criterion = TwoBranchMultitaskLoss.from_config(sample_config["models"])
    assert isinstance(criterion, TwoBranchMultitaskLoss)
    assert criterion.learnable is True
