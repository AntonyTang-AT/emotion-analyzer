"""Two-branch MLP for per-modality VA prediction (self + inter)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn

from src.core.types import VAConfidence
from src.utils.config_loader import get_config, load_config
from src.utils.logger import get_logger

from .base_predictor import BasePredictor

logger = get_logger(__name__)


def _decode_output(raw: Any) -> VAConfidence:
    import torch

    if raw.ndim == 2:
        if raw.shape[0] != 1:
            raise ValueError("predict_* expects a single feature vector, not a batch")
        raw = raw.squeeze(0)

    activated = torch.stack(
        [
            torch.tanh(raw[0]),
            torch.tanh(raw[1]),
            torch.sigmoid(raw[2]),
        ]
    )
    values = activated.detach().cpu().tolist()
    return VAConfidence(
        valence=float(values[0]),
        arousal=float(values[1]),
        confidence=float(values[2]),
    )


class TwoBranchMLP(nn.Module, BasePredictor):
    """Shared-bottom MLP with independent self and inter prediction heads."""

    def __init__(
        self,
        modality: str,
        input_dim: int,
        *,
        shared_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        device: str | Any = "cpu",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        import torch
        import torch.nn as nn

        nn.Module.__init__(self)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        self.modality = modality
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = torch.device(device)

        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
        )
        self.self_branch = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.inter_branch = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 3),
        )

        self._init_weights()
        self.to(self.device)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _init_weights(self) -> None:
        import torch.nn as nn

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _to_tensor(self, features: Any, *, allow_batch: bool = False) -> Any:
        import torch

        if isinstance(features, torch.Tensor):
            tensor = features.detach().float()
        elif isinstance(features, np.ndarray):
            tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        else:
            tensor = torch.tensor(features, dtype=torch.float32)

        if allow_batch and tensor.ndim == 2:
            if tensor.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected feature width {self.input_dim}, got {tensor.shape[1]}"
                )
            return tensor.to(self.device)

        if tensor.ndim != 1:
            raise ValueError(
                f"Expected 1-D features of length {self.input_dim}, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected feature length {self.input_dim}, got {tensor.shape[0]}"
            )
        return tensor.to(self.device)

    def _forward_branch(self, features: Any, branch: Any) -> Any:
        import torch

        tensor = self._to_tensor(features)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        shared = self.shared_fc(tensor)
        return branch(shared).squeeze(0)

    def forward(self, features: Any) -> tuple[Any, Any]:
        """Return raw (self, inter) branch logits before activation."""
        tensor = self._to_tensor(features, allow_batch=True)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        shared = self.shared_fc(tensor)
        self_out = self.self_branch(shared)
        inter_out = self.inter_branch(shared)
        if single:
            return self_out.squeeze(0), inter_out.squeeze(0)
        return self_out, inter_out

    def predict_self(self, features: Any) -> VAConfidence:
        import torch

        self.eval()
        with torch.inference_mode():
            raw = self._forward_branch(features, self.self_branch)
        return _decode_output(raw)

    def predict_inter(self, features: Any) -> VAConfidence:
        import torch

        self.eval()
        with torch.inference_mode():
            raw = self._forward_branch(features, self.inter_branch)
        return _decode_output(raw)

    def load_checkpoint(self, path: str | Path) -> None:
        import torch

        checkpoint_file = Path(path)
        if not checkpoint_file.is_file():
            resolved = get_config().resolve_path(str(path))
            if resolved.is_file():
                checkpoint_file = resolved
            else:
                logger.debug(
                    "Checkpoint not found for modality '%s': %s",
                    self.modality,
                    path,
                )
                return

        state = torch.load(checkpoint_file, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state)
        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        modality: str,
        config: dict[str, Any] | None = None,
        *,
        device: str | Any = "cpu",
    ) -> TwoBranchMLP:
        models_cfg = config or load_config("models")
        l2_cfg = models_cfg["layer2"]["two_branch_mlp"]
        modalities = l2_cfg.get("modalities", {})
        if modality not in modalities:
            raise ValueError(f"No layer2 config for modality '{modality}'")

        modal_cfg = modalities[modality]
        checkpoint = modal_cfg.get("checkpoint")
        checkpoint_path = None
        if checkpoint:
            checkpoint_path = get_config().resolve_path(str(checkpoint))

        return cls(
            modality=modality,
            input_dim=int(modal_cfg["input_dim"]),
            shared_dim=int(l2_cfg.get("shared_dim", 256)),
            hidden_dim=int(l2_cfg.get("branch_hidden", 128)),
            dropout=float(l2_cfg.get("dropout", 0.2)),
            device=device,
            checkpoint_path=checkpoint_path,
        )
