"""Layer 2 single-modality VA prediction (two-branch architecture)."""

from .base_predictor import BasePredictor
from .two_branch_mlp import TwoBranchMLP

__all__ = [
    "BasePredictor",
    "TwoBranchMLP",
]
