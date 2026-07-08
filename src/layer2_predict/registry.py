"""Predictor registry for L2 single-modality VA models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.core.types import MODALITIES
from src.utils.config_loader import load_config

from .base_predictor import BasePredictor
from .two_branch_mlp import TwoBranchMLP


class PredictorRegistry:
    """Map modality names to predictor instances."""

    def __init__(self) -> None:
        self._predictors: dict[str, BasePredictor] = {}

    def register(self, modal_name: str, predictor: BasePredictor) -> None:
        """Register a predictor instance for one modality."""
        name = _normalize_modality(modal_name)
        self._predictors[name] = predictor

    def get(self, modal_name: str) -> BasePredictor:
        """Return the predictor for a modality or raise a clear error."""
        name = _normalize_modality(modal_name)
        try:
            return self._predictors[name]
        except KeyError as exc:
            raise KeyError(f"No L2 predictor registered for modality '{name}'") from exc

    def get_predictor(self, modal_name: str) -> BasePredictor:
        """Compatibility alias for callers that prefer explicit naming."""
        return self.get(modal_name)

    def clear(self) -> None:
        self._predictors.clear()

    def names(self) -> tuple[str, ...]:
        return tuple(self._predictors.keys())

    def items(self) -> tuple[tuple[str, BasePredictor], ...]:
        return tuple(self._predictors.items())

    def __contains__(self, modal_name: object) -> bool:
        return isinstance(modal_name, str) and modal_name in self._predictors

    def __len__(self) -> int:
        return len(self._predictors)


default_registry = PredictorRegistry()


def register(modal_name: str, predictor: BasePredictor) -> None:
    default_registry.register(modal_name, predictor)


def get(modal_name: str) -> BasePredictor:
    return default_registry.get(modal_name)


def get_predictor(modal_name: str) -> BasePredictor:
    return default_registry.get_predictor(modal_name)


def initialize_registry(
    *,
    active_modalities: Iterable[str] | None = None,
    registry: PredictorRegistry | None = None,
    models_config: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
    device: str | Any = "cpu",
) -> PredictorRegistry:
    """Instantiate and register configured TwoBranchMLP predictors.

    If L2 is disabled in ``pipeline.yaml``, the target registry is cleared and
    returned empty. When ``active_modalities`` is omitted, all project
    modalities are registered.
    """
    target = registry or default_registry
    target.clear()

    pipeline = pipeline_config if pipeline_config is not None else load_config("pipeline")
    if not _l2_enabled(pipeline):
        return target

    models = models_config if models_config is not None else load_config("models")
    modalities = tuple(active_modalities) if active_modalities is not None else MODALITIES

    for modality in modalities:
        name = _normalize_modality(modality)
        predictor = TwoBranchMLP.from_config(name, models, device=device)
        target.register(name, predictor)

    return target


def _normalize_modality(modal_name: str) -> str:
    name = str(modal_name).strip().lower()
    if not name:
        raise ValueError("modality name must not be empty")
    return name


def _l2_enabled(pipeline_config: dict[str, Any]) -> bool:
    return bool(
        pipeline_config.get("pipeline", {})
        .get("stages", {})
        .get("L2", {})
        .get("enabled", True)
    )
