"""L1 feature extractor registry and runner."""

from __future__ import annotations

from typing import Any, Callable

from src.core.context import DataContext
from src.core.interfaces import FeatureExtractor
from src.core.types import FeatureDict

from .macro_extractor import MacroExtractor
from .micro_extractor import MicroExpressionExtractor
from .speech_extractor import SpeechExtractor
from .text_extractor import TextExtractor


class StubModalityExtractor(FeatureExtractor):
    """Return placeholder features for a single modality."""

    def __init__(self, modality: str, *, feature_dim: int = 8) -> None:
        self.modality = modality
        self.feature_dim = feature_dim

    def extract(self, context: DataContext) -> FeatureDict:
        return {
            self.modality: [
                {
                    "stub": True,
                    "modality": self.modality,
                    "feature_dim": self.feature_dim,
                    "input_type": context.input_type,
                    "timestamp": 0.0,
                }
            ]
        }

    def extract_raw_visual(self, context: DataContext) -> dict[str, Any]:
        if self.modality in {"macro", "micro"}:
            return {self.modality: [0.0] * self.feature_dim}
        return {}


ExtractorFactory = Callable[[], FeatureExtractor]

_EXTRACTOR_REGISTRY: dict[str, ExtractorFactory] = {
    "text": TextExtractor,
    "speech": SpeechExtractor,
    "macro": MacroExtractor,
    "micro": MicroExpressionExtractor,
}


def register_extractor(name: str, extractor_cls: type[FeatureExtractor]) -> None:
    """Register an extractor class that supports no-argument construction."""
    _EXTRACTOR_REGISTRY[name] = extractor_cls


def get_extractors_for_context(context: DataContext) -> list[FeatureExtractor]:
    """Instantiate extractors listed in context.metadata.active_modalities."""
    modalities = context.active_modalities
    if not modalities:
        raise ValueError("context.metadata.active_modalities is empty")

    extractors: list[FeatureExtractor] = []
    for name in modalities:
        if name not in _EXTRACTOR_REGISTRY:
            raise ValueError(f"No extractor registered for modality '{name}'")
        extractors.append(_EXTRACTOR_REGISTRY[name]())
    return extractors


def run_l1(context: DataContext) -> DataContext:
    """Run all active L1 extractors and merge features into context."""
    features: FeatureDict = dict(context.features)
    raw_visual = dict(context.raw_visual_features)

    for extractor in get_extractors_for_context(context):
        extracted = extractor.extract(context)
        features.update(extracted)
        raw_visual.update(extractor.extract_raw_visual(context))

    context.set_stage(
        "L1",
        {
            "features": features,
            "raw_visual_features": raw_visual,
        },
    )
    return context