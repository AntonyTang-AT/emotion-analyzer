"""L1 feature extractor registry and runner."""

from __future__ import annotations

from typing import Any, Callable

from src.core.context import DataContext
from src.core.interfaces import FeatureExtractor
from src.core.types import FeatureDict
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

from .macro_extractor import MacroExtractor
from .micro_extractor import MicroExpressionExtractor
from .speech_extractor import SpeechExtractor
from .text_extractor import TextExtractor

logger = get_logger(__name__)


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

_EXTRACTOR_REGISTRY: dict[str, ExtractorFactory] = {}


class FeatureExtractorFactory:
    """Registry and factory for L1 modality extractors."""

    @staticmethod
    def register(name: str, extractor_cls: type[FeatureExtractor]) -> None:
        register_extractor(name, extractor_cls)

    @staticmethod
    def create(name: str) -> FeatureExtractor:
        if name not in _EXTRACTOR_REGISTRY:
            raise ValueError(f"No extractor registered for modality '{name}'")
        return _EXTRACTOR_REGISTRY[name]()

    @staticmethod
    def registered_names() -> tuple[str, ...]:
        return tuple(_EXTRACTOR_REGISTRY)

    @classmethod
    def register_defaults(cls) -> None:
        cls.register("text", TextExtractor)
        cls.register("speech", SpeechExtractor)
        cls.register("macro", MacroExtractor)
        cls.register("micro", MicroExpressionExtractor)


def register_extractor(name: str, extractor_cls: type[FeatureExtractor]) -> None:
    """Register an extractor class that supports no-argument construction."""
    _EXTRACTOR_REGISTRY[name] = extractor_cls


def _resolve_modalities(
    config: dict[str, Any] | None,
    context: DataContext | None,
) -> list[str]:
    if context is not None and context.active_modalities:
        return list(context.active_modalities)

    if context is not None:
        metadata = context.metadata
        profile = metadata.get("input_profile_data")
        if isinstance(profile, dict) and profile.get("l1_extractors"):
            return list(profile["l1_extractors"])

        profile_name = metadata.get("input_profile")
        if profile_name:
            from src.pipeline.input_profile import load_input_profiles

            profiles = load_input_profiles()
            if profile_name in profiles:
                extractors = profiles[profile_name].get("l1_extractors")
                if isinstance(extractors, list) and extractors:
                    return list(extractors)

        if metadata.get("active_modalities"):
            return list(metadata["active_modalities"])

    pipeline_cfg = config or load_config("pipeline")
    stages = pipeline_cfg.get("pipeline", {}).get("stages", {})
    l1_extractors = stages.get("L1", {}).get("feature_extractors", [])
    if not isinstance(l1_extractors, list) or not l1_extractors:
        raise ValueError("No L1 extractors found in context or pipeline config")
    return list(l1_extractors)


def get_extractors_from_config(
    config: dict[str, Any] | None = None,
    context: DataContext | None = None,
) -> list[FeatureExtractor]:
    """Instantiate extractors from profile context or pipeline config."""
    modalities = _resolve_modalities(config, context)
    return [FeatureExtractorFactory.create(name) for name in modalities]


def get_extractors_for_context(context: DataContext) -> list[FeatureExtractor]:
    """Instantiate extractors listed in context.metadata.active_modalities."""
    return get_extractors_from_config(context=context)


def _extractor_modality_name(extractor: FeatureExtractor) -> str:
    modality = getattr(extractor, "modality", None)
    if isinstance(modality, str) and modality:
        return modality
    return type(extractor).__name__


def run_l1(context: DataContext) -> DataContext:
    """Run all active L1 extractors and merge features into context."""
    features: FeatureDict = dict(context.features)
    raw_visual = dict(context.raw_visual_features)

    for extractor in get_extractors_for_context(context):
        modality = _extractor_modality_name(extractor)
        try:
            extracted = extractor.extract(context)
            features.update(extracted)
            raw_visual.update(extractor.extract_raw_visual(context))
        except Exception:
            logger.exception(
                "L1 extractor failed for modality '%s', skipping",
                modality,
            )

    context.set_stage(
        "L1",
        {
            "features": features,
            "raw_visual_features": raw_visual,
        },
    )
    return context


FeatureExtractorFactory.register_defaults()
