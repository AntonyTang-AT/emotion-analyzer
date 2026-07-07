"""Abstract base classes for pipeline layers L1-L6."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .context import DataContext
from .types import ContradictionResult, FeatureDict, MemoryHit, VAConfidence


class LayerModule(ABC):
    """Base contract: each layer reads and mutates a shared DataContext."""

    @abstractmethod
    def run(self, context: DataContext) -> DataContext:
        """Execute this layer and return the updated context."""


class FeatureExtractor(LayerModule):
    """L1: extract multimodal features from raw audio/video."""

    @abstractmethod
    def extract(self, context: DataContext) -> FeatureDict:
        """Return structured features for all enabled modalities."""

    def extract_raw_visual(self, context: DataContext) -> dict[str, Any]:
        """Return pure-visual bypass features (override in visual extractors)."""
        return {}

    def run(self, context: DataContext) -> DataContext:
        features = {**context.features, **self.extract(context)}
        raw_visual = {
            **context.raw_visual_features,
            **self.extract_raw_visual(context),
        }
        context.set_stage(
            "L1",
            {
                "features": features,
                "raw_visual_features": raw_visual,
            },
        )
        return context


class VAPredictor(ABC):
    """L2: two-branch VA prediction for a single modality."""

    modality: str

    @abstractmethod
    def predict_self(self, features: Any) -> VAConfidence:
        """Self-modeling branch for baseline calibration and cold start."""

    @abstractmethod
    def predict_inter(self, features: Any) -> VAConfidence:
        """Interaction branch aligned with other modalities."""


class VAPredictionLayer(LayerModule):
    """L2 orchestrator that fills both VA branches in the context."""

    @abstractmethod
    def predict_all(self, context: DataContext) -> tuple[dict[str, VAConfidence], dict[str, VAConfidence]]:
        """Return (va_self_predictions, va_inter_predictions)."""

    def run(self, context: DataContext) -> DataContext:
        va_self, va_inter = self.predict_all(context)
        context.set_stage(
            "L2",
            {
                "va_self_predictions": va_self,
                "va_inter_predictions": va_inter,
            },
        )
        return context


class SegmentController(LayerModule):
    """L3: dynamic segmentation using VA_inter."""

    @abstractmethod
    def segment(self, context: DataContext) -> list:
        """Return a list of Fragment objects."""

    def run(self, context: DataContext) -> DataContext:
        segments = self.segment(context)
        context.set_stage(
            "L3",
            {
                "segments": segments,
                "memory_retrieved": context.memory_retrieved,
            },
        )
        return context


class MemoryRetriever(LayerModule):
    """L3: retrieve similar historical fragments from vector memory."""

    @abstractmethod
    def retrieve_memory(self, context: DataContext) -> list[MemoryHit]:
        """Return memory hits for the current context."""

    def run(self, context: DataContext) -> DataContext:
        context.set_stage(
            "L3",
            {
                "segments": context.segments,
                "memory_retrieved": self.retrieve_memory(context),
            },
        )
        return context


class PersonalizationModule(ABC):
    """L3 optional helpers: baseline calibration and cold start.

    These flows may update metadata, external storage, or VA predictions rather
    than the canonical ``STAGE_FIELDS`` keys. Use ``MemoryRetriever`` for
    ``memory_retrieved`` updates.
    """

    @abstractmethod
    def process(self, context: DataContext) -> DataContext:
        """Apply personalization logic and return the updated context."""


class ContradictionDetector(LayerModule):
    """L4: detect cross-modal contradictions and produce fusion weights."""

    @abstractmethod
    def detect(self, context: DataContext) -> ContradictionResult:
        """Analyze VA_inter predictions and return routing metadata."""

    def run(self, context: DataContext) -> DataContext:
        result = self.detect(context)
        context.set_stage("L4", {"contradiction": result})
        return context


class ReportGenerator(LayerModule):
    """L5: generate segment/overall reports and visualization artifacts."""

    @abstractmethod
    def generate(self, context: DataContext):
        """Return a ReportBundle."""

    def run(self, context: DataContext) -> DataContext:
        reports = self.generate(context)
        context.set_stage("L5", {"reports": reports})
        return context


class PersonalityRegressor(LayerModule):
    """L6: infer Big-Five personality traits from aggregated pipeline outputs."""

    @abstractmethod
    def predict(self, context: DataContext):
        """Return a PersonalityResult."""

    def run(self, context: DataContext) -> DataContext:
        personality = self.predict(context)
        context.set_stage("L6", {"personality": personality})
        return context
