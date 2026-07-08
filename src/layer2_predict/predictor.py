"""L2 unified prediction entry point."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.context import DataContext
from src.core.types import VAConfidence
from src.utils.logger import get_logger

from .registry import PredictorRegistry, initialize_registry

logger = get_logger(__name__)

FEATURE_KEYS: dict[str, str] = {
    "text": "text_embedding",
    "speech": "speech_feature",
    "micro": "micro_feature",
}

TIME_KEYS: dict[str, str] = {
    "text": "start_time",
    "speech": "timestamp",
    "micro": "start_time",
}


def _extract_feature_vectors(modality: str, items: list[Any]) -> list[np.ndarray]:
    """Convert L1 modality items into ordered feature vectors."""
    name = modality.strip().lower()
    if name == "macro":
        sorted_items = sorted(items, key=lambda item: float(item[1]))
        return [np.asarray(item[0], dtype=np.float32) for item in sorted_items]

    feature_key = FEATURE_KEYS.get(name)
    if feature_key is None:
        raise ValueError(f"Unsupported L2 modality '{modality}'")

    sort_key = TIME_KEYS[name]
    sorted_items = sorted(items, key=lambda item: float(item.get(sort_key, 0.0)))
    return [np.asarray(item[feature_key], dtype=np.float32) for item in sorted_items]


def run_l2(context: DataContext) -> DataContext:
    """Run L2 predictors for active modalities and store VA time series."""
    registry = initialize_registry(
        active_modalities=context.active_modalities,
        registry=PredictorRegistry(),
    )
    if len(registry) == 0:
        return context

    va_self: dict[str, list[VAConfidence]] = {}
    va_inter: dict[str, list[VAConfidence]] = {}
    failed: list[str] = []

    for modality in context.active_modalities:
        items = context.features.get(modality)
        if not items:
            logger.warning("L2 skip '%s': no L1 features", modality)
            failed.append(modality)
            continue
        try:
            predictor = registry.get_predictor(modality)
            vectors = _extract_feature_vectors(modality, items)
            va_self[modality] = [predictor.predict_self(vector) for vector in vectors]
            va_inter[modality] = [predictor.predict_inter(vector) for vector in vectors]
        except KeyError:
            logger.warning("L2 skip '%s': predictor not registered", modality)
            failed.append(modality)
        except Exception:
            logger.exception("L2 prediction failed for modality '%s', skipping", modality)
            failed.append(modality)

    if failed:
        context.metadata["l2_failures"] = failed

    if not va_self:
        context.mark_stage_failed(
            "L2",
            f"all modalities failed: {', '.join(failed) if failed else 'no active modalities'}",
        )
        return context

    context.set_stage(
        "L2",
        {
            "va_self_predictions": va_self,
            "va_inter_predictions": va_inter,
        },
    )
    if failed:
        context.metadata["l2_partial"] = True
    return context
