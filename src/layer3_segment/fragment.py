"""Fragment embedding helpers and the unified L3 entry point."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.core.context import DataContext
from src.core.types import Fragment, MemoryHit
from src.layer3_segment.baseline_calibrator import (
    BaselineConfig,
    apply_baseline,
    load_baseline,
    va_self_to_vector,
)
from src.layer3_segment.cold_start import (
    ColdStartConfig,
    delta_vector_to_dict,
    get_cold_start_delta,
)
from src.layer3_segment.memory_store import MemoryConfig, MemoryStore
from src.layer3_segment.segment_controller import (
    SegmentationConfig,
    segment_from_context,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_va_self_embedding(fragment: Fragment) -> list[float]:
    """Return a stable 8-d ``VA_self`` embedding in ``MODALITIES`` order."""
    return va_self_to_vector(fragment.va_self)


def get_va_inter_embedding(fragment: Fragment) -> list[float]:
    """Return a stable 8-d ``VA_inter`` embedding in ``MODALITIES`` order."""
    return va_self_to_vector(fragment.va_inter)


def _pipeline_config(context: DataContext) -> dict[str, Any] | None:
    config = context.metadata.get("config", {})
    if not isinstance(config, dict):
        return None
    pipeline = config.get("pipeline")
    return pipeline if isinstance(pipeline, dict) else None


def _apply_personalization(
    fragments: list[Fragment],
    user_id: str,
    *,
    baseline_config: BaselineConfig,
    cold_start_config: ColdStartConfig,
) -> tuple[list[Fragment], str]:
    """Apply one consistent baseline/cold-start delta to all fragments."""
    if not fragments:
        return [], "none"

    delta = (
        load_baseline(user_id, config=baseline_config)
        if baseline_config.enabled
        else None
    )
    source = "baseline" if delta else "none"
    if delta is None and cold_start_config.enabled:
        delta_vector = get_cold_start_delta(
            user_id,
            fragments[0].va_self,
            config=cold_start_config,
        )
        if delta_vector is not None:
            delta = delta_vector_to_dict(delta_vector)
            source = "cold_start"

    if delta is None:
        return fragments, source

    return (
        [
            replace(
                fragment,
                va_self=apply_baseline(
                    fragment.va_self,
                    user_id,
                    delta=delta,
                    config=baseline_config,
                ),
            )
            for fragment in fragments
        ],
        source,
    )


def _query_memory(
    fragments: list[Fragment],
    user_id: str,
    store: MemoryStore,
    config: MemoryConfig,
) -> list[MemoryHit]:
    hits: list[MemoryHit] = []
    for fragment in fragments:
        embedding = (
            get_va_self_embedding(fragment)
            if config.embedding_type == "self"
            else get_va_inter_embedding(fragment)
        )
        for hit in store.query_similar(
            user_id,
            embedding,
            top_k=config.top_k,
            embedding_type=config.embedding_type,
        ):
            metadata = dict(hit.metadata)
            metadata["query_fragment_id"] = fragment.id
            hits.append(replace(hit, metadata=metadata))
    return hits


def _persist_fragments(
    fragments: list[Fragment],
    context: DataContext,
    user_id: str,
    store: MemoryStore,
) -> None:
    timestamp = context.metadata.get("timestamp")
    session_id = str(context.metadata.get("session_id", "unknown-session"))
    for fragment in fragments:
        memory_fragment_id = f"{session_id}:{fragment.id}"
        store.add_fragment(
            user_id,
            get_va_self_embedding(fragment),
            get_va_inter_embedding(fragment),
            {
                "fragment_id": memory_fragment_id,
                "source_fragment_id": fragment.id,
                "session_id": session_id,
                "timestamp": timestamp,
                "start_time": fragment.start_time,
                "end_time": fragment.end_time,
                "input_type": context.input_type,
            },
            fragment_id=memory_fragment_id,
        )


def run_l3(
    context: DataContext,
    *,
    memory_store: MemoryStore | None = None,
    persist_memory: bool = True,
) -> DataContext:
    """Segment L2 output, personalize VA_self, and retrieve/persist memory."""
    if not context.va_inter_predictions:
        context.mark_stage_failed("L3", "no L2 VA_inter predictions available")
        return context

    pipeline_config = _pipeline_config(context)
    segmentation_config = SegmentationConfig.from_pipeline(pipeline_config)
    baseline_config = BaselineConfig.from_pipeline(pipeline_config)
    cold_start_config = ColdStartConfig.from_pipeline(pipeline_config)
    memory_config = MemoryConfig.from_pipeline(pipeline_config)

    try:
        fragments = segment_from_context(context, config=segmentation_config)
    except Exception as exc:  # noqa: BLE001 - stage boundary records failures
        logger.exception("L3 segmentation failed")
        context.mark_stage_failed("L3", f"segmentation failed: {exc}")
        return context

    if not fragments:
        context.mark_stage_failed("L3", "segmentation produced no fragments")
        return context

    user_id = str(context.metadata.get("user_id", "anonymous")).strip() or "anonymous"
    fragments, personalization_source = _apply_personalization(
        fragments,
        user_id,
        baseline_config=baseline_config,
        cold_start_config=cold_start_config,
    )
    context.metadata["l3_personalization"] = personalization_source

    hits: list[MemoryHit] = []
    store = memory_store
    if memory_config.enabled:
        try:
            store = store or MemoryStore(memory_config)
            hits = _query_memory(fragments, user_id, store, memory_config)
            # Query before writing so a fragment never retrieves itself.
            if persist_memory:
                _persist_fragments(fragments, context, user_id, store)
        except Exception as exc:  # noqa: BLE001 - memory is an optional L3 enhancement
            logger.exception("L3 memory unavailable; continuing without memory")
            context.metadata["l3_memory_error"] = str(exc)
            hits = []

    context.set_stage("L3", {"segments": fragments, "memory_retrieved": hits})
    return context


__all__ = ["get_va_inter_embedding", "get_va_self_embedding", "run_l3"]
