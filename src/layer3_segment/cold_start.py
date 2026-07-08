"""Collaborative cold-start (CRM) for new users without static baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.core.types import MODALITIES, ModalityVADict
from src.layer3_segment.baseline_calibrator import (
    DeltaVA,
    apply_baseline,
    load_baseline,
    va_self_to_vector,
)
from src.utils.config_loader import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_USER_LIBRARY_PATH = Path("data/user_db/cold_start_users.json")
_VECTOR_DIM = len(MODALITIES) * 2
_EPS = 1e-8


@dataclass(frozen=True)
class ColdStartConfig:
    enabled: bool = True
    embedding_source: str = "self"
    top_k_users: int = 5
    user_library_path: Path = DEFAULT_USER_LIBRARY_PATH

    @classmethod
    def from_pipeline(cls, pipeline_config: dict[str, Any] | None = None) -> ColdStartConfig:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        cold_start = pipeline_config.get("pipeline", {}).get("stages", {}).get("L3", {}).get(
            "cold_start", {}
        )
        return cls(
            enabled=bool(cold_start.get("enabled", True)),
            embedding_source=str(cold_start.get("embedding_source", "self")),
            top_k_users=int(cold_start.get("top_k_users", 5)),
        )


@dataclass(frozen=True)
class ColdStartUserRecord:
    user_id: str
    first_session_avg_va_self: list[float]
    delta_va: list[float]


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return get_project_root() / path


def _as_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (_VECTOR_DIM,):
        raise ValueError(f"Expected {_VECTOR_DIM}-d vector, got shape {vector.shape}")
    return vector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two 8-d VA_self vectors."""
    va = _as_vector(a)
    vb = _as_vector(b)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a <= _EPS or norm_b <= _EPS:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def load_user_library(path: Path | None = None) -> list[ColdStartUserRecord]:
    """Load cold-start user library from JSON."""
    target = _resolve_project_path(path or DEFAULT_USER_LIBRARY_PATH)
    if not target.is_file():
        return []
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records: list[ColdStartUserRecord] = []
    for item in payload.get("users", []):
        records.append(
            ColdStartUserRecord(
                user_id=str(item["user_id"]),
                first_session_avg_va_self=[float(v) for v in item["first_session_avg_va_self"]],
                delta_va=[float(v) for v in item["delta_va"]],
            )
        )
    return records


def find_similar_users(
    query_vector: Sequence[float],
    library: list[ColdStartUserRecord],
    *,
    top_k: int,
    exclude_user_id: str | None = None,
) -> list[tuple[ColdStartUserRecord, float]]:
    """Return Top-K users ranked by cosine similarity to the query vector."""
    scored: list[tuple[ColdStartUserRecord, float]] = []
    for record in library:
        if exclude_user_id is not None and record.user_id == exclude_user_id:
            continue
        similarity = cosine_similarity(query_vector, record.first_session_avg_va_self)
        if similarity > 0.0:
            scored.append((record, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(0, top_k)]


def blend_delta_va(
    neighbors: list[tuple[ColdStartUserRecord, float]],
) -> list[float] | None:
    """Weighted average of neighbor delta_va vectors: sum(sim_i * delta_i) / sum(sim_i)."""
    if not neighbors:
        return None
    weight_sum = sum(sim for _, sim in neighbors)
    if weight_sum <= _EPS:
        return None
    blended = np.zeros(_VECTOR_DIM, dtype=float)
    for record, similarity in neighbors:
        blended += similarity * _as_vector(record.delta_va)
    return (blended / weight_sum).tolist()


def delta_vector_to_dict(vector: Sequence[float]) -> DeltaVA:
    """Convert 8-d delta vector to per-modality DeltaVA dict."""
    values = _as_vector(vector)
    return {
        modality: (float(values[index * 2]), float(values[index * 2 + 1]))
        for index, modality in enumerate(MODALITIES)
    }


def get_cold_start_delta(
    user_id: str,
    first_fragment_va_self: ModalityVADict,
    *,
    config: ColdStartConfig | None = None,
    library: list[ColdStartUserRecord] | None = None,
) -> list[float] | None:
    """Compute temporary delta VA for a new user via Top-K collaborative fusion."""
    cfg = config or ColdStartConfig.from_pipeline()
    if not cfg.enabled:
        return None
    if cfg.embedding_source != "self":
        logger.warning("Cold start v1 only supports embedding_source='self'; skipping")
        return None
    if load_baseline(user_id) is not None:
        return None

    query_vector = va_self_to_vector(first_fragment_va_self)
    records = library if library is not None else load_user_library(cfg.user_library_path)
    neighbors = find_similar_users(
        query_vector,
        records,
        top_k=cfg.top_k_users,
        exclude_user_id=user_id,
    )
    delta = blend_delta_va(neighbors)
    if delta is None:
        logger.info("No cold-start neighbors for user '%s'", user_id)
        return None
    logger.info(
        "Cold-start delta for user '%s' from %d neighbor(s)",
        user_id,
        len(neighbors),
    )
    return delta


def apply_cold_start(
    va_self: ModalityVADict,
    user_id: str,
    first_fragment_va_self: ModalityVADict,
    *,
    config: ColdStartConfig | None = None,
    library: list[ColdStartUserRecord] | None = None,
) -> ModalityVADict:
    """Apply collaborative cold-start delta to VA_self predictions."""
    delta_vec = get_cold_start_delta(
        user_id,
        first_fragment_va_self,
        config=config,
        library=library,
    )
    if delta_vec is None:
        return apply_baseline(va_self, user_id, delta=None)
    return apply_baseline(va_self, user_id, delta=delta_vector_to_dict(delta_vec))


__all__ = [
    "ColdStartConfig",
    "ColdStartUserRecord",
    "apply_cold_start",
    "blend_delta_va",
    "cosine_similarity",
    "delta_vector_to_dict",
    "find_similar_users",
    "get_cold_start_delta",
    "load_user_library",
]
