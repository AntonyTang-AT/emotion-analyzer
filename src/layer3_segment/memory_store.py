"""Persistent L3 fragment memory backed by two Chroma collections.

Each fragment is written under the same ID to a ``self`` and an ``inter``
collection.  Callers can therefore choose the VA branch at query time without
duplicating or reshaping fragment metadata.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.types import MODALITIES, MemoryHit
from src.utils.config_loader import get_project_root, load_config

_EMBEDDING_DIM = len(MODALITIES) * 2
_EMBEDDING_TYPES = frozenset({"self", "inter"})
_PAYLOAD_KEY = "memory_payload_json"


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for the persistent fragment memory."""

    enabled: bool = True
    db_path: Path = Path("data/chroma")
    decay_alpha: float = 0.1
    top_k: int = 3
    embedding_type: str = "inter"

    def __post_init__(self) -> None:
        if self.decay_alpha < 0:
            raise ValueError("decay_alpha must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        _validate_embedding_type(self.embedding_type)

    @classmethod
    def from_pipeline(cls, pipeline_config: dict[str, Any] | None = None) -> MemoryConfig:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        memory = (
            pipeline_config.get("pipeline", {})
            .get("stages", {})
            .get("L3", {})
            .get("memory", {})
        )
        return cls(
            enabled=bool(memory.get("enabled", True)),
            db_path=Path(str(memory.get("db_path", "data/chroma"))),
            decay_alpha=float(memory.get("decay_alpha", 0.1)),
            top_k=int(memory.get("top_k", 3)),
            embedding_type=str(memory.get("embedding_type", "inter")),
        )


class MemoryStore:
    """Store and retrieve 8-dimensional VA fragment embeddings with Chroma."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or MemoryConfig.from_pipeline()
        self.db_path = _resolve_project_path(self.config.db_path)

        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - exercised without extras
                raise RuntimeError(
                    "chromadb is required for MemoryStore; install project requirements"
                ) from exc
            self.db_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.db_path))

        self._client = client
        self.collection_self = self._get_or_create_collection("l3_memory_self")
        self.collection_inter = self._get_or_create_collection("l3_memory_inter")

    def _get_or_create_collection(self, name: str) -> Any:
        """Create a cosine collection across both legacy and current Chroma APIs."""
        common = {"name": name, "embedding_function": None}
        try:
            return self._client.get_or_create_collection(
                **common,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except TypeError:
            # Chroma < 1.x configured HNSW through collection metadata.
            return self._client.get_or_create_collection(
                **common,
                metadata={"hnsw:space": "cosine"},
            )

    def add_fragment(
        self,
        user_id: str,
        embedding_self: Sequence[float],
        embedding_inter: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        *,
        fragment_id: str | None = None,
    ) -> str:
        """Upsert both VA embeddings and return their shared fragment ID."""
        user_id = _validate_user_id(user_id)
        vector_self = _validate_embedding(embedding_self)
        vector_inter = _validate_embedding(embedding_inter)
        payload = dict(metadata or {})

        candidate_id = fragment_id or payload.get("fragment_id") or payload.get("id")
        resolved_id = str(candidate_id or uuid.uuid4())
        if not resolved_id.strip():
            raise ValueError("fragment_id must be non-empty")

        timestamp = _normalise_timestamp(payload.get("timestamp"))
        payload["timestamp"] = timestamp
        payload.setdefault("fragment_id", resolved_id)
        record_metadata = _record_metadata(
            user_id=user_id,
            fragment_id=resolved_id,
            timestamp=timestamp,
            payload=payload,
        )
        # Namespace Chroma IDs by user so shared business IDs (e.g. seg-0000)
        # from different users do not overwrite each other.
        chroma_id = _storage_id(user_id, resolved_id)

        upsert_args = {
            "ids": [chroma_id],
            "metadatas": [record_metadata],
        }
        self.collection_self.upsert(embeddings=[vector_self], **upsert_args)
        self.collection_inter.upsert(embeddings=[vector_inter], **upsert_args)
        return resolved_id

    def query_similar(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int | None = None,
        time_decay: bool = True,
        embedding_type: str | None = None,
        *,
        now: datetime | None = None,
    ) -> list[MemoryHit]:
        """Return a user's most similar historical fragments.

        Chroma returns cosine distance, which is converted to cosine similarity.
        When enabled, similarity is multiplied by ``exp(-alpha * age_days)`` and
        candidates are re-ranked by that adjusted score.
        """
        user_id = _validate_user_id(user_id)
        vector = _validate_embedding(query_embedding)
        resolved_type = embedding_type or self.config.embedding_type
        _validate_embedding_type(resolved_type)
        resolved_top_k = self.config.top_k if top_k is None else int(top_k)
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")

        collection = self._collection_for(resolved_type)
        total = int(collection.count())
        if total == 0:
            return []

        # Oversample before applying recency so decay can materially change rank.
        candidate_count = min(total, max(resolved_top_k, resolved_top_k * 10))
        result = collection.query(
            query_embeddings=[vector],
            n_results=candidate_count,
            where={"user_id": user_id},
            include=["metadatas", "distances"],
        )

        ids = _first_result_list(result.get("ids"))
        metadatas = _first_result_list(result.get("metadatas"))
        distances = _first_result_list(result.get("distances"))
        current_time = _as_utc(now or datetime.now(timezone.utc))

        hits: list[MemoryHit] = []
        for index, storage_id in enumerate(ids):
            raw_metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            score = similarity
            metadata = _decode_payload(raw_metadata or {})
            fragment_id = str(
                metadata.get("fragment_id")
                or raw_metadata.get("fragment_id")
                or _business_fragment_id(str(storage_id), user_id)
            )
            if time_decay:
                age_days = _age_in_days(metadata.get("timestamp"), current_time)
                score *= math.exp(-self.config.decay_alpha * age_days)
            hits.append(
                MemoryHit(
                    fragment_id=fragment_id,
                    score=float(score),
                    metadata=metadata,
                    embedding_type=resolved_type,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.fragment_id))
        return hits[:resolved_top_k]

    def _collection_for(self, embedding_type: str) -> Any:
        if embedding_type == "self":
            return self.collection_self
        return self.collection_inter


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return get_project_root() / path


def _validate_user_id(user_id: str) -> str:
    value = str(user_id).strip()
    if not value:
        raise ValueError("user_id must be non-empty")
    return value


def _storage_id(user_id: str, fragment_id: str) -> str:
    """Build a Chroma document ID namespaced by user."""
    return f"{user_id}::{fragment_id}"


def _business_fragment_id(storage_id: str, user_id: str) -> str:
    """Recover the caller-facing fragment ID from a namespaced storage ID."""
    prefix = f"{user_id}::"
    if storage_id.startswith(prefix):
        return storage_id[len(prefix) :]
    if "::" in storage_id:
        return storage_id.split("::", 1)[1]
    return storage_id


def _validate_embedding_type(embedding_type: str) -> None:
    if embedding_type not in _EMBEDDING_TYPES:
        raise ValueError("embedding_type must be 'self' or 'inter'")


def _validate_embedding(values: Sequence[float]) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != _EMBEDDING_DIM:
        raise ValueError(f"Expected {_EMBEDDING_DIM}-d embedding, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    return vector


def _normalise_timestamp(value: Any) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        timestamp = _as_utc(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        timestamp = _parse_timestamp(value)
    else:
        raise ValueError("timestamp must be an ISO-8601 string, Unix time, or datetime")
    return timestamp.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp must be non-empty")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _as_utc(datetime.fromisoformat(text))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _record_metadata(
    *,
    user_id: str,
    fragment_id: str,
    timestamp: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "fragment_id": fragment_id,
        "timestamp": timestamp,
        _PAYLOAD_KEY: json.dumps(payload, ensure_ascii=False, default=_json_default),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _normalise_timestamp(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Metadata value of type {type(value).__name__} is not JSON serializable")


def _decode_payload(raw_metadata: Mapping[str, Any]) -> dict[str, Any]:
    encoded = raw_metadata.get(_PAYLOAD_KEY)
    if isinstance(encoded, str):
        try:
            payload = json.loads(encoded)
            if isinstance(payload, dict):
                result = payload
            else:
                result = {}
        except json.JSONDecodeError:
            result = {}
    else:
        result = {}
    result.setdefault("timestamp", raw_metadata.get("timestamp"))
    result.setdefault("user_id", raw_metadata.get("user_id"))
    result.setdefault("fragment_id", raw_metadata.get("fragment_id"))
    return result


def _age_in_days(timestamp: Any, now: datetime) -> float:
    if not isinstance(timestamp, str):
        return 0.0
    try:
        stored_at = _parse_timestamp(timestamp)
    except ValueError:
        return 0.0
    return max(0.0, (now - stored_at).total_seconds() / 86_400.0)


def _first_result_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return []


__all__ = ["MemoryConfig", "MemoryStore"]
