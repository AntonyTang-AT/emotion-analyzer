"""Pipeline-wide data context for inter-layer communication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .types import (
    ContradictionResult,
    FeatureDict,
    Fragment,
    MemoryHit,
    PersonalityResult,
    ReportBundle,
    VAConfidence,
)

STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "L1": ("features", "raw_visual_features"),
    "L2": ("va_self_predictions", "va_inter_predictions"),
    "L3": ("segments", "memory_retrieved"),
    "L4": ("contradiction",),
    "L5": ("reports",),
    "L6": ("personality",),
}
"""Map pipeline stage names to DataContext fields.

Layer ``run()`` implementations should pass every field listed for their stage
to ``set_stage`` (preserve existing values when a submodule only updates part
of the stage, e.g. ``memory_retrieved`` when segmenting).
"""

VALID_STAGES = frozenset(STAGE_FIELDS.keys())


def _encode_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if _is_ndarray(value):
        return {
            "__ndarray__": True,
            "data": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    if isinstance(value, VAConfidence):
        return value.to_dict()
    if isinstance(value, ContradictionResult):
        return value.to_dict()
    if isinstance(value, Fragment):
        return value.to_dict()
    if isinstance(value, MemoryHit):
        return value.to_dict()
    if isinstance(value, ReportBundle):
        return value.to_dict()
    if isinstance(value, PersonalityResult):
        return value.to_dict()

    if isinstance(value, dict):
        return {str(k): _encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]

    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            array = _import_numpy().array(value["data"], dtype=value["dtype"])
            return array.reshape(value["shape"])
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _is_ndarray(value: Any) -> bool:
    return value.__class__.__module__ == "numpy" and value.__class__.__name__ == "ndarray"


def _import_numpy() -> Any:
    import numpy as np

    return np


@dataclass
class DataContext:
    """Unified state container passed through the L1-L6 pipeline."""

    metadata: dict[str, Any] = field(default_factory=dict)
    raw_data: dict[str, str] = field(default_factory=dict)
    features: FeatureDict = field(default_factory=dict)
    raw_visual_features: dict[str, Any] = field(default_factory=dict)
    va_self_predictions: dict[str, VAConfidence] = field(default_factory=dict)
    va_inter_predictions: dict[str, VAConfidence] = field(default_factory=dict)
    segments: list[Fragment] = field(default_factory=list)
    memory_retrieved: list[MemoryHit] = field(default_factory=list)
    contradiction: ContradictionResult | None = None
    reports: ReportBundle | None = None
    personality: PersonalityResult | None = None

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        session_id: str | None = None,
        output_dir: str | Path | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> DataContext:
        session = session_id or str(uuid4())
        metadata: dict[str, Any] = {
            "session_id": session,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage_status": {stage: "pending" for stage in VALID_STAGES},
        }
        if output_dir is not None:
            metadata["output_dir"] = str(output_dir)
        if config_snapshot is not None:
            metadata["config"] = config_snapshot

        raw_data = {"video_path": str(video_path)}
        if audio_path is not None:
            raw_data["audio_path"] = str(audio_path)

        return cls(metadata=metadata, raw_data=raw_data)

    def set_stage(self, stage_name: str, data: dict[str, Any]) -> None:
        if stage_name not in VALID_STAGES:
            raise ValueError(
                f"Unknown stage '{stage_name}'. Valid stages: {', '.join(sorted(VALID_STAGES))}"
            )

        for field_name in STAGE_FIELDS[stage_name]:
            if field_name not in data:
                continue
            setattr(self, field_name, data[field_name])

        stage_status = self.metadata.setdefault("stage_status", {})
        stage_status[stage_name] = "completed"

    def get_stage(self, stage_name: str) -> dict[str, Any]:
        if stage_name not in VALID_STAGES:
            raise ValueError(
                f"Unknown stage '{stage_name}'. Valid stages: {', '.join(sorted(VALID_STAGES))}"
            )

        result: dict[str, Any] = {}
        for field_name in STAGE_FIELDS[stage_name]:
            value = getattr(self, field_name)
            if field_name in {"va_self_predictions", "va_inter_predictions"}:
                result[field_name] = {
                    k: v.to_dict() if isinstance(v, VAConfidence) else v
                    for k, v in value.items()
                }
            elif field_name == "segments":
                result[field_name] = [f.to_dict() for f in value]
            elif field_name == "memory_retrieved":
                result[field_name] = [m.to_dict() for m in value]
            elif field_name == "contradiction" and value is not None:
                result[field_name] = value.to_dict()
            elif field_name == "reports" and value is not None:
                result[field_name] = value.to_dict()
            elif field_name == "personality" and value is not None:
                result[field_name] = value.to_dict()
            else:
                result[field_name] = _encode_value(value)
        return result

    def mark_stage_failed(self, stage_name: str, error: str) -> None:
        stage_status = self.metadata.setdefault("stage_status", {})
        stage_status[stage_name] = "failed"
        errors = self.metadata.setdefault("errors", {})
        errors[stage_name] = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": _encode_value(self.metadata),
            "raw_data": dict(self.raw_data),
            "features": _encode_value(self.features),
            "raw_visual_features": _encode_value(self.raw_visual_features),
            "va_self_predictions": {
                k: v.to_dict() for k, v in self.va_self_predictions.items()
            },
            "va_inter_predictions": {
                k: v.to_dict() for k, v in self.va_inter_predictions.items()
            },
            "segments": [segment.to_dict() for segment in self.segments],
            "memory_retrieved": [hit.to_dict() for hit in self.memory_retrieved],
            "contradiction": (
                self.contradiction.to_dict() if self.contradiction else None
            ),
            "reports": self.reports.to_dict() if self.reports else None,
            "personality": self.personality.to_dict() if self.personality else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataContext:
        return cls(
            metadata=_decode_value(data.get("metadata", {})),
            raw_data=dict(data.get("raw_data", {})),
            features=_decode_value(data.get("features", {})),
            raw_visual_features=_decode_value(data.get("raw_visual_features", {})),
            va_self_predictions={
                k: VAConfidence.from_dict(v)
                for k, v in data.get("va_self_predictions", {}).items()
            },
            va_inter_predictions={
                k: VAConfidence.from_dict(v)
                for k, v in data.get("va_inter_predictions", {}).items()
            },
            segments=[Fragment.from_dict(item) for item in data.get("segments", [])],
            memory_retrieved=[
                MemoryHit.from_dict(item) for item in data.get("memory_retrieved", [])
            ],
            contradiction=(
                ContradictionResult.from_dict(data["contradiction"])
                if data.get("contradiction")
                else None
            ),
            reports=(
                ReportBundle.from_dict(data["reports"])
                if data.get("reports")
                else None
            ),
            personality=(
                PersonalityResult.from_dict(data["personality"])
                if data.get("personality")
                else None
            ),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> DataContext:
        return cls.from_dict(json.loads(payload))

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> DataContext:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
