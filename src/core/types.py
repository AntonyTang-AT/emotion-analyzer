"""Core data types shared across pipeline layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

MODALITIES: tuple[str, ...] = ("text", "speech", "macro", "micro")
WEIGHT_SUM_TOLERANCE = 0.01


class Modality(str, Enum):
    TEXT = "text"
    SPEECH = "speech"
    MACRO = "macro"
    MICRO = "micro"


class ContradictionType(str, Enum):
    MASKING = "masking"
    SARCASM = "sarcasm"
    HIDDEN_EMOTION = "hidden_emotion"
    INTENSITY_MISMATCH = "intensity_mismatch"
    CONSISTENT = "consistent"


FeatureDict = dict[str, Any]
ModalityVADict = dict[str, "VAConfidence"]


@dataclass
class VAConfidence:
    valence: float
    arousal: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VAConfidence:
        return cls(
            valence=float(data["valence"]),
            arousal=float(data["arousal"]),
            confidence=float(data["confidence"]),
        )


@dataclass
class ContradictionResult:
    contradiction_type: ContradictionType | str
    contradiction_intensity: float
    involved_modalities: list[str]
    suggested_fusion_weights: list[float]
    routing_confidence: float

    def __post_init__(self) -> None:
        if isinstance(self.contradiction_type, str):
            self.contradiction_type = ContradictionType(self.contradiction_type)

        if len(self.suggested_fusion_weights) != len(MODALITIES):
            raise ValueError(
                "suggested_fusion_weights must have "
                f"{len(MODALITIES)} elements for {MODALITIES}"
            )

        total = sum(float(w) for w in self.suggested_fusion_weights)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"suggested_fusion_weights must sum to 1.0 (±{WEIGHT_SUM_TOLERANCE}), "
                f"got {total:.4f}"
            )

        if not 0.0 <= float(self.routing_confidence) <= 1.0:
            raise ValueError("routing_confidence must be in [0, 1]")

        if float(self.contradiction_intensity) < 0.0:
            raise ValueError("contradiction_intensity must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_type": self.contradiction_type.value,
            "contradiction_intensity": float(self.contradiction_intensity),
            "involved_modalities": list(self.involved_modalities),
            "suggested_fusion_weights": [
                float(w) for w in self.suggested_fusion_weights
            ],
            "routing_confidence": float(self.routing_confidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContradictionResult:
        return cls(
            contradiction_type=data["contradiction_type"],
            contradiction_intensity=float(data["contradiction_intensity"]),
            involved_modalities=list(data["involved_modalities"]),
            suggested_fusion_weights=[
                float(w) for w in data["suggested_fusion_weights"]
            ],
            routing_confidence=float(data["routing_confidence"]),
        )


@dataclass
class Fragment:
    id: str
    start_time: float
    end_time: float
    va_self: ModalityVADict = field(default_factory=dict)
    va_inter: ModalityVADict = field(default_factory=dict)
    contradiction: ContradictionResult | None = None

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "va_self": {k: v.to_dict() for k, v in self.va_self.items()},
            "va_inter": {k: v.to_dict() for k, v in self.va_inter.items()},
            "contradiction": (
                self.contradiction.to_dict() if self.contradiction else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fragment:
        return cls(
            id=str(data["id"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            va_self={
                k: VAConfidence.from_dict(v) for k, v in data.get("va_self", {}).items()
            },
            va_inter={
                k: VAConfidence.from_dict(v)
                for k, v in data.get("va_inter", {}).items()
            },
            contradiction=(
                ContradictionResult.from_dict(data["contradiction"])
                if data.get("contradiction")
                else None
            ),
        )


@dataclass
class MemoryHit:
    fragment_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_type: str = "inter"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryHit:
        return cls(
            fragment_id=str(data["fragment_id"]),
            score=float(data["score"]),
            metadata=dict(data.get("metadata", {})),
            embedding_type=str(data.get("embedding_type", "inter")),
        )


@dataclass
class ReportBundle:
    segment_reports: list[str] = field(default_factory=list)
    overall_report: str = ""
    figures_paths: dict[str, str] = field(default_factory=dict)
    calm_index: float | None = None
    anxiety_index: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportBundle:
        return cls(
            segment_reports=list(data.get("segment_reports", [])),
            overall_report=str(data.get("overall_report", "")),
            figures_paths=dict(data.get("figures_paths", {})),
            calm_index=(
                float(data["calm_index"]) if data.get("calm_index") is not None else None
            ),
            anxiety_index=(
                float(data["anxiety_index"])
                if data.get("anxiety_index") is not None
                else None
            ),
        )


@dataclass
class PersonalityResult:
    O: float
    C: float
    E: float
    A: float
    N: float
    confidence_interval: list[float] = field(default_factory=list)
    behavioral_evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "personality": {
                "O": float(self.O),
                "C": float(self.C),
                "E": float(self.E),
                "A": float(self.A),
                "N": float(self.N),
            },
            "confidence_interval": [float(v) for v in self.confidence_interval],
            "behavioral_evidence": self.behavioral_evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityResult:
        personality = data.get("personality", data)
        return cls(
            O=float(personality["O"]),
            C=float(personality["C"]),
            E=float(personality["E"]),
            A=float(personality["A"]),
            N=float(personality["N"]),
            confidence_interval=[
                float(v) for v in data.get("confidence_interval", [])
            ],
            behavioral_evidence=str(data.get("behavioral_evidence", "")),
        )
