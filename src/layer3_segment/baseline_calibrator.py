"""Static baseline calibration using VA_self offsets (delta VA)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.core.types import MODALITIES, Fragment, ModalityVADict, VAConfidence
from src.utils.config_loader import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_POPULATION_BASELINE_PATH = Path("data/standard_stimuli/population_baseline.json")

DeltaVA = dict[str, tuple[float, float]]


@dataclass(frozen=True)
class BaselineConfig:
    enabled: bool = True
    use_va_type: str = "self"
    calibration_videos: int = 3
    video_duration_sec: int = 15
    delta_va_storage: Path = Path("data/processed/delta_va")
    population_baseline_path: Path = DEFAULT_POPULATION_BASELINE_PATH
    stimuli_dir: Path = Path("data/standard_stimuli")

    @classmethod
    def from_pipeline(cls, pipeline_config: dict[str, Any] | None = None) -> BaselineConfig:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        baseline = pipeline_config.get("pipeline", {}).get("stages", {}).get("L3", {}).get(
            "baseline", {}
        )
        return cls(
            enabled=bool(baseline.get("enabled", True)),
            use_va_type=str(baseline.get("use_va_type", "self")),
            calibration_videos=int(baseline.get("calibration_videos", 3)),
            video_duration_sec=int(baseline.get("video_duration_sec", 15)),
            delta_va_storage=Path(str(baseline.get("delta_va_storage", "data/processed/delta_va"))),
        )


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return get_project_root() / path


def _clip_va(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _va_from_mapping(data: dict[str, Any]) -> VAConfidence:
    return VAConfidence(
        valence=float(data["valence"]),
        arousal=float(data["arousal"]),
        confidence=float(data.get("confidence", 1.0)),
    )


def _modality_dict_from_json(data: dict[str, Any]) -> ModalityVADict:
    result: ModalityVADict = {}
    for modality, payload in data.items():
        if isinstance(payload, dict) and "valence" in payload and "arousal" in payload:
            result[str(modality)] = _va_from_mapping(payload)
    return result


def _delta_to_json(delta_va: DeltaVA) -> dict[str, dict[str, float]]:
    return {
        modality: {"valence": float(values[0]), "arousal": float(values[1])}
        for modality, values in delta_va.items()
    }


def _delta_from_json(data: dict[str, Any]) -> DeltaVA:
    delta: DeltaVA = {}
    for modality, payload in data.items():
        if isinstance(payload, dict):
            delta[str(modality)] = (
                float(payload["valence"]),
                float(payload["arousal"]),
            )
    return delta


def va_self_to_vector(va_dict: ModalityVADict) -> list[float]:
    """Flatten VA_self to 8-d vector: [text_V, text_A, speech_V, speech_A, ...]."""
    vector: list[float] = []
    for modality in MODALITIES:
        item = va_dict.get(modality)
        if item is None:
            vector.extend([0.0, 0.0])
        else:
            vector.extend([float(item.valence), float(item.arousal)])
    return vector


def vector_to_va_self(vector: list[float]) -> ModalityVADict:
    """Parse 8-d vector back into a partial modality VA dict."""
    result: ModalityVADict = {}
    for index, modality in enumerate(MODALITIES):
        offset = index * 2
        if offset + 1 >= len(vector):
            break
        result[modality] = VAConfidence(
            valence=float(vector[offset]),
            arousal=float(vector[offset + 1]),
            confidence=1.0,
        )
    return result


def compute_delta_va(
    user_avg: ModalityVADict,
    population_avg: ModalityVADict,
) -> DeltaVA:
    """Compute per-modality delta: user minus population."""
    delta: DeltaVA = {}
    for modality in MODALITIES:
        user_item = user_avg.get(modality)
        pop_item = population_avg.get(modality)
        if user_item is None or pop_item is None:
            continue
        delta[modality] = (
            float(user_item.valence) - float(pop_item.valence),
            float(user_item.arousal) - float(pop_item.arousal),
        )
    return delta


def load_population_baseline(path: Path | None = None) -> ModalityVADict:
    """Load population-average VA_self baseline from JSON."""
    target = _resolve_project_path(path or DEFAULT_POPULATION_BASELINE_PATH)
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    population = payload.get("population_average", payload)
    return _modality_dict_from_json(population)


def _baseline_file_path(user_id: str, storage_dir: Path) -> Path:
    safe_id = quote(str(user_id).strip(), safe="")
    if not safe_id:
        raise ValueError("user_id must not be empty")
    directory = _resolve_project_path(storage_dir)
    return directory / f"{safe_id}.json"


def load_baseline(
    user_id: str,
    *,
    storage_dir: Path | None = None,
    config: BaselineConfig | None = None,
) -> DeltaVA | None:
    """Load stored delta VA for a user, or None if missing."""
    cfg = config or BaselineConfig.from_pipeline()
    path = _baseline_file_path(user_id, storage_dir or cfg.delta_va_storage)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    delta = _delta_from_json(payload.get("delta_va", payload))
    return delta or None


def save_baseline(
    user_id: str,
    delta_va: DeltaVA,
    *,
    storage_dir: Path | None = None,
    config: BaselineConfig | None = None,
) -> Path:
    """Persist delta VA for a user."""
    cfg = config or BaselineConfig.from_pipeline()
    path = _baseline_file_path(user_id, storage_dir or cfg.delta_va_storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_id": user_id,
        "delta_va": _delta_to_json(delta_va),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def apply_baseline(
    va_self: ModalityVADict,
    user_id: str,
    *,
    delta: DeltaVA | None = None,
    config: BaselineConfig | None = None,
) -> ModalityVADict:
    """Subtract stored delta VA from VA_self predictions."""
    cfg = config or BaselineConfig.from_pipeline()
    if delta is not None:
        resolved_delta = delta
    elif cfg.enabled:
        resolved_delta = load_baseline(user_id, config=cfg)
    else:
        resolved_delta = None
    if not resolved_delta:
        return {modality: VAConfidence(item.valence, item.arousal, item.confidence) for modality, item in va_self.items()}

    calibrated: ModalityVADict = {}
    for modality, item in va_self.items():
        offsets = resolved_delta.get(modality)
        if offsets is None:
            calibrated[modality] = VAConfidence(item.valence, item.arousal, item.confidence)
            continue
        delta_v, delta_a = offsets
        calibrated[modality] = VAConfidence(
            valence=_clip_va(item.valence - delta_v),
            arousal=_clip_va(item.arousal - delta_a),
            confidence=item.confidence,
        )
    return calibrated


def calibrate_from_responses(
    user_id: str,
    user_avg: ModalityVADict,
    population_avg: ModalityVADict | None = None,
    *,
    config: BaselineConfig | None = None,
) -> DeltaVA:
    """Compute delta VA from user averages and persist for the user."""
    cfg = config or BaselineConfig.from_pipeline()
    population = population_avg or load_population_baseline(cfg.population_baseline_path)
    delta_va = compute_delta_va(user_avg, population)
    if delta_va:
        save_baseline(user_id, delta_va, config=cfg)
        logger.info("Saved baseline calibration for user '%s' (%d modalities)", user_id, len(delta_va))
    else:
        logger.info("No overlapping modalities for user '%s'; baseline not saved", user_id)
    return delta_va


def apply_baseline_to_fragment(fragment: Fragment, user_id: str, *, config: BaselineConfig | None = None) -> Fragment:
    """Return a copy of fragment with calibrated va_self values."""
    calibrated = apply_baseline(fragment.va_self, user_id, config=config)
    return replace(fragment, va_self=calibrated)


def run_calibration_session(
    user_id: str,
    stimulus_paths: list[Path],
    *,
    config: BaselineConfig | None = None,
) -> DeltaVA:
    """Run L1+L2 on standard stimuli and persist delta VA (future integration).

    v1 leaves this as a documented hook; callers should use
    ``calibrate_from_responses`` with mocked or precomputed VA_self averages.
    """
    raise NotImplementedError(
        "Full stimulus calibration requires L1+L2 pipeline integration; "
        "use calibrate_from_responses() in v1."
    )


__all__ = [
    "BaselineConfig",
    "DeltaVA",
    "apply_baseline",
    "apply_baseline_to_fragment",
    "calibrate_from_responses",
    "compute_delta_va",
    "load_baseline",
    "load_population_baseline",
    "run_calibration_session",
    "save_baseline",
    "va_self_to_vector",
    "vector_to_va_self",
]
