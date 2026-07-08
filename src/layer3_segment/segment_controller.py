"""Dynamic segmentation controller based on VA_inter time series."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.core.context import DataContext
from src.core.interfaces import SegmentController as SegmentControllerBase
from src.core.types import Fragment, SegmentationMode, VAConfidence
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

TIME_KEYS: dict[str, str] = {
    "text": "start_time",
    "speech": "timestamp",
    "micro": "start_time",
}

SIGN_EPSILON = 1e-6


@dataclass(frozen=True)
class SegmentationConfig:
    arousal_threshold: float = 0.3
    max_fragment_length: float = 30.0
    polarity_flip: bool = True
    modality_distance_threshold: float | None = 0.6
    use_va_type: str = "inter"
    use_crm_window: bool = False
    crm_min_window_sec: float = 2.0
    crm_max_window_sec: float = 30.0

    @classmethod
    def from_pipeline(cls, pipeline_config: dict[str, Any] | None = None) -> SegmentationConfig:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        seg = pipeline_config.get("pipeline", {}).get("stages", {}).get("L3", {}).get(
            "segmentation", {}
        )
        distance = seg.get("modality_distance_threshold")
        if distance is not None:
            distance = float(distance)
            if distance <= 0:
                distance = None
        return cls(
            arousal_threshold=float(seg.get("arousal_threshold", 0.3)),
            max_fragment_length=float(seg.get("max_fragment_length", 30.0)),
            polarity_flip=bool(seg.get("polarity_flip", True)),
            modality_distance_threshold=distance,
            use_va_type=str(seg.get("use_va_type", "inter")),
            use_crm_window=bool(seg.get("use_crm_window", False)),
        )


@dataclass
class TimelineFrame:
    time: float
    va_inter: dict[str, VAConfidence] = field(default_factory=dict)
    va_self: dict[str, VAConfidence] = field(default_factory=dict)
    fresh_modalities: set[str] = field(default_factory=set)


def _sign(value: float) -> int:
    if value > SIGN_EPSILON:
        return 1
    if value < -SIGN_EPSILON:
        return -1
    return 0


def _modality_times(modality: str, items: list[Any]) -> list[float]:
    name = modality.strip().lower()
    if name == "macro":
        return [float(item[1]) for item in items]
    key = TIME_KEYS.get(name)
    if key is None:
        return [float(index) for index in range(len(items))]
    if name in {"text", "micro"}:
        return [
            (float(item.get("start_time", 0.0)) + float(item.get("end_time", 0.0))) / 2.0
            for item in items
        ]
    return [float(item.get(key, index)) for index, item in enumerate(items)]


def _resolve_segmentation_mode(context: DataContext) -> str:
    mode = context.metadata.get("segmentation_mode", SegmentationMode.DYNAMIC.value)
    return str(mode).strip().lower()


def _branch_predictions(context: DataContext, branch: str) -> dict[str, list[VAConfidence]]:
    if branch == "self":
        return dict(context.va_self_predictions)
    return dict(context.va_inter_predictions)


def build_timeline(
    context: DataContext,
    *,
    branch: str = "inter",
) -> list[TimelineFrame]:
    """Align per-modality VA series onto a shared time axis."""
    predictions = _branch_predictions(context, branch)
    other_branch = "self" if branch == "inter" else "inter"
    other_predictions = _branch_predictions(context, other_branch)

    if not predictions:
        return []

    modality_times: dict[str, list[float]] = {}
    for modality, series in predictions.items():
        items = context.features.get(modality)
        if items and len(items) >= len(series):
            times = _modality_times(modality, items[: len(series)])
        else:
            times = [float(index) for index in range(len(series))]
        modality_times[modality] = times

    unique_times = sorted({time for times in modality_times.values() for time in times})
    if not unique_times:
        return []

    frames: list[TimelineFrame] = []
    for time in unique_times:
        frame = TimelineFrame(time=time)
        for modality, series in predictions.items():
            times = modality_times[modality]
            index = _index_at_or_before(times, time)
            if index is None:
                continue
            if abs(times[index] - time) <= 1e-9:
                frame.fresh_modalities.add(modality)
            if branch == "inter":
                frame.va_inter[modality] = series[index]
                self_series = other_predictions.get(modality)
                if self_series and index < len(self_series):
                    frame.va_self[modality] = self_series[index]
            else:
                frame.va_self[modality] = series[index]
                inter_series = other_predictions.get(modality)
                if inter_series and index < len(inter_series):
                    frame.va_inter[modality] = inter_series[index]
        frames.append(frame)
    return frames


def _index_at_or_before(times: list[float], target: float) -> int | None:
    if not times:
        return None
    chosen: int | None = None
    for index, value in enumerate(times):
        if value <= target + 1e-9:
            chosen = index
        else:
            break
    return chosen


def combined_va(values: Iterable[VAConfidence]) -> tuple[float, float]:
    total_conf = 0.0
    weighted_v = 0.0
    weighted_a = 0.0
    for item in values:
        conf = max(float(item.confidence), 0.0)
        total_conf += conf
        weighted_v += conf * float(item.valence)
        weighted_a += conf * float(item.arousal)
    if total_conf <= 0:
        items = list(values)
        if not items:
            return 0.0, 0.0
        return (
            sum(item.valence for item in items) / len(items),
            sum(item.arousal for item in items) / len(items),
        )
    return weighted_v / total_conf, weighted_a / total_conf


def max_modality_distance(values: dict[str, VAConfidence]) -> float:
    modalities = list(values.keys())
    if len(modalities) < 2:
        return 0.0
    max_distance = 0.0
    for left in range(len(modalities)):
        for right in range(left + 1, len(modalities)):
            a = values[modalities[left]]
            b = values[modalities[right]]
            distance = math.hypot(a.valence - b.valence, a.arousal - b.arousal)
            max_distance = max(max_distance, distance)
    return max_distance


def _average_va(values: list[VAConfidence]) -> VAConfidence:
    if not values:
        return VAConfidence(0.0, 0.0, 0.0)
    return VAConfidence(
        valence=sum(item.valence for item in values) / len(values),
        arousal=sum(item.arousal for item in values) / len(values),
        confidence=sum(item.confidence for item in values) / len(values),
    )


def _frames_for_segment(
    frames: list[TimelineFrame],
    start_time: float,
    end_time: float,
) -> list[TimelineFrame]:
    if not frames:
        return []
    return [
        frame
        for frame in frames
        if start_time <= frame.time <= end_time
    ]


def _build_fragment(
    *,
    index: int,
    start_time: float,
    end_time: float,
    frames: list[TimelineFrame],
) -> Fragment:
    inter_by_modality: dict[str, list[VAConfidence]] = {}
    self_by_modality: dict[str, list[VAConfidence]] = {}
    for frame in frames:
        for modality, value in frame.va_inter.items():
            inter_by_modality.setdefault(modality, []).append(value)
        for modality, value in frame.va_self.items():
            self_by_modality.setdefault(modality, []).append(value)
    return Fragment(
        id=f"seg-{index:04d}",
        start_time=start_time,
        end_time=end_time,
        va_inter={
            modality: _average_va(series)
            for modality, series in inter_by_modality.items()
        },
        va_self={
            modality: _average_va(series)
            for modality, series in self_by_modality.items()
        },
    )


def segment_single(frames: list[TimelineFrame]) -> list[Fragment]:
    if not frames:
        return []
    start = frames[0].time
    end = frames[-1].time
    return [_build_fragment(index=0, start_time=start, end_time=end, frames=frames)]


def segment_utterance(context: DataContext, frames: list[TimelineFrame]) -> list[Fragment]:
    if not frames:
        return []

    primary = _primary_modality(context)
    predictions = context.va_inter_predictions.get(primary, [])
    if not predictions:
        return segment_single(frames)

    items = context.features.get(primary, [])
    if items and len(items) >= len(predictions):
        boundaries = _utterance_boundaries(primary, items[: len(predictions)])
    else:
        boundaries = [
            (float(index), float(index + 1)) for index in range(len(predictions))
        ]

    fragments: list[Fragment] = []
    for index, (start, end) in enumerate(boundaries):
        segment_frames = _frames_for_segment(frames, start, end)
        if not segment_frames:
            segment_frames = [frames[min(index, len(frames) - 1)]]
        fragments.append(
            _build_fragment(
                index=index,
                start_time=start,
                end_time=end,
                frames=segment_frames,
            )
        )
    return fragments


def _primary_modality(context: DataContext) -> str:
    for candidate in ("text", "speech", "macro", "micro"):
        if context.va_inter_predictions.get(candidate):
            return candidate
    return next(iter(context.va_inter_predictions))


def _utterance_boundaries(modality: str, items: list[Any]) -> list[tuple[float, float]]:
    name = modality.strip().lower()
    if name in {"text", "micro"}:
        return [
            (float(item.get("start_time", index)), float(item.get("end_time", index + 1)))
            for index, item in enumerate(items)
        ]
    if name == "speech":
        boundaries: list[tuple[float, float]] = []
        for index, item in enumerate(items):
            start = float(item.get("timestamp", index))
            if index + 1 < len(items):
                end = float(items[index + 1].get("timestamp", start + 1.0))
            else:
                end = start + 1.0
            boundaries.append((start, end))
        return boundaries
    if name == "macro":
        return [(float(item[1]), float(item[1]) + 1.0) for item in items]
    return [(float(index), float(index + 1)) for index in range(len(items))]


def _should_cut(
    *,
    current: TimelineFrame,
    combined_previous: tuple[float, float],
    combined_current: tuple[float, float],
    config: SegmentationConfig,
) -> bool:
    _, prev_a = combined_previous
    _, curr_a = combined_current
    if abs(curr_a - prev_a) > config.arousal_threshold:
        return True

    if config.polarity_flip:
        prev_sign = _sign(combined_previous[0])
        curr_sign = _sign(combined_current[0])
        if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
            return True

    if config.modality_distance_threshold is not None:
        fresh_values = {
            modality: current.va_inter[modality]
            for modality in current.fresh_modalities
            if modality in current.va_inter
        }
        if len(fresh_values) >= 2:
            distance = max_modality_distance(fresh_values)
            if distance > config.modality_distance_threshold:
                return True
    return False


def _apply_crm_refinement(
    cut_indices: list[int],
    frames: list[TimelineFrame],
    config: SegmentationConfig,
) -> list[int]:
    if not config.use_crm_window or not cut_indices:
        return cut_indices

    refined: list[int] = []
    segment_start = 0
    for cut in cut_indices:
        best_index = cut
        best_score = float("inf")
        for candidate in range(max(segment_start + 1, cut - 2), min(len(frames) - 1, cut + 2) + 1):
            duration = frames[candidate].time - frames[segment_start].time
            if duration < config.crm_min_window_sec:
                continue
            if duration > config.crm_max_window_sec:
                continue
            window = frames[segment_start : candidate + 1]
            score = sum(max_modality_distance(frame.va_inter) for frame in window) / len(
                window
            )
            if score < best_score:
                best_score = score
                best_index = candidate
        refined.append(best_index)
        segment_start = best_index
    return sorted(set(refined))


def segment_dynamic(frames: list[TimelineFrame], config: SegmentationConfig) -> list[Fragment]:
    if not frames:
        return []
    if len(frames) == 1:
        return segment_single(frames)

    cut_indices: list[int] = []
    combined_values = [combined_va(frame.va_inter.values()) for frame in frames]

    segment_start_index = 0
    for index in range(1, len(frames)):
        duration = frames[index].time - frames[segment_start_index].time
        if duration >= config.max_fragment_length:
            cut_indices.append(index)
            segment_start_index = index
            continue

        if _should_cut(
            current=frames[index],
            combined_previous=combined_values[index - 1],
            combined_current=combined_values[index],
            config=config,
        ):
            cut_indices.append(index)
            segment_start_index = index

    cut_indices = _apply_crm_refinement(cut_indices, frames, config)

    boundaries = [0] + cut_indices + [len(frames)]
    unique_boundaries: list[int] = []
    for value in boundaries:
        if not unique_boundaries or unique_boundaries[-1] != value:
            unique_boundaries.append(value)

    fragments: list[Fragment] = []
    for fragment_index in range(len(unique_boundaries) - 1):
        start_index = unique_boundaries[fragment_index]
        end_index = unique_boundaries[fragment_index + 1] - 1
        if end_index < start_index:
            continue
        start_time = frames[start_index].time
        end_time = frames[end_index].time
        segment_frames = frames[start_index : end_index + 1]
        fragments.append(
            _build_fragment(
                index=fragment_index,
                start_time=start_time,
                end_time=end_time,
                frames=segment_frames,
            )
        )
    return fragments


def segment_from_context(
    context: DataContext,
    *,
    config: SegmentationConfig | None = None,
) -> list[Fragment]:
    """Segment L2 VA predictions into fragments for the given context."""
    if config is None:
        pipeline_config = context.metadata.get("config", {}).get("pipeline")
        config = SegmentationConfig.from_pipeline(pipeline_config)

    branch = config.use_va_type if config.use_va_type in {"self", "inter"} else "inter"
    frames = build_timeline(context, branch=branch)
    if not frames:
        logger.warning("L3 segmentation skipped: no VA timeline available")
        return []

    mode = _resolve_segmentation_mode(context)
    if mode == SegmentationMode.SINGLE.value:
        return segment_single(frames)
    if mode == SegmentationMode.UTTERANCE.value:
        return segment_utterance(context, frames)
    return segment_dynamic(frames, config)


class DynamicSegmentController(SegmentControllerBase):
    """Concrete L3 segment controller driven by pipeline configuration."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self._config = config

    def segment(self, context: DataContext) -> list[Fragment]:
        return segment_from_context(context, config=self._config)


__all__ = [
    "DynamicSegmentController",
    "SegmentationConfig",
    "TimelineFrame",
    "build_timeline",
    "segment_dynamic",
    "segment_from_context",
    "segment_single",
    "segment_utterance",
]
