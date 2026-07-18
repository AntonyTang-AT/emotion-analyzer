"""Tests for L3 dynamic segment controller (task 3.1)."""

from __future__ import annotations

import pytest

from src.core import DataContext, VAConfidence
from src.core.types import SegmentationMode
from src.layer3_segment.segment_controller import (
    SegmentationConfig,
    build_timeline,
    max_modality_distance,
    segment_dynamic,
    segment_from_context,
    segment_single,
)


def _va(v: float, a: float, c: float = 0.9) -> VAConfidence:
    return VAConfidence(v, a, c)


def _context_with_series(
    *,
    inter_series: dict[str, list[VAConfidence]],
    self_series: dict[str, list[VAConfidence]] | None = None,
    features: dict | None = None,
    segmentation_mode: str = SegmentationMode.DYNAMIC.value,
) -> DataContext:
    ctx = DataContext.create(
        user_id="seg-test",
        input_type="video",
        video_path="data/raw/test.mp4",
        profile_metadata={
            "active_modalities": list(inter_series.keys()),
            "segmentation_mode": segmentation_mode,
        },
    )
    ctx.set_stage(
        "L2",
        {
            "va_inter_predictions": inter_series,
            "va_self_predictions": self_series or inter_series,
        },
    )
    if features is not None:
        ctx.set_stage(
            "L1",
            {"features": features, "raw_visual_features": {}},
        )
    return ctx


def test_build_timeline_aligns_modalities_by_timestamp():
    context = _context_with_series(
        inter_series={
            "text": [_va(0.1, 0.2), _va(0.3, 0.4), _va(0.5, 0.6)],
            "speech": [_va(0.5, 0.6), _va(0.7, 0.8), _va(0.9, 1.0)],
        },
    )

    frames = build_timeline(context)

    assert len(frames) == 3
    assert frames[0].va_inter["text"].valence == pytest.approx(0.1)
    assert frames[2].va_inter["speech"].valence == pytest.approx(0.9)


def test_arousal_spike_splits_dynamic(mock_arousal_spike_va_sequence):
    frames = build_timeline(
        _context_with_series(
            inter_series={"text": mock_arousal_spike_va_sequence},
        )
    )
    config = SegmentationConfig(
        arousal_threshold=0.3,
        polarity_flip=False,
        modality_distance_threshold=None,
        max_fragment_length=100.0,
    )

    fragments = segment_dynamic(frames, config)

    assert len(fragments) == 2
    assert fragments[0].end_time == pytest.approx(1.0)
    assert fragments[1].start_time == pytest.approx(2.0)


def test_polarity_flip_splits_dynamic():
    frames = build_timeline(
        _context_with_series(
            inter_series={"text": [_va(0.6, 0.1), _va(-0.6, 0.1)]},
            features={
                "text": [
                    {"text_embedding": [0.0], "start_time": 0.0, "end_time": 1.0},
                    {"text_embedding": [0.0], "start_time": 1.0, "end_time": 2.0},
                ],
            },
        )
    )
    config = SegmentationConfig(
        arousal_threshold=1.0,
        polarity_flip=True,
        modality_distance_threshold=None,
    )

    fragments = segment_dynamic(frames, config)

    assert len(fragments) == 2


def test_modality_distance_rule_respects_threshold():
    near = max_modality_distance(
        {"text": _va(0.1, 0.1), "speech": _va(0.2, 0.15)}
    )
    far = max_modality_distance(
        {"text": _va(1.0, 1.0), "speech": _va(-1.0, -1.0)}
    )
    assert near < 0.6
    assert far > 0.6

    frames = build_timeline(
        _context_with_series(
            inter_series={
                "text": [_va(0.0, 0.0), _va(1.0, 1.0)],
                "speech": [_va(0.0, 0.0), _va(-1.0, -1.0)],
            },
        )
    )
    config = SegmentationConfig(
        arousal_threshold=1.0,
        polarity_flip=False,
        modality_distance_threshold=0.6,
    )

    fragments = segment_dynamic(frames, config)

    assert len(fragments) == 2


def test_modality_distance_ignores_forward_filled_stale_values():
    """Only compare modalities updated at the same timestamp."""
    frames = build_timeline(
        _context_with_series(
            inter_series={
                "text": [_va(0.0, 0.0), _va(1.0, 1.0)],
                "speech": [_va(0.0, 0.0), _va(-1.0, -1.0)],
            },
            features={
                "text": [
                    {"text_embedding": [0.0], "start_time": 0.0, "end_time": 1.0},
                    {"text_embedding": [0.0], "start_time": 2.0, "end_time": 3.0},
                ],
                "speech": [
                    {"speech_feature": [0.0], "timestamp": 0.0},
                    {"speech_feature": [0.0], "timestamp": 1.0},
                ],
            },
        )
    )
    config = SegmentationConfig(
        arousal_threshold=1.0,
        polarity_flip=False,
        modality_distance_threshold=0.6,
    )

    fragments = segment_dynamic(frames, config)

    # speech updates at t=1 with stale forward-filled text -> no distance cut
    assert len(fragments) == 1


def test_max_fragment_length_forces_cut():
    series = [_va(0.0, 0.0) for _ in range(5)]
    features = [
        {"text_embedding": [0.0], "start_time": float(i * 10), "end_time": float(i * 10 + 1)}
        for i in range(5)
    ]
    frames = build_timeline(
        _context_with_series(
            inter_series={"text": series},
            features={"text": features},
        )
    )
    config = SegmentationConfig(
        arousal_threshold=1.0,
        polarity_flip=False,
        modality_distance_threshold=None,
        max_fragment_length=30.0,
    )

    fragments = segment_dynamic(frames, config)

    assert len(fragments) >= 2
    assert all(fragment.end_time - fragment.start_time <= 30.0 for fragment in fragments)


def test_single_mode_one_fragment():
    context = _context_with_series(
        inter_series={"text": [_va(0.2, 0.3), _va(0.4, 0.5)]},
        features={
            "text": [
                {"text_embedding": [0.0], "start_time": 0.0, "end_time": 1.0},
                {"text_embedding": [0.0], "start_time": 1.0, "end_time": 2.0},
            ],
        },
        segmentation_mode=SegmentationMode.SINGLE.value,
    )

    fragments = segment_from_context(context)

    assert len(fragments) == 1
    assert fragments[0].start_time == pytest.approx(0.5)
    assert fragments[0].end_time == pytest.approx(1.5)


def test_utterance_mode_one_fragment_per_step():
    context = _context_with_series(
        inter_series={"text": [_va(0.1, 0.2), _va(0.3, 0.4), _va(0.5, 0.6)]},
        features={
            "text": [
                {"text_embedding": [0.0], "start_time": 0.0, "end_time": 1.0},
                {"text_embedding": [0.0], "start_time": 1.0, "end_time": 2.0},
                {"text_embedding": [0.0], "start_time": 2.0, "end_time": 3.0},
            ],
        },
        segmentation_mode=SegmentationMode.UTTERANCE.value,
    )

    fragments = segment_from_context(context)

    assert len(fragments) == 3
    assert fragments[1].start_time == pytest.approx(1.0)


def test_empty_va_returns_empty_list():
    context = DataContext.create(user_id="empty", input_type="text", text_content="hi")

    assert segment_from_context(context) == []


def test_segment_single_aggregates_modality_means():
    frames = build_timeline(
        _context_with_series(
            inter_series={"text": [_va(0.2, 0.4), _va(0.6, 0.8)]},
            features={
                "text": [
                    {"text_embedding": [0.0], "start_time": 0.0, "end_time": 1.0},
                    {"text_embedding": [0.0], "start_time": 1.0, "end_time": 2.0},
                ],
            },
        )
    )

    fragments = segment_single(frames)

    assert len(fragments) == 1
    assert fragments[0].va_inter["text"].valence == pytest.approx(0.4)
    assert fragments[0].va_inter["text"].arousal == pytest.approx(0.6)


def test_segmentation_config_from_pipeline():
    config = SegmentationConfig.from_pipeline(
        {
            "pipeline": {
                "stages": {
                    "L3": {
                        "segmentation": {
                            "arousal_threshold": 0.25,
                            "max_fragment_length": 20,
                            "polarity_flip": False,
                            "modality_distance_threshold": 0,
                            "use_va_type": "inter",
                        }
                    }
                }
            }
        }
    )

    assert config.arousal_threshold == 0.25
    assert config.max_fragment_length == 20.0
    assert config.polarity_flip is False
    assert config.modality_distance_threshold is None
