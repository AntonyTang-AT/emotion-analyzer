"""Tests for the Issue #10 macro-expression extractor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core import DataContext
from src.layer1_feature import factory
from src.layer1_feature.factory import get_extractors_for_context
from src.layer1_feature.macro_extractor import (
    MacroExpressionExtractor,
    MacroExtractor,
)
from src.pipeline import run_pipeline


def macro_config(**overrides):
    config = {
        "roi_size": 32,
        "clip_frames": 4,
        "clip_stride": 2,
        "embedding_dim": 512,
        "backend": "simple",
        "align_faces": False,
    }
    config.update(overrides)
    return config


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "face.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    assert cv2.imwrite(str(path), image)
    return path


def test_video_returns_second_aligned_512d_features(sample_video_path):
    result = MacroExtractor(macro_config()).extract_video(sample_video_path)

    assert [timestamp for _, timestamp in result] == [0.0, 1.0]
    assert all(feature.shape == (512,) for feature, _ in result)
    assert all(feature.dtype == np.float32 for feature, _ in result)


def test_image_is_padded_to_one_clip(sample_image_path):
    observed_lengths = []

    def encoder(clip):
        observed_lengths.append(len(clip))
        return np.arange(512, dtype=np.float32)

    extractor = MacroExtractor(
        macro_config(clip_frames=16),
        aligner=lambda frame: frame,
        encoder=encoder,
    )
    result = extractor.extract_image(sample_image_path)

    assert observed_lengths == [16]
    assert result[0][0].shape == (512,)
    assert result[0][1] == 0.0


def test_run_preserves_features_and_populates_visual_bypass(sample_video_path):
    context = DataContext.create(user_id="test", video_path=sample_video_path)
    context.features["text"] = ["kept"]
    extractor = MacroExtractor(
        macro_config(),
        aligner=lambda frame: frame,
        encoder=lambda clip: np.arange(512, dtype=np.float32),
    )

    assert extractor.run(context) is context
    assert context.features["text"] == ["kept"]
    assert len(context.features["macro"]) == 2
    assert context.raw_visual_features["macro"] is context.features["macro"]


@pytest.mark.parametrize(
    "override",
    [{"roi_size": 0}, {"clip_frames": 0}, {"clip_stride": 0}, {"embedding_dim": 0}],
)
def test_rejects_invalid_window_config(override):
    with pytest.raises(ValueError, match="must be positive"):
        MacroExtractor(macro_config(**override))


def test_missing_profile_path_is_reported():
    video_context = DataContext.create(user_id="test", input_type="video")
    image_context = DataContext.create(user_id="test", input_type="image")
    extractor = MacroExtractor(macro_config())

    with pytest.raises(ValueError, match="video_path"):
        extractor.extract(video_context)
    with pytest.raises(ValueError, match="image_path"):
        extractor.extract(image_context)


def test_dependencies_are_lazy_until_first_extraction():
    extractor = MacroExtractor(macro_config(backend="videomae"))

    assert extractor._processor is None
    assert extractor._model is None
    assert extractor._aligner is None


def test_factory_registers_real_macro_extractor(sample_video_path):
    context = DataContext.create(
        user_id="test",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["macro"]},
    )
    extractors = get_extractors_for_context(context)

    assert len(extractors) == 1
    assert isinstance(extractors[0], MacroExtractor)
    assert MacroExpressionExtractor is MacroExtractor


@pytest.mark.parametrize("input_type", ["video", "image"])
def test_run_pipeline_profiles_produce_macro_features(
    monkeypatch,
    input_type,
    sample_video_path,
    sample_image_path,
):
    extractor = MacroExtractor(
        macro_config(),
        aligner=lambda frame: frame,
        encoder=lambda clip: np.arange(512, dtype=np.float32),
    )
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "macro", lambda: extractor)
    kwargs = (
        {"video_path": sample_video_path}
        if input_type == "video"
        else {"image_path": sample_image_path}
    )

    result = run_pipeline(input_type, user_id="test", **kwargs)

