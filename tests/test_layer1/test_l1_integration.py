"""L1 layer integration acceptance tests (task 1.7)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.layer1_feature import factory
from src.layer1_feature.factory import run_l1
from tests.test_layer1.conftest import (
    FEATURE_KEYS,
    MODALITY_DIMS,
    context_for_profile,
    mock_text_extractor,
)


@patch("src.layer1_feature.text_extractor.extract_audio_from_video")
@patch("src.layer1_feature.speech_extractor.SpeechExtractor._wav2vec_frames")
def test_run_l1_video_four_modalities_dims(
    mock_wav2vec,
    mock_extract_audio,
    monkeypatch,
    mock_video_l1_extractors,
    sample_video_path,
    sample_wav_path,
    sample_config,
):
    mock_wav2vec.return_value = np.ones((49, 1024), dtype=np.float32)
    mock_extract_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_video_l1_extractors()

    context = context_for_profile(
        "video",
        sample_config,
        video_path=str(sample_video_path),
        audio_path=str(sample_wav_path),
    )
    result = run_l1(context)

    text_item = result.features["text"][0]
    assert text_item[FEATURE_KEYS["text"]].shape == (MODALITY_DIMS["text"],)
    assert "stub" not in text_item

    speech_item = result.features["speech"][0]
    assert speech_item[FEATURE_KEYS["speech"]].shape == (MODALITY_DIMS["speech"],)
    assert "stub" not in speech_item

    macro_item = result.features["macro"][0]
    assert macro_item[0].shape == (MODALITY_DIMS["macro"],)

    micro_item = result.features["micro"][0]
    assert micro_item[FEATURE_KEYS["micro"]].shape == (MODALITY_DIMS["micro"],)
    assert "stub" not in micro_item

    assert result.raw_visual_features["macro"] is result.features["macro"]
    assert result.raw_visual_features["micro"][0].shape == (MODALITY_DIMS["micro"],)
    assert result.metadata["stage_status"]["L1"] == "completed"


def test_run_l1_text_profile_only_text(monkeypatch, sample_config):
    extractor = mock_text_extractor(monkeypatch)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    context = context_for_profile(
        "text",
        sample_config,
        text_subtype="descriptive",
        text_content="Hello world.",
    )
    result = run_l1(context)

    assert set(result.features.keys()) == {"text"}
    assert result.features["text"][0][FEATURE_KEYS["text"]].shape == (
        MODALITY_DIMS["text"],
    )
    assert result.metadata["stage_status"]["L1"] == "completed"


def test_run_l1_image_profile_macro_micro_bypass(sample_config, tmp_path):
    image_path = tmp_path / "face.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    import cv2

    assert cv2.imwrite(str(image_path), image)

    context = context_for_profile(
        "image",
        sample_config,
        image_path=str(image_path),
    )
    result = run_l1(context)

    assert set(result.features.keys()) == {"macro", "micro"}
    assert result.features["macro"][0][0].shape == (MODALITY_DIMS["macro"],)
    assert (
        result.features["micro"][0][FEATURE_KEYS["micro"]].shape
        == (MODALITY_DIMS["micro"],)
    )
    assert "macro" in result.raw_visual_features
    assert "micro" in result.raw_visual_features
    assert result.raw_visual_features["micro"][0].shape == (MODALITY_DIMS["micro"],)


def test_run_l1_partial_failure_records_metadata(monkeypatch, sample_config):
    extractor = mock_text_extractor(monkeypatch)

    def _boom(_context):
        raise RuntimeError("simulated extractor failure")

    monkeypatch.setattr(extractor, "extract", _boom)
    ok = factory.StubModalityExtractor("speech", feature_dim=16)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "speech", lambda: ok)

    context = context_for_profile(
        "audio",
        sample_config,
        audio_path="data/raw/test.wav",
    )
    result = run_l1(context)

    assert result.metadata["l1_failures"] == ["text"]
    assert result.metadata["l1_partial"] is True
    assert "speech" in result.features
    assert result.metadata["stage_status"]["L1"] == "completed"
