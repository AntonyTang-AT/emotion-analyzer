"""Pipeline integration tests for L1 → L2 handoff."""

from __future__ import annotations

import cv2
import numpy as np

from src.layer1_feature import factory
from src.pipeline import run_pipeline
from tests.test_layer1.conftest import (
    MODALITY_DIMS,
    FEATURE_KEYS,
    mock_macro_extractor,
    mock_text_extractor,
)


def test_pipeline_text_l2_completed(monkeypatch):
    extractor = mock_text_extractor(monkeypatch)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    result = run_pipeline(
        input_type="text",
        text_content="test text input",
        text_subtype="dialogue",
    )

    assert result["input_profile"] == "text_dialogue"
    assert result["stage_status"]["L2"] == "completed"
    assert len(result["context"]["va_self_predictions"]["text"]) >= 1
    assert len(result["context"]["va_inter_predictions"]["text"]) >= 1
    self_len = len(result["context"]["va_self_predictions"]["text"])
    inter_len = len(result["context"]["va_inter_predictions"]["text"])
    assert self_len == inter_len
    assert self_len == len(result["features"]["text"])


def test_pipeline_audio_l2_two_modalities(monkeypatch, sample_wav_path):
    text_extractor = mock_text_extractor(monkeypatch)
    monkeypatch.setattr(
        text_extractor,
        "_transcribe_audio",
        lambda source: {
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
        },
    )
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "text",
        lambda: text_extractor,
    )
    monkeypatch.setattr(
        "src.layer1_feature.speech_extractor.SpeechExtractor._wav2vec_frames",
        lambda self, waveform, sample_rate: np.ones((49, 1024), dtype=np.float32),
    )

    result = run_pipeline(
        input_type="audio",
        audio_path=sample_wav_path,
    )

    assert result["stage_status"]["L1"] == "completed"
    assert result["stage_status"]["L2"] == "completed"
    for modality in ("text", "speech"):
        assert modality in result["context"]["va_self_predictions"]
        assert len(result["context"]["va_self_predictions"][modality]) == len(
            result["features"][modality]
        )
        assert len(result["context"]["va_inter_predictions"][modality]) == len(
            result["context"]["va_self_predictions"][modality]
        )
        key = FEATURE_KEYS[modality]
        assert result["features"][modality][0][key].shape == (MODALITY_DIMS[modality],)


def test_pipeline_image_l2_visual_modalities(monkeypatch, tmp_path):
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "macro",
        lambda: mock_macro_extractor(),
    )

    image_path = tmp_path / "face.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    assert cv2.imwrite(str(image_path), image)

    result = run_pipeline(
        input_type="image",
        image_path=str(image_path),
    )

    assert result["stage_status"]["L1"] == "completed"
    assert result["stage_status"]["L2"] == "completed"
    for modality in ("macro", "micro"):
        assert modality in result["context"]["va_self_predictions"]
        assert len(result["context"]["va_self_predictions"][modality]) == len(
            result["features"][modality]
        )
        assert len(result["context"]["va_inter_predictions"][modality]) == len(
            result["context"]["va_self_predictions"][modality]
        )


def test_pipeline_skips_l2_when_disabled(monkeypatch):
    extractor = mock_text_extractor(monkeypatch)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    result = run_pipeline(
        input_type="text",
        text_content="hello",
        text_subtype="dialogue",
        config_overrides={"pipeline": {"stages": {"L2": {"enabled": False}}}},
    )

    assert result["stage_status"]["L1"] == "completed"
    assert result["stage_status"]["L2"] == "skipped"
    assert result["skipped_stages"]["L2"] == "disabled_by_profile"
    assert not result["context"]["va_self_predictions"]
