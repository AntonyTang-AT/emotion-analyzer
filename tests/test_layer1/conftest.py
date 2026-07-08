"""Shared constants and helpers for L1 layer tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import DataContext
from src.layer1_feature import factory
from src.layer1_feature.macro_extractor import MacroExtractor
from src.layer1_feature.text_extractor import TextExtractor
from src.pipeline.input_profile import resolve_input_profile
from src.pipeline.stage_manager import StageManager

MODALITY_DIMS = {
    "text": 768,
    "speech": 1040,
    "macro": 512,
    "micro": 256,
}

FEATURE_KEYS = {
    "text": "text_embedding",
    "speech": "speech_feature",
    "micro": "micro_feature",
}


def text_config() -> dict:
    return {
        "text": {
            "enabled": True,
            "whisper_model": "base",
            "bert_model": "bert-base-uncased",
            "embedding_dim": MODALITY_DIMS["text"],
            "language": "auto",
        },
        "speech": {"sample_rate": 16000},
    }


def mock_text_extractor(monkeypatch) -> TextExtractor:
    extractor = TextExtractor(config=text_config())
    monkeypatch.setattr(
        extractor,
        "_embed_text",
        lambda text: np.ones(MODALITY_DIMS["text"], dtype=np.float32),
    )
    return extractor


def macro_config() -> dict:
    return {
        "roi_size": 32,
        "clip_frames": 4,
        "clip_stride": 2,
        "embedding_dim": MODALITY_DIMS["macro"],
        "backend": "simple",
        "align_faces": False,
    }


def mock_macro_extractor() -> MacroExtractor:
    return MacroExtractor(
        macro_config(),
        aligner=lambda frame: frame,
        encoder=lambda clip: np.arange(MODALITY_DIMS["macro"], dtype=np.float32),
    )


def context_for_profile(
    input_type: str,
    sample_config: dict,
    *,
    text_subtype: str | None = None,
    video_path: str | None = None,
    audio_path: str | None = None,
    text_content: str | None = None,
    image_path: str | None = None,
) -> DataContext:
    _, profile = resolve_input_profile(input_type, text_subtype)
    manager = StageManager(
        pipeline_config=sample_config["pipeline"],
        profile=profile,
    )
    return DataContext.create(
        user_id="l1-integration",
        input_type=input_type,
        video_path=video_path,
        audio_path=audio_path,
        text_content=text_content,
        text_subtype=text_subtype,
        image_path=image_path,
        profile_metadata=manager.to_metadata_patch(),
    )


@pytest.fixture
def mock_video_l1_extractors(monkeypatch):
    """Register mocked text/speech/macro; micro uses the real lightweight extractor."""
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
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "macro",
        lambda: mock_macro_extractor(),
    )

    def _apply_speech_mock():
        monkeypatch.setattr(
            "src.layer1_feature.speech_extractor.SpeechExtractor._wav2vec_frames",
            lambda self, waveform, sample_rate: np.ones(
                (49, 1024), dtype=np.float32
            ),
        )

    return _apply_speech_mock
