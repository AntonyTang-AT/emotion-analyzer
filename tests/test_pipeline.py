"""Pipeline integration tests for multi-input routing."""

from __future__ import annotations

import numpy as np

from src.layer1_feature import factory
from src.layer1_feature.text_extractor import TextExtractor
from src.pipeline import run_pipeline


def _text_config() -> dict:
    return {
        "text": {
            "enabled": True,
            "whisper_model": "base",
            "bert_model": "bert-base-uncased",
            "embedding_dim": 768,
            "language": "auto",
        },
        "speech": {"sample_rate": 16000},
    }


def test_run_pipeline_text_minimal(monkeypatch):
    extractor = TextExtractor(config=_text_config())
    monkeypatch.setattr(
        extractor,
        "_embed_text",
        lambda text: np.ones(768, dtype=np.float32),
    )
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    result = run_pipeline(
        input_type="text",
        text_content="test text input",
        text_subtype="dialogue",
    )
    assert result["input_profile"] == "text_dialogue"
    assert result["segmentation_mode"] == "utterance"
    assert result["features"]["text"][0]["text_embedding"].shape == (768,)
    assert "stub" not in result["features"]["text"][0]


def test_run_pipeline_execute_false(sample_wav_path):
    result = run_pipeline(
        input_type="audio",
        audio_path=sample_wav_path,
        execute=False,
    )
    assert result["stage_status"]["L1"] == "pending"
    assert result["pipeline_complete"] is False
    assert not result["features"]