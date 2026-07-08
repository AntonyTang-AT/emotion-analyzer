"""Unit tests for L1 text feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core import DataContext
from src.layer1_feature import factory
from src.layer1_feature.factory import run_l1
from src.layer1_feature.macro_extractor import MacroExtractor
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


def _mock_embedding(value: float = 1.0) -> np.ndarray:
    return np.full(768, value, dtype=np.float32)


def _text_extractor(monkeypatch=None) -> TextExtractor:
    extractor = TextExtractor(config=_text_config())
    if monkeypatch is not None:
        monkeypatch.setattr(extractor, "_embed_text", lambda text: _mock_embedding())
    return extractor


def test_extract_text_content_descriptive(monkeypatch):
    extractor = _text_extractor(monkeypatch)
    context = DataContext.create(
        user_id="test",
        input_type="text",
        text_content="First paragraph.\n\nSecond paragraph.",
        text_subtype="descriptive",
        profile_metadata={"active_modalities": ["text"]},
    )

    result = extractor.extract(context)

    items = result["text"]
    assert len(items) == 2
    assert items[0]["text"] == "First paragraph."
    assert items[0]["text_embedding"].shape == (768,)
    assert items[0]["text_embedding"].dtype == np.float32
    assert items[0]["start_time"] == 0.0
    assert items[0]["end_time"] == 1.0
    assert "stub" not in items[0]


def test_extract_text_path_dialogue(monkeypatch, tmp_path: Path):
    path = tmp_path / "dialogue.txt"
    path.write_text("Alice: Hello there.\nBob: I am fine.", encoding="utf-8")
    extractor = _text_extractor(monkeypatch)
    context = DataContext.create(
        user_id="test",
        input_type="text",
        text_path=path,
        text_subtype="dialogue",
        profile_metadata={"active_modalities": ["text"]},
    )

    items = extractor.extract(context)["text"]

    assert [item["text"] for item in items] == ["Hello there.", "I am fine."]
    assert [item["start_time"] for item in items] == [0.0, 1.0]
    assert [item["end_time"] for item in items] == [1.0, 2.0]


def test_extract_audio_uses_whisper_segments(monkeypatch, sample_wav_path):
    extractor = _text_extractor(monkeypatch)
    monkeypatch.setattr(
        extractor,
        "_transcribe_audio",
        lambda source: {
            "segments": [
                {"text": "hello", "start": 0.25, "end": 0.75},
                {"text": "world", "start": 0.75, "end": 1.5},
            ]
        },
    )
    context = DataContext.create(
        user_id="test",
        input_type="audio",
        audio_path=sample_wav_path,
        profile_metadata={"active_modalities": ["text"]},
    )

    items = extractor.extract(context)["text"]

    assert len(items) == 2
    assert items[0]["text"] == "hello"
    assert items[0]["start_time"] == 0.25
    assert items[0]["end_time"] == 0.75
    assert items[0]["text_embedding"].shape == (768,)


def test_extract_video_routes_through_audio_extraction(monkeypatch, sample_video_path):
    extractor = _text_extractor(monkeypatch)
    waveform = np.zeros(16000, dtype=np.float32)
    seen: dict[str, object] = {}

    def fake_extract_audio(video_path, *, target_sr=None):
        seen["video_path"] = video_path
        seen["target_sr"] = target_sr
        return waveform

    monkeypatch.setattr(
        "src.layer1_feature.text_extractor.extract_audio_from_video",
        fake_extract_audio,
    )
    monkeypatch.setattr(
        extractor,
        "_transcribe_audio",
        lambda source: {"segments": [{"text": "from video", "start": 0.0, "end": 1.0}]},
    )
    context = DataContext.create(
        user_id="test",
        input_type="video",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["text"]},
    )

    items = extractor.extract(context)["text"]

    assert seen["video_path"] == str(sample_video_path)
    assert seen["target_sr"] == 16000
    assert items[0]["text"] == "from video"


def test_extract_returns_only_text_modality(monkeypatch, sample_wav_path):
    context = DataContext.create(
        user_id="test",
        input_type="audio",
        audio_path=sample_wav_path,
        profile_metadata={"active_modalities": ["text"]},
    )
    context.features["speech"] = ["kept"]
    extractor = _text_extractor(monkeypatch)
    monkeypatch.setattr(
        extractor,
        "_transcribe_audio",
        lambda source: {"segments": [{"text": "hi", "start": 0.0, "end": 1.0}]},
    )

    result = extractor.extract(context)

    assert set(result.keys()) == {"text"}
    assert len(result["text"]) == 1


def test_run_preserves_features_and_populates_text(monkeypatch):
    context = DataContext.create(
        user_id="test",
        input_type="text",
        text_content="hello",
        text_subtype="descriptive",
        profile_metadata={"active_modalities": ["text"]},
    )
    context.features["speech"] = ["kept"]
    extractor = _text_extractor(monkeypatch)

    assert extractor.run(context) is context
    assert context.features["speech"] == ["kept"]
    assert context.features["text"][0]["text_embedding"].shape == (768,)


def test_extract_skips_image_input():
    context = DataContext.create(
        user_id="test",
        input_type="image",
        image_path="dummy.png",
        profile_metadata={"active_modalities": ["text"]},
    )

    result = TextExtractor(config=_text_config()).extract(context)

    assert result == {"text": []}


def test_run_l1_uses_registered_text_extractor(monkeypatch):
    extractor = _text_extractor(monkeypatch)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)
    context = DataContext.create(
        user_id="test",
        input_type="text",
        text_content="hello",
        text_subtype="descriptive",
        profile_metadata={"active_modalities": ["text"]},
    )

    run_l1(context)

    assert context.metadata["stage_status"]["L1"] == "completed"
    assert context.features["text"][0]["text_embedding"].shape == (768,)
    assert "stub" not in context.features["text"][0]


def test_run_pipeline_text_profile_produces_text_features(monkeypatch):
    extractor = _text_extractor(monkeypatch)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    result = run_pipeline(
        input_type="text",
        text_content="Hello.\n\nWorld.",
        text_subtype="descriptive",
        user_id="test",
        execute=True,
    )

    assert result["input_profile"] == "text_descriptive"
    assert result["stage_status"]["L1"] == "completed"
    assert len(result["features"]["text"]) == 2
    assert result["features"]["text"][0]["text_embedding"].shape == (768,)
    assert "stub" not in result["features"]["text"][0]


def test_run_pipeline_video_profile_produces_text_features(
    monkeypatch,
    sample_video_path,
):
    extractor = _text_extractor(monkeypatch)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "speech",
        lambda: factory.StubModalityExtractor("speech"),
    )
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "macro",
        lambda: MacroExtractor(
            {
                "roi_size": 32,
                "clip_frames": 4,
                "clip_stride": 2,
                "embedding_dim": 512,
                "backend": "simple",
                "align_faces": False,
            },
            aligner=lambda frame: frame,
            encoder=lambda clip: np.arange(512, dtype=np.float32),
        ),
    )
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "micro",
        lambda: factory.StubModalityExtractor("micro"),
    )
    monkeypatch.setattr(
        extractor,
        "_transcribe_audio",
        lambda source: {"segments": [{"text": "video line", "start": 0.0, "end": 1.0}]},
    )
    monkeypatch.setattr(
        "src.layer1_feature.text_extractor.extract_audio_from_video",
        lambda video_path, target_sr=None: np.zeros(16000, dtype=np.float32),
    )

    result = run_pipeline(
        input_type="video",
        user_id="test",
        video_path=sample_video_path,
        execute=True,
    )

    assert result["stage_status"]["L1"] == "completed"
    assert result["features"]["text"][0]["text_embedding"].shape == (768,)
    assert result["features"]["text"][0]["text"] == "video line"
    assert "stub" not in result["features"]["text"][0]


@pytest.mark.slow
def test_embed_text_real_bert():
    extractor = TextExtractor(config=_text_config())
    embedding = extractor._embed_text("hello world")

    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
