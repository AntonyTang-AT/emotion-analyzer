"""Tests for multi-input source profiles and routing."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.core import DataContext, InputType, TextSubtype
from src.layer1_feature import factory
from src.layer1_feature.speech_extractor import SpeechExtractor
from src.layer1_feature.text_extractor import TextExtractor
from src.pipeline import run_pipeline
from src.pipeline.input_profile import resolve_input_profile, resolve_profile_name
from src.pipeline.io_handler import InputValidationError, load_input
from src.pipeline.stage_manager import StageManager
from src.utils.config_loader import load_config


@pytest.mark.parametrize(
    ("input_type", "text_subtype", "expected_profile"),
    [
        ("video", None, "video_default"),
        ("audio", None, "audio_default"),
        ("text", "descriptive", "text_descriptive"),
        ("text", "dialogue", "text_dialogue"),
        ("text", None, "text_descriptive"),
        ("image", None, "image_default"),
    ],
)
def test_resolve_profile_name(input_type, text_subtype, expected_profile):
    assert resolve_profile_name(input_type, text_subtype) == expected_profile


def test_text_profile_l4_disabled():
    _, profile = resolve_input_profile("text", "descriptive")
    assert profile["l4"]["enabled"] is False
    assert profile["l1_extractors"] == ["text"]


def test_stage_manager_text_skips_l4_l6(sample_config):
    _, profile = resolve_input_profile("text", "dialogue")
    manager = StageManager(
        pipeline_config=sample_config["pipeline"],
        profile=profile,
        profile_name="text_dialogue",
    )
    assert manager.is_stage_enabled("L1") is True
    assert manager.is_stage_enabled("L4") is False
    assert manager.is_stage_enabled("L6") is False
    assert manager.get_segmentation_mode() == "utterance"
    assert "L4" not in manager.get_enabled_stages()


def test_stage_manager_video_full_pipeline(sample_config):
    _, profile = resolve_input_profile("video")
    manager = StageManager(
        pipeline_config=sample_config["pipeline"],
        profile=profile,
        profile_name="video_default",
    )
    assert manager.get_l1_extractors() == ["text", "speech", "macro", "micro"]
    assert manager.is_stage_enabled("L4") is True
    assert manager.is_stage_enabled("L6") is True


def test_data_context_create_text():
    ctx = DataContext.create(
        user_id="u1",
        input_type="text",
        text_content="hello world",
        text_subtype="dialogue",
        profile_metadata={"active_modalities": ["text"]},
    )
    assert ctx.input_type == "text"
    assert ctx.text_subtype == "dialogue"
    assert ctx.raw_data["text_content"] == "hello world"


def test_load_input_text_no_file():
    ctx = load_input("text", user_id="u1", text_content="sample text")
    assert ctx.metadata["input_profile"] == "text_descriptive"
    assert ctx.active_modalities == ["text"]
    assert ctx.metadata["l4_enabled"] is False


def test_load_input_video(sample_video_path):
    ctx = load_input("video", user_id="u1", video_path=sample_video_path)
    assert ctx.metadata["input_profile"] == "video_default"
    assert set(ctx.active_modalities) == {"text", "speech", "macro", "micro"}
    assert "media_info" in ctx.metadata


def test_load_input_audio(sample_wav_path):
    ctx = load_input("audio", user_id="u1", audio_path=sample_wav_path)
    assert ctx.metadata["input_profile"] == "audio_default"
    assert ctx.active_modalities == ["text", "speech"]


def test_load_input_missing_video_raises():
    with pytest.raises(InputValidationError, match="video_path"):
        load_input("video", video_path="missing.mp4")


def test_run_pipeline_text_stub(monkeypatch):
    extractor = TextExtractor()
    monkeypatch.setattr(
        extractor, "_embed_text", lambda text: np.ones(768, dtype=np.float32)
    )
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    result = run_pipeline(
        "text",
        user_id="test",
        text_content="I feel happy today.",
        text_subtype="descriptive",
    )
    assert result["input_type"] == "text"
    assert result["l4_enabled"] is False
    assert result["stage_status"]["L1"] == "completed"
    assert result["stage_status"]["L4"] == "skipped"
    assert "text" in result["features"]
    assert result["features"]["text"][0]["text_embedding"].shape == (768,)
    assert "stub" not in result["features"]["text"][0]
    assert result["pipeline_complete"] is True


@patch.object(SpeechExtractor, "_wav2vec_frames")
@patch.object(TextExtractor, "_transcribe_audio")
@patch("src.layer1_feature.text_extractor.extract_audio_from_video")
def test_run_pipeline_video_stub(
    mock_extract_audio,
    mock_transcribe,
    mock_wav2vec,
    sample_video_path,
    sample_wav_path,
    monkeypatch,
):
    mock_wav2vec.return_value = np.ones((49, 1024), dtype=np.float32)
    mock_extract_audio.return_value = np.zeros(16000, dtype=np.float32)
    mock_transcribe.return_value = {
        "segments": [{"text": "hello", "start": 0.0, "end": 1.0}]
    }
    extractor = TextExtractor()
    monkeypatch.setattr(
        extractor, "_embed_text", lambda text: np.ones(768, dtype=np.float32)
    )
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)

    result = run_pipeline(
        "video",
        user_id="test",
        video_path=sample_video_path,
        audio_path=sample_wav_path,
        execute=True,
    )
    assert result["l4_enabled"] is True
    assert result["stage_status"]["L1"] == "completed"
    assert len(result["features"]) == 4
    assert "stub" not in result["features"]["speech"][0]
    assert result["features"]["speech"][0]["speech_feature"].shape == (1040,)
    assert result["features"]["text"][0]["text_embedding"].shape == (768,)


def test_config_loader_includes_input_profiles(sample_config):
    assert "input_profiles" in sample_config
    assert "video_default" in sample_config["input_profiles"]["profiles"]


def test_load_config_input_profiles_directly():
    data = load_config("input_profiles")
    assert "profiles" in data
