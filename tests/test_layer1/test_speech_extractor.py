"""Unit tests for L1 speech feature extractor."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.core import DataContext
from src.layer1_feature.speech_extractor import PROSODY_DIM, SpeechExtractor
from src.pipeline import run_pipeline


def _speech_config() -> dict:
    return {
        "speech": {
            "enabled": True,
            "wav2vec2_model": "facebook/wav2vec2-large-960h",
            "sample_rate": 16000,
            "embedding_dim": 1024,
            "fox": {"enabled": True, "gamma": 0.1},
            "prosody": {"enabled": True, "features": ["f0", "energy", "zcr", "mfcc"]},
        }
    }


def test_fox_pool_weights():
    extractor = SpeechExtractor(config=_speech_config())
    frame_feats = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    pooled = extractor.fox_pool(frame_feats)
    assert pooled.shape == (2,)
    assert not np.allclose(pooled, frame_feats.mean(axis=0))

    extractor_mean = SpeechExtractor(
        config={
            "speech": {
                **_speech_config()["speech"],
                "fox": {"enabled": False, "gamma": 0.1},
            }
        }
    )
    mean_pooled = extractor_mean.fox_pool(frame_feats)
    assert np.allclose(mean_pooled, frame_feats.mean(axis=0))


def test_prosody_vector_dim(sample_waveform):
    waveform, sample_rate = sample_waveform
    extractor = SpeechExtractor(config=_speech_config())
    chunk = waveform[:sample_rate]
    prosody = extractor.prosody_vector(chunk, sample_rate)
    assert prosody.shape == (PROSODY_DIM,)
    assert prosody.dtype == np.float32


@patch.object(SpeechExtractor, "_wav2vec_frames")
def test_extract_output_schema_mocked(mock_wav2vec, sample_wav_path):
    mock_wav2vec.return_value = np.ones((49, 1024), dtype=np.float32)
    extractor = SpeechExtractor(config=_speech_config())
    context = DataContext.create(
        user_id="test",
        input_type="audio",
        audio_path=sample_wav_path,
        profile_metadata={"active_modalities": ["speech"]},
    )
    result = extractor.extract(context)
    speech_items = result["speech"]
    assert len(speech_items) == 2
    for item in speech_items:
        assert "speech_feature" in item
        assert "timestamp" in item
        assert "stub" not in item
        assert item["speech_feature"].shape == (1040,)
        assert item["speech_feature"].dtype == np.float32
    assert speech_items[0]["timestamp"] == 0.0
    assert speech_items[1]["timestamp"] == 1.0


@pytest.mark.slow
def test_extract_real_model(sample_wav_path):
    extractor = SpeechExtractor(config=_speech_config())
    context = DataContext.create(
        user_id="test",
        input_type="audio",
        audio_path=sample_wav_path,
        profile_metadata={"active_modalities": ["speech"]},
    )
    result = extractor.extract(context)
    speech_items = result["speech"]
    assert len(speech_items) == 2
    for item in speech_items:
        feat = item["speech_feature"]
        assert feat.shape == (1040,)
        assert np.isfinite(feat).all()


@pytest.mark.slow
def test_run_pipeline_audio_l1_speech(sample_wav_path):
    result = run_pipeline(
        input_type="audio",
        audio_path=sample_wav_path,
        execute=True,
    )
    speech_items = result["features"]["speech"]
    assert len(speech_items) >= 1
    assert "stub" not in speech_items[0]
    assert speech_items[0]["speech_feature"].shape == (1040,)
