"""Pipeline integration tests for multi-input routing."""

from __future__ import annotations

from src.pipeline import run_pipeline


def test_run_pipeline_text_minimal():
    result = run_pipeline(
        input_type="text",
        text_content="测试文本输入",
        text_subtype="dialogue",
    )
    assert result["input_profile"] == "text_dialogue"
    assert result["segmentation_mode"] == "utterance"
    assert result["features"]["text"][0]["stub"] is True


def test_run_pipeline_execute_false(sample_wav_path):
    result = run_pipeline(
        input_type="audio",
        audio_path=sample_wav_path,
        execute=False,
    )
    assert result["stage_status"]["L1"] == "pending"
    assert result["pipeline_complete"] is False
    assert not result["features"]
