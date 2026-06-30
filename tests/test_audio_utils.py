"""Tests for audio utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.audio_utils import (
    align_timestamps,
    build_second_grid,
    frame_audio,
    get_audio_duration,
    get_audio_sample_rate,
    load_audio,
)


def test_load_audio_resamples_to_16k(sample_wav_path):
    waveform, sample_rate = load_audio(sample_wav_path)
    assert sample_rate == 16000
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert len(waveform) > 0


def test_get_audio_duration_and_sample_rate(sample_wav_path):
    duration = get_audio_duration(sample_wav_path)
    sample_rate = get_audio_sample_rate(sample_wav_path)
    assert sample_rate == 16000
    assert 2.4 <= duration <= 2.6


def test_frame_audio_one_second_windows(sample_waveform):
    waveform, sample_rate = sample_waveform
    frames = frame_audio(waveform, sample_rate, frame_length_sec=1.0, hop_length_sec=1.0)
    assert len(frames) == 2
    assert frames[0][1] == 0.0
    assert frames[1][1] == 1.0
    assert all(chunk.shape[0] == sample_rate for chunk, _ in frames)


def test_build_second_grid():
    grid = build_second_grid(2.5, step=1.0)
    np.testing.assert_allclose(grid, [0.0, 1.0, 2.0])


def test_align_timestamps_clamps_requested_times():
    result = align_timestamps(10.0, 9.5, times_sec=[0.5, 12.0], tolerance_sec=0.1)
    assert result["aligned_duration"] == 9.5
    assert result["within_tolerance"] is False
    assert result["aligned_times"] == [0.5, 9.5]


def test_load_audio_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")
