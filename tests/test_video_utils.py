"""Tests for video utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.video_utils import (
    extract_frame_clip,
    extract_frames_at_times,
    get_video_meta,
    read_frame,
    sample_frame_indices,
)


def test_get_video_meta(sample_video_path):
    meta = get_video_meta(sample_video_path)
    assert meta.frame_count == 20
    assert meta.width == 64
    assert meta.height == 48
    assert meta.fps == pytest.approx(10.0)
    assert meta.duration_sec == pytest.approx(2.0)


def test_read_frame_rgb(sample_video_path):
    frame = read_frame(sample_video_path, 5)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8


def test_extract_frames_at_times(sample_video_path):
    frames = extract_frames_at_times(sample_video_path, [0.0, 0.5, 1.0])
    assert len(frames) == 3
    assert all(isinstance(frame, np.ndarray) for frame, _ in frames)
    assert frames[0][1] == pytest.approx(0.0)
    assert frames[1][1] == pytest.approx(0.5)


def test_extract_frames_at_times_grayscale_micro_size(sample_video_path):
    frames = extract_frames_at_times(
        sample_video_path,
        [0.2],
        return_gray=True,
        target_size=(112, 112),
    )
    assert frames[0][0].shape == (112, 112)


def test_sample_frame_indices(sample_video_path):
    meta = get_video_meta(sample_video_path)
    indices = sample_frame_indices(meta.fps, meta.duration_sec, clip_frames=4)
    assert indices == [0, 1, 2, 3]


def test_extract_frame_clip(sample_video_path):
    clip = extract_frame_clip(sample_video_path, start_frame=2, num_frames=3)
    assert clip.shape == (3, 48, 64, 3)


def test_get_video_meta_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_video_meta(tmp_path / "missing.mp4")
