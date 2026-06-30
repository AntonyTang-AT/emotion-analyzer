"""Audio loading, resampling, and framing utilities for L1 extractors."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np

from .config_loader import load_config
from .logger import get_logger

logger = get_logger(__name__)


def _default_sample_rate() -> int:
    return int(load_config("features")["speech"]["sample_rate"])


def get_audio_sample_rate(path: str | Path) -> int:
    """Return the native sample rate of an audio file without full decoding."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return int(librosa.get_samplerate(path))


def get_audio_duration(path: str | Path) -> float:
    """Return audio duration in seconds."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return float(librosa.get_duration(path=path))


def load_audio(
    path: str | Path,
    *,
    target_sr: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 waveform resampled to ``target_sr``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sample_rate = target_sr if target_sr is not None else _default_sample_rate()
    waveform, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return waveform.astype(np.float32), int(sr)


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    target_sr: int | None = None,
) -> np.ndarray | Path:
    """Extract audio track from a video file using ffmpeg when available."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    sample_rate = target_sr if target_sr is not None else _default_sample_rate()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg_extract(video_path, output_path, sample_rate)
        return output_path

    temp_path = video_path.with_suffix(".extracted.wav")
    try:
        _run_ffmpeg_extract(video_path, temp_path, sample_rate)
        waveform, _ = load_audio(temp_path, target_sr=sample_rate)
        return waveform
    finally:
        if temp_path.is_file():
            temp_path.unlink()


def _run_ffmpeg_extract(
    video_path: Path, output_path: Path, sample_rate: int
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to extract audio from video. "
            "Install ffmpeg and ensure it is on PATH."
        )

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}: {result.stderr.strip()}"
        )


def frame_audio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_length_sec: float = 1.0,
    hop_length_sec: float = 1.0,
) -> list[tuple[np.ndarray, float]]:
    """Split waveform into fixed-length frames with start timestamps in seconds."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_length_sec <= 0 or hop_length_sec <= 0:
        raise ValueError("frame_length_sec and hop_length_sec must be positive")

    frame_length = max(1, int(round(frame_length_sec * sample_rate)))
    hop_length = max(1, int(round(hop_length_sec * sample_rate)))

    frames: list[tuple[np.ndarray, float]] = []
    if len(waveform) == 0:
        return frames

    for start in range(0, len(waveform), hop_length):
        end = start + frame_length
        if end > len(waveform):
            break
        chunk = waveform[start:end]
        frames.append((chunk.astype(np.float32), start / sample_rate))

    return frames


def build_second_grid(duration_sec: float, step: float = 1.0) -> np.ndarray:
    """Build a second-level timestamp grid aligned with L3 VA granularity."""
    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")
    if step <= 0:
        raise ValueError("step must be positive")
    if duration_sec == 0:
        return np.array([0.0], dtype=np.float64)
    return np.arange(0.0, duration_sec, step, dtype=np.float64)


def align_timestamps(
    video_duration: float,
    audio_duration: float,
    *,
    times_sec: Sequence[float] | None = None,
    tolerance_sec: float = 0.1,
) -> dict[str, float | list[float] | bool]:
    """Align audio/video durations and optionally clamp requested timestamps."""
    drift = abs(video_duration - audio_duration)
    if drift > tolerance_sec:
        logger.warning(
            "Audio/video duration drift %.3fs exceeds tolerance %.3fs",
            drift,
            tolerance_sec,
        )

    aligned_duration = min(video_duration, audio_duration)
    aligned_times: list[float] = []
    if times_sec is not None:
        aligned_times = [
            float(max(0.0, min(t, aligned_duration))) for t in times_sec
        ]

    return {
        "video_duration": float(video_duration),
        "audio_duration": float(audio_duration),
        "aligned_duration": float(aligned_duration),
        "drift_sec": float(drift),
        "within_tolerance": drift <= tolerance_sec,
        "aligned_times": aligned_times,
    }
