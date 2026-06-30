"""Shared pytest fixtures for emotion-analyzer tests.

Layer-specific tests (test_layer1/, etc.) should reuse ``sample_data_context``
and ``config_manager`` from this module.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf

from src.core import (
    ContradictionResult,
    ContradictionType,
    DataContext,
    Fragment,
    MemoryHit,
    PersonalityResult,
    ReportBundle,
    VAConfidence,
)
from src.utils.config_loader import ConfigManager, get_project_root


@pytest.fixture
def project_root() -> Path:
    return get_project_root()


@pytest.fixture
def isolated_env(monkeypatch):
    """Remove environment variables that override configuration."""
    for key in ("LOG_LEVEL", "DB_PATH", "DEVICE", "EMOTION_ROOT"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def config_manager(isolated_env):
    """Provide a fresh ConfigManager singleton for each test."""
    ConfigManager._instance = None
    manager = ConfigManager()
    yield manager
    ConfigManager._instance = None


@pytest.fixture
def sample_config(config_manager):
    """Load all project YAML configuration files."""
    return config_manager.load_all()


@pytest.fixture
def sample_data_context() -> DataContext:
    ctx = DataContext.create(
        user_id="test-user",
        video_path="data/raw/test.mp4",
        audio_path="data/raw/test.wav",
    )
    ctx.set_stage(
        "L2",
        {
            "va_self_predictions": {"text": VAConfidence(0.1, 0.2, 0.9)},
            "va_inter_predictions": {"text": VAConfidence(0.3, 0.4, 0.8)},
        },
    )
    return ctx


@pytest.fixture
def sample_contradiction() -> ContradictionResult:
    return ContradictionResult(
        contradiction_type=ContradictionType.MASKING,
        contradiction_intensity=0.72,
        involved_modalities=["text", "micro"],
        suggested_fusion_weights=[0.1, 0.2, 0.2, 0.5],
        routing_confidence=0.85,
    )


@pytest.fixture
def full_data_context(sample_contradiction) -> DataContext:
    ctx = DataContext.create(user_id="full-user", video_path="data/raw/full.mp4")
    ctx.set_stage(
        "L1",
        {
            "features": {"text": [{"embedding_dim": 768}]},
            "raw_visual_features": {"micro": np.array([1.0, 2.0, 3.0])},
        },
    )
    ctx.set_stage(
        "L2",
        {
            "va_self_predictions": {"text": VAConfidence(0.1, 0.2, 0.9)},
            "va_inter_predictions": {"text": VAConfidence(0.3, 0.4, 0.8)},
        },
    )
    fragment = Fragment(
        id="seg-1",
        start_time=0.0,
        end_time=5.0,
        va_self={"text": VAConfidence(0.1, 0.2, 0.9)},
        va_inter={"text": VAConfidence(0.3, 0.4, 0.8)},
        contradiction=sample_contradiction,
    )
    ctx.set_stage(
        "L3",
        {
            "segments": [fragment],
            "memory_retrieved": [MemoryHit("hist-1", 0.91, {"label": "happy"})],
        },
    )
    ctx.set_stage("L4", {"contradiction": sample_contradiction})
    ctx.set_stage(
        "L5",
        {"reports": ReportBundle(segment_reports=["seg"], overall_report="overall")},
    )
    ctx.set_stage(
        "L6",
        {
            "personality": PersonalityResult(
                7.0, 6.0, 5.0, 8.0, 3.0, [0.7] * 5, "evidence"
            )
        },
    )
    return ctx


@pytest.fixture
def temp_config_root(tmp_path, project_root, isolated_env, monkeypatch):
    """Copy project config/ into a temporary EMOTION_ROOT for mutation tests."""
    config_dst = tmp_path / "config"
    shutil.copytree(project_root / "config", config_dst)
    monkeypatch.setenv("EMOTION_ROOT", str(tmp_path))
    ConfigManager._instance = None
    yield tmp_path
    ConfigManager._instance = None


@pytest.fixture
def reset_loggers():
    """Clear cached loggers between tests."""
    import src.utils.logger as logger_module

    logger_module._LOGGERS.clear()
    yield
    logger_module._LOGGERS.clear()


@pytest.fixture
def sample_waveform() -> tuple[np.ndarray, int]:
    sample_rate = 16000
    duration_sec = 2.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return waveform, sample_rate


@pytest.fixture
def sample_wav_path(tmp_path: Path, sample_waveform) -> Path:
    waveform, sample_rate = sample_waveform
    path = tmp_path / "sample.wav"
    sf.write(path, waveform, sample_rate)
    return path


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    width, height, fps = 64, 48, 10.0
    frame_count = 20
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable on this platform")

    for index in range(frame_count):
        frame = np.full((height, width, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
