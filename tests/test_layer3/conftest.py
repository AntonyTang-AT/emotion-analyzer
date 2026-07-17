"""Shared fixtures for L3 unit and pipeline tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.core import VAConfidence


@pytest.fixture
def mock_arousal_spike_va_sequence() -> list[VAConfidence]:
    return [
        VAConfidence(0.0, 0.10, 0.9),
        VAConfidence(0.0, 0.15, 0.9),
        VAConfidence(0.0, 0.80, 0.9),
    ]


@pytest.fixture
def run_l3_text_pipeline(monkeypatch: pytest.MonkeyPatch) -> Callable[[], dict[str, Any]]:
    def _run() -> dict[str, Any]:
        from src.layer1_feature import factory
        from src.pipeline import run_pipeline
        from tests.test_layer1.conftest import mock_text_extractor
        from tests.test_layer2.conftest import init_test_registry

        extractor = mock_text_extractor(monkeypatch)
        monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: extractor)
        monkeypatch.setattr(
            "src.layer2_predict.predictor.initialize_registry",
            lambda **kwargs: init_test_registry(active_modalities=("text",)),
        )
        return run_pipeline(
            input_type="text",
            user_id="l3-pipeline-test",
            text_content="test text input",
            text_subtype="dialogue",
            config_overrides={
                "pipeline": {
                    "stages": {
                        "L3": {
                            "baseline": {"enabled": False},
                            "cold_start": {"enabled": False},
                            "memory": {"enabled": False},
                        },
                        "L4": {"enabled": False},
                        "L5": {"enabled": False},
                        "L6": {"enabled": False},
                    }
                }
            },
        )

    return _run
