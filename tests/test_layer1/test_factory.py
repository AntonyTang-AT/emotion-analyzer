"""Unit tests for L1 feature extractor factory."""

from __future__ import annotations

import pytest

from src.core import DataContext
from src.layer1_feature import factory
from src.layer1_feature.factory import (
    FeatureExtractorFactory,
    StubModalityExtractor,
    get_extractors_from_config,
    get_extractors_for_context,
    run_l1,
)
from src.layer1_feature.macro_extractor import MacroExtractor
from src.layer1_feature.micro_extractor import MicroExpressionExtractor
from src.layer1_feature.speech_extractor import SpeechExtractor
from src.layer1_feature.text_extractor import TextExtractor
from src.pipeline.input_profile import resolve_input_profile
from src.pipeline.stage_manager import StageManager


def test_factory_register_and_create():
    name = "_test_stub_modality"

    class _TmpExtractor(StubModalityExtractor):
        def __init__(self) -> None:
            super().__init__(name)

    FeatureExtractorFactory.register(name, _TmpExtractor)
    try:
        extractor = FeatureExtractorFactory.create(name)
        assert isinstance(extractor, StubModalityExtractor)
        assert extractor.modality == name
    finally:
        factory._EXTRACTOR_REGISTRY.pop(name, None)


def test_registered_defaults_four_modalities():
    names = set(FeatureExtractorFactory.registered_names())
    assert names >= {"text", "speech", "macro", "micro"}


@pytest.mark.parametrize(
    ("input_type", "text_subtype", "expected_modalities"),
    [
        ("video", None, ["text", "speech", "macro", "micro"]),
        ("text", "descriptive", ["text"]),
        ("image", None, ["macro", "micro"]),
    ],
)
def test_get_extractors_from_profile_modalities(
    input_type,
    text_subtype,
    expected_modalities,
    sample_config,
):
    _, profile = resolve_input_profile(input_type, text_subtype)
    manager = StageManager(
        pipeline_config=sample_config["pipeline"],
        profile=profile,
    )
    context = DataContext.create(
        user_id="factory-test",
        input_type=input_type,
        text_content="hello" if input_type == "text" else None,
        text_subtype=text_subtype,
        profile_metadata=manager.to_metadata_patch(),
    )
    extractors = get_extractors_for_context(context)
    assert [type(e).__name__ for e in extractors] == [
        {
            "text": "TextExtractor",
            "speech": "SpeechExtractor",
            "macro": "MacroExtractor",
            "micro": "MicroExpressionExtractor",
        }[name]
        for name in expected_modalities
    ]


def test_get_extractors_from_config_pipeline_fallback(sample_config):
    extractors = get_extractors_from_config(config=sample_config["pipeline"])
    assert len(extractors) == 4
    assert isinstance(extractors[0], TextExtractor)
    assert isinstance(extractors[1], SpeechExtractor)
    assert isinstance(extractors[2], MacroExtractor)
    assert isinstance(extractors[3], MicroExpressionExtractor)


def test_run_l1_skips_failed_extractor(monkeypatch):
    failing = StubModalityExtractor("text")

    def _boom(_context):
        raise RuntimeError("simulated extractor failure")

    monkeypatch.setattr(failing, "extract", _boom)
    ok = StubModalityExtractor("speech", feature_dim=16)

    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "text", lambda: failing)
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "speech", lambda: ok)

    context = DataContext.create(
        user_id="skip-test",
        input_type="audio",
        audio_path="data/raw/test.wav",
        profile_metadata={"active_modalities": ["text", "speech"]},
    )
    result = run_l1(context)

    assert "text" not in result.features
    assert "speech" in result.features
    assert result.features["speech"][0]["stub"] is True


def test_run_l1_merges_raw_visual():
    context = DataContext.create(
        user_id="raw-visual-test",
        input_type="image",
        image_path="data/raw/test.png",
        profile_metadata={"active_modalities": ["macro", "micro"]},
    )
    macro = StubModalityExtractor("macro", feature_dim=4)
    micro = StubModalityExtractor("micro", feature_dim=6)

    factory._EXTRACTOR_REGISTRY["macro"] = lambda: macro
    factory._EXTRACTOR_REGISTRY["micro"] = lambda: micro
    try:
        result = run_l1(context)
        assert "macro" in result.raw_visual_features
        assert "micro" in result.raw_visual_features
        assert len(result.raw_visual_features["macro"]) == 4
        assert len(result.raw_visual_features["micro"]) == 6
    finally:
        FeatureExtractorFactory.register_defaults()
