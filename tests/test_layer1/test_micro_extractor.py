"""Unit tests for L1 micro-expression feature extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core import DataContext
from src.layer1_feature import factory
from src.layer1_feature.dgm import DisplacementFieldGenerator
from src.layer1_feature.factory import run_l1
from src.layer1_feature.gcn import MultiViewMicroProjector
from src.layer1_feature.micro_extractor import MicroExpressionExtractor
from src.pipeline import run_pipeline


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "face.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    assert cv2.imwrite(str(path), image)
    return path


def _micro_config() -> dict:
    return {
        "micro": {
            "enabled": True,
            "input_size": 112,
            "embedding_dim": 256,
            "weak_signal_amplify": {
                "enabled": True,
                "au_coefficients": 2.5,
                "key_aus": ["AU12", "AU15"],
            },
            "preserve_raw_visual": True,
        }
    }


def _micro_extractor(**overrides) -> MicroExpressionExtractor:
    config = _micro_config()
    config["micro"].update(overrides)
    return MicroExpressionExtractor(config=config)


def test_displacement_field_shape():
    onset = np.zeros((112, 112), dtype=np.float32)
    apex = onset.copy()
    cv2.circle(apex, (56, 56), 10, 1.0, -1)

    flow = DisplacementFieldGenerator(size=112).compute(onset, apex)

    assert flow.shape == (112, 112, 2)
    assert flow.dtype == np.float32
    assert np.isfinite(flow).all()


def test_multiview_projector_returns_256_dims():
    flow = np.zeros((112, 112, 2), dtype=np.float32)
    flow[45:70, 40:72, 0] = 0.5
    flow[45:70, 40:72, 1] = -0.25

    feature = MultiViewMicroProjector().project(flow)

    assert feature.shape == (256,)
    assert feature.dtype == np.float32
    assert np.isfinite(feature).all()


def test_extract_video_output_schema(sample_video_path):
    extractor = _micro_extractor()
    context = DataContext.create(
        user_id="test",
        input_type="video",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["micro"]},
    )

    result = extractor.extract(context)
    micro_items = result["micro"]

    assert len(micro_items) == 1
    item = micro_items[0]
    assert item["micro_feature"].shape == (256,)
    assert item["micro_feature"].dtype == np.float32
    assert item["start_time"] <= item["end_time"]
    assert "stub" not in item

    raw = extractor.extract_raw_visual(context)["micro"]
    assert len(raw) == 1
    assert raw[0].shape == (256,)


def test_extract_returns_only_micro_modality(sample_video_path):
    context = DataContext.create(
        user_id="test",
        input_type="video",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["micro"]},
    )
    context.features["text"] = ["kept"]

    result = _micro_extractor().extract(context)

    assert set(result.keys()) == {"micro"}
    assert len(result["micro"]) == 1


def test_run_preserves_features_and_populates_visual_bypass(sample_video_path):
    context = DataContext.create(
        user_id="test",
        input_type="video",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["micro"]},
    )
    context.features["text"] = ["kept"]
    extractor = _micro_extractor()

    assert extractor.run(context) is context
    assert context.features["text"] == ["kept"]
    assert context.features["micro"][0]["micro_feature"].shape == (256,)
    assert context.raw_visual_features["micro"][0].shape == (256,)


def test_resolve_events_uses_input_type_not_stray_paths(sample_video_path, tmp_path):
    image_path = tmp_path / "face.png"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    extractor = _micro_extractor()

    video_context = DataContext.create(
        user_id="test",
        input_type="video",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["micro"]},
    )
    video_context.raw_data["image_path"] = image_path

    events, _fps = extractor._resolve_events(video_context)

    assert len(events) == 1
    assert events[0].onset_index >= 0


def test_extract_image_single_event(tmp_path):
    image_path = tmp_path / "face.png"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 24), (44, 42), (255, 255, 255), -1)
    cv2.imwrite(str(image_path), image)

    extractor = _micro_extractor()
    context = DataContext.create(
        user_id="test",
        input_type="image",
        image_path=image_path,
        profile_metadata={"active_modalities": ["micro"]},
    )

    item = extractor.extract(context)["micro"][0]

    assert item["micro_feature"].shape == (256,)
    assert item["start_time"] == 0.0
    assert item["end_time"] == 0.0


def test_run_l1_uses_registered_micro_extractor(sample_video_path):
    context = DataContext.create(
        user_id="test",
        input_type="video",
        video_path=sample_video_path,
        profile_metadata={"active_modalities": ["micro"]},
    )

    run_l1(context)

    assert context.metadata["stage_status"]["L1"] == "completed"
    assert context.features["micro"][0]["micro_feature"].shape == (256,)
    assert context.raw_visual_features["micro"][0].shape == (256,)


@pytest.mark.parametrize("input_type", ["video", "image"])
def test_run_pipeline_profiles_produce_micro_features(
    monkeypatch,
    input_type,
    sample_video_path,
    sample_image_path,
):
    extractor = _micro_extractor()
    monkeypatch.setitem(factory._EXTRACTOR_REGISTRY, "micro", lambda: extractor)
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "macro",
        lambda: factory.StubModalityExtractor("macro"),
    )
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "speech",
        lambda: factory.StubModalityExtractor("speech"),
    )
    monkeypatch.setitem(
        factory._EXTRACTOR_REGISTRY,
        "text",
        lambda: factory.StubModalityExtractor("text"),
    )
    kwargs = (
        {"video_path": sample_video_path}
        if input_type == "video"
        else {"image_path": sample_image_path}
    )

    result = run_pipeline(input_type, user_id="test", execute=True, **kwargs)

    assert result["stage_status"]["L1"] == "completed"
    micro_features = result["features"]["micro"]
    assert len(micro_features) == 1
    item = micro_features[0]
    assert item["micro_feature"].shape == (256,)
    assert item["micro_feature"].dtype == np.float32
    if input_type == "image":
        assert item["start_time"] == 0.0
        assert item["end_time"] == 0.0
