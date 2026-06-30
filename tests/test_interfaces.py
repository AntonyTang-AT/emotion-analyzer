"""Smoke tests for pipeline layer abstract interfaces."""

from __future__ import annotations

import pytest

from src.core import (
    ContradictionDetector,
    ContradictionResult,
    ContradictionType,
    DataContext,
    FeatureDict,
    FeatureExtractor,
    LayerModule,
    PersonalityResult,
    ReportBundle,
    ReportGenerator,
    SegmentController,
    VAConfidence,
    VAPredictionLayer,
    VAPredictor,
    PersonalityRegressor,
    Fragment,
)


def test_layer_module_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LayerModule()  # type: ignore[abstract]


def test_feature_extractor_run_updates_l1(sample_data_context):
    class DummyExtractor(FeatureExtractor):
        def extract(self, context: DataContext) -> FeatureDict:
            return {"text": [{"dim": 768}]}

    ctx = DummyExtractor().run(sample_data_context)
    assert ctx.metadata["stage_status"]["L1"] == "completed"
    assert "text" in ctx.features
    assert ctx.get_stage("L1").keys() == {"features", "raw_visual_features"}


def test_va_prediction_layer_run_updates_l2(sample_data_context):
    class DummyLayer(VAPredictionLayer):
        def predict_all(self, context: DataContext):
            va = VAConfidence(0.0, 0.0, 1.0)
            return {"text": va}, {"text": va}

    ctx = DummyLayer().run(sample_data_context)
    assert ctx.metadata["stage_status"]["L2"] == "completed"


def test_contradiction_detector_run(sample_data_context, sample_contradiction):
    class DummyDetector(ContradictionDetector):
        def detect(self, context: DataContext) -> ContradictionResult:
            return sample_contradiction

    ctx = DummyDetector().run(sample_data_context)
    assert ctx.contradiction.contradiction_type == ContradictionType.MASKING


def test_segment_controller_run(sample_data_context):
    class DummySegmenter(SegmentController):
        def segment(self, context: DataContext):
            return [
                Fragment(id="s1", start_time=0.0, end_time=1.0),
            ]

    ctx = DummySegmenter().run(sample_data_context)
    assert len(ctx.segments) == 1
    assert ctx.get_stage("L3").keys() == {"segments", "memory_retrieved"}


def test_report_generator_run(sample_data_context):
    class DummyReporter(ReportGenerator):
        def generate(self, context: DataContext):
            return ReportBundle(segment_reports=["ok"])

    ctx = DummyReporter().run(sample_data_context)
    assert ctx.reports.segment_reports == ["ok"]


def test_personality_regressor_run(sample_data_context):
    class DummyRegressor(PersonalityRegressor):
        def predict(self, context: DataContext):
            return PersonalityResult(5, 5, 5, 5, 5)

    ctx = DummyRegressor().run(sample_data_context)
    assert ctx.personality.E == pytest.approx(5.0)


def test_va_predictor_requires_both_branches():
    class IncompletePredictor(VAPredictor):
        modality = "text"

        def predict_self(self, features):
            return VAConfidence(0, 0, 1)

    with pytest.raises(TypeError):
        IncompletePredictor()  # type: ignore[abstract]
