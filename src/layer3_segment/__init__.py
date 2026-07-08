"""L3 adaptive segmentation and personalization."""

from src.layer3_segment.baseline_calibrator import (
    BaselineConfig,
    apply_baseline,
    apply_baseline_to_fragment,
    calibrate_from_responses,
    compute_delta_va,
    load_baseline,
    load_population_baseline,
    run_calibration_session,
    save_baseline,
    va_self_to_vector,
    vector_to_va_self,
)
from src.layer3_segment.segment_controller import (
    DynamicSegmentController,
    SegmentationConfig,
    TimelineFrame,
    build_timeline,
    segment_dynamic,
    segment_from_context,
    segment_single,
    segment_utterance,
)

__all__ = [
    "BaselineConfig",
    "DynamicSegmentController",
    "SegmentationConfig",
    "TimelineFrame",
    "apply_baseline",
    "apply_baseline_to_fragment",
    "build_timeline",
    "calibrate_from_responses",
    "compute_delta_va",
    "load_baseline",
    "load_population_baseline",
    "run_calibration_session",
    "save_baseline",
    "segment_dynamic",
    "segment_from_context",
    "segment_single",
    "segment_utterance",
    "va_self_to_vector",
    "vector_to_va_self",
]
