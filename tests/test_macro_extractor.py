import numpy as np
import pytest
from src.core import DataContext
from src.layer1_feature.macro_extractor import MacroExpressionExtractor

def config(**extra):
    value={"roi_size":32,"clip_frames":4,"clip_stride":2,"embedding_dim":512,"backend":"simple","align_faces":False}
    value.update(extra); return value

def test_returns_second_aligned_512d_features(sample_video_path):
    result=MacroExpressionExtractor(config()).extract_video(sample_video_path)
    assert [stamp for _,stamp in result]==[0.,1.]
    assert all(feature.shape==(512,) and feature.dtype==np.float32 for feature,_ in result)

def test_run_integrates_macro_and_preserves_features(sample_video_path):
    context=DataContext.create(user_id="test",video_path=sample_video_path)
    context.features["text"]=["kept"]
    extractor=MacroExpressionExtractor(config(),aligner=lambda frame:frame,encoder=lambda clip:np.arange(512,dtype=np.float32))
    assert extractor.run(context) is context
    assert context.features["text"]==["kept"]
    assert len(context.features["macro"])==2
    assert context.raw_visual_features["macro"] is context.features["macro"]

def test_rejects_invalid_window_config():
    with pytest.raises(ValueError,match="positive"):
        MacroExpressionExtractor(config(clip_stride=0))
