# L1 测试媒体（fixtures）

本目录用于存放可选的固定测试媒体文件。当前默认策略如下：

## 默认：动态生成（推荐）

根目录 [`tests/conftest.py`](../conftest.py) 在运行时生成：

- `sample_wav_path`：约 2.5 秒 16 kHz 正弦波
- `sample_video_path`：约 2 秒、低分辨率 MP4

L1 单元测试与集成测试优先使用上述 fixture，**无需提交大二进制文件**。

## 慢速测试（`@pytest.mark.slow`）

以下测试会下载或加载真实模型，默认不纳入 `pytest -m "not slow"`：

- `tests/test_layer1/test_speech_extractor.py`
- `tests/test_layer1/test_text_extractor.py`

本地全量验证：

```bash
pytest -m slow
```

## 未来可选

若 CI 需要完全固定的输入，可在此目录添加小于 1 MB 的 `sample.wav` / `sample.mp4`，并在对应测试中显式引用路径。
