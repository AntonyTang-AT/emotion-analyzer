# L3 测试缺口补齐设计

## 背景与目标

GitHub Issue #54 要求在 `tests/test_layer3/` 覆盖动态分段、基线校准、冷启动、长期记忆与 `run_l3` 端到端行为。当前 `main` 已包含五个 L3 测试文件，`pytest tests/test_layer3/ -m "not slow"` 基线结果为 44 个测试通过。

本次工作采用最小补缺策略：保留现有有效测试，只补齐共享 fixture 和 L3 pipeline 联调的结构性缺口，并避免为了目录形式而重写已通过的用例。

## 范围

### 包含

- 新建 `tests/test_layer3/conftest.py`，集中提供确实被多个 L3 测试使用的轻量 fixture。
- 共享 fixture 包括模拟 VA 数据、带 L2 输出的 `DataContext`、临时基线/相似用户数据和临时 Chroma 路径；具体只提取实施时确认存在复用价值的部分。
- 新建 `tests/test_layer3/test_run_l3_pipeline.py`，覆盖 L1 stub 经 L2 到 L3 的 pipeline 联调。
- 联调测试禁用 L3 持久记忆，复用轻量 extractor/registry 替身，不下载大模型，也不依赖外部服务。
- 对照 Issue #54 的验收项目检查现有测试覆盖；只有在验收行为尚未被覆盖时才增加新用例。

### 不包含

- 不整体重写、重命名或搬迁已有五个 L3 测试文件。
- 不删除 `tests/test_layer2/test_run_l2_pipeline.py` 中现有的 L3 冒烟覆盖，以免扩大改动范围。
- 不修改生产代码，除非新增的有效回归测试稳定暴露真实缺陷；若发生这种情况，先记录根因，再以最小修复处理。
- 不引入新的运行时或测试依赖。

## 测试结构

`tests/test_layer3/conftest.py` 负责 L3 测试输入与临时存储的公共构造，测试文件继续负责各模块特有的场景和断言。fixture 保持函数级作用域，确保基线 JSON、用户库和 Chroma 数据在测试间隔离。

`tests/test_layer3/test_run_l3_pipeline.py` 从公开的 `run_pipeline` 入口发起调用，通过替换 L1 extractor 与 L2 predictor registry 构造确定性输入。断言至少包括：L1、L2、L3 状态完成，L3 产生非空 `segments`，以及片段包含预期的稳定标识和 VA 数据。配置显式关闭 memory，避免产生持久化副作用。

## 错误与隔离策略

- fixture 写入仅使用 pytest 的 `tmp_path`。
- Chroma 测试继续通过 `pytest.importorskip("chromadb")` 兼容未安装可选依赖的环境。
- pipeline 测试通过 `monkeypatch` 恢复所有 registry 与 extractor 替换。
- 测试不得访问网络、下载模型或写入仓库内的运行时数据目录。

## 验收

必须执行并通过：

```powershell
pytest tests/test_layer3/ -m "not slow"
pytest -m "not slow"
```

独立验证代理还需检查：

- Issue #54 中分段、基线、冷启动、记忆和 `run_l3` 主路径均有明确覆盖。
- 新 fixture 没有隐藏跨测试状态或引入过宽作用域。
- pipeline 联调不依赖大模型、网络和持久化记忆库。
- 变更未包含与 Issue 无关的生产代码或测试重构。

## 代理职责分离

- 实现代理：按测试驱动方式创建 fixture 与缺口测试，运行 L3 定向测试并自审差异。
- 验证代理：不参与实现，独立阅读 Issue、设计、差异和测试输出，运行定向与全量非 slow 测试并报告发现。
