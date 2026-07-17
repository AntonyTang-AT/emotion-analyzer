# L3 Test Gap Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete GitHub Issue #54 with a minimal shared L3 fixture layer and an isolated L1 stub → L2 → L3 pipeline integration test.

**Architecture:** Keep the five existing L3 test modules intact except for one segmentation test that consumes a shared VA-sequence fixture. Put reusable L3-only setup in `tests/test_layer3/conftest.py`, and add one pipeline test that uses lightweight L1/L2 test doubles while disabling L3 memory and all downstream stages.

**Tech Stack:** Python 3.10+, pytest, NumPy, PyTorch, existing `DataContext`, `run_pipeline`, L1 extractor registry, and L2 predictor registry.

## Global Constraints

- Do not modify production code unless a new deterministic test exposes a real production defect and the root cause is documented first.
- Do not add dependencies or download models.
- All fixture-owned files must live under pytest `tmp_path`; the pipeline case must not create a persistent Chroma store.
- Preserve the existing L3 tests and the existing `tests/test_layer2/test_run_l2_pipeline.py` L3 smoke case.
- The implementation agent writes tests and fixtures; an independent verification agent performs the final review and acceptance runs.
- The default local Python at `D:\Anaconda\python.exe` currently lacks the declared `torch` dependency. A dependency-complete environment must run the pipeline test; the default environment may report that pre-existing collection blocker rather than masking it.

---

### Task 1: Add the shared L3 fixture layer

**Files:**
- Create: `tests/test_layer3/conftest.py`
- Modify: `tests/test_layer3/test_segment_controller.py`
- Test: `tests/test_layer3/test_segment_controller.py::test_arousal_spike_splits_dynamic`

**Interfaces:**
- Produces: pytest fixture `mock_arousal_spike_va_sequence() -> list[VAConfidence]`.
- Produces: pytest fixture `run_l3_text_pipeline(monkeypatch) -> Callable[[], dict[str, Any]]` for Task 2.
- Consumes: `tests.test_layer1.conftest.mock_text_extractor`, `tests.test_layer2.conftest.init_test_registry`, `src.layer1_feature.factory._EXTRACTOR_REGISTRY`, and public `src.pipeline.run_pipeline`.

- [ ] **Step 1: Record the green characterization baseline**

Run:

```powershell
pytest tests/test_layer3/test_segment_controller.py::test_arousal_spike_splits_dynamic -q
```

Expected: `1 passed`. This is a test-organization task, so preserve behavior rather than inventing a failing production test.

- [ ] **Step 2: Create the shared fixtures**

Create `tests/test_layer3/conftest.py` with this behavior:

```python
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
```

Keep all imports that require PyTorch inside `_run`; collecting the other L3 unit tests must remain possible in the current lightweight environment.

- [ ] **Step 3: Consume the shared VA sequence in the existing segmentation case**

Change only `test_arousal_spike_splits_dynamic`:

```python
def test_arousal_spike_splits_dynamic(mock_arousal_spike_va_sequence):
    frames = build_timeline(
        _context_with_series(
            inter_series={"text": mock_arousal_spike_va_sequence},
        )
    )
```

Leave the configuration and assertions unchanged.

- [ ] **Step 4: Verify the fixture refactor**

Run:

```powershell
pytest tests/test_layer3/test_segment_controller.py -q
pytest tests/test_layer3/ -m "not slow" -q
```

Expected in the default environment before Task 2: all existing L3 tests pass, with 44 passed when `chromadb` is installed.

- [ ] **Step 5: Commit Task 1**

```powershell
git add tests/test_layer3/conftest.py tests/test_layer3/test_segment_controller.py
git commit -m "test(L3): add shared layer3 fixtures"
```

---

### Task 2: Add the isolated L1 → L2 → L3 pipeline test

**Files:**
- Create: `tests/test_layer3/test_run_l3_pipeline.py`
- Test: `tests/test_layer3/test_run_l3_pipeline.py`

**Interfaces:**
- Consumes: `run_l3_text_pipeline() -> dict[str, Any]` from Task 1.
- Verifies: `stage_status` for L1/L2/L3, serialized `context.segments`, stable segment IDs and VA dictionaries, empty `memory_retrieved`, and disabled downstream stages.

- [ ] **Step 1: Add the dependency-gated pipeline test**

Create `tests/test_layer3/test_run_l3_pipeline.py`:

```python
"""L1 stub → L2 → L3 pipeline integration coverage for Issue #54."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="L2 predictors require the declared torch dependency")


def test_text_pipeline_produces_l3_segments_without_persistent_memory(
    run_l3_text_pipeline,
):
    result = run_l3_text_pipeline()

    assert result["stage_status"]["L1"] == "completed"
    assert result["stage_status"]["L2"] == "completed"
    assert result["stage_status"]["L3"] == "completed"
    assert result["stage_status"]["L4"] == "skipped"
    assert result["stage_status"]["L5"] == "skipped"
    assert result["stage_status"]["L6"] == "skipped"

    context = result["context"]
    assert context["memory_retrieved"] == []
    assert len(context["segments"]) >= 1
    first_segment = context["segments"][0]
    assert first_segment["id"].startswith("seg-")
    assert set(first_segment["va_self"]) == {"text"}
    assert set(first_segment["va_inter"]) == {"text"}
    assert set(first_segment["va_self"]["text"]) == {
        "valence",
        "arousal",
        "confidence",
    }
```

- [ ] **Step 2: Run the new test in a dependency-complete environment**

Run:

```powershell
python -m pytest tests/test_layer3/test_run_l3_pipeline.py -q
```

Expected with PyTorch installed: `1 passed`. Expected in the current default environment: `1 skipped` with the explicit PyTorch reason; do not report that skip as proof that the pipeline behavior passed.

- [ ] **Step 3: Run the L3 acceptance suite**

Run:

```powershell
python -m pytest tests/test_layer3/ -m "not slow" -q
```

Expected with all declared dependencies installed: 45 tests pass. In the current default environment, 44 tests pass and the pipeline module is explicitly skipped for missing PyTorch.

- [ ] **Step 4: Run the repository non-slow suite**

Run:

```powershell
python -m pytest -m "not slow" -q
```

Expected in a dependency-complete environment: exit code 0 with no failures. The current default environment is known to fail while collecting the pre-existing `tests/test_layer2/conftest.py` because `torch` is missing; record that exact environmental blocker if no dependency-complete interpreter is available.

- [ ] **Step 5: Review scope and commit Task 2**

Confirm `git diff --stat main...HEAD` contains only the approved design/plan and test files, then commit:

```powershell
git add tests/test_layer3/test_run_l3_pipeline.py
git commit -m "test(L3): cover lightweight pipeline integration"
```

---

### Task 3: Independent acceptance review

**Files:**
- Review: `docs/superpowers/specs/2026-07-17-l3-test-gap-coverage-design.md`
- Review: `tests/test_layer3/conftest.py`
- Review: `tests/test_layer3/test_segment_controller.py`
- Review: `tests/test_layer3/test_run_l3_pipeline.py`

**Interfaces:**
- Consumes: the complete branch diff from `main` to `HEAD` and the implementation agent's test report.
- Produces: a verification report separating code defects from missing-environment blockers.

- [ ] **Step 1: Check Issue #54 coverage and scope**

Verify that existing tests still explicitly cover arousal spike, polarity flip, max length, baseline delta/application, cold-start Top-K blending, memory add/query, embedding type switching, time decay, and `run_l3`. Verify the new pipeline test covers the public pipeline handoff without persistent memory or model downloads.

- [ ] **Step 2: Inspect fixture isolation**

Confirm fixtures are function-scoped, `monkeypatch` restores registries, no test writes outside `tmp_path`, and importing non-pipeline L3 tests does not import PyTorch.

- [ ] **Step 3: Run fresh verification**

Run:

```powershell
python -m pytest tests/test_layer3/ -m "not slow" -q
python -m pytest -m "not slow" -q
```

Report exact pass/fail/skip counts and full blocker text. Do not treat a dependency skip as pipeline execution evidence.

- [ ] **Step 4: Report verdict**

Return one of: approved; changes required with file/line findings; or code approved but acceptance blocked by the missing declared PyTorch dependency. Do not modify files during this review task.
