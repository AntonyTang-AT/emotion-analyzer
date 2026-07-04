"""Pipeline runner with profile-aware stage skipping."""

from __future__ import annotations

from typing import Any

from src.core.context import DataContext, VALID_STAGES
from src.layer1_feature.factory import run_l1
from src.pipeline.stage_manager import StageManager


class PipelineRunner:
    """Execute enabled pipeline stages; unimplemented layers are marked skipped."""

    def __init__(self, stage_manager: StageManager) -> None:
        self.stage_manager = stage_manager

    def run(self, context: DataContext) -> DataContext:
        skipped = context.metadata.setdefault("skipped_stages", {})
        for stage in VALID_STAGES:
            if not self.stage_manager.is_stage_enabled(stage):
                context.metadata.setdefault("stage_status", {})[stage] = "skipped"
                skipped[stage] = "disabled_by_profile"

        for stage in self.stage_manager.get_execution_order():
            if stage == "L1":
                context = run_l1(context)
            else:
                context.metadata.setdefault("stage_status", {})[stage] = "pending"
                skipped[stage] = "not_implemented"

        context.metadata["pipeline_complete"] = True
        return context
