"""Pipeline entry points for multi-input emotion analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.context import DataContext
from src.core.types import InputType, TextSubtype
from src.pipeline.io_handler import load_input, save_output
from src.pipeline.runner import PipelineRunner
from src.pipeline.stage_manager import StageManager
from src.utils.config_loader import load_config


def run_pipeline(
    input_type: str | InputType,
    *,
    user_id: str = "anonymous",
    video_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    text_content: str | None = None,
    text_path: str | Path | None = None,
    image_path: str | Path | None = None,
    text_subtype: str | TextSubtype | None = None,
    config_overrides: dict[str, Any] | None = None,
    save_output_flag: bool = False,
    output_dir: str | Path | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    """Load input, optionally run stub pipeline, and return result dictionary."""
    context = load_input(
        input_type,
        user_id=user_id,
        video_path=video_path,
        audio_path=audio_path,
        text_content=text_content,
        text_path=text_path,
        image_path=image_path,
        text_subtype=text_subtype,
        output_dir=output_dir,
        config_overrides=config_overrides,
    )

    if execute:
        pipeline_config = context.metadata.get("config", {}).get("pipeline")
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        profile_name = context.metadata.get("input_profile")
        from src.pipeline.input_profile import resolve_input_profile

        _, profile = resolve_input_profile(context.input_type, context.text_subtype)
        stage_manager = StageManager(
            pipeline_config=pipeline_config,
            profile=profile,
            profile_name=profile_name,
        )
        context = PipelineRunner(stage_manager).run(context)

    if save_output_flag and output_dir is not None:
        save_output(context, output_dir)

    return build_pipeline_result(context)


def run_pipeline_from_context(
    context: DataContext,
    *,
    execute: bool = True,
) -> dict[str, Any]:
    """Continue pipeline execution from an existing context."""
    if execute:
        pipeline_config = context.metadata.get("config", {}).get("pipeline")
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        from src.pipeline.input_profile import resolve_input_profile

        profile_name, profile = resolve_input_profile(
            context.input_type, context.text_subtype
        )
        stage_manager = StageManager(
            pipeline_config=pipeline_config,
            profile=profile,
            profile_name=profile_name or context.metadata.get("input_profile"),
        )
        context = PipelineRunner(stage_manager).run(context)
    return build_pipeline_result(context)


def build_pipeline_result(context: DataContext) -> dict[str, Any]:
    """Summarize context for API / CLI consumers."""
    return {
        "session_id": context.metadata.get("session_id"),
        "input_type": context.input_type,
        "text_subtype": context.text_subtype,
        "input_profile": context.metadata.get("input_profile"),
        "active_modalities": context.active_modalities,
        "enabled_stages": context.metadata.get("enabled_stages", []),
        "segmentation_mode": context.metadata.get("segmentation_mode"),
        "l4_enabled": context.metadata.get("l4_enabled"),
        "l6_enabled": context.metadata.get("l6_enabled"),
        "stage_status": dict(context.metadata.get("stage_status", {})),
        "skipped_stages": dict(context.metadata.get("skipped_stages", {})),
        "features": context.features,
        "raw_visual_features": context.raw_visual_features,
        "pipeline_complete": context.metadata.get("pipeline_complete", False),
        "context": context.to_dict(),
    }


__all__ = [
    "PipelineRunner",
    "build_pipeline_result",
    "load_input",
    "run_pipeline",
    "run_pipeline_from_context",
    "save_output",
]
