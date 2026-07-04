"""Load raw inputs and initialize DataContext for the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.context import DataContext
from src.core.types import InputType, TextSubtype
from src.pipeline.input_profile import resolve_input_profile
from src.pipeline.stage_manager import StageManager
from src.utils.config_loader import load_config


class InputValidationError(ValueError):
    """Raised when required fields for an input_type are missing or invalid."""


def load_input(
    input_type: str | InputType,
    *,
    user_id: str = "anonymous",
    video_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    text_content: str | None = None,
    text_path: str | Path | None = None,
    image_path: str | Path | None = None,
    text_subtype: str | TextSubtype | None = None,
    session_id: str | None = None,
    output_dir: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> DataContext:
    """Validate input fields, resolve profile, and build a DataContext."""
    if isinstance(input_type, InputType):
        input_type = input_type.value
    input_type = str(input_type).lower()

    _validate_input_fields(
        input_type=input_type,
        video_path=video_path,
        audio_path=audio_path,
        text_content=text_content,
        text_path=text_path,
        image_path=image_path,
        text_subtype=text_subtype,
    )

    profile_name, profile = resolve_input_profile(input_type, text_subtype)
    pipeline_config = load_config("pipeline")
    if config_overrides:
        pipeline_config = _merge_pipeline_overrides(pipeline_config, config_overrides)

    stage_manager = StageManager(
        pipeline_config=pipeline_config,
        profile=profile,
        profile_name=profile_name,
    )

    context = DataContext.create(
        user_id=user_id,
        input_type=input_type,
        video_path=video_path,
        audio_path=audio_path,
        text_content=text_content,
        text_path=text_path,
        image_path=image_path,
        text_subtype=text_subtype,
        session_id=session_id,
        output_dir=output_dir,
        config_snapshot={
            "pipeline": pipeline_config,
            "input_profile": profile_name,
        },
        profile_metadata=stage_manager.to_metadata_patch(),
    )

    _attach_media_metadata(context)
    return context


def save_output(context: DataContext, output_dir: str | Path) -> Path:
    """Persist context JSON under output_dir/{session_id}/context.json."""
    session_id = context.metadata.get("session_id", "unknown")
    target_dir = Path(output_dir) / str(session_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    return context.save_json(target_dir / "context.json")


def _validate_input_fields(
    *,
    input_type: str,
    video_path: str | Path | None,
    audio_path: str | Path | None,
    text_content: str | None,
    text_path: str | Path | None,
    image_path: str | Path | None,
    text_subtype: str | TextSubtype | None,
) -> None:
    if input_type == InputType.VIDEO.value:
        if video_path is None:
            raise InputValidationError("video input requires video_path")
        _require_existing_file(video_path, "video_path")
        if audio_path is not None:
            _require_existing_file(audio_path, "audio_path")
        return

    if input_type == InputType.AUDIO.value:
        if audio_path is None:
            raise InputValidationError("audio input requires audio_path")
        _require_existing_file(audio_path, "audio_path")
        return

    if input_type == InputType.TEXT.value:
        if text_content is None and text_path is None:
            raise InputValidationError(
                "text input requires text_content or text_path"
            )
        if text_path is not None:
            _require_existing_file(text_path, "text_path")
        if text_content is not None and not str(text_content).strip():
            raise InputValidationError("text_content must be non-empty")
        if text_subtype is not None:
            subtype = (
                text_subtype.value
                if isinstance(text_subtype, TextSubtype)
                else str(text_subtype).lower()
            )
            if subtype not in {t.value for t in TextSubtype}:
                raise InputValidationError(f"invalid text_subtype '{subtype}'")
        return

    if input_type == InputType.IMAGE.value:
        if image_path is None:
            raise InputValidationError("image input requires image_path")
        _require_existing_file(image_path, "image_path")
        return

    raise InputValidationError(f"unknown input_type '{input_type}'")


def _require_existing_file(path: str | Path, field_name: str) -> None:
    target = Path(path)
    if not target.is_file():
        raise InputValidationError(f"{field_name} does not exist: {target}")


def _merge_pipeline_overrides(
    pipeline_config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(pipeline_config)
    base_pipeline = dict(merged.get("pipeline", {}))
    base_stages = dict(base_pipeline.get("stages", {}))

    override_pipeline = overrides.get("pipeline", overrides)
    override_stages = override_pipeline.get("stages", {})
    if isinstance(override_stages, dict):
        for stage, stage_cfg in override_stages.items():
            if stage in base_stages and isinstance(stage_cfg, dict):
                base_stages[stage] = {**base_stages[stage], **stage_cfg}
            else:
                base_stages[stage] = stage_cfg

    base_pipeline["stages"] = base_stages
    merged["pipeline"] = base_pipeline
    return merged


def _attach_media_metadata(context: DataContext) -> None:
    """Probe media files when utilities are available; failures are non-fatal."""
    raw = context.raw_data
    media_info: dict[str, Any] = {}

    if "video_path" in raw:
        try:
            from src.utils.video_utils import get_video_meta

            meta = get_video_meta(raw["video_path"])
            media_info["video"] = {
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "width": meta.width,
                "height": meta.height,
                "duration_sec": meta.duration_sec,
            }
        except Exception as exc:  # noqa: BLE001 — optional enrichment
            media_info["video_error"] = str(exc)

    if "audio_path" in raw:
        try:
            from src.utils.audio_utils import get_audio_duration, get_audio_sample_rate

            media_info["audio"] = {
                "sample_rate": get_audio_sample_rate(raw["audio_path"]),
                "duration_sec": get_audio_duration(raw["audio_path"]),
            }
        except Exception as exc:  # noqa: BLE001
            media_info["audio_error"] = str(exc)

    if "text_path" in raw and "text_content" not in raw:
        text = Path(raw["text_path"]).read_text(encoding="utf-8")
        context.raw_data["text_content"] = text
        media_info["text_length"] = len(text)

    if "text_content" in raw:
        media_info["text_length"] = len(raw["text_content"])

    if media_info:
        context.metadata["media_info"] = media_info
