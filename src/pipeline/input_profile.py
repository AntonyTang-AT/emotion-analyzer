"""Resolve input source profiles from config/input_profiles.yaml."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.core.types import InputType, SegmentationMode, TextSubtype
from src.utils.config_loader import load_config

PROFILE_BY_INPUT: dict[tuple[str, str | None], str] = {
    (InputType.VIDEO.value, None): "video_default",
    (InputType.AUDIO.value, None): "audio_default",
    (InputType.TEXT.value, TextSubtype.DESCRIPTIVE.value): "text_descriptive",
    (InputType.TEXT.value, TextSubtype.DIALOGUE.value): "text_dialogue",
    (InputType.TEXT.value, None): "text_descriptive",
    (InputType.IMAGE.value, None): "image_default",
}

VALID_SEGMENTATION_MODES = frozenset(mode.value for mode in SegmentationMode)


def load_input_profiles(*, reload: bool = False) -> dict[str, Any]:
    """Load and return the profiles mapping from input_profiles.yaml."""
    data = load_config("input_profiles")
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("input_profiles.profiles must be a non-empty mapping")
    return profiles


def resolve_profile_name(
    input_type: str | InputType,
    text_subtype: str | TextSubtype | None = None,
) -> str:
    """Map input_type (+ optional text_subtype) to a profile key."""
    if isinstance(input_type, InputType):
        input_type = input_type.value
    if isinstance(text_subtype, TextSubtype):
        text_subtype = text_subtype.value

    input_type = str(input_type).lower()
    if input_type not in {t.value for t in InputType}:
        raise ValueError(
            f"Unknown input_type '{input_type}'. "
            f"Must be one of: {', '.join(t.value for t in InputType)}"
        )

    if input_type == InputType.TEXT.value and text_subtype is not None:
        text_subtype = str(text_subtype).lower()
        if text_subtype not in {t.value for t in TextSubtype}:
            raise ValueError(
                f"Unknown text_subtype '{text_subtype}'. "
                f"Must be one of: {', '.join(t.value for t in TextSubtype)}"
            )

    key = (input_type, text_subtype if input_type == InputType.TEXT.value else None)
    profile_name = PROFILE_BY_INPUT.get(key)
    if profile_name is None:
        raise ValueError(
            f"No profile mapping for input_type={input_type!r}, "
            f"text_subtype={text_subtype!r}"
        )
    return profile_name


def resolve_input_profile(
    input_type: str | InputType,
    text_subtype: str | TextSubtype | None = None,
    *,
    reload: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Return (profile_name, profile_dict) for the given input source."""
    profile_name = resolve_profile_name(input_type, text_subtype)
    profiles = load_input_profiles(reload=reload)
    if profile_name not in profiles:
        raise ValueError(
            f"Profile '{profile_name}' not found in input_profiles.yaml"
        )

    profile = deepcopy(profiles[profile_name])
    expected_type = profile.get("input_type")
    if expected_type != (
        input_type.value if isinstance(input_type, InputType) else str(input_type).lower()
    ):
        raise ValueError(
            f"Profile '{profile_name}' input_type mismatch: "
            f"expected {expected_type!r}"
        )

    if text_subtype is not None and profile.get("text_subtype") is not None:
        subtype_val = (
            text_subtype.value
            if isinstance(text_subtype, TextSubtype)
            else str(text_subtype).lower()
        )
        if profile["text_subtype"] != subtype_val:
            raise ValueError(
                f"Profile '{profile_name}' text_subtype mismatch: "
                f"expected {profile['text_subtype']!r}, got {subtype_val!r}"
            )

    _validate_profile(profile_name, profile)
    return profile_name, profile


def _validate_profile(name: str, profile: dict[str, Any]) -> None:
    extractors = profile.get("l1_extractors")
    if not isinstance(extractors, list) or not extractors:
        raise ValueError(f"Profile '{name}' must define non-empty l1_extractors")

    l3 = profile.get("l3", {})
    if not isinstance(l3, dict):
        raise ValueError(f"Profile '{name}' l3 must be a mapping")
    mode = l3.get("segmentation_mode")
    if mode is not None and mode not in VALID_SEGMENTATION_MODES:
        raise ValueError(
            f"Profile '{name}' has invalid segmentation_mode '{mode}'"
        )

    for layer in ("l4", "l5", "l6"):
        section = profile.get(layer, {})
        if section is not None and not isinstance(section, dict):
            raise ValueError(f"Profile '{name}' {layer} must be a mapping")
