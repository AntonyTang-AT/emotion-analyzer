"""Pipeline stage orchestration with input-profile overrides."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.core.context import VALID_STAGES
from src.utils.config_loader import load_config

STAGE_ORDER: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "L1": (),
    "L2": ("L1",),
    "L3": ("L2",),
    "L4": ("L3",),
    "L5": ("L3",),  # L4 optional when disabled; validated at runtime
    "L6": ("L5",),
}


class StageManager:
    """Merge pipeline.yaml defaults with an input profile for stage routing."""

    def __init__(
        self,
        pipeline_config: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
        *,
        profile_name: str | None = None,
    ) -> None:
        if pipeline_config is None:
            pipeline_config = load_config("pipeline")
        self._pipeline = deepcopy(pipeline_config.get("pipeline", {}))
        self._stages = deepcopy(self._pipeline.get("stages", {}))
        self._profile = deepcopy(profile) if profile else {}
        self._profile_name = profile_name
        self._enabled = self._compute_enabled_stages()
        self._execution_order = self._compute_execution_order()

    @property
    def profile_name(self) -> str | None:
        return self._profile_name

    @property
    def profile(self) -> dict[str, Any]:
        return deepcopy(self._profile)

    def get_l1_extractors(self) -> list[str]:
        if self._profile.get("l1_extractors"):
            return list(self._profile["l1_extractors"])
        l1 = self._stages.get("L1", {})
        return list(l1.get("feature_extractors", []))

    def preserve_raw_visual(self) -> bool:
        if "preserve_raw_visual" in self._profile:
            return bool(self._profile["preserve_raw_visual"])
        return bool(self._stages.get("L1", {}).get("preserve_raw_visual", False))

    def get_segmentation_mode(self) -> str:
        l3_profile = self._profile.get("l3", {})
        if isinstance(l3_profile, dict) and l3_profile.get("segmentation_mode"):
            return str(l3_profile["segmentation_mode"])
        return "dynamic"

    def l5_use_l4_weights(self) -> bool:
        l5_profile = self._profile.get("l5", {})
        if isinstance(l5_profile, dict) and "use_l4_weights" in l5_profile:
            return bool(l5_profile["use_l4_weights"])
        return bool(self._stages.get("L5", {}).get("use_l4_weights", True))

    def l6_min_fragments(self) -> int | None:
        l6_profile = self._profile.get("l6", {})
        if isinstance(l6_profile, dict) and "min_fragments" in l6_profile:
            return int(l6_profile["min_fragments"])
        l6 = self._stages.get("L6", {})
        value = l6.get("min_fragments")
        return int(value) if value is not None else None

    def is_stage_enabled(self, stage: str) -> bool:
        if stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage '{stage}'")
        return self._enabled.get(stage, False)

    def get_enabled_stages(self) -> list[str]:
        return [stage for stage in STAGE_ORDER if self.is_stage_enabled(stage)]

    def get_execution_order(self) -> list[str]:
        return list(self._execution_order)

    def validate_dependencies(self) -> list[str]:
        """Return warning messages for dependency violations (auto-fix enabled)."""
        warnings: list[str] = []
        enabled = set(self.get_enabled_stages())

        for stage in self.get_execution_order():
            deps = STAGE_DEPENDENCIES.get(stage, ())
            if stage == "L5" and not self.is_stage_enabled("L4"):
                deps = ("L3",)
            missing = [dep for dep in deps if dep not in enabled]
            if missing and self.is_stage_enabled(stage):
                warnings.append(
                    f"{stage} enabled but dependencies disabled: {', '.join(missing)}"
                )
        return warnings

    def to_metadata_patch(self) -> dict[str, Any]:
        """Fields to merge into DataContext.metadata after profile resolution."""
        patch: dict[str, Any] = {
            "active_modalities": self.get_l1_extractors(),
            "segmentation_mode": self.get_segmentation_mode(),
            "l4_enabled": self.is_stage_enabled("L4"),
            "l6_enabled": self.is_stage_enabled("L6"),
            "l5_use_l4_weights": self.l5_use_l4_weights(),
            "enabled_stages": self.get_enabled_stages(),
        }
        if self._profile_name:
            patch["input_profile"] = self._profile_name
        min_fragments = self.l6_min_fragments()
        if min_fragments is not None:
            patch["l6_min_fragments"] = min_fragments
        l5_profile = self._profile.get("l5", {})
        if isinstance(l5_profile, dict) and l5_profile.get("prompt_profile"):
            patch["prompt_profile"] = l5_profile["prompt_profile"]
        return patch

    def _layer_enabled_in_profile(self, layer_key: str) -> bool | None:
        section = self._profile.get(layer_key, {})
        if isinstance(section, dict) and "enabled" in section:
            return bool(section["enabled"])
        return None

    def _compute_enabled_stages(self) -> dict[str, bool]:
        enabled: dict[str, bool] = {}
        for stage in STAGE_ORDER:
            base = bool(self._stages.get(stage, {}).get("enabled", True))
            profile_override = None
            if stage == "L3":
                profile_override = self._layer_enabled_in_profile("l3")
            elif stage == "L4":
                profile_override = self._layer_enabled_in_profile("l4")
            elif stage == "L6":
                profile_override = self._layer_enabled_in_profile("l6")
            enabled[stage] = profile_override if profile_override is not None else base
        return enabled

    def _compute_execution_order(self) -> tuple[str, ...]:
        order: list[str] = []
        for stage in STAGE_ORDER:
            if not self.is_stage_enabled(stage):
                continue
            deps = list(STAGE_DEPENDENCIES[stage])
            if stage == "L5" and not self.is_stage_enabled("L4"):
                deps = ["L3"]
            for dep in deps:
                if dep not in order and self.is_stage_enabled(dep):
                    order.append(dep)
            if stage not in order:
                order.append(stage)
        return tuple(order)
