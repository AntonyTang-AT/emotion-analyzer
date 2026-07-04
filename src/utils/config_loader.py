"""Unified configuration loading for emotion-analyzer."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAMES: tuple[str, ...] = (
    "global",
    "features",
    "models",
    "pipeline",
    "weight_table",
    "input_profiles",
)

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
VALID_DEVICES = frozenset({"cuda", "cpu", "auto"})
VALID_VA_TYPES = frozenset({"self", "inter"})
VALID_INPUT_TYPES = frozenset({"video", "audio", "text", "image"})
VALID_TEXT_SUBTYPES = frozenset({"descriptive", "dialogue"})
VALID_SEGMENTATION_MODES = frozenset({"dynamic", "single", "utterance"})
WEIGHT_TABLE_KEYS = frozenset({
    "masking",
    "sarcasm",
    "hidden_emotion",
    "intensity_mismatch",
    "consistent",
})

ENV_OVERRIDES: dict[str, tuple[str, tuple[str, ...]]] = {
    "LOG_LEVEL": ("global", ("logging", "level")),
    "DB_PATH": ("global", ("database", "path")),
    "DEVICE": ("global", ("device",)),
}


def get_project_root() -> Path:
    """Resolve project root from EMOTION_ROOT or package location."""
    env_root = os.getenv("EMOTION_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


class ConfigManager:
    """Singleton that caches loaded YAML configuration files."""

    _instance: ConfigManager | None = None

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._cache: dict[str, dict[str, Any]] = {}
            instance._root = get_project_root()
            cls._instance = instance
        return cls._instance

    @property
    def root(self) -> Path:
        return self._root

    @property
    def config_dir(self) -> Path:
        return self._root / "config"

    def load(self, name: str, *, reload: bool = False) -> dict[str, Any]:
        if name not in CONFIG_NAMES:
            raise ValueError(
                f"Unknown config '{name}'. Valid names: {', '.join(CONFIG_NAMES)}"
            )

        if not reload and name in self._cache:
            return deepcopy(self._cache[name])

        path = self.config_dir / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Config '{name}' must be a YAML mapping, got {type(data)}")

        data = self._apply_env_overrides(name, data)
        self._validate(name, data)
        self._cache[name] = deepcopy(data)
        return deepcopy(data)

    def load_all(self, *, reload: bool = False) -> dict[str, dict[str, Any]]:
        return {name: self.load(name, reload=reload) for name in CONFIG_NAMES}

    def reload(self) -> None:
        self._cache.clear()

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against project root."""
        return (self.root / relative).resolve()

    def _apply_env_overrides(
        self, name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(data)
        for env_var, (cfg_name, key_path) in ENV_OVERRIDES.items():
            if cfg_name != name:
                continue
            value = os.getenv(env_var)
            if value is not None:
                _set_nested(result, key_path, value)
        return result

    def _validate(self, name: str, data: dict[str, Any]) -> None:
        if name == "global":
            self._validate_global(data)
        elif name == "pipeline":
            self._validate_pipeline(data)
        elif name == "weight_table":
            self._validate_weight_table(data)
        elif name == "input_profiles":
            self._validate_input_profiles(data)

    def _validate_global(self, data: dict[str, Any]) -> None:
        level = str(data.get("logging", {}).get("level", "INFO")).upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid logging.level '{level}'. "
                f"Must be one of: {', '.join(sorted(VALID_LOG_LEVELS))}"
            )

        device = str(data.get("device", "auto")).lower()
        if device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}'. "
                f"Must be one of: {', '.join(sorted(VALID_DEVICES))}"
            )

    def _validate_pipeline(self, data: dict[str, Any]) -> None:
        stages = data.get("pipeline", {}).get("stages")
        if not isinstance(stages, dict) or not stages:
            raise ValueError("pipeline.stages must be a non-empty mapping")

        l3 = stages.get("L3", {})
        seg = l3.get("segmentation", {})
        threshold = seg.get("arousal_threshold")
        if threshold is not None:
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("L3.segmentation.arousal_threshold must be in [0, 1]")

        va_type = seg.get("use_va_type")
        if va_type is not None and va_type not in VALID_VA_TYPES:
            raise ValueError("L3.segmentation.use_va_type must be 'self' or 'inter'")

        baseline = l3.get("baseline", {})
        baseline_va = baseline.get("use_va_type")
        if baseline_va is not None and baseline_va not in VALID_VA_TYPES:
            raise ValueError("L3.baseline.use_va_type must be 'self' or 'inter'")

        cold = l3.get("cold_start", {})
        source = cold.get("embedding_source")
        if source is not None and source not in VALID_VA_TYPES:
            raise ValueError("L3.cold_start.embedding_source must be 'self' or 'inter'")

        memory = l3.get("memory", {})
        emb_type = memory.get("embedding_type")
        if emb_type is not None and emb_type not in VALID_VA_TYPES:
            raise ValueError("L3.memory.embedding_type must be 'self' or 'inter'")

        l4 = stages.get("L4", {})
        strategy = l4.get("weight_strategy")
        if strategy is not None and strategy not in ("rule_table", "small_nn"):
            raise ValueError("L4.weight_strategy must be 'rule_table' or 'small_nn'")

    def _validate_weight_table(self, data: dict[str, Any]) -> None:
        missing = WEIGHT_TABLE_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"weight_table missing required keys: {', '.join(sorted(missing))}"
            )

        for key, weights in data.items():
            if not isinstance(weights, list) or len(weights) != 4:
                raise ValueError(f"weight_table.{key} must be a list of 4 floats")
            try:
                values = [float(w) for w in weights]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"weight_table.{key} contains non-numeric values") from exc
            if any(v < 0 for v in values):
                raise ValueError(f"weight_table.{key} weights must be non-negative")
            total = sum(values)
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"weight_table.{key} must sum to 1.0 (±0.01), got {total:.4f}"
                )

    def _validate_input_profiles(self, data: dict[str, Any]) -> None:
        profiles = data.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("input_profiles.profiles must be a non-empty mapping")

        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"input_profiles.profiles.{name} must be a mapping")

            input_type = profile.get("input_type")
            if input_type not in VALID_INPUT_TYPES:
                raise ValueError(
                    f"Profile '{name}' has invalid input_type '{input_type}'"
                )

            extractors = profile.get("l1_extractors")
            if not isinstance(extractors, list) or not extractors:
                raise ValueError(
                    f"Profile '{name}' must define non-empty l1_extractors"
                )

            l3 = profile.get("l3", {})
            if isinstance(l3, dict):
                mode = l3.get("segmentation_mode")
                if mode is not None and mode not in VALID_SEGMENTATION_MODES:
                    raise ValueError(
                        f"Profile '{name}' has invalid segmentation_mode '{mode}'"
                    )

            text_subtype = profile.get("text_subtype")
            if text_subtype is not None and text_subtype not in VALID_TEXT_SUBTYPES:
                raise ValueError(
                    f"Profile '{name}' has invalid text_subtype '{text_subtype}'"
                )


def _set_nested(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        nested = current.setdefault(key, {})
        if not isinstance(nested, dict):
            raise ValueError(f"Cannot set nested key at '{key}': parent is not a mapping")
        current = nested
    current[keys[-1]] = value


def get_config() -> ConfigManager:
    return ConfigManager()


def load_config(config_name: str) -> dict[str, Any]:
    return get_config().load(config_name)


def get_global_config() -> dict[str, Any]:
    return load_config("global")
