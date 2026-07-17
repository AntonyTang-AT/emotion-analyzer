"""Tests for configuration loading utilities."""

from __future__ import annotations

import pytest

from src.utils.config_loader import ConfigManager, load_config


def test_load_all_configs(sample_config):
    assert set(sample_config) == {
        "global",
        "features",
        "fusion_policy",
        "models",
        "pipeline",
        "weight_table",
        "input_profiles",
    }


def test_load_pipeline_threshold(sample_config):
    seg = sample_config["pipeline"]["pipeline"]["stages"]["L3"]["segmentation"]
    assert seg["arousal_threshold"] == 0.3
    assert seg["max_fragment_length"] == 30


def test_pipeline_l4_l5_fusion_defaults(sample_config):
    stages = sample_config["pipeline"]["pipeline"]["stages"]
    l4 = stages["L4"]
    l5 = stages["L5"]
    assert l4["fusion_policy_path"] == "config/fusion_policy.yaml"
    assert l4["disagreement_score_divisor"] == 1.2
    assert l5["llm_backend"] == "local"
    assert l5["llm_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert l5["allow_rewrite_upstream"] is False


def test_fusion_policy_defaults(sample_config):
    policy = sample_config["fusion_policy"]
    assert policy["ser"]["max_switch_rate"] == 0.10
    assert policy["dtrb"]["reason_guided"] is False
    assert policy["llm"]["allow_rewrite_upstream"] is False


def test_weight_table_sums_to_one(sample_config):
    table = sample_config["weight_table"]
    for name, weights in table.items():
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) <= 0.01, name


def test_env_override_log_level(config_manager, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    config_manager.reload()
    global_cfg = config_manager.load("global", reload=True)
    assert global_cfg["logging"]["level"] == "DEBUG"


def test_env_override_db_path(config_manager, monkeypatch):
    monkeypatch.setenv("DB_PATH", "data/test/custom.db")
    config_manager.reload()
    global_cfg = config_manager.load("global", reload=True)
    assert global_cfg["database"]["path"] == "data/test/custom.db"


def test_unknown_config_raises(config_manager):
    with pytest.raises(ValueError, match="Unknown config"):
        config_manager.load("missing")


def test_missing_file_raises(config_manager, tmp_path, monkeypatch):
    monkeypatch.setenv("EMOTION_ROOT", str(tmp_path))
    ConfigManager._instance = None
    manager = ConfigManager()
    with pytest.raises(FileNotFoundError):
        manager.load("global")


def test_invalid_weight_table_raises(temp_config_root):
    bad_table = temp_config_root / "config" / "weight_table.yaml"
    bad_table.write_text(
        "masking: [0.5, 0.5, 0.5, 0.5]\n"
        "sarcasm: [0.4, 0.4, 0.1, 0.1]\n"
        "hidden_emotion: [0.1, 0.1, 0.1, 0.7]\n"
        "intensity_mismatch: [0.3, 0.2, 0.3, 0.2]\n"
        "consistent: [0.25, 0.25, 0.25, 0.25]\n",
        encoding="utf-8",
    )
    manager = ConfigManager()
    with pytest.raises(ValueError, match="must sum to 1.0"):
        manager.load("weight_table")


def test_config_cache_and_reload(config_manager):
    first = config_manager.load("global")
    second = config_manager.load("global")
    assert first == second
    assert first is not second

    config_manager.reload()
    third = config_manager.load("global")
    assert third == first


def test_resolve_path(config_manager, project_root):
    resolved = config_manager.resolve_path("data/raw")
    assert resolved == (project_root / "data/raw").resolve()


def test_load_config_module_helper(isolated_env):
    ConfigManager._instance = None
    pipeline = load_config("pipeline")
    assert pipeline["pipeline"]["stages"]["L1"]["enabled"] is True
