"""Unified logging utilities for emotion-analyzer."""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_DEFAULT_LOGGING = {
    "level": "INFO",
    "file": "logs/pipeline.log",
    "max_bytes": 10 * 1024 * 1024,
    "backup_count": 5,
    "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    "json_enabled": False,
}

_LOGGERS: dict[str, logging.Logger] = {}


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_logging_settings() -> dict[str, Any]:
    """Load logging settings from global config, with safe fallbacks."""
    settings = dict(_DEFAULT_LOGGING)
    try:
        from .config_loader import get_config, get_global_config

        global_cfg = get_global_config()
        logging_cfg = global_cfg.get("logging", {})
        root = get_config().root

        file_rel = logging_cfg.get("file", settings["file"])
        settings.update({
            "level": logging_cfg.get("level", settings["level"]),
            "file": str(root / file_rel),
            "max_bytes": logging_cfg.get("max_bytes", settings["max_bytes"]),
            "backup_count": logging_cfg.get("backup_count", settings["backup_count"]),
            "format": logging_cfg.get("format", settings["format"]),
            "json_enabled": logging_cfg.get("json_enabled", settings["json_enabled"]),
        })
    except Exception:
        settings["file"] = str(Path(settings["file"]).resolve())

    return settings


def setup_logger(
    name: str,
    log_file: str | None = None,
    level: str | None = None,
) -> logging.Logger:
    """Create or reconfigure a logger with console and rotating file handlers."""
    cfg = _resolve_logging_settings()
    log_level = str(level or cfg["level"]).upper()
    file_path = log_file or cfg["file"]

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if name in _LOGGERS and logger.handlers:
        logger.setLevel(log_level)
        return logger

    logger.handlers.clear()

    if cfg.get("json_enabled"):
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(cfg["format"])

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=int(cfg["max_bytes"]),
        backupCount=int(cfg["backup_count"]),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name)
