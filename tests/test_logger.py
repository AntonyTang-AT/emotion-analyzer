"""Tests for logging utilities."""

from __future__ import annotations

import logging

from src.utils.logger import get_logger, setup_logger


def test_setup_logger_writes_to_file(reset_loggers, tmp_path):
    log_file = tmp_path / "unit.log"
    logger = setup_logger("tests.logger.file", log_file=str(log_file), level="INFO")
    logger.info("hello from test")

    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "hello from test" in content


def test_get_logger_is_cached(reset_loggers, tmp_path):
    log_file = tmp_path / "cached.log"
    first = setup_logger("tests.logger.cache", log_file=str(log_file))
    second = get_logger("tests.logger.cache")
    assert first is second


def test_get_logger_does_not_duplicate_handlers(reset_loggers, tmp_path):
    log_file = tmp_path / "handlers.log"
    logger = setup_logger("tests.logger.handlers", log_file=str(log_file))
    handler_count = len(logger.handlers)
    again = get_logger("tests.logger.handlers")
    assert len(again.handlers) == handler_count


def test_logger_respects_debug_level(reset_loggers, tmp_path):
    log_file = tmp_path / "debug.log"
    logger = setup_logger("tests.logger.debug", log_file=str(log_file), level="DEBUG")
    assert logger.level == logging.DEBUG
