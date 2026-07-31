"""Tests for the logging system."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import AppSettings
from app.utils.logging import LoggingManager


class TestLoggingManager:
    """Suite for LoggingManager behaviour."""

    def test_bootstrap_adds_handlers(self) -> None:
        """The root logger should have handlers after bootstrap."""
        settings = AppSettings(log_to_file=False)
        manager = LoggingManager(settings)
        manager.bootstrap()
        root = logging.getLogger()
        assert len(root.handlers) >= 1

    def test_bootstrap_is_idempotent(self) -> None:
        """Calling bootstrap twice should not duplicate handlers."""
        settings = AppSettings(log_to_file=False)
        manager = LoggingManager(settings)
        manager.bootstrap()
        count_after_first = len(logging.getLogger().handlers)
        manager.bootstrap()
        count_after_second = len(logging.getLogger().handlers)
        assert count_after_second == count_after_first

    def test_get_logger_returns_named_logger(self) -> None:
        """get_logger should return a logger matching the given name."""
        settings = AppSettings(log_to_file=False)
        manager = LoggingManager(settings)
        logger = manager.get_logger("test.module")
        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_file_handler_writes_to_disk(self, tmp_path: Path) -> None:
        """A rotating file handler should write log output to disk."""
        log_dir = tmp_path / "logs"
        settings = AppSettings(
            log_to_console=False,
            log_to_file=True,
            log_level="DEBUG",
            log_dir=log_dir,
            project_root=tmp_path,
        )
        manager = LoggingManager(settings)
        manager.bootstrap()

        logger = manager.get_logger("test.file")
        logger.info("Hello, log file!")

        # Force flush
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = log_dir / "generalai.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Hello, log file!" in content
