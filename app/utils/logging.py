"""Centralised logging system.

Provides a :class:`LoggingManager` that bootstraps Python's ``logging``
package with consistent formatting, rotating file handlers, and
per-module log-level overrides.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

from app.config.defaults import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_DATE_FORMAT,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
)
from app.config.settings import AppSettings


class LoggingManager:
    """Bootstraps and manages application-wide logging.

    Usage::

        settings = AppSettings()
        manager = LoggingManager(settings)
        logger = manager.get_logger(__name__)
        logger.info("Application started")
    """

    def __init__(self, settings: AppSettings) -> None:
        """Initialise the logging manager.

        Args:
            settings: Application settings used to derive log levels,
                file paths, and output targets.
        """
        self._settings = settings
        self._initialised = False

    def bootstrap(self) -> None:
        """Configure the root logger.

        Call this once at application startup.  Idempotent — safe to
        invoke multiple times.
        """
        if self._initialised:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(self._settings.log_level)

        self._clear_handlers(root_logger)
        self._add_console_handler(root_logger)
        self._add_file_handler(root_logger)

        # Suppress noisy third-party loggers in non-debug mode
        if not self._settings.debug:
            for noisy in ("httpx", "urllib3", "asyncio"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

        self._initialised = True

    def get_logger(self, name: str) -> logging.Logger:
        """Return a logger for the given *name*.

        Args:
            name: Usually ``__name__`` of the calling module.

        Returns:
            A configured :class:`logging.Logger` instance.
        """
        return logging.getLogger(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_handlers(logger: logging.Logger) -> None:
        """Remove all existing handlers from *logger*."""
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _add_console_handler(self, logger: logging.Logger) -> None:
        """Add a stream handler writing to stderr."""
        if not self._settings.log_to_console:
            return

        handler = logging.StreamHandler(self._stream())
        handler.setLevel(self._settings.log_level)
        handler.setFormatter(self._formatter())
        logger.addHandler(handler)

    def _add_file_handler(self, logger: logging.Logger) -> None:
        """Add a rotating file handler."""
        if not self._settings.log_to_file:
            return

        log_file = self._log_file_path()
        handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setLevel(self._settings.log_level)
        handler.setFormatter(self._formatter())
        logger.addHandler(handler)

    def _log_file_path(self) -> Path:
        """Derive the path for the application log file."""
        resolved = self._settings.resolve_path(self._settings.log_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved / f"{self._settings.app_name.lower()}.log"

    def _formatter(self) -> logging.Formatter:
        """Build a standard log formatter."""
        return logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_LOG_DATE_FORMAT,
        )

    @staticmethod
    def _stream() -> TextIO:
        """Return the output stream (stderr)."""
        return sys.stderr
