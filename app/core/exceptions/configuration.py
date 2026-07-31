"""Configuration-related exceptions."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class ConfigurationError(GeneralAIError):
    """Raised when application configuration is invalid or missing."""


class ConfigValidationError(ConfigurationError):
    """Raised when a configuration value fails validation."""


class ConfigLoadError(ConfigurationError):
    """Raised when configuration cannot be loaded from a source."""
