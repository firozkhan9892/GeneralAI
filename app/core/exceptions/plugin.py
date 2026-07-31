"""Plugin system exceptions."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class PluginError(GeneralAIError):
    """Base for all plugin errors."""


class PluginDiscoveryError(PluginError):
    """Raised when plugin discovery fails."""


class PluginLoadError(PluginError):
    """Raised when loading a plugin module fails."""


class PluginDependencyError(PluginError):
    """Raised when a plugin's dependencies cannot be satisfied."""


class PluginValidationError(PluginError):
    """Raised when plugin metadata is invalid."""


class PluginDisabledError(PluginError):
    """Raised when attempting to load a disabled plugin."""
