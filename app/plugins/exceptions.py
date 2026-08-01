"""Plugin system exceptions.

All plugin errors derive from :class:`PluginError` which in turn derives
from the platform-wide :class:`GeneralAIError`, so callers can catch and
report plugin failures uniformly.
"""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class PluginError(GeneralAIError):
    """Base for all plugin system errors."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin name is not found in the registry."""


class PluginInstallError(PluginError):
    """Raised when installing a plugin fails."""


class PluginLoadError(PluginError):
    """Raised when loading a plugin module fails."""


class PluginUnloadError(PluginError):
    """Raised when unloading a plugin fails."""


class PluginEnableError(PluginError):
    """Raised when enabling a plugin fails."""


class PluginDisableError(PluginError):
    """Raised when disabling a plugin fails."""


class PluginDependencyError(PluginError):
    """Raised when a plugin's dependencies cannot be satisfied."""


class PluginValidationError(PluginError):
    """Raised when plugin metadata is invalid."""


class PluginVersionError(PluginError):
    """Raised when a version constraint is not satisfied."""


class PluginSandboxError(PluginError):
    """Raised when the sandbox blocks a plugin operation."""


class PluginRegistrationError(PluginError):
    """Raised when a plugin fails to register its capabilities."""
