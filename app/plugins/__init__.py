"""Phase 10: Plugin & Extension System.

Public API:

    PluginBase       — abstract base class plugins must implement
    PluginContext     — capability bundle handed to plugin hooks
    PluginManager     — lifecycle orchestrator (install/load/enable/disable/unload/uninstall)
    PluginLoader      — discovers and instantiates plugins
    PluginRegistry    — tracks instances, states, registrations
    PluginSandbox     — restricted execution environment

Models:
    PluginManifest
    PluginMetadata  (re-exported from core)
    PluginType
    PluginLifecycleState
    PluginDependency
    PluginRegistration
    PluginState

Exceptions:
    PluginError, PluginNotFoundError, PluginInstallError,
    PluginLoadError, PluginUnloadError, PluginEnableError,
    PluginDisableError, PluginDependencyError, PluginValidationError,
    PluginVersionError, PluginSandboxError, PluginRegistrationError
"""

from __future__ import annotations

from app.core.plugins.plugin_metadata import PluginMetadata
from app.plugins.base import PluginBase, PluginContext
from app.plugins.exceptions import (
    PluginDependencyError,
    PluginDisableError,
    PluginEnableError,
    PluginError,
    PluginInstallError,
    PluginLoadError,
    PluginNotFoundError,
    PluginRegistrationError,
    PluginSandboxError,
    PluginUnloadError,
    PluginValidationError,
    PluginVersionError,
)
from app.plugins.manager import PluginManager
from app.plugins.loader import PluginLoader
from app.plugins.models import (
    PluginDependency,
    PluginLifecycleState,
    PluginManifest,
    PluginRegistration,
    PluginState,
    PluginType,
)
from app.plugins.registry import PluginRegistry
from app.plugins.sandbox import PluginSandbox

__all__ = [
    # Core
    "PluginBase",
    "PluginContext",
    "PluginManager",
    "PluginLoader",
    "PluginRegistry",
    "PluginSandbox",
    # Models
    "PluginManifest",
    "PluginMetadata",
    "PluginType",
    "PluginLifecycleState",
    "PluginDependency",
    "PluginRegistration",
    "PluginState",
    # Exceptions
    "PluginError",
    "PluginNotFoundError",
    "PluginInstallError",
    "PluginLoadError",
    "PluginUnloadError",
    "PluginEnableError",
    "PluginDisableError",
    "PluginDependencyError",
    "PluginValidationError",
    "PluginVersionError",
    "PluginSandboxError",
    "PluginRegistrationError",
]
