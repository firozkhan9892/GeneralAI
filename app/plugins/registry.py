"""Plugin registry.

The :class:`PluginRegistry` is the single source of truth for:

- active plugin instances (``name -> PluginBase``)
- lifecycle state (``name -> PluginLifecycleState``)
- per-plugin registration tracking (``name -> list[PluginRegistration]``)

It deliberately does **not** duplicate the capabilities of
:class:`ToolRegistry`, :class:`ProviderRegistry`, etc. — it only tracks
*which plugin registered what*, so that ``disable``/``unload`` can undo
registrations cleanly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.plugins.base import PluginBase
from app.plugins.exceptions import PluginNotFoundError
from app.plugins.models import (
    PluginLifecycleState,
    PluginRegistration,
    PluginState,
    PluginType,
)

log = logging.getLogger(__name__)


class PluginRegistry:
    """Thread-safe registry of plugin instances and their registrations.

    Attributes:
        _plugins: ``name -> PluginBase`` for loaded instances.
        _states: ``name -> PluginState`` for lifecycle tracking.
        _registrations: ``name -> list[PluginRegistration]``.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginBase] = {}
        self._states: dict[str, PluginState] = {}
        self._registrations: dict[str, list[PluginRegistration]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Plugin instance management
    # ------------------------------------------------------------------

    def register_plugin(self, plugin: PluginBase, state: PluginState) -> None:
        """Store a plugin instance and its initial state.

        Args:
            plugin: The plugin instance.
            state: The initial runtime state.

        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        with self._lock:
            if state.name in self._plugins:
                raise ValueError(f"Plugin '{state.name}' is already registered")
            self._plugins[state.name] = plugin
            self._states[state.name] = state
            self._registrations[state.name] = []

    def unregister_plugin(self, name: str) -> PluginBase | None:
        """Remove a plugin instance from the registry.

        Args:
            name: Plugin name.

        Returns:
            The removed plugin, or ``None`` if not found.
        """
        with self._lock:
            plugin = self._plugins.pop(name, None)
            self._states.pop(name, None)
            self._registrations.pop(name, None)
            if plugin is not None:
                log.debug("Unregistered plugin '%s'", name)
            return plugin

    def get_plugin(self, name: str) -> PluginBase | None:
        """Return a loaded plugin by name, or ``None``."""
        with self._lock:
            return self._plugins.get(name)

    def has_plugin(self, name: str) -> bool:
        """Return ``True`` if *name* is in the registry."""
        with self._lock:
            return name in self._plugins

    def list_plugins(self) -> list[str]:
        """Return all registered plugin names."""
        with self._lock:
            return list(self._plugins.keys())

    def list_by_type(self, plugin_type: PluginType) -> list[str]:
        """Return names of plugins matching *plugin_type*."""
        with self._lock:
            return [
                name
                for name, state in self._states.items()
                if state.plugin_type == plugin_type
            ]

    # ------------------------------------------------------------------
    # Lifecycle state management
    # ------------------------------------------------------------------

    def get_state(self, name: str) -> PluginState:
        """Return the runtime state for *name*.

        Raises:
            PluginNotFoundError: If the plugin is not registered.
        """
        with self._lock:
            state = self._states.get(name)
            if state is None:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not found in registry",
                    module="plugins.registry",
                    context={"plugin": name},
                )
            return state

    def set_state(self, name: str, state: PluginLifecycleState) -> None:
        """Transition a plugin to a new lifecycle state.

        Raises:
            PluginNotFoundError: If the plugin is not registered.
        """
        with self._lock:
            if name not in self._states:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not found in registry",
                    module="plugins.registry",
                    context={"plugin": name},
                )
            current = self._states[name]
            self._states[name] = current.model_copy(update={"lifecycle_state": state})
            log.debug("Plugin '%s' transitioned to %s", name, state.value)

    def add_state(self, name: str, state: PluginState) -> None:
        """Store or replace a plugin's state without registering a plugin instance."""
        with self._lock:
            self._states[name] = state
            if name not in self._registrations:
                self._registrations[name] = []

    # ------------------------------------------------------------------
    # Registration tracking
    # ------------------------------------------------------------------

    def add_registration(
        self, plugin_name: str, registration: PluginRegistration
    ) -> None:
        """Record a capability registration made by *plugin_name*."""
        with self._lock:
            self._registrations[plugin_name].append(registration)

    def get_registrations(self, plugin_name: str) -> list[PluginRegistration]:
        """Return all registrations tracked for *plugin_name*."""
        with self._lock:
            return list(self._registrations.get(plugin_name, []))

    def clear_registrations(self, plugin_name: str) -> list[PluginRegistration]:
        """Remove all registrations for *plugin_name*.

        Returns:
            The removed registrations.
        """
        with self._lock:
            removed = self._registrations.pop(plugin_name, [])
            self._registrations[plugin_name] = []
            return removed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def plugins(self) -> dict[str, PluginBase]:
        """Return a shallow copy of all registered plugins."""
        with self._lock:
            return dict(self._plugins)

    @property
    def states(self) -> dict[str, PluginState]:
        """Return a shallow copy of all plugin states."""
        with self._lock:
            return dict(self._states)

    def total_count(self) -> int:
        """Return the total number of registered plugins."""
        with self._lock:
            return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._plugins

    def __len__(self) -> int:
        with self._lock:
            return len(self._plugins)

    def __iter__(self) -> Any:
        with self._lock:
            return iter(list(self._plugins.keys()))
