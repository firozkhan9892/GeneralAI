"""Plugin manager — full lifecycle orchestration.

The :class:`PluginManager` is the top-level orchestrator.  It coordinates
::

    - :class:`~app.plugins.loader.PluginLoader` for discovery and instantiation.
    - :class:`~app.plugins.registry.PluginRegistry` for instance/state tracking.
    - :class:`~app.plugins.sandbox.PluginSandbox` for restricted execution.
    - System registries (ToolRegistry, ProviderRegistry, etc.) via
      :class:`~app.plugins.base.PluginContext`.

It does **not** duplicate registry internals — it delegates registration
calls to the appropriate system registry and only tracks *what each
plugin registered* for clean teardown.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Coroutine

from app import __version__ as GENERALAI_VERSION
from app.plugins.base import PluginBase, PluginContext
from app.plugins.exceptions import (
    PluginDependencyError,
    PluginEnableError,
    PluginLoadError,
    PluginNotFoundError,
    PluginUnloadError,
    PluginValidationError,
    PluginVersionError,
)
from app.plugins.loader import PluginLoader
from app.plugins.models import (
    PluginLifecycleState,
    PluginManifest,
    PluginRegistration,
    PluginState,
    PluginType,
)
from app.plugins.registry import PluginRegistry
from app.plugins.sandbox import PluginSandbox

log = logging.getLogger(__name__)


def _run_coroutine(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine, handling both sync and async contexts."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


async def _async_enable(plugin: PluginBase, context: PluginContext) -> list[str]:
    """Async portion of enable — runs load + enable hooks."""
    await plugin.load(context)
    result = await plugin.enable(context)
    if not isinstance(result, list):
        return []
    return result


async def _async_disable(plugin: PluginBase, context: PluginContext) -> None:
    """Async portion of disable — runs disable + unregister hooks."""
    await plugin.disable(context)
    await plugin.unregister(context)


async def _async_unload(plugin: PluginBase, context: PluginContext) -> None:
    """Async portion of unload — runs unload + cleanup hooks."""
    await plugin.unload(context)
    await plugin.cleanup(context)


async def _async_uninstall(plugin: PluginBase, context: PluginContext) -> None:
    """Async portion of uninstall — runs uninstall + cleanup hooks."""
    await plugin.uninstall(context)
    await plugin.cleanup(context)


class PluginManager:
    """Orchestrates plugin discovery, installation, loading, and lifecycle.

    Usage::

        manager = PluginManager(context=my_context)
        manager.discover()
        manager.install("my_plugin")
        manager.load("my_plugin")
        manager.enable("my_plugin")
    """

    def __init__(
        self,
        context: PluginContext | None = None,
        plugin_dirs: list[str] | None = None,
        sandbox: PluginSandbox | None = None,
    ) -> None:
        """Initialise the plugin manager.

        Args:
            context: The :class:`PluginContext` handed to every plugin hook.
            plugin_dirs: Directories to scan for plugin manifests.
            sandbox: Optional :class:`PluginSandbox` for restricted execution.
        """
        self._loader = PluginLoader(plugin_dirs=plugin_dirs)
        self._registry = PluginRegistry()
        self._context: PluginContext | None = context
        self._sandbox: PluginSandbox | None = sandbox
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    @property
    def context(self) -> PluginContext | None:
        """Return the plugin context."""
        return self._context

    @context.setter
    def context(self, value: PluginContext) -> None:
        """Set the plugin context (before any plugins are enabled)."""
        self._context = value

    @property
    def loader(self) -> PluginLoader:
        """Return the plugin loader."""
        return self._loader

    @property
    def registry(self) -> PluginRegistry:
        """Return the plugin registry."""
        return self._registry

    @property
    def sandbox(self) -> PluginSandbox | None:
        """Return the plugin sandbox, if configured."""
        return self._sandbox

    # ------------------------------------------------------------------
    # Discovery & manifests
    # ------------------------------------------------------------------

    def discover(self) -> dict[str, PluginManifest]:
        """Discover available plugins and return their manifests.

        Returns:
            Dict mapping plugin name to :class:`PluginManifest`.
        """
        with self._lock:
            return self._loader.discover()

    def get_manifest(self, name: str) -> PluginManifest | None:
        """Return the discovered manifest for *name*, if any."""
        manifests = self._loader.discover()
        return manifests.get(name)

    # ------------------------------------------------------------------
    # Version compatibility
    # ------------------------------------------------------------------

    def check_version_compatibility(self, manifest: PluginManifest) -> None:
        """Verify the plugin's GeneralAI version requirement is met.

        Raises:
            PluginVersionError: If the constraint is not satisfied.
        """
        if not manifest.generalai_version:
            return
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            spec = SpecifierSet(manifest.generalai_version)
            if Version(GENERALAI_VERSION) not in spec:
                raise PluginVersionError(
                    f"Plugin '{manifest.name}' requires GeneralAI "
                    f"{manifest.generalai_version}, but running {GENERALAI_VERSION}",
                    module="plugins.manager",
                    context={
                        "plugin": manifest.name,
                        "required": manifest.generalai_version,
                        "actual": GENERALAI_VERSION,
                    },
                )
        except PluginVersionError:
            raise
        except Exception as exc:
            raise PluginVersionError(
                f"Invalid version specifier '{manifest.generalai_version}' "
                f"for plugin '{manifest.name}'",
                module="plugins.manager",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def resolve_load_order(
        self, manifests: dict[str, PluginManifest] | None = None
    ) -> list[str]:
        """Topologically sort plugins by dependency graph.

        Args:
            manifests: Optional pre-discovered manifests.  If ``None``,
                calls :meth:`discover`.

        Returns:
            Plugin names in dependency-first order.

        Raises:
            PluginDependencyError: On circular or missing dependencies.
        """
        if manifests is None:
            manifests = self._loader.discover()

        visited: set[str] = set()
        visiting: set[str] = set()
        order: list[str] = []

        depth_limit = 20
        deps_by_plugin: dict[str, list[str]] = {
            name: [d.name for d in m.effective_dependencies]
            for name, m in manifests.items()
        }

        def _visit(name: str, depth: int) -> None:
            if depth > depth_limit:
                raise PluginDependencyError(
                    f"Dependency chain too deep for '{name}'",
                    module="plugins.manager",
                    context={"plugin": name},
                )
            if name not in manifests:
                raise PluginDependencyError(
                    f"Plugin '{name}' depends on unknown plugin",
                    module="plugins.manager",
                    context={"plugin": name},
                )
            if name in visited:
                return
            if name in visiting:
                raise PluginDependencyError(
                    f"Circular dependency detected involving '{name}'",
                    module="plugins.manager",
                    context={"plugin": name},
                )
            visiting.add(name)
            for dep_name in deps_by_plugin.get(name, []):
                _visit(dep_name, depth + 1)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in manifests:
            if name not in visited:
                _visit(name, 0)

        return order

    def _check_dependency_versions(
        self,
        name: str,
        manifest: PluginManifest,
        discovered: dict[str, PluginManifest],
    ) -> None:
        """Verify all dependencies' versions are satisfied."""
        for dep in manifest.effective_dependencies:
            dep_manifest = discovered.get(dep.name)
            if dep_manifest is None:
                raise PluginDependencyError(
                    f"Plugin '{name}' depends on '{dep.name}' which is not discovered",
                    module="plugins.manager",
                    context={"plugin": name, "dependency": dep.name},
                )
            if not dep.matches(dep_manifest.version):
                raise PluginDependencyError(
                    f"Plugin '{name}' requires '{dep.name}' "
                    f"{dep.version_spec}, but found version "
                    f"{dep_manifest.version}",
                    module="plugins.manager",
                    context={
                        "plugin": name,
                        "dependency": dep.name,
                        "required": dep.version_spec,
                        "found": dep_manifest.version,
                    },
                )

    # ------------------------------------------------------------------
    # Lifecycle: INSTALL
    # ------------------------------------------------------------------

    def install(
        self,
        name: str,
        manifest: PluginManifest | None = None,
        plugin: PluginBase | None = None,
    ) -> PluginState:
        """Install a plugin (validate + instantiate + INSTALLED state).

        Does not load or enable — just validates metadata, checks
        version and dependencies.

        Args:
            name: Plugin name.
            manifest: Optional pre-resolved manifest.
            plugin: Optional pre-instantiated plugin instance.  When
                provided, the plugin is registered immediately so that
                :meth:`load` can return it without re-importing.

        Returns:
            The plugin's initial :class:`PluginState`.

        Raises:
            PluginNotFoundError: If the plugin is not discovered.
            PluginValidationError: If the manifest is invalid.
            PluginVersionError: If GeneralAI version constraint fails.
            PluginDependencyError: If dependencies are unsatisfied.
        """
        with self._lock:
            if manifest is None:
                manifests = self._loader.discover()
                manifest = manifests.get(name)
            if manifest is None:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not found",
                    module="plugins.manager",
                    context={"plugin": name},
                )

            if not manifest.enabled:
                raise PluginValidationError(
                    f"Plugin '{name}' is marked disabled",
                    module="plugins.manager",
                    context={"plugin": name},
                )

            self.check_version_compatibility(manifest)
            self._check_dependency_versions(name, manifest, self._loader.discover())

            state = PluginState(
                name=name,
                version=manifest.version,
                plugin_type=manifest.plugin_type,
                lifecycle_state=PluginLifecycleState.INSTALLED,
                installed_at=datetime.utcnow(),
                manifest=manifest,
            )
            if plugin is not None:
                self._registry.register_plugin(plugin, state)
            else:
                self._registry.add_state(name, state)
            log.info("Installed plugin '%s' v%s", name, manifest.version)
            return state

    # ------------------------------------------------------------------
    # Lifecycle: LOAD
    # ------------------------------------------------------------------

    def load(self, name: str) -> PluginBase:
        """Load (import + instantiate) a previously installed plugin.

        Args:
            name: Plugin name.

        Returns:
            The loaded :class:`PluginBase` instance.

        Raises:
            PluginNotFoundError: If not installed or discovered.
            PluginLoadError: If module import or instantiation fails.
        """
        with self._lock:
            try:
                state = self._registry.get_state(name)
            except PluginNotFoundError:
                raise PluginLoadError(
                    f"Plugin '{name}' not installed",
                    module="plugins.manager",
                    context={"plugin": name},
                )
            if state.lifecycle_state != PluginLifecycleState.INSTALLED:
                raise PluginLoadError(
                    f"Plugin '{name}' must be INSTALLED before LOAD "
                    f"(current: {state.lifecycle_state.value})",
                    module="plugins.manager",
                    context={
                        "plugin": name,
                        "current_state": state.lifecycle_state.value,
                    },
                )

            manifest = state.manifest
            if manifest is None:
                raise PluginLoadError(
                    f"Plugin '{name}' has no manifest",
                    module="plugins.manager",
                )

            existing = self._registry.get_plugin(name)
            if existing is not None:
                plugin = existing
            else:
                try:
                    plugin = self._loader.instantiate(manifest)
                except Exception as exc:
                    self._registry.set_state(name, PluginLifecycleState.ERROR)
                    raise PluginLoadError(
                        f"Failed to load plugin '{name}': {exc}",
                        module="plugins.manager",
                        cause=exc,
                        context={"plugin": name},
                    ) from exc

            if existing is None:
                self._registry.register_plugin(plugin, state)
            self._registry.set_state(name, PluginLifecycleState.LOADED)
            log.info("Loaded plugin '%s'", name)
            return plugin

    # ------------------------------------------------------------------
    # Lifecycle: ENABLE
    # ------------------------------------------------------------------

    def enable(self, name: str) -> list[str]:
        """Enable a loaded plugin — activates its capabilities.

        Args:
            name: Plugin name.

        Returns:
            List of registration IDs returned by ``plugin.enable()``.

        Raises:
            PluginNotFoundError: If not in registry.
            PluginEnableError: If the plugin is not in LOADED state.
        """
        with self._lock:
            plugin = self._registry.get_plugin(name)
            if plugin is None:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not loaded",
                    module="plugins.manager",
                    context={"plugin": name},
                )

            state = self._registry.get_state(name)
            if state.lifecycle_state not in (
                PluginLifecycleState.LOADED,
                PluginLifecycleState.DISABLED,
            ):
                raise PluginEnableError(
                    f"Plugin '{name}' must be LOADED or DISABLED "
                    f"before ENABLE (current: {state.lifecycle_state.value})",
                    module="plugins.manager",
                    context={
                        "plugin": name,
                        "current_state": state.lifecycle_state.value,
                    },
                )

            context = self._context
            if context is None:
                context = PluginContext()

            if self._sandbox is not None:
                try:
                    plugin_module = self._loader.load_module(name)
                    self._sandbox.wrap_module(plugin_module)
                except Exception as exc:
                    raise PluginEnableError(
                        f"Failed to apply sandbox to '{name}': {exc}",
                        module="plugins.manager",
                        cause=exc,
                        context={"plugin": name},
                    ) from exc

            try:
                registrations = _run_coroutine(_async_enable(plugin, context))
                if not isinstance(registrations, list):
                    registrations = []
            except Exception as exc:
                self._registry.set_state(name, PluginLifecycleState.ERROR)
                raise PluginEnableError(
                    f"Failed to enable plugin '{name}': {exc}",
                    module="plugins.manager",
                    cause=exc,
                    context={"plugin": name},
                ) from exc

            self._record_registrations(name, plugin, registrations, context)
            self._registry.set_state(name, PluginLifecycleState.ENABLED)
            state = self._registry.get_state(name)
            self._registry._states[name] = state.model_copy(
                update={"enabled_at": datetime.utcnow()}
            )
            log.info("Enabled plugin '%s'", name)
            return registrations

    # ------------------------------------------------------------------
    # Lifecycle: DISABLE
    # ------------------------------------------------------------------

    def disable(self, name: str) -> None:
        """Disable an enabled plugin — deactivates its capabilities.

        Args:
            name: Plugin name.

        Raises:
            PluginNotFoundError: If not in registry.
            PluginEnableError: If the plugin is not in ENABLED state.
        """
        with self._lock:
            state = self._registry.get_state(name)
            if state.lifecycle_state != PluginLifecycleState.ENABLED:
                raise PluginEnableError(
                    f"Plugin '{name}' must be ENABLED before DISABLE "
                    f"(current: {state.lifecycle_state.value})",
                    module="plugins.manager",
                    context={
                        "plugin": name,
                        "current_state": state.lifecycle_state.value,
                    },
                )

            plugin = self._registry.get_plugin(name)
            if plugin is None:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not found",
                    module="plugins.manager",
                    context={"plugin": name},
                )

            context = self._context or PluginContext()

            try:
                _run_coroutine(_async_disable(plugin, context))
            except Exception as exc:
                self._registry.set_state(name, PluginLifecycleState.ERROR)
                raise PluginEnableError(
                    f"Failed to disable plugin '{name}': {exc}",
                    module="plugins.manager",
                    cause=exc,
                    context={"plugin": name},
                ) from exc

            self._registry.set_state(name, PluginLifecycleState.DISABLED)
            state = self._registry.get_state(name)
            self._registry._states[name] = state.model_copy(
                update={"disabled_at": datetime.utcnow()}
            )
            log.info("Disabled plugin '%s'", name)

    # ------------------------------------------------------------------
    # Lifecycle: UNLOAD
    # ------------------------------------------------------------------

    def unload(self, name: str) -> None:
        """Unload a loaded (or disabled) plugin.

        Args:
            name: Plugin name.

        Raises:
            PluginNotFoundError: If not in registry.
            PluginUnloadError: If the plugin is in the wrong state.
        """
        with self._lock:
            state = self._registry.get_state(name)
            if state.lifecycle_state not in (
                PluginLifecycleState.LOADED,
                PluginLifecycleState.DISABLED,
            ):
                raise PluginUnloadError(
                    f"Plugin '{name}' must be LOADED or DISABLED "
                    f"before UNLOAD (current: {state.lifecycle_state.value})",
                    module="plugins.manager",
                    context={
                        "plugin": name,
                        "current_state": state.lifecycle_state.value,
                    },
                )

            plugin = self._registry.get_plugin(name)
            if plugin is None:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not found",
                    module="plugins.manager",
                    context={"plugin": name},
                )

            context = self._context or PluginContext()

            try:
                _run_coroutine(_async_unload(plugin, context))
            except Exception as exc:
                self._registry.set_state(name, PluginLifecycleState.ERROR)
                raise PluginUnloadError(
                    f"Failed to unload plugin '{name}': {exc}",
                    module="plugins.manager",
                    cause=exc,
                    context={"plugin": name},
                ) from exc

            self._registry.clear_registrations(name)
            self._registry.unregister_plugin(name)
            log.info("Unloaded plugin '%s'", name)

    # ------------------------------------------------------------------
    # Lifecycle: UNINSTALL
    # ------------------------------------------------------------------

    def uninstall(self, name: str) -> None:
        """Fully remove a plugin from the system.

        Calls UNLOAD (if loaded), then the plugin's ``uninstall`` hook.

        Args:
            name: Plugin name.
        """
        with self._lock:
            if self._registry.has_plugin(name):
                state = self._registry.get_state(name)
                if state.lifecycle_state in (
                    PluginLifecycleState.ENABLED,
                    PluginLifecycleState.LOADED,
                    PluginLifecycleState.DISABLED,
                ):
                    try:
                        self._do_unload(name)
                    except PluginUnloadError:
                        pass

            plugin = self._registry.get_plugin(name)
            context = self._context or PluginContext()

            if plugin is not None:
                try:
                    _run_coroutine(_async_uninstall(plugin, context))
                except Exception as exc:
                    raise PluginLoadError(
                        f"Failed to uninstall plugin '{name}': {exc}",
                        module="plugins.manager",
                        cause=exc,
                        context={"plugin": name},
                    ) from exc

            if self._registry.has_plugin(name):
                self._registry.clear_registrations(name)
                self._registry.unregister_plugin(name)
            log.info("Uninstalled plugin '%s'", name)

    def _do_unload(self, name: str) -> None:
        """Internal unload without the lock (called from uninstall)."""
        state = self._registry.get_state(name)
        if state.lifecycle_state not in (
            PluginLifecycleState.LOADED,
            PluginLifecycleState.DISABLED,
        ):
            raise PluginUnloadError(
                f"Plugin '{name}' must be LOADED or DISABLED before UNLOAD",
                module="plugins.manager",
                context={"plugin": name},
            )

        plugin = self._registry.get_plugin(name)
        if plugin is None:
            raise PluginNotFoundError(
                f"Plugin '{name}' not found",
                module="plugins.manager",
                context={"plugin": name},
            )

        context = self._context or PluginContext()
        try:
            _run_coroutine(_async_unload(plugin, context))
        except Exception as exc:
            self._registry.set_state(name, PluginLifecycleState.ERROR)
            raise PluginUnloadError(
                f"Failed to unload plugin '{name}': {exc}",
                module="plugins.manager",
                cause=exc,
                context={"plugin": name},
            ) from exc

        self._registry.clear_registrations(name)
        self._registry.unregister_plugin(name)

    # ------------------------------------------------------------------
    # Registration tracking
    # ------------------------------------------------------------------

    def _record_registrations(
        self,
        name: str,
        plugin: PluginBase,
        registration_ids: list[str],
        context: PluginContext,
    ) -> None:
        """Track what the plugin registered so disable can undo it."""
        for reg_id in registration_ids:
            reg = PluginRegistration(
                plugin_name=name,
                plugin_type=plugin.plugin_type,
                registration_id=reg_id,
                registry_target=reg_id,
            )
            self._registry.add_registration(name, reg)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def discover_and_install_all(self) -> list[PluginState]:
        """Discover all plugins and install every enabled one.

        Returns:
            List of installed plugin states, in dependency order.
        """
        manifests = self.discover()
        order = self.resolve_load_order(manifests)

        installed: list[PluginState] = []
        for name in order:
            manifest = manifests[name]
            if not manifest.enabled:
                log.info("Skipping disabled plugin '%s'", name)
                continue
            try:
                state = self.install(name, manifest)
                installed.append(state)
            except Exception as exc:
                log.error("Failed to install '%s': %s", name, exc)
                continue
        return installed

    def enable_all(self) -> list[str]:
        """Enable all installed plugins in dependency order.

        Returns:
            List of enabled plugin names.
        """
        installed: list[str] = []
        for name in self._registry.list_plugins():
            state = self._registry.get_state(name)
            if state.lifecycle_state == PluginLifecycleState.INSTALLED:
                installed.append(name)

        manifests: dict[str, PluginManifest] = {}
        for name in installed:
            state = self._registry.get_state(name)
            if state.manifest is not None:
                manifests[name] = state.manifest
        order = self._sort_by_deps(manifests)

        enabled: list[str] = []
        for name in order:
            try:
                self.enable(name)
                enabled.append(name)
            except Exception as exc:
                log.error("Failed to enable '%s': %s", name, exc)
                continue
        return enabled

    def _sort_by_deps(self, manifests: dict[str, PluginManifest]) -> list[str]:
        """Sort plugin names by dependency order."""
        visited: set[str] = set()
        order: list[str] = []

        def _visit(name: str) -> None:
            if name in visited or name not in manifests:
                return
            for dep in manifests[name].effective_dependencies:
                if dep.name in manifests:
                    _visit(dep.name)
            visited.add(name)
            order.append(name)

        for name in manifests:
            _visit(name)
        return order

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> PluginBase | None:
        """Return a plugin instance by name, or ``None``."""
        return self._registry.get_plugin(name)

    def get_state(self, name: str) -> PluginState:
        """Return the lifecycle state for *name*."""
        return self._registry.get_state(name)

    def list_plugins(self, plugin_type: PluginType | None = None) -> list[str]:
        """Return plugin names, optionally filtered by type."""
        with self._lock:
            if plugin_type is None:
                return self._registry.list_plugins()
            return self._registry.list_by_type(plugin_type)

    def list_registrations(self, name: str) -> list[PluginRegistration]:
        """Return all registrations for *name*."""
        return self._registry.get_registrations(name)

    @property
    def total_count(self) -> int:
        """Return the total number of registered (loaded) plugins."""
        return self._registry.total_count()
