"""Plugin discovery and loading system.

Discovers plugins via Python entry points or directory scanning,
validates their metadata and dependency graph, and provides the
loaded instances to the rest of the platform.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from app.core.constants.plugins import (
    PLUGIN_ENTRYPOINT_GROUP,
    PLUGIN_MANIFEST_FILENAME,
    PLUGIN_SCAN_DIRS,
)
from app.core.constants.paths import PLUGIN_MAX_DEPENDENCY_DEPTH
from app.core.exceptions.plugin import (
    PluginDependencyError,
    PluginLoadError,
)
from app.core.interfaces.iplugin import IPlugin
from app.core.plugins.plugin_metadata import PluginMetadata

log = logging.getLogger(__name__)


class PluginLoader:
    """Discovers, validates, and loads plugins.

    Usage::

        loader = PluginLoader()
        loader.discover()
        loader.load_all()
        plugin = loader.get_plugin("my-plugin")
    """

    def __init__(self, plugin_dirs: list[str] | None = None) -> None:
        self._plugin_dirs: list[str] = plugin_dirs or list(PLUGIN_SCAN_DIRS)
        self._discovered: dict[str, PluginMetadata] = {}
        self._loaded: dict[str, IPlugin] = {}
        self._failed: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> dict[str, PluginMetadata]:
        """Discover available plugins.

        Searches:
        1. Python entry points (``generalai.plugins`` group).
        2. Plugin directories listed in :attr:`_plugin_dirs`.

        Returns:
            Dict mapping plugin name to metadata.
        """
        self._discovered.clear()
        self._discovered.update(self._discover_entry_points())
        self._discovered.update(self._discover_directories())
        log.info("Discovered %d plugin(s)", len(self._discovered))
        return dict(self._discovered)

    def _discover_entry_points(self) -> dict[str, PluginMetadata]:
        """Discover plugins registered as setuptools entry points."""
        discovered: dict[str, PluginMetadata] = {}
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group=PLUGIN_ENTRYPOINT_GROUP)
            for ep in eps:
                metadata = PluginMetadata(
                    name=ep.name,
                    module=ep.value,
                    package=ep.dist.name if ep.dist else "",
                    version=ep.dist.version if ep.dist else "0.1.0",
                )
                discovered[ep.name] = metadata
        except Exception as exc:
            log.warning("Entry-point discovery failed: %s", exc)
        return discovered

    def _discover_directories(self) -> dict[str, PluginMetadata]:
        """Discover plugins by scanning filesystem directories."""
        discovered: dict[str, PluginMetadata] = {}
        for dir_path_str in self._plugin_dirs:
            dir_path = Path(dir_path_str)
            if not dir_path.is_dir():
                continue
            for candidate in dir_path.iterdir():
                if not candidate.is_dir():
                    continue
                manifest = candidate / PLUGIN_MANIFEST_FILENAME
                if manifest.is_file():
                    try:
                        import json

                        data = json.loads(manifest.read_text(encoding="utf-8"))
                        metadata = PluginMetadata(**data)
                        discovered[metadata.name] = metadata
                    except Exception as exc:
                        log.warning("Failed to load manifest %s: %s", manifest, exc)
        return discovered

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> dict[str, IPlugin]:
        """Load all discovered and enabled plugins.

        Validates the dependency graph before loading.  Loaded
        plugins are available via :meth:`get_plugin`.

        Returns:
            Dict mapping plugin name to loaded instance.
        """
        self._validate_dependencies()
        self._loaded.clear()
        self._failed.clear()

        # Load in dependency order
        ordered = self._resolve_load_order()
        for name in ordered:
            meta = self._discovered[name]
            if not meta.enabled:
                log.info("Skipping disabled plugin '%s'", name)
                continue
            try:
                plugin = self._load_single(meta)
                self._loaded[name] = plugin
                log.info("Loaded plugin '%s' v%s", name, meta.version)
            except Exception as exc:
                self._failed[name] = str(exc)
                log.error("Failed to load plugin '%s': %s", name, exc)

        return dict(self._loaded)

    def _load_single(self, metadata: PluginMetadata) -> IPlugin:
        """Import and instantiate a single plugin."""
        try:
            mod = importlib.import_module(metadata.module)
        except ImportError as exc:
            raise PluginLoadError(
                f"Cannot import module '{metadata.module}'",
                module="plugins",
                cause=exc,
                context={"plugin": metadata.name},
            ) from exc

        # Look for a well-known factory or class
        plugin_cls = getattr(mod, "Plugin", None) or getattr(mod, "plugin", None)
        if plugin_cls is None:
            # Walk module attributes for an IPlugin subclass
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, IPlugin)
                    and attr is not IPlugin
                ):
                    plugin_cls = attr
                    break

        if plugin_cls is None:
            raise PluginLoadError(
                f"No IPlugin subclass found in module '{metadata.module}'",
                module="plugins",
                context={"plugin": metadata.name},
            )

        instance = plugin_cls()
        return instance

    # ------------------------------------------------------------------
    # Dependency validation
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Check that all declared plugin dependencies exist."""
        for name, meta in self._discovered.items():
            for dep in meta.dependencies:
                if dep not in self._discovered:
                    raise PluginDependencyError(
                        f"Plugin '{name}' depends on '{dep}' which is not discovered",
                        module="plugins",
                        context={"plugin": name, "dependency": dep},
                    )

    def _resolve_load_order(self) -> list[str]:
        """Topological sort of plugins by dependency graph.

        Returns:
            Plugin names in load order (dependencies first).
        """
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str, depth: int = 0) -> None:
            if depth > PLUGIN_MAX_DEPENDENCY_DEPTH:
                raise PluginDependencyError(
                    f"Dependency chain too deep for '{name}' "
                    f"(max {PLUGIN_MAX_DEPENDENCY_DEPTH})",
                    module="plugins",
                )
            if name in visited:
                return
            visited.add(name)
            meta = self._discovered.get(name)
            if meta:
                for dep in meta.dependencies:
                    if dep not in visited:
                        visit(dep, depth + 1)
            order.append(name)

        for name in self._discovered:
            if name not in visited:
                visit(name)

        return order

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> IPlugin | None:
        """Return a loaded plugin by name."""
        return self._loaded.get(name)

    def get_metadata(self, name: str) -> PluginMetadata | None:
        """Return discovered metadata for *name*."""
        return self._discovered.get(name)

    @property
    def discovered(self) -> dict[str, PluginMetadata]:
        """Return all discovered plugin metadata."""
        return dict(self._discovered)

    @property
    def loaded(self) -> dict[str, IPlugin]:
        """Return all successfully loaded plugin instances."""
        return dict(self._loaded)

    @property
    def failed(self) -> dict[str, str]:
        """Return plugins that failed to load and their error messages."""
        return dict(self._failed)
