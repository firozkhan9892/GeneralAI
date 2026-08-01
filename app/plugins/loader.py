"""Plugin loader — discovery and module instantiation.

Reuses :class:`app.core.plugins.plugin_loader.PluginLoader` for the
core discovery logic (entry points + directory scan) and extends it to:

- Produce rich :class:`PluginManifest` objects (with :class:`PluginType`
  and structured :class:`PluginDependency` versions).
- Instantiate :class:`PluginBase` subclasses via ``importlib``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.plugins.plugin_loader import PluginLoader as CorePluginLoader
from app.core.plugins.plugin_metadata import PluginMetadata
from app.plugins.exceptions import (
    PluginLoadError,
    PluginNotFoundError,
    PluginValidationError,
)
from app.plugins.models import PluginManifest

if TYPE_CHECKING:
    from app.plugins.base import PluginBase

log = logging.getLogger(__name__)


class PluginLoader:
    """Discovers plugins and loads their modules.

    Wraps :class:`~app.core.plugins.plugin_loader.PluginLoader` for
    backward-compatible discovery, adding manifest parsing and module
    import capabilities.

    Usage::

        loader = PluginLoader(plugin_dirs=["my_plugins"])
        manifests = loader.discover()
        plugin_module = loader.load_module(manifests["my_plugin"])
    """

    def __init__(self, plugin_dirs: list[str] | None = None) -> None:
        self._core_loader = CorePluginLoader(plugin_dirs=plugin_dirs)
        self._plugin_dirs: list[str] = plugin_dirs or list(
            __import__(
                "app.core.constants.plugins", fromlist=["PLUGIN_SCAN_DIRS"]
            ).PLUGIN_SCAN_DIRS
        )

    # ------------------------------------------------------------------
    # Discovery (delegates to core PluginLoader)
    # ------------------------------------------------------------------

    def discover(self) -> dict[str, PluginManifest]:
        """Discover available plugins.

        Returns:
            Dict mapping plugin name to :class:`PluginManifest`.
        """
        core_metadata = self._core_loader.discover()
        manifests: dict[str, PluginManifest] = {}

        for name, meta in core_metadata.items():
            manifests[name] = PluginManifest.from_metadata(meta)

        # Augment with directory-based manifests that may contain
        # plugin_type and structured dependencies.
        manifests.update(self._discover_directory_manifests())

        log.info("Discovered %d plugin(s)", len(manifests))
        return dict(manifests)

    def _discover_directory_manifests(self) -> dict[str, PluginManifest]:
        """Scan plugin directories for enhanced ``plugin.json`` manifests."""
        from app.core.constants.plugins import PLUGIN_MANIFEST_FILENAME

        manifests: dict[str, PluginManifest] = {}
        for dir_path_str in self._plugin_dirs:
            dir_path = Path(dir_path_str)
            if not dir_path.is_dir():
                continue
            for candidate in dir_path.iterdir():
                if not candidate.is_dir():
                    continue
                manifest_file = candidate / PLUGIN_MANIFEST_FILENAME
                if not manifest_file.is_file():
                    continue
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    log.warning("Failed to load manifest %s: %s", manifest_file, exc)
                    continue

                try:
                    manifest = PluginManifest(**data)
                    manifests[manifest.name] = manifest
                except Exception as exc:
                    log.warning("Invalid manifest %s: %s", manifest_file, exc)
                    continue
        return manifests

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def load_module(self, name_or_manifest: str | PluginManifest) -> object:
        """Import the module for a discovered plugin.

        Args:
            name_or_manifest: Plugin name (looked up in discovered) or
                a :class:`PluginManifest`.

        Returns:
            The imported module object.

        Raises:
            PluginNotFoundError: If the name is not discovered.
            PluginValidationError: If the manifest has no module path.
            PluginLoadError: If the import fails.
        """
        manifest = self._resolve_manifest(name_or_manifest)
        if not manifest.module:
            raise PluginValidationError(
                f"Plugin '{manifest.name}' has no module path",
                module="plugins.loader",
                context={"plugin": manifest.name},
            )
        try:
            import importlib

            module = importlib.import_module(manifest.module)
        except ImportError as exc:
            raise PluginLoadError(
                f"Cannot import module '{manifest.module}'",
                module="plugins.loader",
                cause=exc,
                context={"plugin": manifest.name, "module": manifest.module},
            ) from exc
        return module

    def instantiate(self, name_or_manifest: str | PluginManifest) -> "PluginBase":
        """Load the module and instantiate the plugin class.

        Looks for ``Plugin`` (or ``plugin``) attribute, or the first
        :class:`PluginBase` subclass on the module.

        Args:
            name_or_manifest: Plugin name or manifest.

        Returns:
            A :class:`PluginBase` instance.

        Raises:
            PluginNotFoundError: If the name is not discovered.
            PluginLoadError: If the class cannot be found or instantiated.
        """
        from app.plugins.base import PluginBase

        module = self.load_module(name_or_manifest)

        plugin_cls = getattr(module, "Plugin", None) or getattr(module, "plugin", None)
        if plugin_cls is None:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, PluginBase)
                    and attr is not PluginBase
                ):
                    plugin_cls = attr
                    break

        if plugin_cls is None:
            raise PluginLoadError(
                f"No PluginBase subclass found in module "
                f"'{self._resolve_manifest(name_or_manifest).module}'",
                module="plugins.loader",
                context={
                    "plugin": name_or_manifest
                    if isinstance(name_or_manifest, str)
                    else name_or_manifest.name
                },
            )

        try:
            instance = plugin_cls()
        except Exception as exc:
            raise PluginLoadError(
                f"Failed to instantiate plugin class: {exc}",
                module="plugins.loader",
                cause=exc,
            ) from exc

        return instance

    def _resolve_manifest(
        self, name_or_manifest: str | PluginManifest
    ) -> PluginManifest:
        """Resolve *name_or_manifest* to a :class:`PluginManifest`."""
        if isinstance(name_or_manifest, PluginManifest):
            return name_or_manifest
        discovered = self._core_loader.discovered
        core_meta = discovered.get(name_or_manifest)
        if core_meta is None:
            raise PluginNotFoundError(
                f"Plugin '{name_or_manifest}' not found in discovered plugins",
                module="plugins.loader",
                context={"plugin": name_or_manifest},
            )
        return PluginManifest.from_metadata(core_meta)

    # ------------------------------------------------------------------
    # Delegation to core loader
    # ------------------------------------------------------------------

    @property
    def discovered(self) -> dict[str, PluginMetadata]:
        """Return discovered metadata from the core loader."""
        return self._core_loader.discovered

    @property
    def loaded(self) -> dict[str, Any]:
        """Return loaded IPlugin instances from the core loader."""
        return self._core_loader.loaded

    @property
    def failed(self) -> dict[str, str]:
        """Return plugins that failed to load and their error messages."""
        return self._core_loader.failed
