"""DI bootstrap for the plugin system.

Registers :class:`PluginManager`, :class:`PluginLoader`,
:class:`PluginRegistry`, and :class:`PluginSandbox` as singletons
in the :class:`DependencyContainer`.
"""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.plugins.manager import PluginManager
from app.plugins.loader import PluginLoader
from app.plugins.registry import PluginRegistry
from app.plugins.sandbox import PluginSandbox

log = logging.getLogger(__name__)


def register_plugin_components(
    container: DependencyContainer,
    plugin_dirs: list[str] | None = None,
    allowed_modules: set[str] | None = None,
) -> None:
    """Register plugin system components with the DI container.

    Idempotent: re-calling is safe (uses ``container.has`` guards).

    Args:
        container: The application's :class:`DependencyContainer`.
        plugin_dirs: Optional plugin scan directories.
        allowed_modules: Optional allowed module set for sandbox.
    """
    if not container.has(PluginRegistry):
        container.register_singleton(PluginRegistry, factory=PluginRegistry)
    if not container.has(PluginLoader):
        container.register_singleton(
            PluginLoader, factory=lambda: PluginLoader(plugin_dirs=plugin_dirs)
        )
    if not container.has(PluginSandbox):
        container.register_singleton(
            PluginSandbox,
            factory=lambda: PluginSandbox(allowed_modules=allowed_modules),
        )
    if not container.has(PluginManager):
        container.register_singleton(
            PluginManager,
            factory=lambda: PluginManager(
                context=None,
                plugin_dirs=plugin_dirs,
                sandbox=container.resolve(PluginSandbox),
            ),
        )
    log.info("Registered %d plugin system components", 4)
