"""Dependency-injection wiring for the tool framework."""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.tools.executor import ToolExecutor
from app.tools.permissions import PermissionSystem
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def register_tool_components(container: DependencyContainer) -> None:
    """Register tool framework components with a DI container.

    Registers the tool registry, permission system, and executor as
    singletons.  The executor is wired to the registered registry and
    permission system.  The registry starts empty; callers populate it
    via :meth:`ToolRegistry.discover` or :meth:`ToolRegistry.register`.

    Args:
        container: The application's :class:`DependencyContainer`.
    """

    def _build_executor() -> ToolExecutor:
        return ToolExecutor(
            registry=container.resolve(ToolRegistry),
            permission_system=container.resolve(PermissionSystem),
        )

    container.register_singleton(ToolRegistry)
    container.register_singleton(PermissionSystem)
    container.register_singleton(ToolExecutor, factory=_build_executor)
    log.info("Registered %d tool components", 3)
