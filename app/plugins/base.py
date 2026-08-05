"""Plugin base classes.

Defines :class:`PluginBase` — the abstract contract every plugin must
satisfy — and :class:`PluginContext` — the capability bundle handed to
plugins during lifecycle hooks for dynamic registration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.plugins.models import PluginManifest, PluginType

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginContext:
    """Bundle of system handles available to a plugin during hooks.

    All attributes are optional — plugins may use only the subset
    relevant to their :class:`PluginType`.

    Attributes:
        tool_registry: :class:`app.tools.registry.ToolRegistry` instance.
        agent_manager: :class:`app.agents.manager.AgentManager` instance.
        provider_registry: :class:`app.llm.registry.ProviderRegistry` instance.
        provider_factory: :class:`app.llm.factory.ProviderFactory` instance.
        memory_engine: :class:`app.kernel.memory.engine.MemoryEngine` instance.
        fastapi_app: :class:`fastapi.FastAPI` instance (for API_ROUTE plugins).
        container: :class:`app.core.container.DependencyContainer` instance.
        workflow_service: :class:`app.automation.workflow.WorkflowService` instance.
        workflow_registry: :class:`app.automation.registries.WorkflowRegistry` instance.
        step_type_registry: :class:`app.automation.registries.StepTypeRegistry` instance.
        logger: Logger scoped to the plugin.
    """

    tool_registry: Any | None = None
    agent_manager: Any | None = None
    provider_registry: Any | None = None
    provider_factory: Any | None = None
    memory_engine: Any | None = None
    fastapi_app: Any | None = None
    container: Any | None = None
    workflow_service: Any | None = None
    workflow_registry: Any | None = None
    step_type_registry: Any | None = None
    logger: logging.Logger = field(default_factory=lambda: log)

    def log_info(self, msg: str, *args: Any) -> None:
        """Log an info message scoped to this context."""
        self.logger.info(msg, *args)

    def log_warning(self, msg: str, *args: Any) -> None:
        """Log a warning message scoped to this context."""
        self.logger.warning(msg, *args)

    def log_error(self, msg: str, *args: Any) -> None:
        """Log an error message scoped to this context."""
        self.logger.error(msg, *args)


class PluginBase(ABC):
    """Abstract base class that every plugin must implement.

    Lifecycle hooks are intentionally async so plugins can perform
    I/O-bound setup (network connections, DB migrations, etc.).

    Subclasses must set ``self.metadata`` (a :class:`PluginManifest`)
    either as a class attribute or by overriding the property.
    """

    metadata: PluginManifest

    def __init__(self) -> None:
        self._initialized: bool = False

    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return the manifest describing this plugin."""

    @property
    def plugin_type(self) -> PluginType:
        """Return the plugin's capability type from its manifest."""
        return self.manifest.plugin_type

    @property
    def is_enabled(self) -> bool:
        """Whether this plugin is currently enabled."""
        return self.manifest.enabled

    async def initialize(self) -> None:
        """Called once after instantiation during INSTALL.

        Override to perform heavy setup (schema creation, etc.).
        Must set ``self._initialized = True`` on success.
        """
        self._initialized = True

    async def install(self, context: PluginContext) -> None:
        """Perform one-time installation (filesystem, DB, etc.).

        Called once during the INSTALL lifecycle stage, before
        ``load``.  Safe to call multiple times — no-op if already
        installed.
        """

    async def load(self, context: PluginContext) -> None:
        """Bind capabilities to system registries.

        Called during LOAD.  Override to register tools, agents,
        routes, providers, etc.  Return value is ignored.
        """

    async def enable(self, context: PluginContext) -> list[str]:
        """Activate the plugin's capabilities.

        Called during ENABLE.  Should call ``context.tool_registry``,
        ``context.fastapi_app``, etc. to register functionality.

        Returns:
            List of registration IDs (e.g. tool names) for later
            unregistration.
        """
        return []

    async def disable(self, context: PluginContext) -> None:
        """Deactivate the plugin's capabilities.

        Called during DISABLE.  Should gracefully deregister everything
        registered in :meth:`enable`.
        """

    async def unregister(self, context: PluginContext) -> None:
        """Remove all registrations made during enable.

        Called during DISABLE before :meth:`disable`.  Default
        implementation deregisters tracked registrations from the
        container's :class:`PluginRegistry`.
        """

    async def unload(self, context: PluginContext) -> None:
        """Tear down resources created in :meth:`load`.

        Called during UNLOAD.  Override to unregister tools, close
        connections, clear state.
        """

    async def uninstall(self, context: PluginContext) -> None:
        """Remove everything the plugin created (filesystem, DB, etc.).

        Called during UNINSTALL.  Should be idempotent.
        """

    async def cleanup(self, context: PluginContext) -> None:
        """Final cleanup called after UNLOAD or UNINSTALL.

        Ensures resources are freed even if other hooks raise.
        """
