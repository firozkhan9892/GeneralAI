"""Dependency-injection wiring for the agent manager.

Registers the :class:`AgentManager` and its collaborators (session store,
session registry, and the agent runtime) with the application's
:class:`DependencyContainer`.
"""

from __future__ import annotations

import logging

from app.agents.manager import AgentManager
from app.agents.persistence import InMemorySessionStore, SessionStore
from app.agents.registry import SessionRegistry
from app.core.container import DependencyContainer
from app.kernel.agent.bootstrap import register_agent_components
from app.kernel.agent.runtime import AgentRuntime

log = logging.getLogger(__name__)


def register_agent_manager_components(container: DependencyContainer) -> None:
    """Register agent manager components with the DI container.

    Ensures the agent runtime (and its engines/tools) are registered,
    then wires the session store, registry, and manager as singletons.

    Args:
        container: The application's ``DependencyContainer``.
    """
    register_agent_components(container)

    if not container.has(SessionStore):
        container.register_singleton(SessionStore, factory=_make_session_store)
    if not container.has(SessionRegistry):
        container.register_singleton(
            SessionRegistry, factory=_make_session_registry(container)
        )
    if not container.has(AgentManager):
        container.register_singleton(
            AgentManager, factory=_make_agent_manager(container)
        )
    log.info("Registered agent manager components with DI container")


def _make_session_store() -> SessionStore:
    """Return a default in-memory session store."""
    return InMemorySessionStore()


def _make_session_registry(container: DependencyContainer):
    """Return a factory building a SessionRegistry from the container."""

    def _factory() -> SessionRegistry:
        store = container.resolve(SessionStore)  # type: ignore[type-abstract]
        return SessionRegistry(store=store)

    return _factory


def _make_agent_manager(container: DependencyContainer):
    """Return a factory building an AgentManager from the container."""

    def _factory() -> AgentManager:
        return AgentManager(
            runtime=container.resolve(AgentRuntime),
            registry=container.resolve(SessionRegistry),
        )

    return _factory
