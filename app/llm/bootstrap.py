"""Dependency-injection wiring for the LLM provider layer."""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.llm.factory import ProviderFactory
from app.llm.registry import ProviderRegistry
from app.llm.router_bootstrap import register_router_components

log = logging.getLogger(__name__)


def register_llm_components(container: DependencyContainer) -> None:
    """Register LLM provider components with a DI container.

    Registers the provider registry and factory as singletons.  The
    factory builds provider instances on demand, keeping credential
    configuration out of the container itself.

    Also registers the Multi-LLM Intelligence Layer (router, health
    monitor, etc.) via :func:`register_router_components`.

    Args:
        container: The application's :class:`DependencyContainer`.
    """
    if not container.has(ProviderRegistry):
        container.register_singleton(ProviderRegistry)
    if not container.has(ProviderFactory):
        container.register_singleton(ProviderFactory)
    register_router_components(container)
    log.info("Registered LLM provider components")
