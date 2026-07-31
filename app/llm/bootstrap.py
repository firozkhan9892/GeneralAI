"""Dependency-injection wiring for the LLM provider layer."""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.llm.factory import ProviderFactory
from app.llm.registry import ProviderRegistry

log = logging.getLogger(__name__)


def register_llm_components(container: DependencyContainer) -> None:
    """Register LLM provider components with a DI container.

    Registers the provider registry and factory as singletons.  The
    factory builds provider instances on demand, keeping credential
    configuration out of the container itself.

    Args:
        container: The application's :class:`DependencyContainer`.
    """
    container.register_singleton(ProviderRegistry)
    container.register_singleton(ProviderFactory)
    log.info("Registered %d LLM provider components", 2)
