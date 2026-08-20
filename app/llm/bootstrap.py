"""Dependency-injection wiring for the LLM provider layer."""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.llm.config import (
    LLMSettings,
    build_llm_settings_from_env,
    is_provider_configured,
)
from app.llm.factory import ProviderFactory
from app.llm.llm_router import LLMRouter
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


def register_default_llm_providers(
    registry: ProviderRegistry,
    factory: ProviderFactory,
    settings: LLMSettings | None = None,
    *,
    router: LLMRouter | None = None,
) -> None:
    """Register default LLM providers based on configuration.

    In ``API_MODE=mock`` (the default) a deterministic mock provider is
    registered and requires no credentials.  In ``API_MODE=real`` only
    providers whose configuration is present in *settings* are registered,
    so the app never starts with a real provider that lacks credentials.

    The function is idempotent: providers already registered under a name
    are left untouched, and real providers with missing configuration are
    skipped.  Credentials are never logged.

    Additionally, if *router* is provided, each registered provider is
    wired into the router's health monitoring, circuit breaker, and
    fallback infrastructure so that provider failures are tracked and
    fallback chains are functional.

    Args:
        registry: The provider registry to populate.
        factory: The factory used to build provider instances.
        settings: LLM configuration.  Defaults to the environment.
        router: Optional LLMRouter to wire providers into.  When ``None``,
            provider registration behaves as before (registry-only).
    """
    settings = settings or build_llm_settings_from_env()

    if settings.api_mode == "mock":
        if not registry.has("mock"):
            provider = factory.create_mock()
            if router is not None:
                router.register_provider(provider)
            else:
                registry.register(provider)
            log.info("Registered default LLM provider 'mock' (API_MODE=mock)")
        return

    for config in settings.providers:
        name = config.name
        if registry.has(name) or not is_provider_configured(config):
            continue
        kwargs: dict[str, str] = {}
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        if config.model is not None:
            kwargs["model"] = config.model
        provider = factory.create(name, **kwargs)
        if router is not None:
            router.register_provider(provider)
        else:
            registry.register(provider)
        log.info("Registered default LLM provider '%s' (API_MODE=real)", name)
