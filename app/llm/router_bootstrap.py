"""Dependency-injection wiring for the Multi-LLM Intelligence Layer.

Registers the router and all its collaborators as singletons in the
application's :class:`DependencyContainer`.  Idempotent — re-calling
is safe (guarded by ``container.has``).
"""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.llm.analytics import LLMAnalytics
from app.llm.capability_matrix import CapabilityMatrix
from app.llm.circuit_breaker import CircuitBreaker
from app.llm.cost_optimizer import CostOptimizer
from app.llm.fallback_manager import FallbackManager
from app.llm.health_monitor import ProviderHealthMonitor
from app.llm.llm_router import LLMRouter
from app.llm.load_balancer import LoadBalancer
from app.llm.policy_engine import PolicyEngine
from app.llm.prompt_cache import PromptCache
from app.llm.registry import ProviderRegistry
from app.llm.request_queue import RequestQueue
from app.llm.unified_streamer import UnifiedStreamer

log = logging.getLogger(__name__)


def register_router_components(container: DependencyContainer) -> None:
    """Register all Multi-LLM Intelligence Layer components.

    Registers each collaborator as a singleton, then wires the
    :class:`LLMRouter` last so it can resolve its dependencies.

    Args:
        container: The application's :class:`DependencyContainer`.
    """
    for component in (
        ProviderHealthMonitor,
        CapabilityMatrix,
        CostOptimizer,
        LoadBalancer,
        CircuitBreaker,
        FallbackManager,
        RequestQueue,
        PromptCache,
        PolicyEngine,
        LLMAnalytics,
        UnifiedStreamer,
    ):
        if not container.has(component):
            container.register_singleton(component, factory=component)

    if not container.has(LLMRouter):
        container.register_singleton(LLMRouter, factory=_make_router(container))

    log.info("Registered %d Multi-LLM Intelligence Layer components", 13)


def _make_router(container: DependencyContainer):
    """Return a factory building an :class:`LLMRouter` from the container."""

    def _factory() -> LLMRouter:
        return LLMRouter(
            registry=container.resolve(ProviderRegistry),
            health_monitor=container.resolve(ProviderHealthMonitor),
            capability_matrix=container.resolve(CapabilityMatrix),
            cost_optimizer=container.resolve(CostOptimizer),
            load_balancer=container.resolve(LoadBalancer),
            circuit_breakers={},
            fallback_manager=container.resolve(FallbackManager),
            request_queue=container.resolve(RequestQueue),
            prompt_cache=container.resolve(PromptCache),
            policy_engine=container.resolve(PolicyEngine),
            analytics=container.resolve(LLMAnalytics),
            unified_streamer=container.resolve(UnifiedStreamer),
        )

    return _factory
