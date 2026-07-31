"""Kernel bootstrap — wires Cognitive Kernel into the application framework.

Connects all kernel modules (orchestrator, engines, registries) to the
DependencyContainer, LifecycleManager, and EventBus from ``app.core``.
"""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.core.events import EventBus
from app.core.lifecycle import LifecycleManager
from app.core.constants.lifecycle import (
    HOOK_BEFORE_INIT,
    HOOK_AFTER_CONFIG,
    HOOK_AFTER_START,
)
from app.kernel.orchestrator import CognitiveOrchestrator
from app.kernel.agent.bootstrap import register_agent_components
from app.kernel.perception.engine import PerceptionEngine
from app.kernel.intent.engine import IntentEngine
from app.kernel.goals.engine import GoalEngine
from app.kernel.planning.engine import PlanningEngine
from app.kernel.reasoning.engine import ReasoningEngine
from app.kernel.decision.engine import DecisionEngine
from app.kernel.capability.manager import CapabilityManager
from app.kernel.policy.engine import PolicyEngine
from app.kernel.skills.executor import SkillSelector, SkillExecutor
from app.kernel.tools.executor import ToolResolver, ToolExecutor
from app.kernel.reflection.engine import ReflectionEngine
from app.kernel.experience.engine import ExperienceEngine, ExperienceStore
from app.kernel.context.manager import ContextManager, ContextBuilder, ContextPruner
from app.kernel.state.manager import StateManager
from app.kernel.pipeline.executor import PipelineExecutor
from app.kernel.models.router import ModelRouter
from app.kernel.response.builder import ResponseBuilder

log = logging.getLogger(__name__)


def register_kernel_components(container: DependencyContainer) -> None:
    """Register all kernel singleton instances with the DI container.

    Args:
        container: The application's ``DependencyContainer``.
    """
    # The orchestrator is registered with a factory that shares the
    # caller's container so it can resolve the AgentRuntime (and its
    # tool registry) from the same DI graph.
    container.register_singleton(
        CognitiveOrchestrator,
        factory=lambda: CognitiveOrchestrator(container=container),
    )
    container.register_singleton(PerceptionEngine)
    container.register_singleton(IntentEngine)
    container.register_singleton(GoalEngine)
    container.register_singleton(PlanningEngine)
    container.register_singleton(ReasoningEngine)
    container.register_singleton(DecisionEngine)
    container.register_singleton(CapabilityManager)
    container.register_singleton(PolicyEngine)
    container.register_singleton(SkillSelector)
    container.register_singleton(SkillExecutor)
    container.register_singleton(ToolResolver)
    container.register_singleton(ToolExecutor)
    container.register_singleton(ReflectionEngine)
    container.register_singleton(ExperienceEngine)
    container.register_singleton(ExperienceStore)
    container.register_singleton(ContextManager)
    container.register_singleton(ContextBuilder)
    container.register_singleton(ContextPruner)
    container.register_singleton(StateManager)
    container.register_singleton(PipelineExecutor)
    container.register_singleton(ModelRouter)
    container.register_singleton(ResponseBuilder)

    # Agent runtime (engines, policies, loop, runtime) — skips any
    # engines and tools already registered above.
    register_agent_components(container)

    log.info("Registered kernel + agent components with DI container")


def register_kernel_lifecycle_hooks(lifecycle: LifecycleManager) -> None:
    """Register kernel hooks with the LifecycleManager.

    Args:
        lifecycle: The application's ``LifecycleManager``.
    """

    async def _init_kernel() -> None:
        log.info("Cognitive Kernel init phase (placeholder)")

    async def _config_kernel() -> None:
        log.info("Cognitive Kernel config phase (placeholder)")

    async def _start_kernel() -> None:
        log.info("Cognitive Kernel start phase (placeholder)")

    lifecycle.register_hook(HOOK_BEFORE_INIT, _init_kernel)
    lifecycle.register_hook(HOOK_AFTER_CONFIG, _config_kernel)
    lifecycle.register_hook(HOOK_AFTER_START, _start_kernel)
    log.info("Registered %d kernel lifecycle hooks", 3)


def bootstrap_kernel(
    container: DependencyContainer,
    lifecycle: LifecycleManager,
    event_bus: EventBus | None = None,
) -> CognitiveOrchestrator:
    """Convenience: wire kernel components, hooks, and return the orchestrator.

    Args:
        container: The application's ``DependencyContainer``.
        lifecycle: The application's ``LifecycleManager``.
        event_bus: Optional ``EventBus`` (reserved for future event wiring).

    Returns:
        The resolved ``CognitiveOrchestrator`` singleton.
    """
    register_kernel_components(container)
    register_kernel_lifecycle_hooks(lifecycle)
    if event_bus:
        log.debug(
            "EventBus provided — kernel event wiring will be added in a future phase"
        )
    orchestrator = container.resolve(CognitiveOrchestrator)
    log.info("Kernel bootstrap complete")
    return orchestrator
