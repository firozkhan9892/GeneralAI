"""Dependency-injection wiring for the agent runtime.

Registers the AgentRuntime (the execution brain) and its collaborating
engines with the application's DependencyContainer.
"""

from __future__ import annotations

import logging

from app.core.container import DependencyContainer
from app.kernel.agent.loop import AgentLoop
from app.kernel.agent.policies import FallbackPolicy, RetryPolicy
from app.kernel.agent.runtime import AgentRuntime
from app.kernel.decision.engine import DecisionEngine
from app.kernel.experience.engine import ExperienceEngine
from app.kernel.goals.engine import GoalEngine
from app.kernel.intent.engine import IntentEngine
from app.kernel.memory.engine import MemoryEngine
from app.kernel.perception.engine import PerceptionEngine
from app.kernel.planning.engine import PlanningEngine
from app.kernel.policy.engine import PolicyEngine
from app.kernel.reasoning.engine import ReasoningEngine
from app.kernel.reflection.engine import ReflectionEngine
from app.kernel.response.builder import ResponseBuilder
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_ENGINES = (
    PerceptionEngine,
    IntentEngine,
    GoalEngine,
    PlanningEngine,
    ReasoningEngine,
    DecisionEngine,
    PolicyEngine,
    MemoryEngine,
    ExperienceEngine,
    ReflectionEngine,
    ResponseBuilder,
)


def register_agent_components(container: DependencyContainer) -> None:
    """Register agent runtime components with the DI container.

    Registers the collaborating engines (skipping any already registered
    by the kernel bootstrap), the Phase-5 tool components, agent
    policies, the agent loop, and the AgentRuntime as singletons.

    Args:
        container: The application's ``DependencyContainer``.
    """
    for engine in _ENGINES:
        if not container.has(engine):
            container.register_singleton(engine)

    if not container.has(ToolRegistry):
        container.register_singleton(ToolRegistry)
    if not container.has(ToolExecutor):
        container.register_singleton(
            ToolExecutor, factory=_make_tool_executor(container)
        )

    if not container.has(RetryPolicy):
        container.register_singleton(RetryPolicy)
    if not container.has(FallbackPolicy):
        container.register_singleton(FallbackPolicy)
    if not container.has(AgentLoop):
        container.register_singleton(AgentLoop, factory=_make_agent_loop(container))
    if not container.has(AgentRuntime):
        container.register_singleton(
            AgentRuntime, factory=_make_agent_runtime(container)
        )
    log.info("Registered agent runtime components with DI container")


def _make_tool_executor(container: DependencyContainer):
    """Return a factory building a ToolExecutor from the container registry."""

    def _factory() -> ToolExecutor:
        return ToolExecutor(registry=container.resolve(ToolRegistry))

    return _factory


def _make_agent_loop(container: DependencyContainer):
    """Return a factory building an AgentLoop from the container."""

    def _factory() -> AgentLoop:
        return AgentLoop(
            decision_engine=container.resolve(DecisionEngine),
            policy_engine=container.resolve(PolicyEngine),
            tool_executor=container.resolve(ToolExecutor),
            tool_registry=container.resolve(ToolRegistry),
            memory_engine=container.resolve(MemoryEngine),
            retry_policy=container.resolve(RetryPolicy),
            fallback_policy=container.resolve(FallbackPolicy),
        )

    return _factory


def _make_agent_runtime(container: DependencyContainer):
    """Return a factory building an AgentRuntime from the container."""

    def _factory() -> AgentRuntime:
        return AgentRuntime(
            perception=container.resolve(PerceptionEngine),
            intent=container.resolve(IntentEngine),
            goals=container.resolve(GoalEngine),
            planning=container.resolve(PlanningEngine),
            reasoning=container.resolve(ReasoningEngine),
            decision=container.resolve(DecisionEngine),
            policy=container.resolve(PolicyEngine),
            memory=container.resolve(MemoryEngine),
            experience=container.resolve(ExperienceEngine),
            reflection=container.resolve(ReflectionEngine),
            response=container.resolve(ResponseBuilder),
            tool_registry=container.resolve(ToolRegistry),
            tool_executor=container.resolve(ToolExecutor),
            loop=container.resolve(AgentLoop),
            retry_policy=container.resolve(RetryPolicy),
            fallback_policy=container.resolve(FallbackPolicy),
        )

    return _factory
