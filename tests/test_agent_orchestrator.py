"""Integration tests wiring the AgentRuntime into the CognitiveOrchestrator.

Verifies that ``CognitiveOrchestrator.run_agent`` / ``cancel_agent`` /
``get_active_agent_sessions`` resolve the agent runtime from the shared
DI container (so tools registered on the container are visible to the
agent loop), and that the kernel bootstrap wires everything end-to-end.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.container import DependencyContainer
from app.core.lifecycle import LifecycleManager
from app.kernel import (
    AgentRequest,
    AgentRunConfig,
    AgentStatus,
    AgentStepStatus,
    CognitiveOrchestrator,
    register_agent_components,
)
from app.kernel.agent.runtime import AgentRuntime
from app.kernel.bootstrap import bootstrap_kernel, register_kernel_components
from app.kernel.orchestrator import CognitiveOrchestrator as _CO
from app.tools.mock import MockTool
from app.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _registry(*tools: MockTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _question_tools() -> tuple[MockTool, ...]:
    return (
        MockTool(name="analyze_question"),
        MockTool(name="retrieve_knowledge"),
        MockTool(name="formulate_answer"),
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def container() -> DependencyContainer:
    c = DependencyContainer()
    register_kernel_components(c)
    return c


@pytest.fixture
def orchestrator(container: DependencyContainer) -> CognitiveOrchestrator:
    registry = container.resolve(ToolRegistry)
    for tool in _question_tools():
        registry.register(tool)
    return container.resolve(CognitiveOrchestrator)


class TestOrchestratorAgentResolution:
    """The orchestrator resolves the runtime from the shared container."""

    def test_orchestrator_is_kernel_type(self) -> None:
        assert CognitiveOrchestrator is _CO

    def test_agent_runtime_resolved_from_container(
        self, container: DependencyContainer
    ) -> None:
        orchestrator = container.resolve(CognitiveOrchestrator)
        runtime = orchestrator.agent_runtime
        assert isinstance(runtime, AgentRuntime)

    def test_runtime_shares_container_registry(
        self, container: DependencyContainer
    ) -> None:
        registry = container.resolve(ToolRegistry)
        for tool in _question_tools():
            registry.register(tool)
        orchestrator = container.resolve(CognitiveOrchestrator)
        assert orchestrator.agent_runtime.registry is registry

    def test_agent_components_registered(self, container: DependencyContainer) -> None:
        assert container.has(AgentRuntime)

    def test_bootstrap_kernel_shares_container(self) -> None:
        container = DependencyContainer()
        lifecycle = LifecycleManager()
        orchestrator = bootstrap_kernel(container, lifecycle)
        assert orchestrator.container is container
        assert orchestrator.agent_runtime.registry is container.resolve(ToolRegistry)


class TestOrchestratorRunAgent:
    """End-to-end agent execution through the orchestrator."""

    @pytest.mark.asyncio
    async def test_run_agent_success(self, orchestrator: CognitiveOrchestrator) -> None:
        response = await orchestrator.run_agent(
            AgentRequest(raw_input="What is 2+2?", session_id="orch-sess-1")
        )
        assert response.success is True
        assert response.status == AgentStatus.SUCCEEDED
        assert response.session_id == "orch-sess-1"
        assert response.error is None
        assert response.summary.total_steps == 3
        assert response.summary.succeeded == 3
        assert response.summary.failed == 0
        assert set(response.summary.tools_invoked) == {
            "analyze_question",
            "retrieve_knowledge",
            "formulate_answer",
        }
        assert all(s.status == AgentStepStatus.SUCCEEDED for s in response.steps)

    @pytest.mark.asyncio
    async def test_run_agent_request_session(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        response = await orchestrator.run_agent(
            AgentRequest(raw_input="Hello", session_id="orch-sess-2"),
            config=AgentRunConfig(session_id="orch-sess-2"),
        )
        assert response.session_id == "orch-sess-2"

    @pytest.mark.asyncio
    async def test_run_agent_without_registered_tools(self) -> None:
        container = DependencyContainer()
        register_kernel_components(container)
        orchestrator = container.resolve(CognitiveOrchestrator)
        response = await orchestrator.run_agent(
            AgentRequest(raw_input="Question without tools", session_id="orch-sess-3")
        )
        assert response.success is False
        assert response.status == AgentStatus.FAILED
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_run_agent_shared_tools_after_bootstrap(self) -> None:
        container = DependencyContainer()
        lifecycle = LifecycleManager()
        orchestrator = bootstrap_kernel(container, lifecycle)
        registry = container.resolve(ToolRegistry)
        for tool in _question_tools():
            registry.register(tool)
        response = await orchestrator.run_agent(
            AgentRequest(raw_input="What is the time?", session_id="orch-sess-4")
        )
        assert response.success is True

    @pytest.mark.asyncio
    async def test_run_agent_default_runtime_without_container(self) -> None:
        orchestrator = CognitiveOrchestrator()
        response = await orchestrator.run_agent(
            AgentRequest(raw_input="Hello", session_id="orch-sess-5")
        )
        assert isinstance(response.status, AgentStatus)

    @pytest.mark.asyncio
    async def test_run_agent_cancellation(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        task = asyncio.create_task(
            orchestrator.run_agent(
                AgentRequest(raw_input="Will be cancelled", session_id="orch-cancel")
            )
        )
        await asyncio.sleep(0)
        orchestrator.cancel_agent("orch-cancel", reason="test")
        response = await task
        assert response.status == AgentStatus.CANCELLED
        assert response.success is False
        assert response.error == "Agent run cancelled"

    @pytest.mark.asyncio
    async def test_run_agent_active_sessions(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        task = asyncio.create_task(
            orchestrator.run_agent(
                AgentRequest(raw_input="Long run", session_id="orch-active")
            )
        )
        await asyncio.sleep(0)
        assert "orch-active" in orchestrator.get_active_agent_sessions()
        await task
        assert "orch-active" not in orchestrator.get_active_agent_sessions()

    def test_cancel_unknown_session_noop(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        orchestrator.cancel_agent("orch-unknown", reason="test")


class TestAgentBootstrapThroughKernel:
    """register_kernel_components wires the agent components too."""

    def test_register_kernel_components_registers_agent(self) -> None:
        container = DependencyContainer()
        register_kernel_components(container)
        assert container.has(AgentRuntime)

    def test_register_agent_components_is_idempotent(self) -> None:
        container = DependencyContainer()
        register_kernel_components(container)
        register_agent_components(container)
        assert container.has(AgentRuntime)

    @pytest.mark.asyncio
    async def test_resolved_runtime_executes_with_container_tools(self) -> None:
        container = DependencyContainer()
        register_kernel_components(container)
        registry = container.resolve(ToolRegistry)
        for tool in _question_tools():
            registry.register(tool)
        runtime = container.resolve(AgentRuntime)
        response = await runtime.run(
            AgentRequest(raw_input="What is a runtime?", session_id="orch-boot")
        )
        assert response.success is True
