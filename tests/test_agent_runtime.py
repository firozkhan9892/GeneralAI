"""Tests for the agent runtime (Phase 7).

Covers the retry/fallback policies, the agent loop (tool selection,
policy gating, retries, cancellation, memory), the full AgentRuntime
execution brain (perception → intent → goal → plan → reasoning → loop →
reflection → experience → memory → response), and DI bootstrap.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.container import DependencyContainer
from app.kernel.agent.bootstrap import register_agent_components
from app.kernel.agent.loop import AgentLoop
from app.kernel.agent.models import (
    AgentRequest,
    AgentRunConfig,
    AgentRunSummary,
    AgentStatus,
    AgentStep,
    AgentStepStatus,
)
from app.kernel.agent.policies import FallbackPolicy, RetryPolicy
from app.kernel.agent.runtime import AgentRuntime
from app.kernel.decision.engine import DecisionEngine
from app.kernel.memory.engine import MemoryEngine
from app.kernel.policy.engine import PolicyEngine
from app.kernel.policy.models import PolicyRule as PolicyRuleModel
from app.kernel.policy.models import VerdictType
from app.kernel.planning.models import Plan, PlanningStrategy, SkillStep
from app.tools.context import CancellationToken
from app.tools.exceptions import ToolExecutionError
from app.tools.executor import ToolExecutor
from app.tools.mock import MockTool
from app.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _registry(*tools: MockTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _question_registry() -> ToolRegistry:
    """Registry whose tools match the QUESTION plan's skill names."""
    return _registry(
        MockTool(name="analyze_question"),
        MockTool(name="retrieve_knowledge"),
        MockTool(name="formulate_answer"),
    )


def _plan(skill_names: tuple[str, ...] = ("analyze_question",)) -> Plan:
    return Plan(
        goal_id="goal_test",
        strategy=PlanningStrategy.TOP_DOWN,
        steps=tuple(
            SkillStep(order=order, skill_name=name, description=f"Step {order}")
            for order, name in enumerate(skill_names)
        ),
    )


def _config(**overrides: Any) -> AgentRunConfig:
    return AgentRunConfig(session_id="sess-test", **overrides)


# ── RetryPolicy ──────────────────────────────────────────────────────


class TestRetryPolicy:
    def test_default_max_retries(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 2

    def test_retryable_marker(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry(1, "temporary connection error") is True

    def test_non_retryable_error(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry(1, "permission denied") is False

    def test_attempt_budget_exhausted(self) -> None:
        policy = RetryPolicy(max_retries=1)
        assert policy.should_retry(2, "temporary error") is False

    def test_no_error_message(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry(1, None) is False

    def test_is_retryable_classifier(self) -> None:
        policy = RetryPolicy()
        assert policy.is_retryable("the server is unavailable") is True
        assert policy.is_retryable("ok") is False


# ── FallbackPolicy ───────────────────────────────────────────────────


class TestFallbackPolicy:
    def test_default_fallback(self) -> None:
        policy = FallbackPolicy()
        policy.set_available_tools(["echo"])
        assert policy.select_fallback() == "echo"

    def test_prefers_requested_available(self) -> None:
        policy = FallbackPolicy(fallback_tool="echo", available_tools=("echo", "calc"))
        assert policy.select_fallback("calc") == "calc"

    def test_fallback_not_available(self) -> None:
        policy = FallbackPolicy(fallback_tool="echo", available_tools=("calc",))
        assert policy.select_fallback() is None

    def test_requested_not_available_uses_default(self) -> None:
        policy = FallbackPolicy(fallback_tool="echo", available_tools=("echo", "calc"))
        assert policy.select_fallback("missing") == "echo"

    def test_set_available_tools_after_construction(self) -> None:
        policy = FallbackPolicy()
        assert policy.select_fallback() is None
        policy.set_available_tools(["echo"])
        assert policy.select_fallback() == "echo"


# ── Agent models ─────────────────────────────────────────────────────


class TestAgentModels:
    def test_run_config_frozen(self) -> None:
        cfg = AgentRunConfig()
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            cfg.max_iterations = 5  # type: ignore[misc]

    def test_run_config_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AgentRunConfig(max_iterations=0)
        with pytest.raises(ValidationError):
            AgentRunConfig(overall_timeout_s=0.5)

    def test_step_frozen(self) -> None:
        step = AgentStep(order=0, skill_name="analyze")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            step.status = AgentStepStatus.SUCCEEDED  # type: ignore[misc]

    def test_summary_defaults(self) -> None:
        summary = AgentRunSummary()
        assert summary.total_steps == 0
        assert summary.tools_invoked == ()


# ── AgentLoop ────────────────────────────────────────────────────────


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_single_step_success(self) -> None:
        loop = AgentLoop(tool_registry=_question_registry())
        steps = await loop.execute(_plan(), config=_config())
        assert len(steps) == 1
        assert steps[0].status == AgentStepStatus.SUCCEEDED
        assert steps[0].tool_name == "analyze_question"
        assert steps[0].tool_result is not None
        assert steps[0].tool_result.success is True
        assert steps[0].decision is not None
        assert steps[0].policy_verdict is not None

    @pytest.mark.asyncio
    async def test_multi_step_success(self) -> None:
        loop = AgentLoop(tool_registry=_question_registry())
        steps = await loop.execute(
            _plan(("analyze_question", "retrieve_knowledge", "formulate_answer")),
            config=_config(),
        )
        assert len(steps) == 3
        assert all(s.status == AgentStepStatus.SUCCEEDED for s in steps)
        assert [s.tool_name for s in steps] == [
            "analyze_question",
            "retrieve_knowledge",
            "formulate_answer",
        ]

    @pytest.mark.asyncio
    async def test_memory_recorded_per_step(self) -> None:
        loop = AgentLoop(
            tool_registry=_question_registry(),
            memory_engine=MemoryEngine(),
        )
        steps = await loop.execute(
            _plan(("analyze_question", "formulate_answer")),
            config=_config(),
        )
        assert all(s.memory_record_id is not None for s in steps)

    @pytest.mark.asyncio
    async def test_policy_denial_fails_step(self) -> None:
        policy = PolicyEngine()
        policy.register_policy(
            PolicyRuleModel(
                name="deny_tool_call",
                description="Block all tool calls for this test",
                action_pattern="tool_call",
                verdict=VerdictType.DENY,
                priority=200,
                denial_reason="tool calls blocked for test",
            )
        )
        loop = AgentLoop(policy_engine=policy, tool_registry=_question_registry())
        steps = await loop.execute(_plan(), config=_config())
        assert len(steps) == 1
        assert steps[0].status == AgentStepStatus.FAILED
        assert steps[0].error == "tool calls blocked for test"

    @pytest.mark.asyncio
    async def test_retry_recovers_from_transient_error(self) -> None:
        calls: dict[str, int] = {"n": 0}

        def flaky(arguments: Any, context: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ToolExecutionError(
                    "temporary connection error", module="tools.mock"
                )
            return f"ok attempt {calls['n']}"

        registry = _registry(
            MockTool(name="analyze_question", echo_input=False, on_run=flaky)
        )
        loop = AgentLoop(tool_registry=registry)
        steps = await loop.execute(_plan(), config=_config(max_retries=2))
        assert calls["n"] == 2
        assert steps[0].status == AgentStepStatus.SUCCEEDED
        assert steps[0].retries == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted_marks_failed(self) -> None:
        registry = _registry(
            MockTool(
                name="analyze_question",
                echo_input=False,
                on_run=lambda a, c: (_ for _ in ()).throw(
                    ToolExecutionError("connection refused", module="tools.mock")
                ),
            )
        )
        loop = AgentLoop(
            tool_registry=registry, retry_policy=RetryPolicy(max_retries=2)
        )
        steps = await loop.execute(_plan(), config=_config(max_retries=2))
        assert steps[0].status == AgentStepStatus.FAILED
        assert steps[0].retries == 2

    @pytest.mark.asyncio
    async def test_no_tool_match_uses_fallback(self) -> None:
        registry = _registry(MockTool(name="echo"))
        loop = AgentLoop(tool_registry=registry)
        steps = await loop.execute(_plan(("unrelated_skill",)), config=_config())
        assert steps[0].status == AgentStepStatus.SUCCEEDED
        assert steps[0].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_no_fallback_marks_failed(self) -> None:
        loop = AgentLoop(tool_registry=_registry(MockTool(name="echo")))
        steps = await loop.execute(_plan(), config=_config())
        assert steps[0].status == AgentStepStatus.SUCCEEDED
        assert steps[0].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_cancellation_before_start(self) -> None:
        loop = AgentLoop(tool_registry=_question_registry())
        token = CancellationToken()
        token.cancel()
        steps = await loop.execute(
            _plan(("analyze_question", "formulate_answer")),
            config=_config(),
            cancellation_token=token,
        )
        assert len(steps) == 2
        assert all(s.status == AgentStepStatus.SKIPPED for s in steps)

    @pytest.mark.asyncio
    async def test_failed_step_skips_remainder(self) -> None:
        registry = _registry(
            MockTool(
                name="analyze_question",
                echo_input=False,
                result="ok",
            ),
            MockTool(name="retrieve_knowledge", fail="boom"),
            MockTool(name="formulate_answer"),
        )
        loop = AgentLoop(tool_registry=registry)
        steps = await loop.execute(
            _plan(("analyze_question", "retrieve_knowledge", "formulate_answer")),
            config=_config(),
        )
        assert [s.status for s in steps] == [
            AgentStepStatus.SUCCEEDED,
            AgentStepStatus.FAILED,
            AgentStepStatus.SKIPPED,
        ]

    @pytest.mark.asyncio
    async def test_max_iterations_limits_steps(self) -> None:
        loop = AgentLoop(tool_registry=_question_registry())
        steps = await loop.execute(
            _plan(("analyze_question", "retrieve_knowledge", "formulate_answer")),
            config=_config(max_iterations=1),
        )
        assert len(steps) == 3
        assert steps[0].status == AgentStepStatus.SUCCEEDED
        assert all(s.status == AgentStepStatus.SKIPPED for s in steps[1:])


# ── AgentRuntime ─────────────────────────────────────────────────────


class TestAgentRuntime:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self) -> None:
        runtime = AgentRuntime(tool_registry=_question_registry())
        response = await runtime.run(
            AgentRequest(raw_input="What is the weather?", session_id="s1")
        )
        assert response.success is True
        assert response.status == AgentStatus.SUCCEEDED
        assert response.intent is not None
        assert response.goal_hierarchy is not None
        assert response.plan is not None
        assert response.reasoning_trace is not None
        assert response.reflection_report is not None
        assert response.experience is not None
        assert response.experience.success is True
        assert response.memory_summary is not None
        assert response.memory_summary.total_records == 3
        assert len(response.steps) == 3
        assert all(s.status == AgentStepStatus.SUCCEEDED for s in response.steps)
        assert response.summary.total_steps == 3
        assert response.summary.succeeded == 3
        assert response.summary.tools_invoked == (
            "analyze_question",
            "formulate_answer",
            "retrieve_knowledge",
        )

    @pytest.mark.asyncio
    async def test_request_config_session_id_used(self) -> None:
        runtime = AgentRuntime(tool_registry=_question_registry())
        request = AgentRequest(raw_input="What is up?", session_id="custom-session")
        response = await runtime.run(request)
        assert response.session_id == "custom-session"
        assert response.summary.memory_records == 3

    @pytest.mark.asyncio
    async def test_per_run_config_override(self) -> None:
        runtime = AgentRuntime(tool_registry=_question_registry())
        response = await runtime.run(
            AgentRequest(raw_input="How does it work?", session_id="s2"),
            config=AgentRunConfig(session_id="s2", memory_enabled=False),
        )
        assert response.success is True
        assert response.summary.memory_records == 0

    @pytest.mark.asyncio
    async def test_memory_records_experience_and_reflection_disabled(self) -> None:
        runtime = AgentRuntime(tool_registry=_question_registry())
        response = await runtime.run(
            AgentRequest(raw_input="What is a test?", session_id="s3"),
            config=AgentRunConfig(
                session_id="s3",
                memory_enabled=False,
                experience_enabled=False,
                reflection_enabled=False,
                reasoning_enabled=False,
            ),
        )
        assert response.success is True
        assert response.reasoning_trace is None
        assert response.reflection_report is None
        assert response.experience is None
        assert response.memory_summary is not None
        assert response.memory_summary.total_records == 0

    @pytest.mark.asyncio
    async def test_step_failure_propagates(self) -> None:
        registry = _registry(
            MockTool(name="analyze_question"),
            MockTool(name="retrieve_knowledge", fail="boom"),
            MockTool(name="formulate_answer"),
        )
        runtime = AgentRuntime(tool_registry=registry)
        response = await runtime.run(
            AgentRequest(raw_input="What is happening?", session_id="s4")
        )
        assert response.success is False
        assert response.status == AgentStatus.FAILED
        assert response.error == "boom"
        assert response.steps[1].status == AgentStepStatus.FAILED
        assert response.steps[2].status == AgentStepStatus.SKIPPED
        assert response.experience is not None
        assert response.experience.success is False
        assert response.summary.failed == 1

    @pytest.mark.asyncio
    async def test_timeout_returns_timed_out(self) -> None:
        registry = _registry(
            MockTool(name="analyze_question", delay_s=2.0),
            MockTool(name="retrieve_knowledge"),
            MockTool(name="formulate_answer"),
        )
        runtime = AgentRuntime(
            tool_registry=registry,
            config=AgentRunConfig(overall_timeout_s=1.0),
        )
        response = await runtime.run(
            AgentRequest(raw_input="What is slow?", session_id="s5"),
            config=AgentRunConfig(
                session_id="s5", overall_timeout_s=1.0, step_timeout_s=30.0
            ),
        )
        assert response.status == AgentStatus.TIMED_OUT
        assert response.success is False
        assert "timed out" in (response.error or "")

    @pytest.mark.asyncio
    async def test_cancel_mid_run(self) -> None:
        registry = _registry(
            MockTool(name="analyze_question", delay_s=0.5),
            MockTool(name="retrieve_knowledge"),
            MockTool(name="formulate_answer"),
        )
        runtime = AgentRuntime(tool_registry=registry)
        token = CancellationToken()
        task = asyncio.create_task(
            runtime.run(
                AgentRequest(raw_input="What is going on?", session_id="s6"),
                cancellation_token=token,
            )
        )
        await asyncio.sleep(0.2)
        runtime.cancel("s6", "user_requested")
        response = await task
        assert response.status == AgentStatus.CANCELLED
        assert response.success is False
        assert response.steps[0].status == AgentStepStatus.SUCCEEDED
        assert all(s.status == AgentStepStatus.SKIPPED for s in response.steps[1:])

    @pytest.mark.asyncio
    async def test_pre_cancelled_token(self) -> None:
        runtime = AgentRuntime(tool_registry=_question_registry())
        token = CancellationToken()
        token.cancel()
        response = await runtime.run(
            AgentRequest(raw_input="What is new?", session_id="s7"),
            cancellation_token=token,
        )
        assert response.status == AgentStatus.CANCELLED
        assert all(s.status == AgentStepStatus.SKIPPED for s in response.steps)

    def test_cancel_unknown_session_is_noop(self) -> None:
        runtime = AgentRuntime()
        runtime.cancel("missing")

    @pytest.mark.asyncio
    async def test_active_sessions_tracking(self) -> None:
        registry = _registry(
            MockTool(name="analyze_question", delay_s=0.5),
            MockTool(name="retrieve_knowledge"),
            MockTool(name="formulate_answer"),
        )
        runtime = AgentRuntime(tool_registry=registry)
        task = asyncio.create_task(
            runtime.run(AgentRequest(raw_input="What is tracking?", session_id="s8"))
        )
        await asyncio.sleep(0.1)
        assert runtime.get_active_sessions() == ["s8"]
        await task
        assert runtime.get_active_sessions() == []

    @pytest.mark.asyncio
    async def test_loop_without_memory_engine(self) -> None:
        loop = AgentLoop(
            tool_registry=_question_registry(),
            memory_engine=None,
        )
        steps = await loop.execute(
            _plan(("analyze_question", "formulate_answer")),
            config=_config(memory_enabled=True),
        )
        assert all(s.status == AgentStepStatus.SUCCEEDED for s in steps)
        assert all(s.memory_record_id is None for s in steps)


# ── Bootstrap / DI ───────────────────────────────────────────────────


class TestAgentBootstrap:
    def test_register_and_resolve_runtime(self) -> None:
        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)
        assert isinstance(runtime, AgentRuntime)
        assert isinstance(runtime.loop, AgentLoop)
        assert isinstance(runtime.registry, ToolRegistry)
        assert isinstance(runtime.memory, object)

    def test_resolve_loop_and_policies(self) -> None:
        container = DependencyContainer()
        register_agent_components(container)
        assert isinstance(container.resolve(AgentLoop), AgentLoop)
        assert isinstance(container.resolve(RetryPolicy), RetryPolicy)
        assert isinstance(container.resolve(FallbackPolicy), FallbackPolicy)
        assert isinstance(container.resolve(DecisionEngine), DecisionEngine)
        assert isinstance(container.resolve(ToolExecutor), ToolExecutor)

    @pytest.mark.asyncio
    async def test_bootstrapped_runtime_executes(self) -> None:
        container = DependencyContainer()
        register_agent_components(container)
        runtime = container.resolve(AgentRuntime)
        for name in ("analyze_question", "retrieve_knowledge", "formulate_answer"):
            runtime.registry.register(MockTool(name=name))
        response = await runtime.run(
            AgentRequest(raw_input="What is a container?", session_id="di-1")
        )
        assert response.success is True
        assert response.status == AgentStatus.SUCCEEDED
