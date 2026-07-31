"""Integration tests for the cognitive pipeline end-to-end."""

from __future__ import annotations

import asyncio

import pytest

from app.core.container import DependencyContainer
from app.kernel.bootstrap import register_kernel_components
from app.kernel.orchestrator import CognitiveOrchestrator
from app.kernel.perception.models import RawMessage
from app.kernel.pipeline.models import PipelineContext
from app.kernel.policy.engine import PolicyEngine
from app.kernel.policy.models import PolicyRule as PolicyRuleModel, VerdictType
from app.kernel.policy.rules.base import ModelPolicyRule
from app.kernel.response.builder import ResponseBuilder


@pytest.fixture
def container() -> DependencyContainer:
    c = DependencyContainer()
    register_kernel_components(c)
    return c


@pytest.fixture
def orchestrator(container: DependencyContainer) -> CognitiveOrchestrator:
    return CognitiveOrchestrator(container=container)


class TestPipelineIntegration:
    """Integration tests for the full cognitive pipeline."""

    @pytest.mark.asyncio
    async def test_normal_request_flow(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = await orchestrator.process(
            RawMessage(content="Hello, who are you?"),
            session_id="test-normal",
        )
        assert result.success is True
        assert result.error is None
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.session_id == "test-normal"

    @pytest.mark.asyncio
    async def test_unknown_intent_flow(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = await orchestrator.process(
            RawMessage(content=""),
            session_id="test-unknown",
        )
        assert result.success is True
        assert result.error is None
        assert isinstance(result.content, str)

    @pytest.mark.asyncio
    async def test_clarification_flow(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = await orchestrator.process(
            RawMessage(content="Hello, who are you?"),
            session_id="test-clarify",
        )
        assert result.success is True
        assert result.error is None
        assert isinstance(result.content, str)
        assert result.session_id == "test-clarify"

    @pytest.mark.asyncio
    async def test_denied_policy(self, container: DependencyContainer) -> None:
        policy_engine = container.resolve(PolicyEngine)
        policy_engine.register_rule(
            ModelPolicyRule(
                PolicyRuleModel(
                    name="deny_all_skills",
                    description="Deny all skill_call actions",
                    action_pattern="skill_call",
                    verdict=VerdictType.DENY,
                    priority=100,
                    denial_reason="Blocked by test policy",
                )
            )
        )
        orch = CognitiveOrchestrator(container=container)
        result = await orch.process(
            RawMessage(content="Hello, who are you?"),
            session_id="test-denied",
        )
        assert result.success is False
        assert result.error is not None
        assert "deny" in result.error.lower() or "blocked" in result.error.lower()
        assert result.metadata["policy_verdict"] == "deny"

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        r1 = await orchestrator.process(
            RawMessage(content="First request"), session_id="test-seq-1"
        )
        assert r1.success is True
        r2 = await orchestrator.process(
            RawMessage(content="Second request"), session_id="test-seq-2"
        )
        assert r2.success is True
        assert r1.session_id == "test-seq-1"
        assert r2.session_id == "test-seq-2"

    @pytest.mark.asyncio
    async def test_cancellation_during_execution(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        task = asyncio.create_task(
            orchestrator.process(
                RawMessage(content="Will be cancelled"),
                session_id="test-cancel",
            )
        )
        await asyncio.sleep(0)
        await orchestrator.cancel("test-cancel", reason="test")
        result = await task
        assert result.success is False
        assert result.error is not None
        assert (
            "cancelled" in result.error.lower() or "no response" in result.error.lower()
        )

    @pytest.mark.asyncio
    async def test_timeout(self, orchestrator: CognitiveOrchestrator) -> None:
        result = await orchestrator.process(
            RawMessage(content="Timeout test"),
            session_id="test-timeout",
            ttl_s=0,
        )
        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_metrics_collection(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        await orchestrator.process(
            RawMessage(content="Metrics test"),
            session_id="test-metrics",
        )
        summary = orchestrator.get_metrics_summary()
        assert summary["stage_count"] == 10
        assert summary["success_count"] == 10
        assert summary["failure_count"] == 0
        assert len(summary["stages"]) == 10

    @pytest.mark.asyncio
    async def test_tracing(self, orchestrator: CognitiveOrchestrator) -> None:
        await orchestrator.process(
            RawMessage(content="Tracing test"),
            session_id="test-tracing",
        )
        summary = orchestrator.get_tracing_summary()
        assert summary is not None

    @pytest.mark.asyncio
    async def test_deterministic_output(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        r1 = await orchestrator.process(
            RawMessage(content="Hello, who are you?"),
            session_id="test-det-1",
        )
        r2 = await orchestrator.process(
            RawMessage(content="Hello, who are you?"),
            session_id="test-det-2",
        )
        assert r1.content == r2.content
        assert r1.success == r2.success

    @pytest.mark.asyncio
    async def test_streaming_response(self, container: DependencyContainer) -> None:
        builder = container.resolve(ResponseBuilder)
        chunk1 = await builder.build_chunk({"content": "Hello", "type": "text"})
        chunk2 = await builder.build_chunk(
            {"content": " world", "type": "text", "finished": True}
        )
        assert chunk1.content == "Hello"
        assert chunk2.content == " world"
        assert chunk2.finished is True

    @pytest.mark.asyncio
    async def test_context_fields_populated(
        self, container: DependencyContainer
    ) -> None:
        orch = CognitiveOrchestrator(container=container)
        context = PipelineContext(session_id="test-context-fields")
        from app.kernel.pipeline.execution_context import ExecutionContext

        exec_ctx = ExecutionContext.create(
            session_id="test-context-fields",
        )
        context.percept = RawMessage(content="Hello, who are you?")
        await orch.pipeline.execute(context, exec_ctx)
        assert context.intent is not None
        assert context.goal_hierarchy is not None
        assert context.plan is not None
        assert context.reasoning_trace is not None
        assert context.decision is not None
        assert context.policy_verdict is not None
        assert context.reflection is not None
        assert context.experience is not None
        assert context.response is not None
