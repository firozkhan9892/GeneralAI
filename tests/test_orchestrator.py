"""Tests for Cognitive Orchestrator infrastructure (Phase 3.4)."""

from __future__ import annotations

import time

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.kernel.contracts.base import ContractRequest, EngineType
from app.kernel.contracts.events import PipelineEvent
from app.kernel.pipeline.dispatcher import EngineDispatcher, StageDefinition
from app.kernel.pipeline.execution_context import CancellationToken, ExecutionContext
from app.kernel.pipeline.executor import PipelineExecutor
from app.kernel.pipeline.models import (
    ErrorPolicy,
    PipelineContext,
    PipelineMetadata,
    PipelineStep,
)
from app.kernel.pipeline.observability import (
    EventPublisher,
    MetricsCollector,
    StageMetrics,
    TracingHook,
)
from app.kernel.pipeline.policies import (
    FailurePolicy,
    PolicySet,
    StagePolicy,
    lenient_policy,
    resilient_policy,
    strict_policy,
)
from app.kernel.pipeline.stages import build_stage_definitions


# ── CancellationToken ──────────────────────────────────────────────────


class TestCancellationToken:
    def test_default_not_cancelled(self) -> None:
        token = CancellationToken()
        assert not token.is_cancelled

    def test_cancel_sets_flag(self) -> None:
        token = CancellationToken()
        token.cancel("test reason")
        assert token.is_cancelled
        assert token.reason == "test reason"

    def test_cancel_is_idempotent(self) -> None:
        token = CancellationToken()
        token.cancel("first")
        token.cancel("second")
        assert token.is_cancelled
        # Latest reason wins
        assert token.reason == "second"

    def test_cancel_no_reason(self) -> None:
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled
        assert token.reason == ""

    def test_frozen(self) -> None:
        token = CancellationToken()
        with pytest.raises(AttributeError):
            token._cancelled = True  # type: ignore[misc]


# ── ExecutionContext ──────────────────────────────────────────────────


class TestExecutionContext:
    def test_create_generates_ids(self) -> None:
        ctx = ExecutionContext.create(session_id="s1")
        assert ctx.session_id == "s1"
        assert ctx.correlation_id != ""
        assert ctx.trace_id != ""
        assert ctx.span_id != ""
        assert ctx.request_id != ""

    def test_create_with_all_params(self) -> None:
        ctx = ExecutionContext.create(
            session_id="s1",
            user_id="user-42",
            pipeline_id="my-pipeline",
            timeout_s=60,
            ttl_s=1800,
            metadata={"key": "val"},
        )
        assert ctx.user_id == "user-42"
        assert ctx.pipeline_id == "my-pipeline"
        assert ctx.timeout_s == 60
        assert ctx.metadata == {"key": "val"}

    def test_default_deadline_is_one_hour(self) -> None:
        ctx = ExecutionContext.create()
        assert ctx.deadline is not None
        delta = ctx.deadline - datetime.now(timezone.utc)
        assert 3590 < delta.total_seconds() < 3610

    def test_is_deadline_expired_false(self) -> None:
        ctx = ExecutionContext.create()
        assert not ctx.is_deadline_expired

    def test_is_deadline_expired_true(self) -> None:
        ctx = ExecutionContext.create(ttl_s=0)
        assert ctx.is_deadline_expired

    def test_is_not_cancelled_by_default(self) -> None:
        ctx = ExecutionContext.create()
        assert not ctx.is_cancelled

    def test_with_stage_returns_new_context(self) -> None:
        from app.kernel.contracts.base import EngineType

        ctx = ExecutionContext.create()
        new_ctx = ctx.with_stage(EngineType.PERCEPTION)
        assert new_ctx.current_stage == EngineType.PERCEPTION
        assert new_ctx.session_id == ctx.session_id
        assert new_ctx is not ctx

    def test_with_retry_increments_count(self) -> None:
        ctx = ExecutionContext.create()
        new_ctx = ctx.with_retry()
        assert new_ctx.retry_count == ctx.retry_count + 1

    def test_with_metadata_adds_entry(self) -> None:
        ctx = ExecutionContext.create()
        new_ctx = ctx.with_metadata("test_key", "test_value")
        assert new_ctx.metadata["test_key"] == "test_value"
        assert ctx.metadata == {}

    def test_frozen(self) -> None:

        ctx = ExecutionContext.create()
        with pytest.raises(AttributeError):
            ctx.session_id = "wrong"  # type: ignore[misc]

    def test_remaining_seconds_no_deadline(self) -> None:
        ctx = ExecutionContext.create()
        ctx_no_deadline = ExecutionContext(
            session_id=ctx.session_id,
            correlation_id=ctx.correlation_id,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=None,
            request_id=ctx.request_id,
            user_id=None,
            pipeline_id=ctx.pipeline_id,
            current_stage=EngineType.UNKNOWN,
            deadline=None,
            timeout_s=ctx.timeout_s,
            retry_count=ctx.retry_count,
            metadata=ctx.metadata,
            cancellation_token=ctx.cancellation_token,
            created_at=ctx.created_at,
        )
        assert ctx_no_deadline.remaining_seconds is None

    def test_cancellation_propagation(self) -> None:
        token = CancellationToken()
        ctx = ExecutionContext.create()
        cancelled_ctx = ExecutionContext(
            session_id=ctx.session_id,
            correlation_id=ctx.correlation_id,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=None,
            request_id=ctx.request_id,
            user_id=None,
            pipeline_id=ctx.pipeline_id,
            current_stage=ctx.current_stage,
            deadline=ctx.deadline,
            timeout_s=ctx.timeout_s,
            retry_count=ctx.retry_count,
            metadata=ctx.metadata,
            cancellation_token=token,
            created_at=ctx.created_at,
        )
        assert not cancelled_ctx.is_cancelled
        token.cancel("user cancelled")
        assert cancelled_ctx.is_cancelled
        assert cancelled_ctx.cancellation_token.reason == "user cancelled"


# ── PipelineContext ────────────────────────────────────────────────────


class TestPipelineContext:
    def test_defaults(self) -> None:
        ctx = PipelineContext()
        assert ctx.session_id == ""
        assert ctx.percept is None
        assert ctx.intent is None
        assert ctx.goal_hierarchy is None
        assert ctx.plan is None
        assert ctx.reasoning_trace is None
        assert ctx.decision is None
        assert ctx.capability is None
        assert ctx.policy_verdict is None
        assert ctx.skill_binding is None
        assert ctx.tool_bindings == []
        assert ctx.execution_results == []
        assert ctx.reflection is None
        assert ctx.experience is None
        assert ctx.response is None
        assert isinstance(ctx.metadata, PipelineMetadata)

    def test_empty_metadata_has_zero_stage(self) -> None:
        ctx = PipelineContext()
        assert ctx.metadata.current_stage == 0


# ── Failure Policies ──────────────────────────────────────────────────


class TestFailurePolicy:
    def test_abort_policy(self) -> None:
        policy = FailurePolicy.ABORT
        assert policy.value == "abort"

    def test_retry_policy(self) -> None:
        policy = FailurePolicy.RETRY
        assert policy.value == "retry"

    def test_continue_policy(self) -> None:
        policy = FailurePolicy.CONTINUE
        assert policy.value == "continue"

    def test_fallback_policy(self) -> None:
        policy = FailurePolicy.FALLBACK
        assert policy.value == "fallback"


class TestStagePolicy:
    def test_defaults(self) -> None:
        policy = StagePolicy()
        assert policy.policy == FailurePolicy.ABORT
        assert policy.max_retries == 3
        assert policy.retry_delay_s == 0.5
        assert policy.fallback_factory is None

    def test_custom(self) -> None:
        policy = StagePolicy(
            policy=FailurePolicy.RETRY,
            max_retries=5,
            retry_delay_s=1.0,
        )
        assert policy.policy == FailurePolicy.RETRY
        assert policy.max_retries == 5
        assert policy.retry_delay_s == 1.0

    def test_fallback_raises_without_factory(self) -> None:
        policy = StagePolicy()
        with pytest.raises(RuntimeError):
            policy.get_fallback()

    def test_fallback_works_with_factory(self) -> None:
        policy = StagePolicy(fallback_factory=lambda: "fallback_value")
        assert policy.get_fallback() == "fallback_value"

    def test_frozen(self) -> None:
        policy = StagePolicy(policy=FailurePolicy.CONTINUE)
        with pytest.raises(Exception):
            policy.policy = FailurePolicy.RETRY  # type: ignore[misc]


class TestPolicySet:
    def test_default_policy(self) -> None:
        ps = PolicySet()
        stage_policy = ps.get_policy_for_stage("unknown_stage")
        assert stage_policy.policy == FailurePolicy.ABORT

    def test_stage_override(self) -> None:
        ps = PolicySet(
            stage_overrides={
                "perception": StagePolicy(policy=FailurePolicy.CONTINUE, max_retries=2)
            }
        )
        perception_policy = ps.get_policy_for_stage("perception")
        assert perception_policy.policy == FailurePolicy.CONTINUE
        assert perception_policy.max_retries == 2

        default_policy = ps.get_policy_for_stage("unknown")
        assert default_policy.policy == FailurePolicy.ABORT

    def test_lenient_policy(self) -> None:
        ps = lenient_policy(max_retries=2)
        policy = ps.get_policy_for_stage("any_stage")
        assert policy.policy == FailurePolicy.RETRY
        assert policy.max_retries == 2

    def test_strict_policy(self) -> None:
        ps = strict_policy()
        policy = ps.get_policy_for_stage("any_stage")
        assert policy.policy == FailurePolicy.ABORT
        assert policy.max_retries == 0

    def test_resilient_policy(self) -> None:
        ps = resilient_policy(max_retries=3)
        assert ps.get_policy_for_stage("response").policy == FailurePolicy.ABORT
        assert ps.get_policy_for_stage("memory").policy == FailurePolicy.ABORT


# ── MetricsCollector ──────────────────────────────────────────────────


class TestMetricsCollector:
    def test_empty(self) -> None:
        mc = MetricsCollector()
        assert mc.stage_count == 0
        assert mc.success_count == 0
        assert mc.failure_count == 0
        assert mc.total_duration_ms == 0
        assert mc.total_retries == 0

    def test_record_start(self) -> None:
        mc = MetricsCollector()
        mc.record_start(EngineType.PERCEPTION)
        assert mc._total_start is not None

    def test_record_end(self) -> None:
        mc = MetricsCollector()
        started = datetime.now(timezone.utc)
        metrics = mc.record_end(EngineType.PERCEPTION, started, success=True)
        assert metrics.success is True
        assert metrics.stage == EngineType.PERCEPTION
        assert mc.stage_count == 1
        assert mc.success_count == 1

    def test_record_failure(self) -> None:
        mc = MetricsCollector()
        started = datetime.now(timezone.utc)
        error_exc = Exception("test error")
        metrics = mc.record_end(
            EngineType.INTENT,
            started,
            success=False,
            error="intent failed",
            exception=error_exc,
        )
        assert metrics.success is False
        assert metrics.error == "intent failed"
        assert metrics.exception is error_exc
        assert mc.failure_count == 1

    def test_summary(self) -> None:
        mc = MetricsCollector()
        started = datetime.now(timezone.utc)
        mc.record_end(EngineType.PERCEPTION, started, success=True)
        mc.record_end(
            EngineType.INTENT,
            started,
            success=False,
            error="err",
            retry_count=2,
        )
        summary = mc.summary()
        assert summary["stage_count"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1
        assert summary["total_retries"] == 2

    def test_get_stage_metrics(self) -> None:
        mc = MetricsCollector()
        started = datetime.now(timezone.utc)
        mc.record_end(EngineType.PERCEPTION, started, success=True)
        mc.record_end(EngineType.INTENT, started, success=True)
        mc.record_end(EngineType.PERCEPTION, started, success=False, error="fail")

        perception_metrics = mc.get_stage_metrics(EngineType.PERCEPTION)
        assert len(perception_metrics) == 2

        intent_metrics = mc.get_stage_metrics(EngineType.INTENT)
        assert len(intent_metrics) == 1


# ── EventPublisher ────────────────────────────────────────────────────


class TestEventPublisher:
    def test_publish_with_no_subscribers(self) -> None:
        pub = EventPublisher()
        pub.publish("test_event", {"data": "value"})  # should not raise

    def test_publish_with_event_bus(self) -> None:
        event_bus = MagicMock()
        event_bus.emit = MagicMock()
        pub = EventPublisher(event_bus=event_bus)
        pub.publish("test_event", {"data": "value"})
        event_bus.emit.assert_called_once_with("test_event", {"data": "value"})

    def test_subscribers_called(self) -> None:
        events_received: list = []

        def callback(event, data):
            events_received.append((event, data))

        pub = EventPublisher()
        pub.subscribe(callback)
        pub.publish("custom_event", {"key": "val"})

        assert len(events_received) == 1
        assert events_received[0][0] == "custom_event"
        assert events_received[0][1] == {"key": "val"}

    def test_publish_stage_start(self) -> None:
        from app.kernel.contracts.events import PipelineEvent

        received: list = []
        pub = EventPublisher()
        pub.subscribe(lambda event, data: received.append((event, data)))

        pub.publish_stage_start("perception", {"extra": "info"})
        assert len(received) == 1
        assert received[0][0] == PipelineEvent.PERCEPTION_STARTED

    def test_publish_stage_complete(self) -> None:
        from app.kernel.contracts.events import PipelineEvent

        received: list = []
        pub = EventPublisher()
        pub.subscribe(lambda event, data: received.append((event, data)))

        pub.publish_stage_complete("perception")
        assert len(received) == 1
        assert received[0][0] == PipelineEvent.PERCEPTION_COMPLETED

    def test_publish_stage_error(self) -> None:
        from app.kernel.contracts.events import PipelineEvent

        received: list = []
        pub = EventPublisher()
        pub.subscribe(lambda event, data: received.append((event, data)))

        pub.publish_stage_error("perception", "test_error")
        assert len(received) == 1
        assert received[0][0] == PipelineEvent.PERCEPTION_FAILED


# ── TracingHook ───────────────────────────────────────────────────────


class TestTracingHook:
    def test_start_span(self) -> None:
        hook = TracingHook()
        span = hook.start_span(
            "perception",
            trace_id="trace-001",
            span_id="span-001",
            parent_span_id=None,
            metadata={"stage": "perception"},
        )
        assert span["trace_id"] == "trace-001"
        assert span["span_id"] == "span-001"
        assert span["stage"] == "perception"
        assert "start_time" in span

    def test_end_span(self) -> None:
        hook = TracingHook()
        span = hook.start_span(
            "intent",
            trace_id="trace-001",
            span_id="span-001",
        )
        time.sleep(0.01)
        hook.end_span(span, success=True)

        assert span["end_time"] is not None
        assert span["duration_ms"] is not None
        assert span["success"] is True
        assert span["end_timestamp"] is not None

    def test_end_span_with_error(self) -> None:
        hook = TracingHook()
        span = hook.start_span(
            "reasoning",
            trace_id="trace-001",
            span_id="span-001",
        )
        hook.end_span(span, success=False, error="reasoning timeout")

        assert span["success"] is False
        assert span["error"] == "reasoning timeout"

    def test_spans_list(self) -> None:
        hook = TracingHook()
        assert len(hook.spans) == 0
        span = hook.start_span("perception", "t1", "s1")
        hook.end_span(span, success=True)
        assert len(hook.spans) == 1

    def test_summary(self) -> None:
        hook = TracingHook()
        span1 = hook.start_span("perception", "t1", "s1")
        hook.end_span(span1, success=True)
        span2 = hook.start_span("intent", "t1", "s2", parent_span_id="s1")
        hook.end_span(span2, success=False, error="failed")
        summary = hook.summary()
        assert summary["span_count"] == 2
        assert summary["total_duration_ms"] >= 0


# ── StageDefinition ───────────────────────────────────────────────────


class TestStageDefinition:
    def test_creation(self) -> None:
        def req_builder(ctx: Any, ec: ExecutionContext) -> ContractRequest:
            return ContractRequest()

        def resp_extractor(resp: object, ctx: Any) -> Any:
            return resp

        stage = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="perception",
            engine_attr="perception",
            method_name="perceive",
            request_field="percept",
            response_field="intent",
            request_builder=req_builder,
            response_extractor=resp_extractor,
        )
        assert stage.engine_type == EngineType.PERCEPTION
        assert stage.name == "perception"
        assert stage.method_name == "perceive"

    def test_stage_name_is_name(self) -> None:
        def req_builder(ctx: Any, ec: ExecutionContext) -> ContractRequest:
            return ContractRequest()

        def resp_extractor(resp: object, ctx: Any) -> Any:
            return resp

        stage = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="my_stage",
            engine_attr="my_engine",
            method_name="process",
            request_field="input",
            response_field="output",
            request_builder=req_builder,
            response_extractor=resp_extractor,
        )
        assert stage.stage_name == "my_stage"


# ── build_stage_definitions ───────────────────────────────────────────


class TestBuildStageDefinitions:
    def test_returns_list(self) -> None:
        stages = build_stage_definitions()
        assert isinstance(stages, list)

    def test_has_expected_stages(self) -> None:
        stages = build_stage_definitions()
        stage_names = [s.name for s in stages]
        assert "perception" in stage_names
        assert "intent" in stage_names
        assert "goal" in stage_names
        assert "planning" in stage_names
        assert "reasoning" in stage_names
        assert "decision" in stage_names
        assert "policy" in stage_names
        assert "reflection" in stage_names
        assert "experience" in stage_names
        assert "response" in stage_names

    def test_stages_in_correct_order(self) -> None:
        stages = build_stage_definitions()
        names = [s.name for s in stages]
        expected_order = [
            "perception",
            "intent",
            "goal",
            "planning",
            "reasoning",
            "decision",
            "policy",
            "reflection",
            "experience",
            "response",
        ]
        assert names == expected_order

    def test_all_have_required_attrs(self) -> None:
        from app.kernel.contracts.base import EngineType

        stages = build_stage_definitions()
        for stage in stages:
            assert stage.engine_type in [e for e in EngineType]
            assert stage.name != ""
            assert stage.method_name != ""
            assert stage.request_builder is not None
            assert stage.response_extractor is not None


# ── PipelineExecutor ──────────────────────────────────────────────────


class TestPipelineExecutor:
    def test_requires_dispatcher(self) -> None:
        executor = PipelineExecutor()
        with pytest.raises(RuntimeError):
            asyncio.run(executor.execute(PipelineContext(), ExecutionContext.create()))

    def test_register_stage(self) -> None:
        executor = PipelineExecutor()
        stage = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="test_stage",
            engine_attr="test",
            method_name="run",
            request_field="input",
            response_field="output",
            request_builder=lambda ctx, ec: ContractRequest(),
            response_extractor=lambda resp, ctx: resp,
        )
        executor.register_stage(stage)
        assert executor.get_stage("test_stage") is stage

    def test_set_dispatcher(self) -> None:
        executor = PipelineExecutor()
        dispatcher = MagicMock()
        executor.set_dispatcher(dispatcher)
        assert executor.dispatcher is dispatcher

    def test_ordered_stages(self) -> None:
        executor = PipelineExecutor()
        s1 = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="s1",
            engine_attr="e1",
            method_name="run",
            request_field="in",
            response_field="out",
            request_builder=lambda ctx, ec: ContractRequest(),
            response_extractor=lambda resp, ctx: resp,
        )
        s2 = StageDefinition(
            engine_type=EngineType.INTENT,
            name="s2",
            engine_attr="e2",
            method_name="run",
            request_field="in",
            response_field="out",
            request_builder=lambda ctx, ec: ContractRequest(),
            response_extractor=lambda resp, ctx: resp,
        )
        executor.register_stage(s1)
        executor.register_stage(s2)
        assert len(executor.stages) == 2
        assert executor.stages[0].name == "s1"
        assert executor.stages[1].name == "s2"


# ── EngineDispatcher ──────────────────────────────────────────────────


class TestEngineDispatcher:
    def test_ordered_stages(self) -> None:
        container = MagicMock()
        dispatcher = EngineDispatcher(container=container, stages=[])
        assert dispatcher.ordered_stages == []

    def test_get_stage(self) -> None:
        from app.kernel.contracts.base import EngineType

        def req_builder(ctx: Any, ec: ExecutionContext) -> ContractRequest:
            return ContractRequest()

        def resp_extractor(resp: object, ctx: Any) -> Any:
            return resp

        stage = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="perception",
            engine_attr="perception",
            method_name="perceive",
            request_field="percept",
            response_field="intent",
            request_builder=req_builder,
            response_extractor=resp_extractor,
        )

        container = MagicMock()
        dispatcher = EngineDispatcher(container=container, stages=[stage])
        found = dispatcher.get_stage(EngineType.PERCEPTION)
        assert found is stage
        assert dispatcher.get_stage(EngineType.INTENT) is None

    def test_get_policy(self) -> None:
        from app.kernel.contracts.base import EngineType

        def req_builder(ctx: Any, ec: ExecutionContext) -> ContractRequest:
            return ContractRequest()

        def resp_extractor(resp: object, ctx: Any) -> Any:
            return resp

        stage = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="perception",
            engine_attr="perception",
            method_name="perceive",
            request_field="percept",
            response_field="intent",
            request_builder=req_builder,
            response_extractor=resp_extractor,
        )

        policy_set = PolicySet(
            stage_overrides={"perception": StagePolicy(policy=FailurePolicy.CONTINUE)}
        )
        container = MagicMock()
        dispatcher = EngineDispatcher(
            container=container,
            stages=[stage],
            policy_set=policy_set,
        )
        policy = dispatcher.get_policy_for_stage("perception")
        assert policy.policy == FailurePolicy.CONTINUE

    def test_missing_container(self) -> None:
        container = MagicMock()

        def req_builder(ctx: Any, ec: ExecutionContext) -> ContractRequest:
            return ContractRequest()

        def resp_extractor(resp: object, ctx: Any) -> Any:
            return resp

        stage = StageDefinition(
            engine_type=EngineType.PERCEPTION,
            name="perception",
            engine_attr="perception",
            method_name="perceive",
            request_field="percept",
            response_field="intent",
            request_builder=req_builder,
            response_extractor=resp_extractor,
        )

        dispatcher = EngineDispatcher(container=container, stages=[stage])
        # Without a dispatcher configured, dispatch should raise
        with pytest.raises(RuntimeError):
            # This will fail because the container was created by default but won't have the engine
            asyncio.run(dispatcher.dispatch(stage, object(), ExecutionContext.create()))

    def test_cancel_session(self) -> None:
        container = MagicMock()
        dispatcher = EngineDispatcher(container=container, stages=[])
        dispatcher.cancel_session("session-123", "user_requested")
        assert "session-123" in dispatcher._cancelled_sessions


# ── CognitiveOrchestrator Integration ─────────────────────────────────


class TestCognitiveOrchestrator:
    def test_creation_with_no_deps(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        assert orch.container is not None
        assert orch.pipeline is not None
        assert orch.dispatcher is not None

    def test_creation_with_container(self) -> None:
        from app.core.container import DependencyContainer
        from app.kernel.orchestrator import CognitiveOrchestrator

        container = DependencyContainer()
        orch = CognitiveOrchestrator(container=container)
        assert orch.container is container

    def test_get_metrics_summary(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        summary = orch.get_metrics_summary()
        assert isinstance(summary, dict)
        assert "stage_count" in summary

    def test_get_active_sessions_empty(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        assert orch.get_active_sessions() == []

    def test_get_tracing_summary(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        summary = orch.get_tracing_summary()
        assert isinstance(summary, dict)

    def test_cancel_unknown_session(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator

        orch = CognitiveOrchestrator()
        # Should not raise
        asyncio.run(orch.cancel("nonexistent_session"))

    def test_process_returns_output_message(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator
        from app.kernel.response.models import OutputMessage

        orch = CognitiveOrchestrator()
        result = asyncio.run(orch.process("test_input"))
        assert isinstance(result, OutputMessage)
        assert result.success is False

    def test_process_with_custom_timeout(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator
        from app.kernel.response.models import OutputMessage

        orch = CognitiveOrchestrator()
        result = asyncio.run(orch.process("test", timeout_s=10))
        assert isinstance(result, OutputMessage)

    def test_process_with_metadata(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator
        from app.kernel.response.models import OutputMessage

        orch = CognitiveOrchestrator()
        result = asyncio.run(orch.process("test", metadata={"key": "value"}))
        assert isinstance(result, OutputMessage)

    def test_process_with_session_id(self) -> None:
        from app.kernel.orchestrator import CognitiveOrchestrator
        from app.kernel.response.models import OutputMessage

        orch = CognitiveOrchestrator()
        result = asyncio.run(orch.process("test", session_id="my-session"))
        assert isinstance(result, OutputMessage)


# ── PipelineStep validation ──────────────────────────────────────────


class TestPipelineStep:
    def test_defaults(self) -> None:
        step = PipelineStep(name="test", order=0)
        assert step.name == "test"
        assert step.order == 0
        assert step.error_policy == FailurePolicy.ABORT
        assert step.max_retries == 0
        assert step.timeout_s is None
        assert step.depends_on == ()

    def test_full(self) -> None:
        step = PipelineStep(
            name="process",
            order=1,
            description="Process data",
            error_policy=ErrorPolicy.RETRY,
            max_retries=3,
            timeout_s=30,
            depends_on=("preprocess",),
        )
        assert step.description == "Process data"
        assert step.error_policy == FailurePolicy.RETRY
        assert step.max_retries == 3
        assert step.timeout_s == 30
        assert step.depends_on == ("preprocess",)

    def test_ge_constraint_timeout(self) -> None:
        with pytest.raises(Exception):
            PipelineStep(name="bad", order=0, timeout_s=0)


# ── PipelineDefinition validation ─────────────────────────────────────


class TestPipelineDefinition:
    def test_required_fields(self) -> None:
        with pytest.raises(Exception):
            from app.kernel.pipeline.models import PipelineDefinition

            PipelineDefinition(**{})

    def test_create(self) -> None:
        from app.kernel.pipeline.models import PipelineDefinition, PipelineStep

        step = PipelineStep(name="s1", order=0)
        pd = PipelineDefinition(
            name="test_pipeline",
            steps=(step,),
            description="test",
        )
        assert pd.name == "test_pipeline"
        assert len(pd.steps) == 1
        assert pd.version == "1.0.0"


# ── PipelineMetadata ──────────────────────────────────────────────────


class TestPipelineMetadata:
    def test_defaults(self) -> None:
        meta = PipelineMetadata()
        assert meta.stage_count == 0
        assert meta.current_stage == 0
        assert meta.errors == []

    def test_updated_fields(self) -> None:
        meta = PipelineMetadata(
            stage_count=5,
            current_stage=3,
            errors=[{"stage": "perception", "error": "timeout"}],
        )
        assert meta.stage_count == 5
        assert meta.current_stage == 3


# ── StageMetrics ──────────────────────────────────────────────────────


class TestStageMetrics:
    def test_defaults(self) -> None:
        now = datetime.now(timezone.utc)
        metrics = StageMetrics(
            stage=EngineType.PERCEPTION,
            started_at=now,
            ended_at=now,
            duration_ms=10,
            success=True,
            error=None,
            retry_count=0,
        )
        assert metrics.is_complete
        assert metrics.success is True

    def test_duration(self) -> None:
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ended = datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        metrics = StageMetrics(
            stage=EngineType.PERCEPTION,
            started_at=started,
            ended_at=ended,
            duration_ms=1000,
            success=True,
            error=None,
            retry_count=0,
        )
        assert metrics.duration_ms == 1000

    def test_failed_with_exception(self) -> None:
        now = datetime.now(timezone.utc)
        exc = RuntimeError("pipeline error")
        metrics = StageMetrics(
            stage=EngineType.PERCEPTION,
            started_at=now,
            ended_at=now,
            duration_ms=0,
            success=False,
            error="pipeline error",
            retry_count=2,
            exception=exc,
        )
        assert metrics.success is False
        assert metrics.retry_count == 2
        assert metrics.exception is exc


# ── Observability integration ─────────────────────────────────────────


class TestObservabilityIntegration:
    def test_metrics_collects_stage_timings(self) -> None:
        mc = MetricsCollector()
        now = datetime.now(timezone.utc)
        mc.record_start(EngineType.PERCEPTION)
        mc.record_end(EngineType.PERCEPTION, now, success=True)
        assert mc.stage_count == 1
        assert mc.total_duration_ms >= 0

    def test_tracing_start_and_end(self) -> None:
        hook = TracingHook()
        span = hook.start_span(
            "perception",
            "trace-001",
            "span-001",
            metadata={"stage": "perception"},
        )
        assert len(hook.spans) == 1
        hook.end_span(span, success=True)
        assert hook.spans[0]["success"] is True

    def test_event_publisher_with_subscriber(self) -> None:
        events = []

        def on_event(event, data):
            events.append((event, data))

        pub = EventPublisher()
        pub.subscribe(on_event)
        pub.publish("custom_event", {"detail": "test"})
        assert len(events) == 1
        assert events[0][0] == "custom_event"

    def test_publisher_with_event_bus(self) -> None:
        mock_bus = MagicMock()
        mock_bus.emit = AsyncMock()
        pub = EventPublisher(event_bus=mock_bus)
        pub.publish(PipelineEvent.PERCEPTION_STARTED, {"stage": "perception"})
        assert mock_bus.emit.called
