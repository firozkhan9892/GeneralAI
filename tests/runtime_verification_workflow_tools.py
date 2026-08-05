"""Comprehensive runtime verification of workflow engine, tool system,
memory subsystem, and event bus.

Run:  python -m pytest tests/runtime_verification_workflow_tools.py -v 2>&1
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


# ===========================================================================
# WORKFLOW ENGINE
# ===========================================================================


class TestWorkflowEngine:
    """End-to-end verification of the workflow automation subsystem."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Bootstrap fresh automation components for every test."""
        from app.automation.bootstrap import register_automation_components
        from app.automation.executor import WorkflowExecutor
        from app.automation.graph import WorkflowGraph
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
            WorkflowStatus,
        )
        from app.automation.registries import (
            WorkflowRegistry,
            WorkflowRunRegistry,
        )
        from app.automation.stores import (
            EventStore,
            InMemoryEventStore,
            InMemoryScheduleStore,
            InMemoryWorkflowRunStore,
            InMemoryWorkflowStore,
            ScheduleStore,
            WorkflowRunStore,
            WorkflowStore,
        )
        from app.automation.validation import WorkflowValidator
        from app.automation.workflow import WorkflowGraphExporter, WorkflowService
        from app.core.container import DependencyContainer

        self.WorkflowDefinition = WorkflowDefinition
        self.WorkflowStep = WorkflowStep
        self.WorkflowStepType = WorkflowStepType
        self.WorkflowStatus = WorkflowStatus
        self.WorkflowGraphExporter = WorkflowGraphExporter
        self.WorkflowGraph = WorkflowGraph

        container = DependencyContainer()
        register_automation_components(container)

        self.registry = container.resolve(WorkflowRegistry)
        self.run_registry = container.resolve(WorkflowRunRegistry)
        self.executor = container.resolve(WorkflowExecutor)
        self.validator = container.resolve(WorkflowValidator)
        self.exporter = container.resolve(WorkflowGraphExporter)
        self.definition_store = container.resolve(WorkflowStore)  # type: ignore[type-abstract]
        self.run_store = container.resolve(WorkflowRunStore)  # type: ignore[type-abstract]
        self.schedule_store = container.resolve(ScheduleStore)  # type: ignore[type-abstract]
        self.event_store = container.resolve(EventStore)  # type: ignore[type-abstract]

        self.service = WorkflowService(
            registry=self.registry,
            run_registry=self.run_registry,
            executor=self.executor,
            validator=self.validator,
            exporter=self.exporter,
            definition_store=InMemoryWorkflowStore(),
            run_store=InMemoryWorkflowRunStore(),
            schedule_store=InMemoryScheduleStore(),
            event_store=InMemoryEventStore(),
        )

    # ------------------------------------------------------------------
    # 1. Create WorkflowDefinition with TRANSFORM steps
    # ------------------------------------------------------------------

    def test_create_workflow_definition(self) -> None:
        """Create a WorkflowDefinition with TRANSFORM steps and expression."""
        definition = self.WorkflowDefinition(
            id="test-wf",
            version="1.0.0",
            name="Test Workflow",
            steps=(
                self.WorkflowStep(
                    id="step1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                ),
                self.WorkflowStep(
                    id="step2",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="2",
                    depends_on=("step1",),
                ),
            ),
        )
        assert definition.id == "test-wf"
        assert definition.version == "1.0.0"
        assert len(definition.steps) == 2
        assert definition.steps[0].type == self.WorkflowStepType.TRANSFORM

    # ------------------------------------------------------------------
    # 2. Register and publish definition
    # ------------------------------------------------------------------

    def test_register_and_publish(self) -> None:
        """Register a definition and publish it through the service."""
        definition = self.WorkflowDefinition(
            id="pub-wf",
            version="1.0.0",
            name="Publishable",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                ),
            ),
        )
        created = self.service.create_definition(definition)
        assert created.id == "pub-wf"

        published = self.service.publish_definition("pub-wf", "1.0.0")
        assert published.status == self.WorkflowStatus.PUBLISHED

    # ------------------------------------------------------------------
    # 3. Execute workflow and verify completion
    # ------------------------------------------------------------------

    def test_execute_workflow(self) -> None:
        """Execute a simple TRANSFORM workflow end-to-end."""
        definition = self.WorkflowDefinition(
            id="exec-wf",
            version="1.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                ),
            ),
        )
        self.service.create_definition(definition)
        self.service.publish_definition("exec-wf", "1.0.0")

        async def _run() -> Any:
            return await self.service.execute("exec-wf")

        run = asyncio.run(_run())
        from app.automation.models import WorkflowRunStatus

        assert run.status in (
            WorkflowRunStatus.SUCCEEDED,
            WorkflowRunStatus.RUNNING,
        )

    # ------------------------------------------------------------------
    # 4. list_definitions / get_definition
    # ------------------------------------------------------------------

    def test_list_and_get_definitions(self) -> None:
        """Verify list_definitions and get_definition return expected data."""
        defn_a = self.WorkflowDefinition(
            id="list-wf-a",
            version="1.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                ),
            ),
        )
        defn_b = self.WorkflowDefinition(
            id="list-wf-b",
            version="1.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="2",
                ),
            ),
        )
        self.service.create_definition(defn_a)
        self.service.create_definition(defn_b)

        all_defs = self.service.list_definitions()
        ids = {d.id for d in all_defs}
        assert "list-wf-a" in ids
        assert "list-wf-b" in ids

        found = self.service.get_definition("list-wf-a", "1.0.0")
        assert found is not None
        assert found.id == "list-wf-a"

    # ------------------------------------------------------------------
    # 5. Workflow cancellation
    # ------------------------------------------------------------------

    def test_cancel_workflow_run(self) -> None:
        """Cancel a pending/running workflow run."""
        definition = self.WorkflowDefinition(
            id="cancel-wf",
            version="1.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                ),
            ),
        )
        self.service.create_definition(definition)
        self.service.publish_definition("cancel-wf", "1.0.0")

        async def _run_and_cancel() -> Any:
            run = await self.service.execute("cancel-wf")
            if not run.is_terminal:
                cancelled = self.service.cancel(run.run_id)
                from app.automation.models import WorkflowRunStatus

                assert cancelled.status == WorkflowRunStatus.CANCELLED
            return run

        asyncio.run(_run_and_cancel())

    # ------------------------------------------------------------------
    # 6. Workflow versioning (create new version)
    # ------------------------------------------------------------------

    def test_workflow_versioning(self) -> None:
        """Create multiple versions of the same workflow."""
        v1 = self.WorkflowDefinition(
            id="ver-wf",
            version="1.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                ),
            ),
        )
        self.service.create_definition(v1)
        self.service.publish_definition("ver-wf", "1.0.0")

        v2 = self.WorkflowDefinition(
            id="ver-wf",
            version="2.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="2",
                ),
            ),
        )
        self.service.create_definition(v2)
        self.service.publish_definition("ver-wf", "2.0.0")

        all_defs = self.service.list_definitions()
        versions = [d.version for d in all_defs if d.id == "ver-wf"]
        assert "1.0.0" in versions
        assert "2.0.0" in versions

        latest = self.service.get_definition("ver-wf")
        assert latest is not None
        assert latest.version == "2.0.0"

    # ------------------------------------------------------------------
    # 7. WorkflowGraphExporter
    # ------------------------------------------------------------------

    def test_workflow_graph_exporter(self) -> None:
        """Export a workflow definition to a JSON-safe graph."""
        definition = self.WorkflowDefinition(
            id="graph-wf",
            version="1.0.0",
            steps=(
                self.WorkflowStep(
                    id="s1",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="1",
                    name="First",
                ),
                self.WorkflowStep(
                    id="s2",
                    type=self.WorkflowStepType.TRANSFORM,
                    expression="2",
                    depends_on=("s1",),
                    name="Second",
                ),
            ),
        )
        exporter = self.WorkflowGraphExporter()
        graph = exporter.export(definition)

        assert graph["workflow_id"] == "graph-wf"
        assert graph["version"] == "1.0.0"
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["source"] == "s1"
        assert graph["edges"][0]["target"] == "s2"
        assert "topological_order" in graph
        assert graph["topological_order"] == ["s1", "s2"]


# ===========================================================================
# TOOL SYSTEM
# ===========================================================================


class TestToolSystem:
    """Verify ToolRegistry, ToolExecutor, built-in tools and MockTool."""

    # ------------------------------------------------------------------
    # 1. ToolRegistry discover and built-in tools
    # ------------------------------------------------------------------

    def test_registry_discover_builtins(self) -> None:
        """Discover default tools and verify known built-in names exist."""
        from app.tools import ToolRegistry

        registry = ToolRegistry()
        count = registry.discover()
        assert count > 0

        names = registry.names()
        for expected in ("calculator", "clock", "echo", "text_utils"):
            assert expected in names, f"Expected tool '{expected}' not found"

    def test_registry_count(self) -> None:
        """Registry count matches discovered tools."""
        from app.tools import ToolRegistry

        registry = ToolRegistry()
        count = registry.discover()
        assert registry.count == count

    def test_registry_has_and_get(self) -> None:
        """has() and get() return consistent results."""
        from app.tools import ToolRegistry

        registry = ToolRegistry()
        registry.discover()

        assert registry.has("calculator")
        tool = registry.get("calculator")
        assert tool is not None
        assert tool.name == "calculator"
        assert not registry.has("nonexistent_tool_xyz")

    # ------------------------------------------------------------------
    # 2. ToolExecutor.execute_async for calculator
    # ------------------------------------------------------------------

    def test_executor_calculator(self) -> None:
        """Execute the calculator tool via ToolExecutor."""
        from app.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()
        registry.discover()
        executor = ToolExecutor(registry=registry)

        async def _run() -> Any:
            result = await executor.execute_async("calculator", {"expression": "1+1"})
            return result

        result = asyncio.run(_run())
        assert result.success is True
        assert result.output == 2

    # ------------------------------------------------------------------
    # 3. ToolExecutor.execute_async for echo
    # ------------------------------------------------------------------

    def test_executor_echo(self) -> None:
        """Execute the echo tool via ToolExecutor."""
        from app.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()
        registry.discover()
        executor = ToolExecutor(registry=registry)

        async def _run() -> Any:
            result = await executor.execute_async("echo", {"text": "hello world"})
            return result

        result = asyncio.run(_run())
        assert result.success is True
        assert result.output == "hello world"

    # ------------------------------------------------------------------
    # 4. ToolContext
    # ------------------------------------------------------------------

    def test_tool_context(self) -> None:
        """Create and exercise a ToolContext with session, memory, token."""
        from app.tools.context import (
            CancellationToken,
            ExecutionContext,
            Memory,
            ToolContext,
            ToolSession,
        )

        session = ToolSession(session_id="test-session")
        memory = Memory(initial={"key": "value"})
        execution = ExecutionContext(attempt=1)
        token = CancellationToken()

        ctx = ToolContext(
            session=session,
            memory=memory,
            execution=execution,
            token=token,
        )

        assert ctx.session_id == "test-session"
        assert ctx.memory.get("key") == "value"
        assert ctx.cancelled is False
        assert ctx.execution.attempt == 1

        ctx.cancel()
        assert ctx.cancelled is True
        assert ctx.token.is_cancelled is True

    # ------------------------------------------------------------------
    # 5. MockTool creation and execution
    # ------------------------------------------------------------------

    def test_mock_tool_creation_and_run(self) -> None:
        """Create a MockTool, verify echo behavior and call counting."""
        from app.tools import MockTool, ToolContext

        tool = MockTool(name="my-mock", echo_input=True)
        ctx = ToolContext()
        output = tool.run({"input": "test-data"}, ctx)

        assert output == "Echo: test-data"
        assert tool.call_count == 1

    def test_mock_tool_fixed_result(self) -> None:
        """MockTool with echo_input=False returns the configured result."""
        from app.tools import MockTool

        tool = MockTool(name="fixed", echo_input=False, result=42)
        output = tool.run({"input": "ignored"})
        assert output == 42

    def test_mock_tool_fail_mode(self) -> None:
        """MockTool configured to fail raises ToolExecutionError."""
        from app.tools import MockTool
        from app.tools.exceptions import ToolExecutionError

        tool = MockTool(name="fail-mock", fail="boom")
        with pytest.raises(ToolExecutionError):
            tool.run({})

    def test_mock_tool_on_run_callback(self) -> None:
        """MockTool with on_run callback delegates to the callable."""
        from app.tools import MockTool

        custom_handler = lambda args, ctx: args.get("x", 0) * 2  # noqa: E731
        tool = MockTool(name="cb-mock", on_run=custom_handler)
        output = tool.run({"x": 10})
        assert output == 20


# ===========================================================================
# MEMORY SUBSYSTEM
# ===========================================================================


class TestMemorySubsystem:
    """Verify MemoryEngine: store, retrieve, search, count, get."""

    @pytest.fixture(autouse=True)
    def _engine(self) -> None:
        from app.kernel.memory.engine import MemoryEngine

        self.engine = MemoryEngine()

    # ------------------------------------------------------------------
    # 1. Store a memory record
    # ------------------------------------------------------------------

    def test_remember(self) -> None:
        """Store a memory and verify it returns an id."""
        from app.kernel.memory.models import MemoryTier

        async def _run() -> str:
            return await self.engine.remember(
                "Python is a programming language",
                tier=MemoryTier.SHORT_TERM,
                tags=("python", "lang"),
                importance=0.8,
            )

        record_id = asyncio.run(_run())
        assert record_id, "remember() should return a non-empty id"

    # ------------------------------------------------------------------
    # 2. Retrieve by id
    # ------------------------------------------------------------------

    def test_get_by_id(self) -> None:
        """Store then retrieve a memory by its id."""
        from app.kernel.memory.models import MemoryTier

        async def _run() -> Any:
            rid = await self.engine.remember(
                "The sky is blue",
                tier=MemoryTier.SHORT_TERM,
                tags=("nature",),
                importance=0.6,
            )
            return await self.engine.get(rid)

        record = asyncio.run(_run())
        assert record is not None
        assert record.content == "The sky is blue"

    # ------------------------------------------------------------------
    # 3. Search by keywords
    # ------------------------------------------------------------------

    def test_search_keywords(self) -> None:
        """Search memory by keywords and verify ranked results."""
        from app.kernel.memory.engine import MemoryQuery
        from app.kernel.memory.models import MemoryTier

        async def _run() -> Any:
            await self.engine.remember(
                "Machine learning is a subset of AI",
                tier=MemoryTier.SHORT_TERM,
                tags=("ml",),
                importance=0.9,
            )
            await self.engine.remember(
                "Deep learning uses neural networks",
                tier=MemoryTier.SHORT_TERM,
                tags=("dl",),
                importance=0.8,
            )
            query = MemoryQuery(
                keywords=("learning",),
                limit=5,
            )
            return await self.engine.search(query)

        hits = asyncio.run(_run())
        assert len(hits) >= 1
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    # ------------------------------------------------------------------
    # 4. Count records
    # ------------------------------------------------------------------

    def test_count(self) -> None:
        """Store several records and verify count matches."""
        from app.kernel.memory.models import MemoryTier

        async def _run() -> int:
            await self.engine.remember("a", tier=MemoryTier.SHORT_TERM)
            await self.engine.remember("b", tier=MemoryTier.SHORT_TERM)
            await self.engine.remember("c", tier=MemoryTier.LONG_TERM)
            return await self.engine.count()

        count = asyncio.run(_run())
        assert count == 3

    # ------------------------------------------------------------------
    # 5. Retrieve with MemoryQuery (tier filter)
    # ------------------------------------------------------------------

    def test_retrieve_with_query(self) -> None:
        """Retrieve records filtered by tier."""
        from app.kernel.memory.engine import MemoryQuery
        from app.kernel.memory.models import MemoryTier

        async def _run() -> Any:
            await self.engine.remember("short-term fact", tier=MemoryTier.SHORT_TERM)
            await self.engine.remember("long-term fact", tier=MemoryTier.LONG_TERM)
            query = MemoryQuery(tier=MemoryTier.LONG_TERM)
            return await self.engine.retrieve(query)

        results = asyncio.run(_run())
        assert len(results) == 1
        assert results[0].content == "long-term fact"

    # ------------------------------------------------------------------
    # 6. Forget a memory
    # ------------------------------------------------------------------

    def test_forget(self) -> None:
        """Forget a record and verify it is removed."""
        from app.kernel.memory.models import MemoryTier

        async def _run() -> None:
            rid = await self.engine.remember("ephemeral", tier=MemoryTier.SHORT_TERM)
            removed = await self.engine.forget(rid)
            assert removed is True
            record = await self.engine.get(rid)
            assert record is None

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # 7. Different query patterns
    # ------------------------------------------------------------------

    def test_search_different_queries(self) -> None:
        """Verify search returns different results for different queries."""
        from app.kernel.memory.engine import MemoryQuery
        from app.kernel.memory.models import MemoryTier

        async def _run() -> tuple[list, list]:
            await self.engine.remember(
                "cats are furry animals",
                tier=MemoryTier.SHORT_TERM,
                tags=("pets",),
            )
            await self.engine.remember(
                "dogs are loyal companions",
                tier=MemoryTier.SHORT_TERM,
                tags=("pets",),
            )
            await self.engine.remember(
                "Python is a programming language",
                tier=MemoryTier.SHORT_TERM,
                tags=("coding",),
            )
            q1 = MemoryQuery(keywords=("cats",), limit=5)
            q2 = MemoryQuery(keywords=("Python",), limit=5)
            return await self.engine.search(q1), await self.engine.search(q2)

        cats_hits, python_hits = asyncio.run(_run())
        assert len(cats_hits) >= 1
        assert "cats" in cats_hits[0].record.content.lower()
        assert len(python_hits) >= 1
        assert "python" in python_hits[0].record.content.lower()


# ===========================================================================
# EVENT SYSTEM
# ===========================================================================


class TestEventSystem:
    """Verify EventBus: subscribe, publish, unsubscribe, multiple handlers."""

    @pytest.fixture(autouse=True)
    def _bus(self) -> None:
        from app.core.events.event_bus import EventBus

        self.bus = EventBus()

    # ------------------------------------------------------------------
    # 1. Subscribe and publish
    # ------------------------------------------------------------------

    def test_subscribe_and_publish(self) -> None:
        """Subscribe a handler and verify it is called on publish."""
        from app.core.interfaces.ievent import Event

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        self.bus.subscribe("test.event", handler)

        async def _run() -> None:
            await self.bus.publish(Event(event_type="test.event", source="test"))

        asyncio.run(_run())
        assert len(received) == 1
        assert received[0].source == "test"

    # ------------------------------------------------------------------
    # 2. Unsubscribe
    # ------------------------------------------------------------------

    def test_unsubscribe(self) -> None:
        """Unsubscribe a handler and verify it is no longer called."""
        from app.core.interfaces.ievent import Event

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        self.bus.subscribe("unsub.event", handler)
        self.bus.unsubscribe("unsub.event", handler)

        async def _run() -> None:
            await self.bus.publish(Event(event_type="unsub.event"))

        asyncio.run(_run())
        assert len(received) == 0

    # ------------------------------------------------------------------
    # 3. Multiple subscribers
    # ------------------------------------------------------------------

    def test_multiple_subscribers(self) -> None:
        """Multiple handlers for the same event type are all called."""
        from app.core.interfaces.ievent import Event

        call_order: list[str] = []

        async def handler_a(event: Event) -> None:
            call_order.append("a")

        async def handler_b(event: Event) -> None:
            call_order.append("b")

        self.bus.subscribe("multi.event", handler_a)
        self.bus.subscribe("multi.event", handler_b)

        async def _run() -> None:
            await self.bus.publish(Event(event_type="multi.event"))

        asyncio.run(_run())
        assert "a" in call_order
        assert "b" in call_order

    # ------------------------------------------------------------------
    # 4. Async handlers
    # ------------------------------------------------------------------

    def test_async_handlers(self) -> None:
        """Async handlers that do I/O are awaited correctly."""
        from app.core.interfaces.ievent import Event

        results: list[int] = []

        async def slow_handler(event: Event) -> None:
            await asyncio.sleep(0.001)
            results.append(42)

        self.bus.subscribe("async.event", slow_handler)

        async def _run() -> None:
            await self.bus.publish(Event(event_type="async.event"))

        asyncio.run(_run())
        assert results == [42]

    # ------------------------------------------------------------------
    # 5. Handler failure does not break others
    # ------------------------------------------------------------------

    def test_handler_failure_isolation(self) -> None:
        """A failing handler does not prevent other handlers from running."""
        from app.core.interfaces.ievent import Event

        surviving: list[str] = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("kaboom")

        async def good_handler(event: Event) -> None:
            surviving.append("ok")

        self.bus.subscribe("fail.event", bad_handler)
        self.bus.subscribe("fail.event", good_handler)

        async def _run() -> None:
            await self.bus.publish(Event(event_type="fail.event"))

        asyncio.run(_run())
        assert surviving == ["ok"]

    # ------------------------------------------------------------------
    # 6. Wildcard subscriber
    # ------------------------------------------------------------------

    def test_wildcard_subscriber(self) -> None:
        """Wildcard '*' handler receives all event types."""
        from app.core.interfaces.ievent import Event

        received: list[str] = []

        async def wildcard(event: Event) -> None:
            received.append(event.event_type)

        self.bus.subscribe("*", wildcard)

        async def _run() -> None:
            await self.bus.publish(Event(event_type="alpha"))
            await self.bus.publish(Event(event_type="beta"))

        asyncio.run(_run())
        assert received == ["alpha", "beta"]

    # ------------------------------------------------------------------
    # 7. No handlers for event type
    # ------------------------------------------------------------------

    def test_no_handlers_does_not_raise(self) -> None:
        """Publishing to an event type with no handlers is a no-op."""
        from app.core.interfaces.ievent import Event

        async def _run() -> None:
            await self.bus.publish(Event(event_type="nonexistent.event.type"))

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # 8. Clear all handlers
    # ------------------------------------------------------------------

    def test_clear_handlers(self) -> None:
        """clear() removes all registered handlers."""
        from app.core.interfaces.ievent import Event

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        self.bus.subscribe("clear.event", handler)
        self.bus.clear()

        async def _run() -> None:
            await self.bus.publish(Event(event_type="clear.event"))

        asyncio.run(_run())
        assert len(received) == 0
