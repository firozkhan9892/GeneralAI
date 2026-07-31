"""Smoke tests for Cognitive Kernel skeleton: imports, wire-up, exports."""

from __future__ import annotations


import pytest

from app.core.container import DependencyContainer
from app.core.lifecycle import LifecycleManager
from app.kernel import CognitiveOrchestrator
from app.kernel.bootstrap import (
    bootstrap_kernel,
    register_kernel_components,
    register_kernel_lifecycle_hooks,
)
from app.kernel.orchestrator import CognitiveOrchestrator as _CO
from app.kernel.perception.engine import PerceptionEngine
from app.kernel.perception.models import ModalityType, Percept, RawMessage
from app.kernel.response.models import OutputMessage
from app.kernel.goals.engine import GoalEngine
from app.kernel.goals.models import Goal, GoalStatus, GoalType
from app.kernel.intent.engine import IntentEngine
from app.kernel.intent.models import Intent, IntentConfidence, IntentType
from app.kernel.planning import Plan, PlanningEngine
from app.kernel.reasoning import ReasoningEngine, ReasoningTrace
from app.kernel.reasoning.models import ReasoningRequest
from app.kernel.decision import Decision, DecisionEngine
from app.kernel.decision.models import ActionCandidate
from app.kernel.capability.manager import CapabilityManager
from app.kernel.policy import PolicyDecision, PolicyEngine, VerdictType
from app.kernel.skills.executor import SkillSelector, SkillExecutor
from app.kernel.tools.executor import ToolResolver, ToolExecutor
from app.kernel.reflection.engine import ReflectionEngine
from app.kernel.experience.engine import ExperienceEngine, ExperienceStore
from app.kernel.experience.models import Experience, ExperienceQuery
from app.kernel.context.manager import ContextManager, ContextBuilder, ContextPruner
from app.kernel.state.manager import StateManager
from app.kernel.pipeline.executor import PipelineExecutor
from app.kernel.models.router import ModelRouter
from app.kernel.response.builder import ResponseBuilder
from app.kernel.events.definitions import (
    EVENT_PERCEPTION_STARTED,
    EVENT_INTENT_IDENTIFIED,
    EVENT_GOAL_CREATED,
    EVENT_PLAN_CREATED,
    EVENT_REASONING_STARTED,
    EVENT_DECISION_MADE,
    EVENT_POLICY_EVALUATED,
    EVENT_EXECUTION_STARTED,
    EVENT_REFLECTION_STARTED,
    EVENT_EXPERIENCE_RECORDED,
    EVENT_STATE_CHANGED,
)


class TestKernelPackageImports:
    """All public kernel symbols should be importable."""

    def test_orchestrator_import(self) -> None:
        assert CognitiveOrchestrator is _CO

    def test_all_engine_imports(self) -> None:
        assert PerceptionEngine is not None
        assert IntentEngine is not None
        assert GoalEngine is not None
        assert PlanningEngine is not None
        assert ReasoningEngine is not None
        assert DecisionEngine is not None
        assert CapabilityManager is not None
        assert PolicyEngine is not None
        assert SkillSelector is not None
        assert SkillExecutor is not None
        assert ToolResolver is not None
        assert ToolExecutor is not None
        assert ReflectionEngine is not None
        assert ExperienceEngine is not None
        assert ExperienceStore is not None
        assert ContextManager is not None
        assert ContextBuilder is not None
        assert ContextPruner is not None
        assert StateManager is not None
        assert PipelineExecutor is not None
        assert ModelRouter is not None
        assert ResponseBuilder is not None

    def test_event_imports(self) -> None:
        assert EVENT_PERCEPTION_STARTED == "kernel.perception.started"
        assert EVENT_INTENT_IDENTIFIED == "kernel.intent.identified"
        assert EVENT_GOAL_CREATED == "kernel.goal.created"
        assert EVENT_PLAN_CREATED == "kernel.plan.created"
        assert EVENT_REASONING_STARTED == "kernel.reasoning.started"
        assert EVENT_DECISION_MADE == "kernel.decision.made"
        assert EVENT_POLICY_EVALUATED == "kernel.policy.evaluated"
        assert EVENT_EXECUTION_STARTED == "kernel.execution.started"
        assert EVENT_REFLECTION_STARTED == "kernel.reflection.started"
        assert EVENT_EXPERIENCE_RECORDED == "kernel.experience.recorded"
        assert EVENT_STATE_CHANGED == "kernel.state.changed"


class TestKernelBootstrap:
    """Verify DI container registration and lifecycle hook wiring."""

    def test_register_all_components(self) -> None:
        container = DependencyContainer()
        register_kernel_components(container)
        assert container.has(CognitiveOrchestrator)
        assert container.has(PerceptionEngine)
        assert container.has(IntentEngine)
        assert container.has(GoalEngine)
        assert container.has(PlanningEngine)
        assert container.has(ReasoningEngine)
        assert container.has(DecisionEngine)
        assert container.has(CapabilityManager)
        assert container.has(PolicyEngine)
        assert container.has(SkillSelector)
        assert container.has(SkillExecutor)
        assert container.has(ToolResolver)
        assert container.has(ToolExecutor)
        assert container.has(ReflectionEngine)
        assert container.has(ExperienceEngine)
        assert container.has(ContextManager)
        assert container.has(StateManager)
        assert container.has(PipelineExecutor)
        assert container.has(ModelRouter)
        assert container.has(ResponseBuilder)

    def test_resolve_orchestrator(self) -> None:
        container = DependencyContainer()
        register_kernel_components(container)
        orchestrator = container.resolve(CognitiveOrchestrator)
        assert isinstance(orchestrator, CognitiveOrchestrator)

    def test_registered_lifecycle_hooks(self) -> None:
        container = DependencyContainer()
        lifecycle = LifecycleManager()
        register_kernel_components(container)
        register_kernel_lifecycle_hooks(lifecycle)

    def test_bootstrap_kernel_convenience(self) -> None:
        container = DependencyContainer()
        lifecycle = LifecycleManager()
        orchestrator = bootstrap_kernel(container, lifecycle)
        assert isinstance(orchestrator, CognitiveOrchestrator)


class TestEngineStubs:
    """Every engine method raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_perception_engine(self) -> None:
        engine = PerceptionEngine()
        percept = await engine.perceive(RawMessage(content="Hello"))
        assert percept.normalized_content == "Hello"

    @pytest.mark.asyncio
    async def test_intent_engine(self) -> None:
        engine = IntentEngine()
        intent = await engine.understand(
            Percept(
                raw=RawMessage(content="Hello"),
                normalized_content="Hello",
                modality=ModalityType.TEXT,
            )
        )
        assert intent.primary is not None

    @pytest.mark.asyncio
    async def test_goal_engine(self) -> None:
        engine = GoalEngine()
        hierarchy = await engine.resolve(
            Intent(
                primary=IntentType.ASK_QUESTION,
                confidence=IntentConfidence(primary=0.9),
            )
        )
        assert hierarchy.root.goal_type == GoalType.QUESTION
        assert hierarchy.root.status == GoalStatus.PROPOSED

    @pytest.mark.asyncio
    async def test_planning_engine(self) -> None:
        engine = PlanningEngine()
        goal = Goal(id="test_g", description="Test")
        plan = await engine.create_plan(goal)
        assert isinstance(plan, Plan)
        assert plan.goal_id == "test_g"
        assert plan.revision == 0

    @pytest.mark.asyncio
    async def test_reasoning_engine(self) -> None:
        engine = ReasoningEngine()
        request = ReasoningRequest(problem="Test")
        trace = await engine.reason(request)
        assert isinstance(trace, ReasoningTrace)
        assert len(trace.steps) > 0

    @pytest.mark.asyncio
    async def test_decision_engine(self) -> None:
        engine = DecisionEngine()
        plan = Plan(goal_id="test_g")
        decision = await engine.decide(plan)
        assert isinstance(decision, Decision)
        assert decision.selected_action.action_type == "noop"

    @pytest.mark.asyncio
    async def test_capability_manager(self) -> None:
        mgr = CapabilityManager()
        try:
            await mgr.resolve(None)  # type: ignore[arg-type]
            assert False, "Expected NotImplementedError"
        except NotImplementedError:
            pass

    @pytest.mark.asyncio
    async def test_policy_engine(self) -> None:
        engine = PolicyEngine()
        decision = Decision(selected_action=ActionCandidate(action_type="respond"))
        result = await engine.evaluate(decision)
        assert isinstance(result, PolicyDecision)
        assert result.verdict in (VerdictType.ALLOW,)

    @pytest.mark.asyncio
    async def test_reflection_engine(self) -> None:
        engine = ReflectionEngine()
        from app.kernel.reflection.models import ReflectionRequest, ReflectionReport

        request = ReflectionRequest(output="test output", mode="fallback")
        report = await engine.evaluate(request)
        assert isinstance(report, ReflectionReport)
        assert report.verdict == "pass"

    @pytest.mark.asyncio
    async def test_experience_engine(self) -> None:
        engine = ExperienceEngine()
        exp = Experience(goal_type=IntentType.ASK_QUESTION)
        exp_id = await engine.record(exp)
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0
        results = await engine.search(ExperienceQuery(limit=10))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_response_builder(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build({"session_id": "s1"})
        assert isinstance(msg, OutputMessage)
        assert msg.session_id == "s1"

    @pytest.mark.asyncio
    async def test_orchestrator(self) -> None:
        orch = CognitiveOrchestrator()
        result = await orch.process(None)  # type: ignore[arg-type]
        # Orchestrator returns a failed OutputMessage when engines are not registered.
        assert isinstance(result, OutputMessage)
        assert result.success is False
