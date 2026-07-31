"""Tests for ReasoningEngine and reasoning domain models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from app.kernel.reasoning import ReasoningEngine, ReasoningTrace
from app.kernel.reasoning.models import (
    ReasoningRequest,
    ReasoningStep,
    ReasoningStrategy,
    StepType,
)
from app.kernel.reasoning.strategies.base import (
    ChainOfThoughtStrategy,
    DecompositionStrategy,
    FallbackStrategy,
)


# ── Domain model tests ───────────────────────────────────────────────────


class TestReasoningModels:
    """Tests for reasoning domain models."""

    def test_reasoning_step_create(self) -> None:
        step = ReasoningStep(id="s1", content="Think")
        assert step.id == "s1"
        assert step.content == "Think"
        assert step.type == StepType.THINK
        assert step.token_cost == 0
        assert step.children == ()
        assert step.metadata == {}

    def test_reasoning_step_frozen(self) -> None:
        step = ReasoningStep(id="s1", content="Test")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            step.content = "changed"  # type: ignore[misc]

    def test_step_type_enum(self) -> None:
        assert StepType.THINK.value == "think"
        assert StepType.OBSERVE.value == "observe"
        assert StepType.ACT.value == "act"
        assert StepType.EVALUATE.value == "evaluate"
        assert StepType.SEARCH.value == "search"
        assert StepType.CALCULATE.value == "calculate"

    def test_reasoning_request_create(self) -> None:
        req = ReasoningRequest(problem="Test problem")
        assert req.problem == "Test problem"
        assert req.context == {}
        assert req.constraints == {}
        assert req.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert req.max_steps == 20
        assert req.token_budget == 2000

    def test_reasoning_request_frozen(self) -> None:
        req = ReasoningRequest(problem="Test")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            req.problem = "changed"  # type: ignore[misc]

    def test_reasoning_trace_create(self) -> None:
        trace = ReasoningTrace()
        assert trace.steps == ()
        assert trace.conclusion is None
        assert trace.strategy_used == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert trace.token_cost == 0
        assert trace.duration_ms == 0
        assert trace.metadata == {}

    def test_reasoning_trace_frozen(self) -> None:
        trace = ReasoningTrace()
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            trace.conclusion = "changed"  # type: ignore[misc]

    def test_strategy_enum_values(self) -> None:
        assert ReasoningStrategy.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningStrategy.TREE_OF_THOUGHT.value == "tree_of_thought"
        assert ReasoningStrategy.REACT.value == "react"
        assert ReasoningStrategy.REFLEXION.value == "reflexion"
        assert ReasoningStrategy.STRAW_MAN.value == "straw_man"
        assert ReasoningStrategy.FIRST_PRINCIPLES.value == "first_principles"
        assert ReasoningStrategy.ANALOGICAL.value == "analogical"

    def test_step_serialization_roundtrip(self) -> None:
        original = ReasoningStep(
            id="s1",
            type=StepType.OBSERVE,
            content="Observed data",
            token_cost=25,
            children=("s2",),
            metadata={"source": "test"},
        )
        data = original.model_dump()
        restored = ReasoningStep.model_validate(data)
        assert restored == original

    def test_trace_serialization_roundtrip(self) -> None:
        step = ReasoningStep(id="s1", content="Step 1")
        original = ReasoningTrace(
            steps=(step,),
            conclusion="Done",
            strategy_used=ReasoningStrategy.DECOMPOSITION,
            token_cost=50,
        )
        data = original.model_dump()
        restored = ReasoningTrace.model_validate(data)
        assert restored == original

    def test_step_equality(self) -> None:
        a = ReasoningStep(id="s1", content="Hello")
        b = ReasoningStep(id="s1", content="Hello")
        assert a == b

    def test_step_inequality(self) -> None:
        a = ReasoningStep(id="s1", content="Hello")
        b = ReasoningStep(id="s2", content="Hello")
        assert a != b


# ── ChainOfThoughtStrategy ───────────────────────────────────────────────


class TestChainOfThoughtStrategy:
    """Tests for ChainOfThoughtStrategy."""

    @pytest.fixture
    def strategy(self) -> ChainOfThoughtStrategy:
        return ChainOfThoughtStrategy()

    @pytest.mark.asyncio
    async def test_name(self, strategy: ChainOfThoughtStrategy) -> None:
        assert strategy.name == "chain_of_thought"

    @pytest.mark.asyncio
    async def test_execute_returns_trace(
        self, strategy: ChainOfThoughtStrategy
    ) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await strategy.execute(req)
        assert isinstance(trace, ReasoningTrace)

    @pytest.mark.asyncio
    async def test_execute_minimal_steps(
        self, strategy: ChainOfThoughtStrategy
    ) -> None:
        req = ReasoningRequest(problem="Solve this")
        trace = await strategy.execute(req)
        assert len(trace.steps) >= 3

    @pytest.mark.asyncio
    async def test_execute_with_context(self, strategy: ChainOfThoughtStrategy) -> None:
        req = ReasoningRequest(problem="Test", context={"key": "value"})
        trace = await strategy.execute(req)
        assert any("key" in s.content for s in trace.steps)

    @pytest.mark.asyncio
    async def test_execute_with_constraints(
        self, strategy: ChainOfThoughtStrategy
    ) -> None:
        req = ReasoningRequest(problem="Test", constraints={"time": "5min"})
        trace = await strategy.execute(req)
        assert any("time" in s.content for s in trace.steps)

    @pytest.mark.asyncio
    async def test_execute_conclusion(self, strategy: ChainOfThoughtStrategy) -> None:
        req = ReasoningRequest(problem="My problem")
        trace = await strategy.execute(req)
        assert trace.conclusion is not None
        assert "My problem" in trace.conclusion

    @pytest.mark.asyncio
    async def test_execute_strategy_used(
        self, strategy: ChainOfThoughtStrategy
    ) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await strategy.execute(req)
        assert trace.strategy_used == ReasoningStrategy.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_execute_token_cost(self, strategy: ChainOfThoughtStrategy) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await strategy.execute(req)
        assert trace.token_cost > 0

    @pytest.mark.asyncio
    async def test_execute_deterministic(
        self, strategy: ChainOfThoughtStrategy
    ) -> None:
        req = ReasoningRequest(problem="Test")
        t1 = await strategy.execute(req)
        t2 = await strategy.execute(req)
        assert t1.steps == t2.steps
        assert t1.conclusion == t2.conclusion
        assert t1.strategy_used == t2.strategy_used
        assert t1.token_cost == t2.token_cost


# ── DecompositionStrategy ────────────────────────────────────────────────


class TestDecompositionStrategy:
    """Tests for DecompositionStrategy."""

    @pytest.fixture
    def strategy(self) -> DecompositionStrategy:
        return DecompositionStrategy()

    @pytest.mark.asyncio
    async def test_name(self, strategy: DecompositionStrategy) -> None:
        assert strategy.name == "decomposition"

    @pytest.mark.asyncio
    async def test_execute_with_short_problem(
        self, strategy: DecompositionStrategy
    ) -> None:
        req = ReasoningRequest(problem="Hello")
        trace = await strategy.execute(req)
        assert len(trace.steps) >= 2

    @pytest.mark.asyncio
    async def test_execute_with_long_problem(
        self, strategy: DecompositionStrategy
    ) -> None:
        req = ReasoningRequest(
            problem="This is a longer problem with many words to decompose"
        )
        trace = await strategy.execute(req)
        # Should produce separate sub-problem steps
        sub_steps = [s for s in trace.steps if "Sub-problem" in s.content]
        assert len(sub_steps) >= 1

    @pytest.mark.asyncio
    async def test_execute_conclusion(self, strategy: DecompositionStrategy) -> None:
        req = ReasoningRequest(problem="Test problem")
        trace = await strategy.execute(req)
        assert trace.conclusion is not None

    @pytest.mark.asyncio
    async def test_execute_deterministic(self, strategy: DecompositionStrategy) -> None:
        req = ReasoningRequest(problem="Test")
        t1 = await strategy.execute(req)
        t2 = await strategy.execute(req)
        assert t1.steps == t2.steps
        assert t1.conclusion == t2.conclusion
        assert t1.token_cost == t2.token_cost


# ── FallbackStrategy ─────────────────────────────────────────────────────


class TestFallbackStrategy:
    """Tests for FallbackStrategy."""

    @pytest.fixture
    def strategy(self) -> FallbackStrategy:
        return FallbackStrategy()

    @pytest.mark.asyncio
    async def test_name(self, strategy: FallbackStrategy) -> None:
        assert strategy.name == "fallback"

    @pytest.mark.asyncio
    async def test_execute_two_steps(self, strategy: FallbackStrategy) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await strategy.execute(req)
        assert len(trace.steps) == 2

    @pytest.mark.asyncio
    async def test_execute_deterministic(self, strategy: FallbackStrategy) -> None:
        req = ReasoningRequest(problem="Test")
        t1 = await strategy.execute(req)
        t2 = await strategy.execute(req)
        assert t1.steps == t2.steps
        assert t1.conclusion == t2.conclusion
        assert t1.strategy_used == t2.strategy_used
        assert t1.token_cost == t2.token_cost


# ── ReasoningEngine — reason() ───────────────────────────────────────────


class TestReasoningEngineReason:
    """Tests for ReasoningEngine.reason()."""

    @pytest.fixture
    def engine(self) -> ReasoningEngine:
        return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_reason_returns_trace(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await engine.reason(req)
        assert isinstance(trace, ReasoningTrace)

    @pytest.mark.asyncio
    async def test_reason_default_strategy(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await engine.reason(req)
        assert trace.strategy_used == ReasoningStrategy.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_reason_with_decomposition(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(
            problem="Complex problem",
            strategy=ReasoningStrategy.DECOMPOSITION,
        )
        trace = await engine.reason(req)
        assert trace.strategy_used == ReasoningStrategy.DECOMPOSITION

    @pytest.mark.asyncio
    async def test_reason_steps_populated(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await engine.reason(req)
        assert len(trace.steps) > 0

    @pytest.mark.asyncio
    async def test_reason_conclusion_present(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Solve this")
        trace = await engine.reason(req)
        assert trace.conclusion is not None

    @pytest.mark.asyncio
    async def test_reason_deterministic(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test")
        t1 = await engine.reason(req)
        t2 = await engine.reason(req)
        assert t1.steps == t2.steps
        assert t1.conclusion == t2.conclusion
        assert t1.strategy_used == t2.strategy_used
        assert t1.token_cost == t2.token_cost

    @pytest.mark.asyncio
    async def test_reason_with_context(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test", context={"data": "important"})
        trace = await engine.reason(req)
        assert "data" in str(trace.steps)

    @pytest.mark.asyncio
    async def test_reason_with_constraints(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test", constraints={"max_results": 10})
        trace = await engine.reason(req)
        assert "max_results" in str(trace.steps)

    @pytest.mark.asyncio
    async def test_reason_token_cost_non_zero(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await engine.reason(req)
        assert trace.token_cost > 0


# ── Strategy registration ───────────────────────────────────────────────


class TestStrategyRegistration:
    """Tests for strategy registration."""

    @pytest.fixture
    def engine(self) -> ReasoningEngine:
        return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_register_custom_strategy(self, engine: ReasoningEngine) -> None:
        strategy = FallbackStrategy()
        engine.register_strategy("custom", strategy)
        assert "custom" in engine._strategies

    @pytest.mark.asyncio
    async def test_register_overwrites(self, engine: ReasoningEngine) -> None:
        class CustomStrategy(ChainOfThoughtStrategy):
            @property
            def name(self) -> str:
                return "custom"

        engine.register_strategy("chain_of_thought", CustomStrategy())
        req = ReasoningRequest(problem="Test")
        trace = await engine.reason(req)
        # Should use the custom strategy
        assert trace.strategy_used == ReasoningStrategy.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_unregister_strategy(self, engine: ReasoningEngine) -> None:
        engine.register_strategy("temp", FallbackStrategy())
        assert "temp" in engine._strategies
        engine.unregister_strategy("temp")
        assert "temp" not in engine._strategies

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, engine: ReasoningEngine) -> None:
        engine.unregister_strategy("nonexistent")  # Should not raise


# ── Fallback for unknown strategy ───────────────────────────────────────


class TestFallbackBehavior:
    """Tests for fallback when strategy is not registered."""

    @pytest.fixture
    def engine(self) -> ReasoningEngine:
        return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_unknown_strategy_fallback(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(
            problem="Test",
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
        )
        trace = await engine.reason(req)
        assert trace.conclusion is not None


# ── refine() ────────────────────────────────────────────────────────────


class TestRefine:
    """Tests for refine()."""

    @pytest.fixture
    def engine(self) -> ReasoningEngine:
        return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_refine_adds_step(self, engine: ReasoningEngine) -> None:
        trace = ReasoningTrace(steps=(), conclusion="Initial")
        refined = await engine.refine(trace, "Check your math")
        assert len(refined.steps) == 1
        assert "Check your math" in refined.steps[0].content

    @pytest.mark.asyncio
    async def test_refine_preserves_conclusion(self, engine: ReasoningEngine) -> None:
        trace = ReasoningTrace(steps=(), conclusion="Original conclusion")
        refined = await engine.refine(trace, "Feedback")
        assert refined.conclusion == "Original conclusion"

    @pytest.mark.asyncio
    async def test_refine_increments_token_cost(self, engine: ReasoningEngine) -> None:
        trace = ReasoningTrace(steps=(), conclusion="C", token_cost=50)
        refined = await engine.refine(trace, "Fix")
        assert refined.token_cost == 55

    @pytest.mark.asyncio
    async def test_refine_frozen_output(self, engine: ReasoningEngine) -> None:
        trace = ReasoningTrace(steps=(), conclusion="C")
        refined = await engine.refine(trace, "Fix")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            refined.conclusion = "Hacked"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_refine_deterministic(self, engine: ReasoningEngine) -> None:
        trace = ReasoningTrace(steps=(), conclusion="C")
        r1 = await engine.refine(trace, "Fix")
        r2 = await engine.refine(trace, "Fix")
        assert r1.steps == r2.steps
        assert r1.conclusion == r2.conclusion
        assert r1.token_cost == r2.token_cost


# ── Edge cases ──────────────────────────────────────────────────────────


class TestReasoningEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def engine(self) -> ReasoningEngine:
        return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_empty_problem(self, engine: ReasoningEngine) -> None:
        req = ReasoningRequest(problem="")
        trace = await engine.reason(req)
        assert isinstance(trace, ReasoningTrace)

    @pytest.mark.asyncio
    async def test_long_problem(self, engine: ReasoningEngine) -> None:
        problem = "word " * 500
        req = ReasoningRequest(problem=problem.strip())
        trace = await engine.reason(req)
        assert isinstance(trace, ReasoningTrace)

    @pytest.mark.asyncio
    async def test_trace_strategy_preserved_on_refine(
        self, engine: ReasoningEngine
    ) -> None:
        trace = ReasoningTrace(
            steps=(),
            conclusion="C",
            strategy_used=ReasoningStrategy.DECOMPOSITION,
        )
        refined = await engine.refine(trace, "Fix")
        assert refined.strategy_used == ReasoningStrategy.DECOMPOSITION

    @pytest.mark.asyncio
    async def test_step_token_cost_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningStep(id="bad", content="bad", token_cost=-1)

    @pytest.mark.asyncio
    async def test_request_max_steps_range(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningRequest(problem="Test", max_steps=0)

    @pytest.mark.asyncio
    async def test_request_max_steps_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningRequest(problem="Test", max_steps=101)

    @pytest.mark.asyncio
    async def test_frozen_reasoning_request_output(
        self, engine: ReasoningEngine
    ) -> None:
        req = ReasoningRequest(problem="Test")
        trace = await engine.reason(req)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            trace.conclusion = "Hacked"  # type: ignore[misc]
