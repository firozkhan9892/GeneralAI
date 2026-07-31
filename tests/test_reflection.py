"""Comprehensive tests for the Reflection Engine (Phase 3.5H)."""

from __future__ import annotations

import pickle

import pytest

from app.kernel.reflection.engine import ReflectionEngine
from app.kernel.reflection.models import (
    ErrorDetail,
    ErrorType,
    Refinement,
    ReflectionReport,
    ReflectionRequest,
    ReflectionScore,
)
from app.kernel.reflection.modes import IReflectionStrategy
from app.kernel.reflection.modes.base import (
    BasicReflectionStrategy,
    ConsistencyReflectionStrategy,
    FallbackReflectionStrategy,
    QualityReflectionStrategy,
)
from app.kernel.reasoning.models import ReasoningStep, ReasoningTrace


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_trace(
    num_steps: int = 3,
    conclusion: str | None = "Done",
) -> ReasoningTrace:
    steps = [
        ReasoningStep(
            id=f"s{i}",
            content=f"Step {i} content",
        )
        for i in range(num_steps)
    ]
    return ReasoningTrace(steps=tuple(steps), conclusion=conclusion)


# ──────────────────────────────────────────────
# Engine lifecycle
# ──────────────────────────────────────────────


class TestReflectionEngineInit:
    """Engine creation and default state."""

    def test_create_engine(self) -> None:
        engine = ReflectionEngine()
        assert hasattr(engine, "_modes")

    def test_default_fallback_registered(self) -> None:
        engine = ReflectionEngine()
        assert "fallback" in engine._modes
        assert isinstance(engine._modes["fallback"], FallbackReflectionStrategy)


class TestReflectionEngineRegister:
    """Strategy registration and unregistration."""

    def test_register_mode(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        assert "basic" in engine._modes

    def test_register_duplicate_overwrites(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        engine.register_mode("basic", ConsistencyReflectionStrategy())
        assert isinstance(engine._modes["basic"], ConsistencyReflectionStrategy)

    def test_unregister_mode(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        engine.unregister_mode("basic")
        assert "basic" not in engine._modes

    def test_unregister_unknown_raises_key_error(self) -> None:
        engine = ReflectionEngine()
        with pytest.raises(KeyError, match="not registered"):
            engine.unregister_mode("nonexistent")

    def test_unregister_fallback(self) -> None:
        engine = ReflectionEngine()
        engine.unregister_mode("fallback")
        assert "fallback" not in engine._modes

    def test_register_multiple_strategies(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        engine.register_mode("consistency", ConsistencyReflectionStrategy())
        engine.register_mode("quality", QualityReflectionStrategy())
        assert len(engine._modes) == 4  # 3 + default fallback

    def test_registration_order_preserved(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("a", BasicReflectionStrategy())
        engine.register_mode("b", ConsistencyReflectionStrategy())
        engine.register_mode("c", QualityReflectionStrategy())
        keys = list(engine._modes.keys())
        assert keys == ["fallback", "a", "b", "c"]


# ──────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────


class TestReflectionEngineEvaluate:
    """Core evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_with_trace(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        trace = _make_trace(num_steps=5)
        request = ReflectionRequest(
            output="result",
            trace=trace,
            mode="basic",
        )
        report = await engine.evaluate(request)
        assert isinstance(report, ReflectionReport)
        assert report.overall_score > 0

    @pytest.mark.asyncio
    async def test_evaluate_without_trace(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        request = ReflectionRequest(
            output="result",
            mode="basic",
        )
        report = await engine.evaluate(request)
        assert isinstance(report, ReflectionReport)

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("quality", QualityReflectionStrategy())
        trace = _make_trace(num_steps=3, conclusion="Final answer")
        request = ReflectionRequest(
            output="Final answer with context",
            trace=trace,
            context={"key": "value"},
            mode="quality",
        )
        report = await engine.evaluate(request)
        assert isinstance(report, ReflectionReport)

    @pytest.mark.asyncio
    async def test_evaluate_deterministic(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        trace = _make_trace(num_steps=4)
        request = ReflectionRequest(output="x", trace=trace, mode="basic")
        r1 = await engine.evaluate(request)
        r2 = await engine.evaluate(request)
        assert r1.overall_score == r2.overall_score
        assert r1.verdict == r2.verdict
        assert r1.errors == r2.errors
        assert r1.dimension_scores == r2.dimension_scores

    @pytest.mark.asyncio
    async def test_evaluate_fallback_on_unknown_mode(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        request = ReflectionRequest(output="x", mode="nonexistent")
        report = await engine.evaluate(request)
        assert report.verdict == "pass"
        assert report.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_no_strategies_at_all(self) -> None:
        engine = ReflectionEngine()
        engine.unregister_mode("fallback")
        request = ReflectionRequest(output="x", mode="anything")
        report = await engine.evaluate(request)
        assert report.overall_score == 0.5
        assert report.verdict == "needs_review"

    @pytest.mark.asyncio
    async def test_evaluate_return_type(self) -> None:
        engine = ReflectionEngine()
        request = ReflectionRequest(output="x", mode="fallback")
        report = await engine.evaluate(request)
        assert isinstance(report, ReflectionReport)

    @pytest.mark.asyncio
    async def test_evaluate_empty_output(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        request = ReflectionRequest(output="", mode="basic")
        report = await engine.evaluate(request)
        assert report.overall_score < 0.5

    @pytest.mark.asyncio
    async def test_evaluate_none_output(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        request = ReflectionRequest(output=None, mode="basic")
        report = await engine.evaluate(request)
        assert report.overall_score == 0.0


# ──────────────────────────────────────────────
# BasicReflectionStrategy
# ──────────────────────────────────────────────


class TestBasicReflectionStrategy:
    """Unit tests for the BasicReflectionStrategy."""

    @pytest.mark.asyncio
    async def test_empty_output_score_zero(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output="")
        report = await strategy.evaluate(request)
        assert report.overall_score == 0.0
        assert report.verdict == "fail"

    @pytest.mark.asyncio
    async def test_none_output_score_zero(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output=None)
        report = await strategy.evaluate(request)
        assert report.overall_score == 0.0
        assert report.verdict == "fail"

    @pytest.mark.asyncio
    async def test_one_step_partial(self) -> None:
        strategy = BasicReflectionStrategy()
        trace = _make_trace(num_steps=1)
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert report.overall_score == pytest.approx(0.6)
        assert report.verdict == "needs_review"

    @pytest.mark.asyncio
    async def test_three_steps_good(self) -> None:
        strategy = BasicReflectionStrategy()
        trace = _make_trace(num_steps=3)
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert report.overall_score == pytest.approx(0.95)
        assert report.verdict == "pass"

    @pytest.mark.asyncio
    async def test_five_steps_perfect(self) -> None:
        strategy = BasicReflectionStrategy()
        trace = _make_trace(num_steps=5)
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert report.overall_score == 1.0
        assert report.verdict == "pass"

    @pytest.mark.asyncio
    async def test_conclusion_bonus(self) -> None:
        strategy = BasicReflectionStrategy()
        trace_with = _make_trace(num_steps=3, conclusion="Yes")
        trace_without = ReasoningTrace(steps=trace_with.steps, conclusion=None)
        r1 = await strategy.evaluate(ReflectionRequest(output="x", trace=trace_with))
        r2 = await strategy.evaluate(ReflectionRequest(output="x", trace=trace_without))
        assert r1.overall_score > r2.overall_score

    @pytest.mark.asyncio
    async def test_incomplete_error_on_empty_output(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output="")
        report = await strategy.evaluate(request)
        assert len(report.errors) == 1
        assert report.errors[0].type == ErrorType.INCOMPLETE

    @pytest.mark.asyncio
    async def test_refinement_on_empty_output(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output="")
        report = await strategy.evaluate(request)
        assert len(report.refinements) == 1

    @pytest.mark.asyncio
    async def test_dimension_scores_present(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output="x", trace=_make_trace(3))
        report = await strategy.evaluate(request)
        assert len(report.dimension_scores) == 2
        dims = {d.dimension for d in report.dimension_scores}
        assert dims == {"completeness", "structure"}


# ──────────────────────────────────────────────
# ConsistencyReflectionStrategy
# ──────────────────────────────────────────────


class TestConsistencyReflectionStrategy:
    """Unit tests for the ConsistencyReflectionStrategy."""

    @pytest.mark.asyncio
    async def test_no_contradictions_clean(self) -> None:
        strategy = ConsistencyReflectionStrategy()
        trace = _make_trace(num_steps=2)
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert report.overall_score >= 0.8

    @pytest.mark.asyncio
    async def test_detects_contradiction(self) -> None:
        strategy = ConsistencyReflectionStrategy()
        steps = (
            ReasoningStep(id="s0", content="This is true"),
            ReasoningStep(id="s1", content="This is false"),
        )
        trace = ReasoningTrace(steps=steps, conclusion="maybe")
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert any(e.type == ErrorType.CONTRADICTION for e in report.errors)

    @pytest.mark.asyncio
    async def test_detects_logical_gap(self) -> None:
        strategy = ConsistencyReflectionStrategy()
        steps = (
            ReasoningStep(id="s0", content="abcdefghij"),
            ReasoningStep(id="s1", content="klmnopqrst"),
        )
        trace = ReasoningTrace(steps=steps, conclusion="mix")
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert any(e.type == ErrorType.LOGICAL_GAP for e in report.errors)

    @pytest.mark.asyncio
    async def test_no_logical_gap_when_words_overlap(self) -> None:
        strategy = ConsistencyReflectionStrategy()
        steps = (
            ReasoningStep(id="s0", content="common words here"),
            ReasoningStep(id="s1", content="common words there"),
        )
        trace = ReasoningTrace(steps=steps, conclusion="ok")
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert not any(e.type == ErrorType.LOGICAL_GAP for e in report.errors)

    @pytest.mark.asyncio
    async def test_empty_trace_no_errors(self) -> None:
        strategy = ConsistencyReflectionStrategy()
        trace = ReasoningTrace(steps=())
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        assert len(report.errors) == 0
        assert report.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_token_cost(self) -> None:
        strategy = ConsistencyReflectionStrategy()
        request = ReflectionRequest(output="x")
        report = await strategy.evaluate(request)
        assert report.token_cost == 15


# ──────────────────────────────────────────────
# QualityReflectionStrategy
# ──────────────────────────────────────────────


class TestQualityReflectionStrategy:
    """Unit tests for the QualityReflectionStrategy."""

    @pytest.mark.asyncio
    async def test_all_dimensions_present(self) -> None:
        strategy = QualityReflectionStrategy()
        trace = _make_trace(num_steps=5, conclusion="Final")
        request = ReflectionRequest(output="result", trace=trace)
        report = await strategy.evaluate(request)
        dims = {d.dimension for d in report.dimension_scores}
        assert dims == {"correctness", "completeness", "clarity", "relevance"}

    @pytest.mark.asyncio
    async def test_perfect_input(self) -> None:
        strategy = QualityReflectionStrategy()
        trace = _make_trace(num_steps=5, conclusion="Concluded")
        request = ReflectionRequest(output="output result", trace=trace)
        report = await strategy.evaluate(request)
        assert report.overall_score >= 0.8

    @pytest.mark.asyncio
    async def test_no_output_no_trace(self) -> None:
        strategy = QualityReflectionStrategy()
        request = ReflectionRequest(output=None)
        report = await strategy.evaluate(request)
        assert report.overall_score < 0.5
        assert report.verdict == "fail"

    @pytest.mark.asyncio
    async def test_context_relevance(self) -> None:
        strategy = QualityReflectionStrategy()
        trace = _make_trace(num_steps=2, conclusion="The answer is 42")
        request = ReflectionRequest(
            output="answer",
            trace=trace,
            context={"answer": "needed"},
        )
        report = await strategy.evaluate(request)
        assert report.overall_score > 0

    @pytest.mark.asyncio
    async def test_no_context_relevance_is_one(self) -> None:
        strategy = QualityReflectionStrategy()
        trace = _make_trace(num_steps=5, conclusion="Done")
        request = ReflectionRequest(output="x", trace=trace)
        report = await strategy.evaluate(request)
        relevance = next(
            d for d in report.dimension_scores if d.dimension == "relevance"
        )
        assert relevance.score == 1.0

    @pytest.mark.asyncio
    async def test_token_cost(self) -> None:
        strategy = QualityReflectionStrategy()
        request = ReflectionRequest(output="x")
        report = await strategy.evaluate(request)
        assert report.token_cost == 20


# ──────────────────────────────────────────────
# FallbackReflectionStrategy
# ──────────────────────────────────────────────


class TestFallbackReflectionStrategy:
    """Unit tests for the FallbackReflectionStrategy."""

    @pytest.mark.asyncio
    async def test_always_pass(self) -> None:
        strategy = FallbackReflectionStrategy()
        request = ReflectionRequest(output=None)
        report = await strategy.evaluate(request)
        assert report.overall_score == 1.0
        assert report.verdict == "pass"

    @pytest.mark.asyncio
    async def test_minimal_cost(self) -> None:
        strategy = FallbackReflectionStrategy()
        request = ReflectionRequest(output="x")
        report = await strategy.evaluate(request)
        assert report.token_cost == 5

    @pytest.mark.asyncio
    async def test_single_dimension(self) -> None:
        strategy = FallbackReflectionStrategy()
        request = ReflectionRequest(output="x")
        report = await strategy.evaluate(request)
        assert len(report.dimension_scores) == 1
        assert report.dimension_scores[0].dimension == "basic"


# ──────────────────────────────────────────────
# Integration: engine + strategies
# ──────────────────────────────────────────────


class TestIntegration:
    """Engine integration with all strategies."""

    @pytest.mark.asyncio
    async def test_basic_mode(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        trace = _make_trace(5, "Final answer")
        r = await engine.evaluate(
            ReflectionRequest(output="result", trace=trace, mode="basic")
        )
        assert r.overall_score > 0.8

    @pytest.mark.asyncio
    async def test_consistency_mode(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("consistency", ConsistencyReflectionStrategy())
        trace = _make_trace(3, "Result")
        r = await engine.evaluate(
            ReflectionRequest(output="result", trace=trace, mode="consistency")
        )
        assert r.verdict in ("pass", "needs_review")

    @pytest.mark.asyncio
    async def test_quality_mode(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("quality", QualityReflectionStrategy())
        trace = _make_trace(5, "Conclusion")
        r = await engine.evaluate(
            ReflectionRequest(output="result", trace=trace, mode="quality")
        )
        assert r.overall_score > 0.7

    @pytest.mark.asyncio
    async def test_fallback_mode(self) -> None:
        engine = ReflectionEngine()
        r = await engine.evaluate(ReflectionRequest(output="x", mode="fallback"))
        assert r.verdict == "pass"

    @pytest.mark.asyncio
    async def test_all_strategies_registered_and_used(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        engine.register_mode("consistency", ConsistencyReflectionStrategy())
        engine.register_mode("quality", QualityReflectionStrategy())
        for mode in ("basic", "consistency", "quality", "fallback"):
            r = await engine.evaluate(ReflectionRequest(output="x", mode=mode))
            assert isinstance(r, ReflectionReport)


# ──────────────────────────────────────────────
# Serialization
# ──────────────────────────────────────────────


class TestSerialization:
    """Model serialization round-trips."""

    def test_report_model_dump(self) -> None:
        report = ReflectionReport(
            overall_score=0.8,
            dimension_scores=(ReflectionScore(dimension="test", score=0.8),),
        )
        data = report.model_dump()
        assert data["overall_score"] == 0.8
        assert data["verdict"] == "pass"

    def test_report_model_dump_json(self) -> None:
        report = ReflectionReport(overall_score=0.5, verdict="needs_review")
        json_str = report.model_dump_json()
        assert "needs_review" in json_str

    def test_report_deserialize(self) -> None:
        data = {
            "overall_score": 0.9,
            "verdict": "pass",
        }
        report = ReflectionReport.model_validate(data)
        assert report.overall_score == 0.9
        assert report.verdict == "pass"

    def test_error_detail_serialization(self) -> None:
        error = ErrorDetail(
            type=ErrorType.CONTRADICTION,
            severity=0.5,
            description="test",
        )
        data = error.model_dump()
        assert data["type"] == "contradiction"

    def test_refinement_serialization(self) -> None:
        ref = Refinement(
            target_step_id="s0",
            modification="fix it",
            priority=1,
        )
        data = ref.model_dump()
        assert data["target_step_id"] == "s0"

    def test_pickle_round_trip(self) -> None:
        report = ReflectionReport(overall_score=0.7)
        restored = pickle.loads(pickle.dumps(report))
        assert restored.overall_score == 0.7


# ──────────────────────────────────────────────
# Frozen models
# ──────────────────────────────────────────────


class TestFrozenModels:
    """Verify all reflection models are immutable."""

    def test_report_frozen(self) -> None:
        report = ReflectionReport()
        with pytest.raises(Exception):
            report.overall_score = 0.5  # type: ignore[misc]

    def test_score_frozen(self) -> None:
        score = ReflectionScore(dimension="x", score=0.5)
        with pytest.raises(Exception):
            score.score = 0.9  # type: ignore[misc]

    def test_error_detail_frozen(self) -> None:
        error = ErrorDetail(type=ErrorType.STYLE, severity=0.3)
        with pytest.raises(Exception):
            error.severity = 0.9  # type: ignore[misc]

    def test_refinement_frozen(self) -> None:
        ref = Refinement(modification="x")
        with pytest.raises(Exception):
            ref.priority = 5  # type: ignore[misc]


# ──────────────────────────────────────────────
# Equality
# ──────────────────────────────────────────────


class TestEquality:
    """Model equality semantics."""

    def test_reports_equal(self) -> None:
        a = ReflectionReport(overall_score=0.5, verdict="needs_review")
        b = ReflectionReport(overall_score=0.5, verdict="needs_review")
        assert a.overall_score == b.overall_score
        assert a.verdict == b.verdict
        assert a.errors == b.errors
        assert a.dimension_scores == b.dimension_scores

    def test_reports_not_equal(self) -> None:
        a = ReflectionReport(overall_score=0.5)
        b = ReflectionReport(overall_score=1.0)
        assert a != b

    def test_scores_equal(self) -> None:
        a = ReflectionScore(dimension="x", score=0.5)
        b = ReflectionScore(dimension="x", score=0.5)
        assert a == b

    def test_errors_equal(self) -> None:
        a = ErrorDetail(type=ErrorType.STYLE, severity=0.3)
        b = ErrorDetail(type=ErrorType.STYLE, severity=0.3)
        assert a == b


# ──────────────────────────────────────────────
# Edge cases & validation
# ──────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_error_type_values(self) -> None:
        assert ErrorType.HALLUCINATION.value == "hallucination"
        assert ErrorType.CONTRADICTION.value == "contradiction"
        assert ErrorType.INCOMPLETE.value == "incomplete"
        assert ErrorType.IRRELEVANT.value == "irrelevant"
        assert ErrorType.LOGICAL_GAP.value == "logical_gap"
        assert ErrorType.FACTUAL_ERROR.value == "factual_error"
        assert ErrorType.STYLE.value == "style"
        assert ErrorType.SAFETY.value == "safety"

    @pytest.mark.asyncio
    async def test_report_with_no_dimensions(self) -> None:
        engine = ReflectionEngine()
        engine.unregister_mode("fallback")
        request = ReflectionRequest(output="x")
        report = await engine.evaluate(request)
        assert isinstance(report, ReflectionReport)

    @pytest.mark.asyncio
    async def test_whitespace_output_treated_as_empty(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output="   ")
        report = await strategy.evaluate(request)
        assert report.overall_score == 0.0

    def test_score_bounds(self) -> None:
        with pytest.raises(Exception):
            ReflectionScore(dimension="x", score=-0.1)
        with pytest.raises(Exception):
            ReflectionScore(dimension="x", score=1.5)

    def test_severity_bounds(self) -> None:
        with pytest.raises(Exception):
            ErrorDetail(type=ErrorType.STYLE, severity=-0.1)
        with pytest.raises(Exception):
            ErrorDetail(type=ErrorType.STYLE, severity=1.5)

    @pytest.mark.asyncio
    async def test_various_conclusion_lengths(self) -> None:
        strategy = BasicReflectionStrategy()
        for conc in ("", "A", "Short", "A" * 100):
            trace = ReasoningTrace(steps=(), conclusion=conc)
            request = ReflectionRequest(output="x", trace=trace)
            report = await strategy.evaluate(request)
            assert isinstance(report, ReflectionReport)

    @pytest.mark.asyncio
    async def test_list_trace(self) -> None:
        strategy = BasicReflectionStrategy()
        steps_list = [
            ReasoningStep(id="s0", content="step 0"),
            ReasoningStep(id="s1", content="step 1"),
        ]
        request = ReflectionRequest(output="x", trace=steps_list)
        report = await strategy.evaluate(request)
        assert report.overall_score > 0

    @pytest.mark.asyncio
    async def test_report_has_created_at(self) -> None:
        strategy = BasicReflectionStrategy()
        request = ReflectionRequest(output="x")
        report = await strategy.evaluate(request)
        assert report.created_at is not None

    def test_empty_errors_tuple_default(self) -> None:
        report = ReflectionReport()
        assert report.errors == ()

    def test_empty_refinements_tuple_default(self) -> None:
        report = ReflectionReport()
        assert report.refinements == ()


# ──────────────────────────────────────────────
# ErrorDetail & Refinement creation
# ──────────────────────────────────────────────


class TestErrorDetailCreation:
    """ErrorDetail model creation and defaults."""

    def test_with_suggested_fix(self) -> None:
        error = ErrorDetail(
            type=ErrorType.HALLUCINATION,
            severity=0.8,
            description="Made up facts",
            location="trace.steps[0]",
            suggested_fix="Verify facts",
        )
        assert error.suggested_fix == "Verify facts"

    def test_default_severity(self) -> None:
        error = ErrorDetail(type=ErrorType.STYLE)
        assert error.severity == 0.5

    def test_default_description(self) -> None:
        error = ErrorDetail(type=ErrorType.SAFETY)
        assert error.description == ""


class TestRefinementCreation:
    """Refinement model creation and defaults."""

    def test_with_target(self) -> None:
        ref = Refinement(
            target_step_id="s2",
            modification="Improve clarity",
            priority=3,
        )
        assert ref.target_step_id == "s2"

    def test_default_target_none(self) -> None:
        ref = Refinement(modification="Fix it")
        assert ref.target_step_id is None

    def test_default_priority_zero(self) -> None:
        ref = Refinement(modification="Fix it")
        assert ref.priority == 0


# ──────────────────────────────────────────────
# ReflectionReport verdict mapping
# ──────────────────────────────────────────────


class TestVerdictMapping:
    """Verify different scores produce expected verdicts."""

    @pytest.mark.asyncio
    async def test_high_score_pass(self) -> None:
        strategy = BasicReflectionStrategy()
        trace = _make_trace(5, "Conclusion")
        report = await strategy.evaluate(
            ReflectionRequest(output="result", trace=trace)
        )
        assert report.verdict == "pass"

    @pytest.mark.asyncio
    async def test_medium_score_needs_review(self) -> None:
        strategy = BasicReflectionStrategy()
        trace = _make_trace(1, "")
        report = await strategy.evaluate(ReflectionRequest(output="x", trace=trace))
        assert report.verdict == "needs_review"

    @pytest.mark.asyncio
    async def test_low_score_fail(self) -> None:
        strategy = BasicReflectionStrategy()
        report = await strategy.evaluate(ReflectionRequest(output=""))
        assert report.verdict == "fail"


# ──────────────────────────────────────────────
# IReflectionStrategy interface
# ──────────────────────────────────────────────


class TestReflectionStrategyInterface:
    """All strategies implement IReflectionStrategy."""

    def test_basic_is_instance(self) -> None:
        assert isinstance(BasicReflectionStrategy(), IReflectionStrategy)

    def test_consistency_is_instance(self) -> None:
        assert isinstance(ConsistencyReflectionStrategy(), IReflectionStrategy)

    def test_quality_is_instance(self) -> None:
        assert isinstance(QualityReflectionStrategy(), IReflectionStrategy)

    def test_fallback_is_instance(self) -> None:
        assert isinstance(FallbackReflectionStrategy(), IReflectionStrategy)

    def test_concrete_strategies_registered_in_engine(self) -> None:
        engine = ReflectionEngine()
        engine.register_mode("basic", BasicReflectionStrategy())
        engine.register_mode("consistency", ConsistencyReflectionStrategy())
        engine.register_mode("quality", QualityReflectionStrategy())
        for name, strat in engine._modes.items():
            assert isinstance(strat, IReflectionStrategy)
