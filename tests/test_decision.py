"""Tests for DecisionEngine and decision domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
import pytest

from app.kernel.decision import ActionCandidate, Decision, DecisionEngine
from app.kernel.decision.models import DecisionReason, DecisionScore
from app.kernel.planning.models import Plan, SkillStep


def _plan(
    *steps: SkillStep,
    goal_id: str = "goal_1",
) -> Plan:
    return Plan(goal_id=goal_id, steps=steps)


def _step(order: int, skill: str = "test_skill", desc: str = "") -> SkillStep:
    return SkillStep(order=order, skill_name=skill, description=desc or f"Step {order}")


def _candidate(
    action_type: str = "respond",
    confidence: float = 0.5,
    cost: int = 0,
    source: str = "test",
    description: str = "",
) -> ActionCandidate:
    return ActionCandidate(
        action_type=action_type,
        confidence=confidence,
        estimated_cost=cost,
        source=source,
        description=description or f"Action {action_type}",
    )


# ── ActionCandidate model ────────────────────────────────────────────────


class TestActionCandidateModel:
    """Tests for ActionCandidate domain model."""

    def test_create_minimal(self) -> None:
        c = ActionCandidate(action_type="test")
        assert c.action_type == "test"
        assert c.description == ""
        assert c.parameters == {}
        assert c.confidence == 0.0
        assert c.estimated_cost == 0
        assert c.source == "reasoning"

    def test_create_full(self) -> None:
        c = ActionCandidate(
            action_type="tool_call",
            description="Call the search tool",
            parameters={"query": "hello"},
            confidence=0.9,
            estimated_cost=100,
            source="planner",
        )
        assert c.action_type == "tool_call"
        assert c.confidence == 0.9
        assert c.estimated_cost == 100

    def test_frozen(self) -> None:
        c = ActionCandidate(action_type="test")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            c.action_type = "changed"  # type: ignore[misc]

    def test_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            ActionCandidate(action_type="bad", confidence=1.5)
        with pytest.raises(ValidationError):
            ActionCandidate(action_type="bad", confidence=-0.1)

    def test_cost_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ActionCandidate(action_type="bad", estimated_cost=-5)

    def test_equality(self) -> None:
        a = ActionCandidate(action_type="test", confidence=0.5)
        b = ActionCandidate(action_type="test", confidence=0.5)
        assert a == b

    def test_inequality(self) -> None:
        a = ActionCandidate(action_type="test", confidence=0.5)
        b = ActionCandidate(action_type="other", confidence=0.5)
        assert a != b

    def test_serialization_roundtrip(self) -> None:
        original = ActionCandidate(
            action_type="tool_call",
            confidence=0.8,
            estimated_cost=50,
        )
        data = original.model_dump()
        restored = ActionCandidate.model_validate(data)
        assert restored == original


# ── Decision model ───────────────────────────────────────────────────────


class TestDecisionModel:
    """Tests for Decision domain model."""

    def test_create_minimal(self) -> None:
        action = ActionCandidate(action_type="test")
        d = Decision(selected_action=action)
        assert d.selected_action == action
        assert d.session_id == ""
        assert d.candidates == ()
        assert d.reason.primary_rationale == ""
        assert d.strategy_used == "greedy"
        assert d.status == "pending"
        assert isinstance(d.created_at, datetime)

    def test_frozen(self) -> None:
        action = ActionCandidate(action_type="test")
        d = Decision(selected_action=action)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            d.status = "approved"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        action = ActionCandidate(action_type="test", confidence=0.9)
        original = Decision(
            selected_action=action,
            candidates=(action,),
            strategy_used="greedy",
            status="pending",
        )
        data = original.model_dump()
        restored = Decision.model_validate(data)
        assert restored == original

    def test_decision_reason_default(self) -> None:
        reason = DecisionReason()
        assert reason.primary_rationale == ""
        assert reason.criteria_scores == ()
        assert reason.trade_offs == ()

    def test_decision_score_defaults(self) -> None:
        score = DecisionScore(criterion_name="test", score=0.5)
        assert score.criterion_name == "test"
        assert score.score == 0.5
        assert score.weight == 1.0
        assert score.rationale == ""

    def test_decision_score_range(self) -> None:
        with pytest.raises(ValidationError):
            DecisionScore(criterion_name="bad", score=1.5)
        with pytest.raises(ValidationError):
            DecisionScore(criterion_name="bad", score=-0.1)


# ── decide() — single candidate ──────────────────────────────────────────


class TestDecideSingle:
    """Tests for decide() with a single step plan."""

    @pytest.fixture
    def engine(self) -> DecisionEngine:
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_decide_returns_decision(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0))
        decision = await engine.decide(plan)
        assert isinstance(decision, Decision)

    @pytest.mark.asyncio
    async def test_decide_selects_action(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "analyze"))
        decision = await engine.decide(plan)
        assert decision.selected_action.action_type == "skill_call"

    @pytest.mark.asyncio
    async def test_decide_candidates_populated(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "analyze"))
        decision = await engine.decide(plan)
        assert len(decision.candidates) == 1

    @pytest.mark.asyncio
    async def test_decide_strategy_greedy(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0))
        decision = await engine.decide(plan)
        assert decision.strategy_used == "greedy"

    @pytest.mark.asyncio
    async def test_decide_status_pending(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0))
        decision = await engine.decide(plan)
        assert decision.status == "pending"

    @pytest.mark.asyncio
    async def test_decide_has_reason(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "analyze"))
        decision = await engine.decide(plan)
        assert decision.reason.primary_rationale
        assert len(decision.reason.trade_offs) == 0  # only one candidate, no trade-offs

    @pytest.mark.asyncio
    async def test_decide_deterministic(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "test"))
        d1 = await engine.decide(plan)
        d2 = await engine.decide(plan)
        assert d1.selected_action == d2.selected_action
        assert d1.strategy_used == d2.strategy_used
        assert d1.status == d2.status


# ── decide() — multiple steps ────────────────────────────────────────────


class TestDecideMultiple:
    """Tests for decide() with multi-step plans."""

    @pytest.fixture
    def engine(self) -> DecisionEngine:
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_decide_multiple_candidates(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "a"), _step(1, "b"), _step(2, "c"))
        decision = await engine.decide(plan)
        assert len(decision.candidates) == 3

    @pytest.mark.asyncio
    async def test_decide_selects_highest_confidence(
        self, engine: DecisionEngine
    ) -> None:
        plan = _plan(_step(0, "first"), _step(1, "second"))
        decision = await engine.decide(plan)
        # Step 0 has confidence 1.0/(0+1) = 1.0, step 1 has 1.0/2 = 0.5
        assert decision.selected_action.action_type == "skill_call"
        assert decision.selected_action.confidence == 1.0

    @pytest.mark.asyncio
    async def test_decide_trade_offs_included(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "a"), _step(1, "b"))
        decision = await engine.decide(plan)
        assert len(decision.reason.trade_offs) == 1
        assert "Rejected" in decision.reason.trade_offs[0]

    @pytest.mark.asyncio
    async def test_decide_criteria_scores(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "analyze"))
        decision = await engine.decide(plan)
        assert len(decision.reason.criteria_scores) == 1
        assert decision.reason.criteria_scores[0].criterion_name == "confidence"


# ── decide() — edge cases ────────────────────────────────────────────────


class TestDecideEdgeCases:
    """Tests for edge cases in decide()."""

    @pytest.fixture
    def engine(self) -> DecisionEngine:
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_decide_empty_plan(self, engine: DecisionEngine) -> None:
        plan = _plan()
        decision = await engine.decide(plan)
        assert decision.selected_action.action_type == "noop"
        assert decision.selected_action.confidence == 0.0

    @pytest.mark.asyncio
    async def test_decide_empty_plan_has_reason(self, engine: DecisionEngine) -> None:
        plan = _plan()
        decision = await engine.decide(plan)
        assert "no steps" in decision.reason.primary_rationale.lower()

    @pytest.mark.asyncio
    async def test_decide_single_step_descriptions(
        self, engine: DecisionEngine
    ) -> None:
        plan = _plan(_step(0, "custom_skill", "Do something custom"))
        decision = await engine.decide(plan)
        assert "Do something custom" in decision.selected_action.description

    @pytest.mark.asyncio
    async def test_decide_frozen_output(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(0, "test"))
        decision = await engine.decide(plan)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            decision.status = "approved"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_decide_many_steps(self, engine: DecisionEngine) -> None:
        steps = tuple(_step(i, f"s{i}") for i in range(10))
        plan = _plan(*steps)
        decision = await engine.decide(plan)
        assert len(decision.candidates) == 10
        assert decision.selected_action.confidence == 1.0  # step 0


# ── rank_candidates() ────────────────────────────────────────────────────


class TestRankCandidates:
    """Tests for rank_candidates()."""

    @pytest.fixture
    def engine(self) -> DecisionEngine:
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_rank_empty_list(self, engine: DecisionEngine) -> None:
        ranked = await engine.rank_candidates([])
        assert ranked == []

    @pytest.mark.asyncio
    async def test_rank_single(self, engine: DecisionEngine) -> None:
        c = _candidate(action_type="test", confidence=0.5)
        ranked = await engine.rank_candidates([c])
        assert len(ranked) == 1
        assert ranked[0] == c

    @pytest.mark.asyncio
    async def test_rank_by_confidence_descending(self, engine: DecisionEngine) -> None:
        low = _candidate("a", confidence=0.3)
        high = _candidate("b", confidence=0.9)
        mid = _candidate("c", confidence=0.6)
        ranked = await engine.rank_candidates([low, high, mid])
        assert [c.action_type for c in ranked] == ["b", "c", "a"]

    @pytest.mark.asyncio
    async def test_rank_tie_confidence_by_cost(self, engine: DecisionEngine) -> None:
        cheap = _candidate("a", confidence=0.5, cost=10)
        expensive = _candidate("b", confidence=0.5, cost=100)
        ranked = await engine.rank_candidates([expensive, cheap])
        assert ranked[0] == cheap
        assert ranked[1] == expensive

    @pytest.mark.asyncio
    async def test_rank_stable_tie_all_equal(self, engine: DecisionEngine) -> None:
        a = _candidate("a", confidence=0.5, cost=10)
        b = _candidate("b", confidence=0.5, cost=10)
        ranked = await engine.rank_candidates([a, b])
        assert ranked[0] == a
        assert ranked[1] == b

    @pytest.mark.asyncio
    async def test_rank_deterministic(self, engine: DecisionEngine) -> None:
        candidates = [
            _candidate("a", confidence=0.7, cost=20),
            _candidate("b", confidence=0.3, cost=5),
            _candidate("c", confidence=0.9, cost=50),
        ]
        r1 = await engine.rank_candidates(candidates)
        r2 = await engine.rank_candidates(candidates)
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_rank_does_not_mutate_input(self, engine: DecisionEngine) -> None:
        candidates = [
            _candidate("a", confidence=0.3),
            _candidate("b", confidence=0.9),
        ]
        original_order = list(candidates)
        await engine.rank_candidates(candidates)
        assert candidates == original_order

    @pytest.mark.asyncio
    async def test_rank_returns_new_list(self, engine: DecisionEngine) -> None:
        candidates = [_candidate("a", confidence=0.5)]
        ranked = await engine.rank_candidates(candidates)
        assert ranked is not candidates


# ── Backward-compatible evaluate() ───────────────────────────────────────


class TestEvaluateAlias:
    """Tests for the evaluate() backward-compatible alias."""

    @pytest.fixture
    def engine(self) -> DecisionEngine:
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_evaluate_returns_decision(self, engine: DecisionEngine) -> None:
        decision = await engine.evaluate(None, None)  # type: ignore[arg-type]
        assert isinstance(decision, Decision)

    @pytest.mark.asyncio
    async def test_evaluate_respond_action(self, engine: DecisionEngine) -> None:
        decision = await engine.evaluate(None, None)  # type: ignore[arg-type]
        assert decision.selected_action.action_type == "respond"

    @pytest.mark.asyncio
    async def test_evaluate_has_candidates(self, engine: DecisionEngine) -> None:
        decision = await engine.evaluate(None, None)  # type: ignore[arg-type]
        assert len(decision.candidates) == 1


# ── Edge cases ───────────────────────────────────────────────────────────


class TestDecisionEdgeCases:
    """Tests for additional edge cases."""

    @pytest.fixture
    def engine(self) -> DecisionEngine:
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_decide_multiple_same_confidence(
        self, engine: DecisionEngine
    ) -> None:
        plan = _plan(_step(0, "a"), _step(1, "b"))
        decision = await engine.decide(plan)
        # Step 0 should win (higher confidence 1.0 vs 0.5)
        assert decision.selected_action.parameters["order"] == 0

    @pytest.mark.asyncio
    async def test_decide_candidate_parameters(self, engine: DecisionEngine) -> None:
        plan = _plan(_step(3, "fetch_data"))
        decision = await engine.decide(plan)
        params = decision.selected_action.parameters
        assert params["skill_name"] == "fetch_data"
        assert params["order"] == 3

    @pytest.mark.asyncio
    async def test_register_criterion(self, engine: DecisionEngine) -> None:
        from unittest.mock import AsyncMock

        mock_criterion = AsyncMock()
        engine.register_criterion("test_criterion", mock_criterion)
        assert "test_criterion" in engine._criteria
