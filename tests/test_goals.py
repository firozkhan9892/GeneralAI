"""Tests for GoalEngine and Goal domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
import pytest

from app.kernel.goals import Goal, GoalEngine, GoalHierarchy
from app.kernel.goals.models import GoalPriority, GoalStatus, GoalType
from app.kernel.intent.models import (
    Intent,
    IntentConfidence,
    IntentType,
)


def _intent(
    primary: IntentType,
    *,
    confidence: float = 0.9,
    sub_intents: tuple[Intent, ...] = (),
) -> Intent:
    return Intent(
        primary=primary,
        confidence=IntentConfidence(primary=confidence),
        sub_intents=sub_intents,
    )


# ── GoalEngine — resolve ─────────────────────────────────────────────────


class TestGoalEngineResolve:
    """Tests for GoalEngine.resolve()."""

    @pytest.fixture
    def engine(self) -> GoalEngine:
        return GoalEngine()

    @pytest.mark.asyncio
    async def test_resolve_returns_goal_hierarchy(self, engine: GoalEngine) -> None:
        intent = _intent(IntentType.ASK_QUESTION)
        hierarchy = await engine.resolve(intent)
        assert isinstance(hierarchy, GoalHierarchy)
        assert isinstance(hierarchy.root, Goal)

    @pytest.mark.asyncio
    async def test_resolve_single_goal(self, engine: GoalEngine) -> None:
        intent = _intent(IntentType.ASK_QUESTION)
        hierarchy = await engine.resolve(intent)
        assert hierarchy.root.description == "Answer the user's question"
        assert hierarchy.root.intent_type == IntentType.ASK_QUESTION
        assert hierarchy.root.goal_type == GoalType.QUESTION
        assert hierarchy.root.status == GoalStatus.PROPOSED
        assert len(hierarchy.children) == 0

    @pytest.mark.asyncio
    async def test_resolve_question_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        assert hierarchy.root.goal_type == GoalType.QUESTION

    @pytest.mark.asyncio
    async def test_resolve_task_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.EXECUTE_TASK))
        assert hierarchy.root.goal_type == GoalType.TASK

    @pytest.mark.asyncio
    async def test_resolve_project_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.PLAN_PROJECT))
        assert hierarchy.root.goal_type == GoalType.PROJECT

    @pytest.mark.asyncio
    async def test_resolve_learning_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.LEARN))
        assert hierarchy.root.goal_type == GoalType.LEARNING

    @pytest.mark.asyncio
    async def test_resolve_exploration_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.EXPLORE))
        assert hierarchy.root.goal_type == GoalType.EXPLORATION

    @pytest.mark.asyncio
    async def test_resolve_debugging_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.DEBUG))
        assert hierarchy.root.goal_type == GoalType.DEBUGGING

    @pytest.mark.asyncio
    async def test_resolve_system_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.META))
        assert hierarchy.root.goal_type == GoalType.SYSTEM

    @pytest.mark.asyncio
    async def test_resolve_unknown_goal_type(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.UNKNOWN))
        assert hierarchy.root.goal_type == GoalType.TASK

    @pytest.mark.asyncio
    async def test_resolve_priority_high(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.SOLVE_PROBLEM))
        assert hierarchy.root.priority == GoalPriority.HIGH

    @pytest.mark.asyncio
    async def test_resolve_priority_normal(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        assert hierarchy.root.priority == GoalPriority.NORMAL

    @pytest.mark.asyncio
    async def test_resolve_priority_low(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.LEARN))
        assert hierarchy.root.priority == GoalPriority.LOW

    @pytest.mark.asyncio
    async def test_resolve_status_is_proposed(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.EXECUTE_TASK))
        assert hierarchy.root.status == GoalStatus.PROPOSED

    @pytest.mark.asyncio
    async def test_resolve_has_timestamps(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        assert isinstance(hierarchy.root.created_at, datetime)
        assert isinstance(hierarchy.root.updated_at, datetime)

    @pytest.mark.asyncio
    async def test_resolve_goal_id_is_unique(self, engine: GoalEngine) -> None:
        h1 = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        h2 = await engine.resolve(_intent(IntentType.EXECUTE_TASK))
        assert h1.root.id != h2.root.id

    @pytest.mark.asyncio
    async def test_resolve_intent_type_stored(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.CREATE_CONTENT))
        assert hierarchy.root.intent_type == IntentType.CREATE_CONTENT

    @pytest.mark.asyncio
    async def test_resolve_goal_type_intent_mapping(self, engine: GoalEngine) -> None:
        cases = [
            (IntentType.ASK_QUESTION, GoalType.QUESTION),
            (IntentType.SOLVE_PROBLEM, GoalType.TASK),
            (IntentType.EXECUTE_TASK, GoalType.TASK),
            (IntentType.PLAN_PROJECT, GoalType.PROJECT),
            (IntentType.LEARN, GoalType.LEARNING),
            (IntentType.CREATE_CONTENT, GoalType.TASK),
            (IntentType.EXPLORE, GoalType.EXPLORATION),
            (IntentType.DEBUG, GoalType.DEBUGGING),
            (IntentType.META, GoalType.SYSTEM),
            (IntentType.CLARIFY, GoalType.SYSTEM),
            (IntentType.UNKNOWN, GoalType.TASK),
        ]
        for intent_type, expected_goal_type in cases:
            hierarchy = await engine.resolve(_intent(intent_type))
            assert hierarchy.root.goal_type == expected_goal_type, (
                f"Expected {intent_type} → {expected_goal_type}, got {hierarchy.root.goal_type}"
            )

    @pytest.mark.asyncio
    async def test_resolve_priority_mapping(self, engine: GoalEngine) -> None:
        cases = [
            (IntentType.ASK_QUESTION, GoalPriority.NORMAL),
            (IntentType.SOLVE_PROBLEM, GoalPriority.HIGH),
            (IntentType.EXECUTE_TASK, GoalPriority.HIGH),
            (IntentType.PLAN_PROJECT, GoalPriority.NORMAL),
            (IntentType.LEARN, GoalPriority.LOW),
            (IntentType.CREATE_CONTENT, GoalPriority.NORMAL),
            (IntentType.EXPLORE, GoalPriority.LOW),
            (IntentType.DEBUG, GoalPriority.HIGH),
            (IntentType.META, GoalPriority.NORMAL),
            (IntentType.CLARIFY, GoalPriority.NORMAL),
            (IntentType.UNKNOWN, GoalPriority.NORMAL),
        ]
        for intent_type, expected_priority in cases:
            hierarchy = await engine.resolve(_intent(intent_type))
            assert hierarchy.root.priority == expected_priority, (
                f"Expected {intent_type} → {expected_priority}, got {hierarchy.root.priority}"
            )

    @pytest.mark.asyncio
    async def test_resolve_multiple_goals(self, engine: GoalEngine) -> None:
        sub = _intent(IntentType.EXECUTE_TASK)
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub,))
        hierarchy = await engine.resolve(intent)
        assert len(hierarchy.children) == 1
        assert hierarchy.children[0].intent_type == IntentType.EXECUTE_TASK

    @pytest.mark.asyncio
    async def test_resolve_multiple_sub_intents(self, engine: GoalEngine) -> None:
        sub1 = _intent(IntentType.EXECUTE_TASK)
        sub2 = _intent(IntentType.CREATE_CONTENT)
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub1, sub2))
        hierarchy = await engine.resolve(intent)
        assert len(hierarchy.children) == 2
        assert hierarchy.children[0].intent_type == IntentType.EXECUTE_TASK
        assert hierarchy.children[1].intent_type == IntentType.CREATE_CONTENT

    @pytest.mark.asyncio
    async def test_resolve_goal_hierarchy_parent_child(
        self, engine: GoalEngine
    ) -> None:
        sub = _intent(IntentType.EXECUTE_TASK)
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub,))
        hierarchy = await engine.resolve(intent)
        assert hierarchy.children[0].parent_id == hierarchy.root.id
        assert hierarchy.root.id in hierarchy.children[0].parent_id

    @pytest.mark.asyncio
    async def test_resolve_sub_goal_ids_referenced(self, engine: GoalEngine) -> None:
        sub = _intent(IntentType.EXECUTE_TASK)
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub,))
        hierarchy = await engine.resolve(intent)
        assert hierarchy.children[0].id in hierarchy.root.sub_goal_ids

    @pytest.mark.asyncio
    async def test_resolve_all_goals_flat_map(self, engine: GoalEngine) -> None:
        sub = _intent(IntentType.EXECUTE_TASK)
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub,))
        hierarchy = await engine.resolve(intent)
        assert hierarchy.root.id in hierarchy.all_goals
        assert hierarchy.children[0].id in hierarchy.all_goals
        assert len(hierarchy.all_goals) == 2

    @pytest.mark.asyncio
    async def test_resolve_no_sub_intents(self, engine: GoalEngine) -> None:
        intent = _intent(IntentType.ASK_QUESTION)
        hierarchy = await engine.resolve(intent)
        assert len(hierarchy.children) == 0
        assert len(hierarchy.root.sub_goal_ids) == 0
        assert hierarchy.root.id in hierarchy.all_goals
        assert len(hierarchy.all_goals) == 1

    @pytest.mark.asyncio
    async def test_update_progress(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        await engine.update_progress(hierarchy.root.id, 0.5)
        updated = engine._active_goals[hierarchy.root.id]
        assert updated.progress == 0.5

    @pytest.mark.asyncio
    async def test_update_progress_invalid_id(self, engine: GoalEngine) -> None:
        with pytest.raises(ValueError, match="Unknown goal"):
            await engine.update_progress("nonexistent", 0.5)

    @pytest.mark.asyncio
    async def test_update_progress_updates_timestamp(self, engine: GoalEngine) -> None:
        hierarchy = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        original = hierarchy.root.updated_at
        await engine.update_progress(hierarchy.root.id, 1.0)
        updated = engine._active_goals[hierarchy.root.id]
        assert updated.updated_at >= original

    @pytest.mark.asyncio
    async def test_resolve_empty_sub_intents(self, engine: GoalEngine) -> None:
        intent = _intent(IntentType.SOLVE_PROBLEM, sub_intents=())
        hierarchy = await engine.resolve(intent)
        assert len(hierarchy.children) == 0


# ── Goal model — frozen validation ────────────────────────────────────────


class TestGoalFrozen:
    """Tests for Goal immutability."""

    def test_goal_is_frozen(self) -> None:
        goal = Goal(description="Test")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            goal.description = "Changed"  # type: ignore[misc]

    def test_goal_hierarchy_is_frozen(self) -> None:
        goal = Goal(description="Root")
        hierarchy = GoalHierarchy(root=goal)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            hierarchy.root = goal  # type: ignore[misc]

    def test_goal_priority_enum_values(self) -> None:
        assert GoalPriority.CRITICAL.value == 100
        assert GoalPriority.HIGH.value == 75
        assert GoalPriority.NORMAL.value == 50
        assert GoalPriority.LOW.value == 25
        assert GoalPriority.BACKGROUND.value == 0

    def test_goal_status_enum_values(self) -> None:
        assert GoalStatus.PROPOSED.value == "proposed"
        assert GoalStatus.ACTIVE.value == "active"
        assert GoalStatus.BLOCKED.value == "blocked"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.FAILED.value == "failed"
        assert GoalStatus.ABANDONED.value == "abandoned"
        assert GoalStatus.SUPERSEDED.value == "superseded"
        assert GoalStatus.PAUSED.value == "paused"

    def test_goal_type_enum_values(self) -> None:
        assert GoalType.QUESTION.value == "question"
        assert GoalType.TASK.value == "task"
        assert GoalType.PROJECT.value == "project"
        assert GoalType.LEARNING.value == "learning"
        assert GoalType.EXPLORATION.value == "exploration"
        assert GoalType.DEBUGGING.value == "debugging"
        assert GoalType.SYSTEM.value == "system"


# ── Serialization ─────────────────────────────────────────────────────────


class TestGoalSerialization:
    """Tests for goal serialization round-trip."""

    @pytest.mark.asyncio
    async def test_goal_serialization_roundtrip(self) -> None:
        engine = GoalEngine()
        hierarchy = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        data = hierarchy.root.model_dump()
        restored = Goal.model_validate(data)
        assert restored.description == hierarchy.root.description
        assert restored.goal_type == hierarchy.root.goal_type
        assert restored.priority == hierarchy.root.priority
        assert restored.status == hierarchy.root.status

    @pytest.mark.asyncio
    async def test_hierarchy_serialization_roundtrip(self) -> None:
        engine = GoalEngine()
        sub = _intent(IntentType.EXECUTE_TASK)
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub,))
        hierarchy = await engine.resolve(intent)
        data = hierarchy.model_dump()
        restored = GoalHierarchy.model_validate(data)
        assert restored.root.description == hierarchy.root.description
        assert len(restored.children) == 1
        assert restored.children[0].intent_type == IntentType.EXECUTE_TASK


# ── Equality ──────────────────────────────────────────────────────────────


class TestGoalEquality:
    """Tests for goal equality semantics."""

    def test_goal_equality_same_fields(self) -> None:
        g1 = Goal(description="Test", id="g1")
        g2 = Goal(description="Test", id="g1")
        assert g1 == g2

    def test_goal_inequality_different_id(self) -> None:
        g1 = Goal(description="Test", id="g1")
        g2 = Goal(description="Test", id="g2")
        assert g1 != g2

    def test_goal_inequality_different_description(self) -> None:
        g1 = Goal(description="One", id="g1")
        g2 = Goal(description="Two", id="g1")
        assert g1 != g2


# ── Invalid / edge case input ─────────────────────────────────────────────


class TestGoalEdgeCases:
    """Tests for edge case inputs to GoalEngine."""

    @pytest.mark.asyncio
    async def test_resolve_unknown_intent(self) -> None:
        engine = GoalEngine()
        intent = _intent(IntentType.UNKNOWN)
        hierarchy = await engine.resolve(intent)
        assert hierarchy.root.goal_type == GoalType.TASK
        assert hierarchy.root.priority == GoalPriority.NORMAL

    @pytest.mark.asyncio
    async def test_resolve_deeply_nested_sub_intents(self) -> None:
        engine = GoalEngine()
        deep_sub = _intent(IntentType.DEBUG)
        sub = _intent(IntentType.SOLVE_PROBLEM, sub_intents=(deep_sub,))
        intent = _intent(IntentType.PLAN_PROJECT, sub_intents=(sub,))
        hierarchy = await engine.resolve(intent)
        # Only direct sub-intents become children
        assert len(hierarchy.children) == 1
        assert hierarchy.children[0].intent_type == IntentType.SOLVE_PROBLEM

    @pytest.mark.asyncio
    async def test_resolve_clarify_intent(self) -> None:
        engine = GoalEngine()
        hierarchy = await engine.resolve(_intent(IntentType.CLARIFY))
        assert hierarchy.root.goal_type == GoalType.SYSTEM

    @pytest.mark.asyncio
    async def test_resolve_meta_intent(self) -> None:
        engine = GoalEngine()
        hierarchy = await engine.resolve(_intent(IntentType.META))
        assert hierarchy.root.goal_type == GoalType.SYSTEM

    @pytest.mark.asyncio
    async def test_resolve_multiple_calls_independent(self) -> None:
        engine = GoalEngine()
        h1 = await engine.resolve(_intent(IntentType.ASK_QUESTION))
        h2 = await engine.resolve(_intent(IntentType.EXECUTE_TASK))
        assert h1.root.id != h2.root.id
        assert h1.root.description == "Answer the user's question"
        assert h2.root.description == "Execute the requested task"
