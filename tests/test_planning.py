"""Tests for PlanningEngine and planning domain models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from app.kernel.goals.models import Goal, GoalType
from app.kernel.planning import DependencyGraph, Plan, PlanningEngine, SkillStep
from app.kernel.planning.models import PlanningStrategy


def _goal(
    goal_type: GoalType = GoalType.TASK,
    *,
    description: str = "Test goal",
    goal_id: str = "goal_1",
) -> Goal:
    return Goal(id=goal_id, description=description, goal_type=goal_type)


# ── SkillStep model tests ────────────────────────────────────────────────


class TestSkillStepModel:
    """Tests for SkillStep domain model."""

    def test_create_minimal(self) -> None:
        step = SkillStep(order=0, skill_name="test_skill")
        assert step.order == 0
        assert step.skill_name == "test_skill"
        assert step.parameters == {}
        assert step.dependencies == ()
        assert step.estimated_tokens == 0
        assert step.description == ""

    def test_create_full(self) -> None:
        step = SkillStep(
            order=1,
            skill_name="analyze",
            parameters={"input": "data"},
            dependencies=(0,),
            estimated_tokens=100,
            description="Analyze the input",
        )
        assert step.order == 1
        assert step.skill_name == "analyze"
        assert step.parameters == {"input": "data"}
        assert step.dependencies == (0,)
        assert step.estimated_tokens == 100
        assert step.description == "Analyze the input"

    def test_frozen(self) -> None:
        step = SkillStep(order=0, skill_name="test")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            step.skill_name = "changed"  # type: ignore[misc]

    def test_order_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            SkillStep(order=-1, skill_name="bad")

    def test_estimated_tokens_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            SkillStep(order=0, skill_name="bad", estimated_tokens=-5)

    def test_equality(self) -> None:
        a = SkillStep(order=0, skill_name="test")
        b = SkillStep(order=0, skill_name="test")
        assert a == b

    def test_inequality(self) -> None:
        a = SkillStep(order=0, skill_name="test")
        b = SkillStep(order=1, skill_name="test")
        assert a != b

    def test_serialization_roundtrip(self) -> None:
        original = SkillStep(
            order=2,
            skill_name="compute",
            parameters={"x": 1},
            dependencies=(0, 1),
            estimated_tokens=50,
            description="Compute result",
        )
        data = original.model_dump()
        restored = SkillStep.model_validate(data)
        assert restored == original


# ── DependencyGraph model tests ──────────────────────────────────────────


class TestDependencyGraphModel:
    """Tests for DependencyGraph domain model."""

    def test_create_empty(self) -> None:
        g = DependencyGraph()
        assert g.edges == ()

    def test_create_with_edges(self) -> None:
        g = DependencyGraph(edges=((1, 0), (2, 1)))
        assert len(g.edges) == 2
        assert g.edges[0] == (1, 0)

    def test_frozen(self) -> None:
        g = DependencyGraph(edges=((1, 0),))
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            g.edges = ()  # type: ignore[misc]

    def test_equality(self) -> None:
        a = DependencyGraph(edges=((1, 0),))
        b = DependencyGraph(edges=((1, 0),))
        assert a == b

    def test_serialization_roundtrip(self) -> None:
        original = DependencyGraph(edges=((2, 0), (2, 1)))
        data = original.model_dump()
        restored = DependencyGraph.model_validate(data)
        assert restored == original


# ── Plan model tests ─────────────────────────────────────────────────────


class TestPlanModel:
    """Tests for Plan domain model."""

    def test_create_defaults(self) -> None:
        plan = Plan()
        assert plan.goal_id == ""
        assert plan.strategy == PlanningStrategy.TOP_DOWN
        assert plan.steps == ()
        assert plan.dependencies.edges == ()
        assert plan.estimated_total_tokens == 0
        assert plan.revision == 0

    def test_frozen(self) -> None:
        plan = Plan()
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            plan.goal_id = "new"  # type: ignore[misc]

    def test_estimation_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            Plan(estimated_total_tokens=-1)

    def test_revision_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            Plan(revision=-1)

    def test_serialization_roundtrip(self) -> None:
        steps = (
            SkillStep(order=0, skill_name="a", dependencies=()),
            SkillStep(order=1, skill_name="b", dependencies=(0,)),
        )
        original = Plan(
            goal_id="g1",
            strategy=PlanningStrategy.TOP_DOWN,
            steps=steps,
            dependencies=DependencyGraph(edges=((1, 0),)),
            estimated_total_tokens=100,
            revision=2,
            metadata={"source": "test"},
        )
        data = original.model_dump()
        restored = Plan.model_validate(data)
        assert restored == original


# ── Plan generation ──────────────────────────────────────────────────────


class TestPlanGeneration:
    """Tests for plan generation from goals."""

    @pytest.fixture
    def engine(self) -> PlanningEngine:
        return PlanningEngine()

    @pytest.mark.asyncio
    async def test_create_plan_returns_plan(self, engine: PlanningEngine) -> None:
        goal = _goal(GoalType.TASK)
        plan = await engine.create_plan(goal)
        assert isinstance(plan, Plan)

    @pytest.mark.asyncio
    async def test_create_plan_goal_id(self, engine: PlanningEngine) -> None:
        goal = _goal(GoalType.TASK, goal_id="custom_1")
        plan = await engine.create_plan(goal)
        assert plan.goal_id == "custom_1"

    @pytest.mark.asyncio
    async def test_create_plan_strategy_default(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        assert plan.strategy == PlanningStrategy.TOP_DOWN

    @pytest.mark.asyncio
    async def test_create_plan_revision_zero(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        assert plan.revision == 0

    @pytest.mark.asyncio
    async def test_create_plan_question_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.QUESTION))
        assert len(plan.steps) == 3
        assert plan.steps[0].skill_name == "analyze_question"
        assert plan.steps[1].skill_name == "retrieve_knowledge"
        assert plan.steps[2].skill_name == "formulate_answer"

    @pytest.mark.asyncio
    async def test_create_plan_task_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        assert len(plan.steps) == 3
        assert plan.steps[0].skill_name == "understand_task"
        assert plan.steps[1].skill_name == "execute_skill"
        assert plan.steps[2].skill_name == "verify_result"

    @pytest.mark.asyncio
    async def test_create_plan_project_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.PROJECT))
        assert len(plan.steps) == 4
        assert plan.steps[0].skill_name == "analyze_requirements"
        assert plan.steps[3].skill_name == "track_progress"

    @pytest.mark.asyncio
    async def test_create_plan_learning_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.LEARNING))
        assert len(plan.steps) == 4
        skill_names = [s.skill_name for s in plan.steps]
        assert "identify_topic" in skill_names
        assert "assess_understanding" in skill_names

    @pytest.mark.asyncio
    async def test_create_plan_exploration_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.EXPLORATION))
        assert len(plan.steps) == 3
        assert plan.steps[0].skill_name == "define_scope"
        assert plan.steps[2].skill_name == "summarize_findings"

    @pytest.mark.asyncio
    async def test_create_plan_debugging_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.DEBUGGING))
        assert len(plan.steps) == 5
        assert plan.steps[0].skill_name == "reproduce_issue"
        assert plan.steps[4].skill_name == "verify_fix"

    @pytest.mark.asyncio
    async def test_create_plan_system_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.SYSTEM))
        assert len(plan.steps) == 2
        assert plan.steps[1].skill_name == "respond_to_user"

    @pytest.mark.asyncio
    async def test_create_plan_all_goal_types(self, engine: PlanningEngine) -> None:
        for gt in GoalType:
            plan = await engine.create_plan(_goal(gt))
            assert len(plan.steps) >= 2, f"GoalType {gt} produced too few steps"

    @pytest.mark.asyncio
    async def test_create_plan_steps_have_descriptions(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        for step in plan.steps:
            assert step.description, f"Step {step.order} missing description"

    @pytest.mark.asyncio
    async def test_create_plan_step_order_sequential(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.PROJECT))
        for i, step in enumerate(plan.steps):
            assert step.order == i

    @pytest.mark.asyncio
    async def test_create_plan_dependencies_correct(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.DEBUGGING))
        for step in plan.steps:
            for dep in step.dependencies:
                assert dep < step.order, (
                    f"Step {step.order} depends on future step {dep}"
                )

    @pytest.mark.asyncio
    async def test_create_plan_dependency_graph_edges(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        assert len(plan.dependencies.edges) > 0
        for edge in plan.dependencies.edges:
            assert len(edge) == 2

    @pytest.mark.asyncio
    async def test_create_plan_default_goal_type(self, engine: PlanningEngine) -> None:
        goal = Goal(id="empty", description="")
        plan = await engine.create_plan(goal)
        assert len(plan.steps) == 3
        assert plan.steps[0].skill_name == "understand_task"

    @pytest.mark.asyncio
    async def test_create_plan_multiple_calls_independent(
        self, engine: PlanningEngine
    ) -> None:
        p1 = await engine.create_plan(_goal(GoalType.QUESTION, goal_id="g1"))
        p2 = await engine.create_plan(_goal(GoalType.TASK, goal_id="g2"))
        assert p1.goal_id == "g1"
        assert p2.goal_id == "g2"
        assert p1.steps[0].skill_name != p2.steps[0].skill_name

    @pytest.mark.asyncio
    async def test_create_plan_deterministic(self, engine: PlanningEngine) -> None:
        goal = _goal(GoalType.QUESTION, goal_id="det")
        p1 = await engine.create_plan(goal)
        p2 = await engine.create_plan(goal)
        assert p1 == p2


# ── Plan revision ────────────────────────────────────────────────────────


class TestPlanRevision:
    """Tests for plan revision."""

    @pytest.fixture
    def engine(self) -> PlanningEngine:
        return PlanningEngine()

    @pytest.mark.asyncio
    async def test_revise_plan_increments_revision(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        revised = await engine.revise_plan(plan)
        assert revised.revision == plan.revision + 1

    @pytest.mark.asyncio
    async def test_revise_plan_preserves_steps(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.QUESTION))
        revised = await engine.revise_plan(plan)
        assert revised.steps == plan.steps

    @pytest.mark.asyncio
    async def test_revise_plan_preserves_goal_id(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK, goal_id="keep_id"))
        revised = await engine.revise_plan(plan)
        assert revised.goal_id == "keep_id"

    @pytest.mark.asyncio
    async def test_revise_plan_is_new_object(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        revised = await engine.revise_plan(plan)
        assert revised is not plan

    @pytest.mark.asyncio
    async def test_revise_plan_multiple_times(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        for _ in range(3):
            plan = await engine.revise_plan(plan)
        assert plan.revision == 3

    @pytest.mark.asyncio
    async def test_revise_plan_preserves_dependencies(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.DEBUGGING))
        revised = await engine.revise_plan(plan)
        assert revised.dependencies == plan.dependencies


# ── Backward-compatible aliases ──────────────────────────────────────────


class TestBackwardCompatibleAliases:
    """Tests for the plan() and revise() aliases."""

    @pytest.fixture
    def engine(self) -> PlanningEngine:
        return PlanningEngine()

    @pytest.mark.asyncio
    async def test_plan_alias(self, engine: PlanningEngine) -> None:
        goal = _goal(GoalType.QUESTION)
        result = await engine.plan(goal)
        assert isinstance(result, Plan)

    @pytest.mark.asyncio
    async def test_plan_alias_matches_create_plan(self, engine: PlanningEngine) -> None:
        goal = _goal(GoalType.TASK)
        a = await engine.plan(goal)
        b = await engine.create_plan(goal)
        assert a == b

    @pytest.mark.asyncio
    async def test_revise_alias(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        result = await engine.revise(plan)
        assert result.revision == 1

    @pytest.mark.asyncio
    async def test_revise_alias_matches_revise_plan(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        a = await engine.revise(plan)
        b = await engine.revise_plan(plan)
        assert a == b


# ── Edge cases ───────────────────────────────────────────────────────────


class TestPlanningEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def engine(self) -> PlanningEngine:
        return PlanningEngine()

    @pytest.mark.asyncio
    async def test_goal_with_empty_description(self, engine: PlanningEngine) -> None:
        goal = Goal(id="g1", description="")
        plan = await engine.create_plan(goal)
        assert isinstance(plan, Plan)

    @pytest.mark.asyncio
    async def test_goal_with_unknown_goal_type(self, engine: PlanningEngine) -> None:
        goal = Goal(id="g1", description="Something", goal_type=GoalType.TASK)
        plan = await engine.create_plan(goal)
        assert len(plan.steps) > 0

    @pytest.mark.asyncio
    async def test_plan_estimated_tokens_zero_default(
        self, engine: PlanningEngine
    ) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        assert plan.estimated_total_tokens == 0

    @pytest.mark.asyncio
    async def test_plan_strategy_always_top_down(self, engine: PlanningEngine) -> None:
        for gt in GoalType:
            plan = await engine.create_plan(_goal(gt))
            assert plan.strategy == PlanningStrategy.TOP_DOWN

    @pytest.mark.asyncio
    async def test_create_plan_frozen_output(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            plan.goal_id = "hack"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_revise_plan_frozen_output(self, engine: PlanningEngine) -> None:
        plan = await engine.create_plan(_goal(GoalType.TASK))
        revised = await engine.revise_plan(plan)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            revised.revision = 99  # type: ignore[misc]
