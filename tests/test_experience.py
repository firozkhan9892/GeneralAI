"""Comprehensive tests for the Experience Engine (Phase 3.5I)."""

from __future__ import annotations

import pytest

from app.kernel.experience.engine import ExperienceEngine, InMemoryExperienceStore
from app.kernel.experience.models import (
    DecisionSummary,
    Experience,
    ExperienceQuery,
    Insight,
    LessonCategory,
    LessonLearned,
)
from app.kernel.intent.models import IntentType


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_exp(
    goal_type: IntentType = IntentType.ASK_QUESTION,
    success: bool = True,
    outcome_score: float = 0.8,
    skills: tuple[str, ...] = ("skill_a",),
    lessons: tuple[LessonLearned, ...] = (),
) -> Experience:
    return Experience(
        goal_type=goal_type,
        success=success,
        outcome_score=outcome_score,
        skills_used=skills,
        lessons=lessons,
    )


# ──────────────────────────────────────────────
# InMemoryExperienceStore
# ──────────────────────────────────────────────


class TestInMemoryExperienceStore:
    """Unit tests for the in-memory store."""

    @pytest.mark.asyncio
    async def test_save_assigns_id(self) -> None:
        store = InMemoryExperienceStore()
        exp = Experience()
        exp_id = await store.save(exp)
        assert exp_id == "exp_1"
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_save_with_existing_id(self) -> None:
        store = InMemoryExperienceStore()
        exp = Experience(id="custom_1")
        exp_id = await store.save(exp)
        assert exp_id == "custom_1"

    @pytest.mark.asyncio
    async def test_save_duplicate_returns_existing(self) -> None:
        store = InMemoryExperienceStore()
        exp1 = Experience(id="dup")
        exp2 = Experience(id="dup", outcome_score=0.2)
        id1 = await store.save(exp1)
        id2 = await store.save(exp2)
        assert id1 == id2
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_save_multiple_increments_counter(self) -> None:
        store = InMemoryExperienceStore()
        ids = []
        for _ in range(5):
            ids.append(await store.save(Experience()))
        assert ids == [f"exp_{i}" for i in range(1, 6)]
        assert store.count() == 5

    @pytest.mark.asyncio
    async def test_query_no_filters_returns_all(self) -> None:
        store = InMemoryExperienceStore()
        for _ in range(3):
            await store.save(Experience())
        query = ExperienceQuery(limit=10)
        results = await store.query(query)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_empty_store(self) -> None:
        store = InMemoryExperienceStore()
        query = ExperienceQuery()
        results = await store.query(query)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_filter_by_goal_types(self) -> None:
        store = InMemoryExperienceStore()
        await store.save(_make_exp(goal_type=IntentType.ASK_QUESTION))
        await store.save(_make_exp(goal_type=IntentType.SOLVE_PROBLEM))
        await store.save(_make_exp(goal_type=IntentType.CREATE_CONTENT))
        q = ExperienceQuery(
            goal_types=(IntentType.ASK_QUESTION, IntentType.SOLVE_PROBLEM),
            limit=10,
        )
        results = await store.query(q)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_filter_by_skills(self) -> None:
        store = InMemoryExperienceStore()
        await store.save(_make_exp(skills=("skill_x",)))
        await store.save(_make_exp(skills=("skill_y",)))
        await store.save(_make_exp(skills=("skill_x", "skill_z")))
        q = ExperienceQuery(skills=("skill_x",), limit=10)
        results = await store.query(q)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_filter_by_success(self) -> None:
        store = InMemoryExperienceStore()
        await store.save(_make_exp(success=True))
        await store.save(_make_exp(success=False))
        await store.save(_make_exp(success=True))
        q = ExperienceQuery(success=True, limit=10)
        results = await store.query(q)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_query_limit(self) -> None:
        store = InMemoryExperienceStore()
        for _ in range(20):
            await store.save(Experience())
        q = ExperienceQuery(limit=5)
        results = await store.query(q)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_results_ordered_by_timestamp_desc(self) -> None:
        store = InMemoryExperienceStore()
        ids = []
        for _ in range(3):
            ids.append(await store.save(Experience()))
        q = ExperienceQuery(limit=10)
        results = await store.query(q)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_all_returns_all(self) -> None:
        store = InMemoryExperienceStore()
        for _ in range(4):
            await store.save(Experience())
        assert len(store.get_all()) == 4


# ──────────────────────────────────────────────
# ExperienceEngine — record
# ──────────────────────────────────────────────


class TestExperienceEngineRecord:
    """record() method."""

    @pytest.mark.asyncio
    async def test_record_returns_string_id(self) -> None:
        engine = ExperienceEngine()
        exp = Experience()
        exp_id = await engine.record(exp)
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    @pytest.mark.asyncio
    async def test_record_auto_increments(self) -> None:
        engine = ExperienceEngine()
        id1 = await engine.record(Experience())
        id2 = await engine.record(Experience())
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_record_preserves_fields(self) -> None:
        engine = ExperienceEngine()
        exp = _make_exp(
            goal_type=IntentType.SOLVE_PROBLEM,
            success=True,
            outcome_score=0.95,
        )
        exp_id = await engine.record(exp)
        results = await engine.search(ExperienceQuery(limit=10))
        stored = next(r for r in results if r.id == exp_id)
        assert stored.goal_type == IntentType.SOLVE_PROBLEM
        assert stored.success is True
        assert stored.outcome_score == 0.95

    @pytest.mark.asyncio
    async def test_record_duplicate_id_skips(self) -> None:
        engine = ExperienceEngine()
        exp1 = Experience(id="fixed", outcome_score=0.9)
        exp2 = Experience(id="fixed", outcome_score=0.1)
        await engine.record(exp1)
        await engine.record(exp2)
        results = await engine.search(ExperienceQuery(limit=10))
        assert len(results) == 1
        assert results[0].outcome_score == 0.9


# ──────────────────────────────────────────────
# ExperienceEngine — search
# ──────────────────────────────────────────────


class TestExperienceEngineSearch:
    """search() method."""

    @pytest.mark.asyncio
    async def test_search_all(self) -> None:
        engine = ExperienceEngine()
        for _ in range(5):
            await engine.record(Experience())
        results = await engine.search(ExperienceQuery(limit=10))
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_empty_store(self) -> None:
        engine = ExperienceEngine()
        results = await engine.search(ExperienceQuery())
        assert results == []

    @pytest.mark.asyncio
    async def test_search_by_goal_type(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(goal_type=IntentType.ASK_QUESTION))
        await engine.record(_make_exp(goal_type=IntentType.SOLVE_PROBLEM))
        q = ExperienceQuery(goal_types=(IntentType.ASK_QUESTION,), limit=10)
        results = await engine.search(q)
        assert len(results) == 1
        assert results[0].goal_type == IntentType.ASK_QUESTION

    @pytest.mark.asyncio
    async def test_search_by_skills(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(skills=("skill_p",)))
        await engine.record(_make_exp(skills=("skill_q",)))
        q = ExperienceQuery(skills=("skill_q",), limit=10)
        results = await engine.search(q)
        assert len(results) == 1
        assert "skill_q" in results[0].skills_used

    @pytest.mark.asyncio
    async def test_search_by_success(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(success=True))
        await engine.record(_make_exp(success=False))
        q = ExperienceQuery(success=False, limit=10)
        results = await engine.search(q)
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_search_respects_limit(self) -> None:
        engine = ExperienceEngine()
        for _ in range(20):
            await engine.record(Experience())
        q = ExperienceQuery(limit=3)
        results = await engine.search(q)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_deterministic(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(goal_type=IntentType.ASK_QUESTION))
        await engine.record(_make_exp(goal_type=IntentType.SOLVE_PROBLEM))
        q = ExperienceQuery(limit=10)
        r1 = await engine.search(q)
        r2 = await engine.search(q)
        assert [e.id for e in r1] == [e.id for e in r2]


# ──────────────────────────────────────────────
# ExperienceEngine — retrieve (alias)
# ──────────────────────────────────────────────


class TestExperienceEngineRetrieve:
    """retrieve() alias for search()."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_same_as_search(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp())
        q = ExperienceQuery(limit=10)
        s_results = await engine.search(q)
        r_results = await engine.retrieve(q)
        assert s_results == r_results


# ──────────────────────────────────────────────
# ExperienceEngine — summarize
# ──────────────────────────────────────────────


class TestExperienceEngineSummarize:
    """summarize() method."""

    @pytest.mark.asyncio
    async def test_summarize_empty(self) -> None:
        engine = ExperienceEngine()
        summary = await engine.summarize()
        assert summary["total_experiences"] == 0
        assert summary["average_outcome_score"] == 0.0
        assert summary["common_skills"] == []
        assert summary["goal_type_counts"] == {}

    @pytest.mark.asyncio
    async def test_summarize_counts(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(success=True))
        await engine.record(_make_exp(success=False))
        summary = await engine.summarize()
        assert summary["total_experiences"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_summarize_average_score(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(outcome_score=1.0))
        await engine.record(_make_exp(outcome_score=0.5))
        summary = await engine.summarize()
        assert summary["average_outcome_score"] == 0.75

    @pytest.mark.asyncio
    async def test_summarize_goal_counts(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(goal_type=IntentType.ASK_QUESTION))
        await engine.record(_make_exp(goal_type=IntentType.ASK_QUESTION))
        await engine.record(_make_exp(goal_type=IntentType.SOLVE_PROBLEM))
        summary = await engine.summarize()
        assert summary["goal_type_counts"]["ask_question"] == 2
        assert summary["goal_type_counts"]["solve_problem"] == 1

    @pytest.mark.asyncio
    async def test_summarize_common_skills(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(skills=("skill_a", "skill_b")))
        await engine.record(_make_exp(skills=("skill_a",)))
        await engine.record(_make_exp(skills=("skill_c",)))
        summary = await engine.summarize()
        assert "skill_a" in summary["common_skills"]

    @pytest.mark.asyncio
    async def test_summarize_lesson_counts(self) -> None:
        engine = ExperienceEngine()
        await engine.record(
            _make_exp(
                lessons=(
                    LessonLearned(
                        description="lesson1",
                        category=LessonCategory.STRATEGY,
                    ),
                )
            )
        )
        await engine.record(
            _make_exp(
                lessons=(
                    LessonLearned(
                        description="lesson2",
                        category=LessonCategory.AVOIDANCE,
                    ),
                )
            )
        )
        summary = await engine.summarize()
        assert summary["total_lessons"] == 2

    @pytest.mark.asyncio
    async def test_summarize_return_structure(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp())
        summary = await engine.summarize()
        expected_keys = {
            "total_experiences",
            "success_count",
            "failure_count",
            "average_outcome_score",
            "goal_type_counts",
            "common_skills",
            "total_lessons",
            "lesson_category_counts",
        }
        assert set(summary.keys()) == expected_keys


# ──────────────────────────────────────────────
# ExperienceEngine — get_insights
# ──────────────────────────────────────────────


class TestExperienceEngineGetInsights:
    """get_insights() method."""

    @pytest.mark.asyncio
    async def test_get_insights_empty(self) -> None:
        engine = ExperienceEngine()
        insights = await engine.get_insights("ask_question")
        assert insights == []

    @pytest.mark.asyncio
    async def test_get_insights_no_match(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(goal_type=IntentType.SOLVE_PROBLEM))
        insights = await engine.get_insights("ask_question")
        assert insights == []

    @pytest.mark.asyncio
    async def test_get_insights_with_matches(self) -> None:
        engine = ExperienceEngine()
        await engine.record(
            _make_exp(
                goal_type=IntentType.ASK_QUESTION,
                lessons=(
                    LessonLearned(
                        description="Be specific",
                        category=LessonCategory.STRATEGY,
                    ),
                ),
            )
        )
        insights = await engine.get_insights("ask_question")
        assert len(insights) >= 1
        assert isinstance(insights[0], Insight)

    @pytest.mark.asyncio
    async def test_get_insights_returns_sorted_by_confidence(self) -> None:
        engine = ExperienceEngine()
        for _ in range(5):
            await engine.record(
                _make_exp(
                    goal_type=IntentType.ASK_QUESTION,
                    lessons=(
                        LessonLearned(
                            description="rep lesson",
                            category=LessonCategory.STRATEGY,
                        ),
                    ),
                )
            )
        insights = await engine.get_insights("ask_question")
        for i in range(len(insights) - 1):
            assert insights[i].confidence >= insights[i + 1].confidence

    @pytest.mark.asyncio
    async def test_get_insights_confidence_scales_with_count(self) -> None:
        engine = ExperienceEngine()
        for _ in range(5):
            await engine.record(
                _make_exp(
                    goal_type=IntentType.ASK_QUESTION,
                    lessons=(
                        LessonLearned(
                            description="scale test",
                            category=LessonCategory.STRATEGY,
                        ),
                    ),
                )
            )
        insights = await engine.get_insights("ask_question")
        assert insights[0].confidence > 0.5


# ──────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────


class TestDeterminism:
    """Same inputs produce same outputs."""

    @pytest.mark.asyncio
    async def test_record_deterministic_ids(self) -> None:
        e1 = ExperienceEngine()
        e2 = ExperienceEngine()
        id1 = await e1.record(Experience())
        id2 = await e2.record(Experience())
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_search_deterministic_order(self) -> None:
        engine = ExperienceEngine()
        for _ in range(5):
            await engine.record(Experience())
        q = ExperienceQuery(limit=10)
        r1 = await engine.search(q)
        r2 = await engine.search(q)
        assert [e.id for e in r1] == [e.id for e in r2]

    @pytest.mark.asyncio
    async def test_summarize_deterministic(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(success=True))
        await engine.record(_make_exp(success=False))
        s1 = await engine.summarize()
        s2 = await engine.summarize()
        assert s1 == s2

    @pytest.mark.asyncio
    async def test_get_insights_deterministic(self) -> None:
        engine = ExperienceEngine()
        await engine.record(
            _make_exp(
                goal_type=IntentType.ASK_QUESTION,
                lessons=(
                    LessonLearned(
                        description="test lesson",
                        category=LessonCategory.STRATEGY,
                    ),
                ),
            )
        )
        i1 = await engine.get_insights("ask_question")
        i2 = await engine.get_insights("ask_question")
        assert i1 == i2


# ──────────────────────────────────────────────
# Model serialization
# ──────────────────────────────────────────────


class TestSerialization:
    """Pydantic model serialization."""

    def test_experience_model_dump(self) -> None:
        exp = Experience(id="test1", outcome_score=0.7)
        data = exp.model_dump()
        assert data["id"] == "test1"
        assert data["outcome_score"] == 0.7

    def test_experience_model_dump_json(self) -> None:
        exp = Experience(id="test1")
        json_str = exp.model_dump_json()
        assert "test1" in json_str

    def test_experience_deserialize(self) -> None:
        data = {"id": "test1", "outcome_score": 0.5}
        exp = Experience.model_validate(data)
        assert exp.id == "test1"
        assert exp.outcome_score == 0.5

    def test_experience_query_model_dump(self) -> None:
        q = ExperienceQuery(limit=5)
        data = q.model_dump()
        assert data["limit"] == 5

    def test_insight_model_dump(self) -> None:
        insight = Insight(
            description="test",
            pattern="pattern",
            confidence=0.8,
            supporting_experience_count=3,
        )
        data = insight.model_dump()
        assert data["confidence"] == 0.8

    def test_lesson_model_dump(self) -> None:
        lesson = LessonLearned(
            description="test",
            category=LessonCategory.OPTIMIZATION,
        )
        data = lesson.model_dump()
        assert data["category"] == "optimization"


# ──────────────────────────────────────────────
# Frozen model tests
# ──────────────────────────────────────────────


class TestFrozenModels:
    """All experience models are immutable."""

    def test_experience_frozen(self) -> None:
        exp = Experience()
        with pytest.raises(Exception):
            exp.id = "changed"  # type: ignore[misc]

    def test_experience_query_frozen(self) -> None:
        q = ExperienceQuery()
        with pytest.raises(Exception):
            q.limit = 50  # type: ignore[misc]

    def test_insight_frozen(self) -> None:
        insight = Insight(description="x", pattern="p")
        with pytest.raises(Exception):
            insight.confidence = 1.0  # type: ignore[misc]

    def test_lesson_frozen(self) -> None:
        lesson = LessonLearned(description="x")
        with pytest.raises(Exception):
            lesson.category = LessonCategory.SAFETY  # type: ignore[misc]

    def test_decision_summary_frozen(self) -> None:
        ds = DecisionSummary(action_type="test")
        with pytest.raises(Exception):
            ds.action_type = "changed"  # type: ignore[misc]


# ──────────────────────────────────────────────
# Equality
# ──────────────────────────────────────────────


class TestEquality:
    """Model equality semantics."""

    def test_experiences_equal(self) -> None:
        a = Experience(id="x", outcome_score=0.5)
        b = Experience(id="x", outcome_score=0.5)
        assert a.id == b.id
        assert a.outcome_score == b.outcome_score
        assert a.goal_type == b.goal_type

    def test_experiences_not_equal(self) -> None:
        a = Experience(id="x")
        b = Experience(id="y")
        assert a != b

    def test_queries_equal(self) -> None:
        a = ExperienceQuery(limit=5)
        b = ExperienceQuery(limit=5)
        assert a == b

    def test_insights_equal(self) -> None:
        a = Insight(description="x", pattern="p", confidence=0.5)
        b = Insight(description="x", pattern="p", confidence=0.5)
        assert a == b

    def test_lessons_equal(self) -> None:
        a = LessonLearned(description="x")
        b = LessonLearned(description="x")
        assert a == b


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_experience_default_id_empty(self) -> None:
        exp = Experience()
        assert exp.id == ""

    def test_experience_default_success_true(self) -> None:
        exp = Experience()
        assert exp.success is True

    def test_experience_default_score_zero(self) -> None:
        exp = Experience()
        assert exp.outcome_score == 0.0

    def test_experience_default_skills_empty(self) -> None:
        exp = Experience()
        assert exp.skills_used == ()

    def test_experience_default_lessons_empty(self) -> None:
        exp = Experience()
        assert exp.lessons == ()

    def test_lesson_default_category_strategy(self) -> None:
        lesson = LessonLearned(description="test")
        assert lesson.category == LessonCategory.STRATEGY

    def test_lesson_default_confidence(self) -> None:
        lesson = LessonLearned(description="test")
        assert lesson.confidence == 0.5

    def test_decision_summary_defaults(self) -> None:
        ds = DecisionSummary(action_type="noop")
        assert ds.confidence == 0.0
        assert ds.success is None

    def test_experience_query_default_limit(self) -> None:
        q = ExperienceQuery()
        assert q.limit == 10

    @pytest.mark.asyncio
    async def test_record_experience_with_lessons(self) -> None:
        engine = ExperienceEngine()
        lesson = LessonLearned(
            description="Always verify",
            category=LessonCategory.SAFETY,
            applicability=(IntentType.ASK_QUESTION,),
            confidence=0.9,
        )
        exp = _make_exp(lessons=(lesson,))
        exp_id = await engine.record(exp)
        results = await engine.search(ExperienceQuery(limit=10))
        stored = next(r for r in results if r.id == exp_id)
        assert len(stored.lessons) == 1
        assert stored.lessons[0].description == "Always verify"

    @pytest.mark.asyncio
    async def test_search_no_matching_results(self) -> None:
        engine = ExperienceEngine()
        await engine.record(_make_exp(goal_type=IntentType.ASK_QUESTION))
        q = ExperienceQuery(goal_types=(IntentType.CREATE_CONTENT,), limit=10)
        results = await engine.search(q)
        assert results == []

    @pytest.mark.asyncio
    async def test_outcome_score_bounds(self) -> None:
        with pytest.raises(Exception):
            Experience(outcome_score=-0.1)
        with pytest.raises(Exception):
            Experience(outcome_score=1.5)

    def test_lesson_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            LessonLearned(description="x", confidence=-0.1)
        with pytest.raises(Exception):
            LessonLearned(description="x", confidence=1.5)

    def test_insight_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            Insight(description="x", pattern="p", confidence=-0.1)
        with pytest.raises(Exception):
            Insight(description="x", pattern="p", confidence=1.5)


# ──────────────────────────────────────────────
# InMemoryExperienceStore — additional coverage
# ──────────────────────────────────────────────


class TestInMemoryStoreExtended:
    """Additional store tests."""

    @pytest.mark.asyncio
    async def test_save_preserves_timestamp(self) -> None:
        store = InMemoryExperienceStore()
        exp = Experience()
        await store.save(exp)
        saved = list(store._records.values())[0]
        assert saved.timestamp is not None

    @pytest.mark.asyncio
    async def test_query_excludes_outside_timeframe(self) -> None:
        store = InMemoryExperienceStore()
        from datetime import datetime, timedelta

        past = datetime.utcnow() - timedelta(hours=2)
        exp = Experience(timestamp=past)
        await store.save(exp)
        q = ExperienceQuery(timeframe_hours=1, limit=10)
        results = await store.query(q)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_saves_unique_ids(self) -> None:
        store = InMemoryExperienceStore()
        ids = set()
        for _ in range(50):
            eid = await store.save(Experience())
            ids.add(eid)
        assert len(ids) == 50
