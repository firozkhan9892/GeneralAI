"""Comprehensive tests for the Memory Engine (Phase 6)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from app.kernel.memory.engine import (
    InMemoryMemoryStore,
    MemoryEngine,
    MemoryStore,
)
from app.kernel.memory.models import (
    MemoryQuery,
    MemoryRecord,
    MemorySearchHit,
    MemoryTier,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_record(
    content: str = "remembered fact",
    tier: MemoryTier = MemoryTier.SHORT_TERM,
    session_id: str = "",
    tags: tuple[str, ...] = (),
    importance: float = 0.5,
    access_count: int = 0,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        tier=tier,
        session_id=session_id,
        tags=tags,
        importance=importance,
        access_count=access_count,
    )


# ──────────────────────────────────────────────
# MemoryRecord model
# ──────────────────────────────────────────────


class TestMemoryRecordModel:
    """Unit tests for the MemoryRecord model."""

    def test_defaults(self) -> None:
        record = MemoryRecord(content="fact")
        assert record.id == ""
        assert record.tier == MemoryTier.SHORT_TERM
        assert record.session_id == ""
        assert record.tags == ()
        assert record.importance == 0.5
        assert record.access_count == 0
        assert record.metadata == {}

    def test_is_frozen(self) -> None:
        record = _make_record()
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            record.content = "mutated"  # type: ignore[misc]

    def test_timestamp_is_set(self) -> None:
        record = _make_record()
        assert isinstance(record.timestamp, datetime)

    def test_importance_range_enforced(self) -> None:
        with pytest.raises(ValueError):
            _make_record(importance=1.5)
        with pytest.raises(ValueError):
            _make_record(importance=-0.1)

    def test_tags_preserve_order(self) -> None:
        record = _make_record(tags=("alpha", "beta"))
        assert record.tags == ("alpha", "beta")

    def test_metadata_dict(self) -> None:
        record = _make_record()
        record2 = record.model_copy(update={"metadata": {"source": "test"}})
        assert record2.metadata == {"source": "test"}


# ──────────────────────────────────────────────
# MemoryQuery model
# ──────────────────────────────────────────────


class TestMemoryQueryModel:
    """Unit tests for the MemoryQuery model."""

    def test_defaults(self) -> None:
        query = MemoryQuery()
        assert query.tier is None
        assert query.session_id is None
        assert query.tags is None
        assert query.keywords is None
        assert query.limit == 10

    def test_limit_bounds(self) -> None:
        with pytest.raises(ValueError):
            MemoryQuery(limit=0)
        with pytest.raises(ValueError):
            MemoryQuery(limit=101)

    def test_is_frozen(self) -> None:
        query = MemoryQuery(limit=5)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            query.limit = 20  # type: ignore[misc]


# ──────────────────────────────────────────────
# InMemoryMemoryStore
# ──────────────────────────────────────────────


class TestInMemoryMemoryStore:
    """Unit tests for the in-memory store."""

    @pytest.mark.asyncio
    async def test_save_assigns_id(self) -> None:
        store = InMemoryMemoryStore()
        record_id = await store.save(_make_record())
        assert record_id == "mem_1"
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_save_with_existing_id(self) -> None:
        store = InMemoryMemoryStore()
        record_id = await store.save(_make_record())
        assert record_id == "mem_1"

    @pytest.mark.asyncio
    async def test_save_multiple_increments_counter(self) -> None:
        store = InMemoryMemoryStore()
        ids = []
        for _ in range(5):
            ids.append(await store.save(_make_record()))
        assert ids == [f"mem_{i}" for i in range(1, 6)]

    @pytest.mark.asyncio
    async def test_save_with_custom_id_keeps_it(self) -> None:
        store = InMemoryMemoryStore()
        record = _make_record()
        custom = record.model_copy(update={"id": "custom_1"})
        record_id = await store.save(custom)
        assert record_id == "custom_1"

    @pytest.mark.asyncio
    async def test_get_returns_record(self) -> None:
        store = InMemoryMemoryStore()
        record_id = await store.save(_make_record(content="hello world"))
        record = await store.get(record_id)
        assert record is not None
        assert record.content == "hello world"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self) -> None:
        store = InMemoryMemoryStore()
        assert await store.get("nope") is None

    @pytest.mark.asyncio
    async def test_delete_removes_record(self) -> None:
        store = InMemoryMemoryStore()
        record_id = await store.save(_make_record())
        assert await store.delete(record_id) is True
        assert await store.get(record_id) is None
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self) -> None:
        store = InMemoryMemoryStore()
        assert await store.delete("nope") is False

    @pytest.mark.asyncio
    async def test_query_no_filters_returns_all(self) -> None:
        store = InMemoryMemoryStore()
        for _ in range(3):
            await store.save(_make_record())
        results = await store.query(MemoryQuery(limit=10))
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_filters_by_tier(self) -> None:
        store = InMemoryMemoryStore()
        await store.save(_make_record(tier=MemoryTier.SHORT_TERM))
        await store.save(_make_record(tier=MemoryTier.LONG_TERM))
        short = await store.query(MemoryQuery(limit=10, tier=MemoryTier.SHORT_TERM))
        long = await store.query(MemoryQuery(limit=10, tier=MemoryTier.LONG_TERM))
        assert len(short) == 1
        assert len(long) == 1
        assert short[0].tier == MemoryTier.SHORT_TERM
        assert long[0].tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_query_filters_by_session(self) -> None:
        store = InMemoryMemoryStore()
        await store.save(_make_record(session_id="alpha"))
        await store.save(_make_record(session_id="beta"))
        results = await store.query(MemoryQuery(limit=10, session_id="alpha"))
        assert len(results) == 1
        assert results[0].session_id == "alpha"

    @pytest.mark.asyncio
    async def test_query_filters_by_tags_all_required(self) -> None:
        store = InMemoryMemoryStore()
        await store.save(_make_record(tags=("a", "b")))
        await store.save(_make_record(tags=("a", "c")))
        results = await store.query(MemoryQuery(limit=10, tags=("a", "b")))
        assert len(results) == 1
        assert results[0].tags == ("a", "b")

    @pytest.mark.asyncio
    async def test_query_respects_limit(self) -> None:
        store = InMemoryMemoryStore()
        for _ in range(5):
            await store.save(_make_record())
        results = await store.query(MemoryQuery(limit=3))
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_returns_newest_first(self) -> None:
        store = InMemoryMemoryStore()
        older = _make_record(content="oldest")
        newer = _make_record(content="newest")
        await store.save(
            older.model_copy(
                update={"timestamp": datetime.utcnow() - timedelta(hours=2)}
            )
        )
        await store.save(newer.model_copy(update={"timestamp": datetime.utcnow()}))
        results = await store.query(MemoryQuery(limit=10))
        assert results[0].content == "newest"
        assert results[1].content == "oldest"

    @pytest.mark.asyncio
    async def test_all_returns_sorted(self) -> None:
        store = InMemoryMemoryStore()
        await store.save(
            _make_record(
                content="first",
            ).model_copy(update={"timestamp": datetime.utcnow() - timedelta(hours=1)})
        )
        await store.save(_make_record(content="second"))
        records = await store.all()
        assert [r.content for r in records] == ["second", "first"]

    @pytest.mark.asyncio
    async def test_clear_removes_everything(self) -> None:
        store = InMemoryMemoryStore()
        await store.save(_make_record())
        await store.save(_make_record())
        await store.clear()
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_raw_records_unsorted(self) -> None:
        store = InMemoryMemoryStore()
        first = await store.save(_make_record(content="first"))
        second = await store.save(_make_record(content="second"))
        raw = [r.id for r in store.raw_records()]
        assert raw == [first, second]


# ──────────────────────────────────────────────
# MemoryEngine — writing
# ──────────────────────────────────────────────


class TestMemoryEngineRemember:
    """Tests for the remember operation."""

    @pytest.mark.asyncio
    async def test_remember_short_term_by_default(self) -> None:
        engine = MemoryEngine()
        record_id = await engine.remember("hello world")
        assert record_id == "mem_1"
        record = await engine.get(record_id)
        assert record is not None
        assert record.content == "hello world"
        assert record.tier == MemoryTier.SHORT_TERM

    @pytest.mark.asyncio
    async def test_remember_long_term(self) -> None:
        engine = MemoryEngine()
        record_id = await engine.remember("important fact", tier=MemoryTier.LONG_TERM)
        record = await engine.get(record_id)
        assert record is not None
        assert record.tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_remember_captures_arguments(self) -> None:
        engine = MemoryEngine()
        record_id = await engine.remember(
            "the user prefers dark mode",
            session_id="session_7",
            tags=("preference", "ui"),
            importance=0.9,
            metadata={"key": "value"},
        )
        record = await engine.get(record_id)
        assert record is not None
        assert record.session_id == "session_7"
        assert record.tags == ("preference", "ui")
        assert record.importance == 0.9
        assert record.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_remember_uses_provided_store(self) -> None:
        store = InMemoryMemoryStore()
        engine = MemoryEngine(store=store)
        record_id = await engine.remember("fact")
        assert record_id == "mem_1"
        assert await store.count() == 1


class TestMemoryEngineForget:
    """Tests for the forget operation."""

    @pytest.mark.asyncio
    async def test_forget_removes_record(self) -> None:
        engine = MemoryEngine()
        record_id = await engine.remember("temporary")
        assert await engine.forget(record_id) is True
        assert await engine.get(record_id) is None
        assert await engine.count() == 0

    @pytest.mark.asyncio
    async def test_forget_missing_returns_false(self) -> None:
        engine = MemoryEngine()
        assert await engine.forget("nope") is False

    @pytest.mark.asyncio
    async def test_clear_removes_all(self) -> None:
        engine = MemoryEngine()
        await engine.remember("one")
        await engine.remember("two")
        await engine.clear()
        assert await engine.count() == 0


class TestMemoryEngineTouch:
    """Tests for the touch operation."""

    @pytest.mark.asyncio
    async def test_touch_increments_access_count(self) -> None:
        engine = MemoryEngine()
        record_id = await engine.remember("fact")
        await engine.touch(record_id)
        await engine.touch(record_id)
        record = await engine.get(record_id)
        assert record is not None
        assert record.access_count == 2

    @pytest.mark.asyncio
    async def test_touch_missing_is_noop(self) -> None:
        engine = MemoryEngine()
        await engine.touch("nope")
        assert await engine.count() == 0


# ──────────────────────────────────────────────
# MemoryEngine — reading
# ──────────────────────────────────────────────


class TestMemoryEngineRetrieve:
    """Tests for the retrieve operation."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_matching(self) -> None:
        engine = MemoryEngine()
        await engine.remember("fact one", session_id="a", tags=("x",))
        await engine.remember("fact two", session_id="b", tags=("y",))
        results = await engine.retrieve(MemoryQuery(limit=10, session_id="a"))
        assert len(results) == 1
        assert results[0].content == "fact one"

    @pytest.mark.asyncio
    async def test_retrieve_empty_when_no_match(self) -> None:
        engine = MemoryEngine()
        await engine.remember("fact")
        results = await engine.retrieve(MemoryQuery(limit=10, tags=("missing",)))
        assert results == []


class TestMemoryEngineSearch:
    """Tests for the search operation."""

    @pytest.mark.asyncio
    async def test_search_returns_scored_hits(self) -> None:
        engine = MemoryEngine()
        await engine.remember("python project setup")
        hits = await engine.search(MemoryQuery(keywords=("python",)))
        assert len(hits) == 1
        assert isinstance(hits[0], MemorySearchHit)
        assert hits[0].record.content == "python project setup"
        assert 0.0 < hits[0].score <= 1.0

    @pytest.mark.asyncio
    async def test_search_ranks_by_keyword_overlap(self) -> None:
        engine = MemoryEngine()
        await engine.remember("cat on the mat")
        await engine.remember("cat and dog")
        hits = await engine.search(MemoryQuery(keywords=("cat", "dog")))
        assert hits[0].record.content == "cat and dog"
        assert hits[0].score > hits[1].score

    @pytest.mark.asyncio
    async def test_search_excludes_zero_score(self) -> None:
        engine = MemoryEngine()
        await engine.remember("unrelated fact")
        hits = await engine.search(MemoryQuery(keywords=("zebra",)))
        assert hits == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self) -> None:
        engine = MemoryEngine()
        for i in range(5):
            await engine.remember(f"subject {i}")
        hits = await engine.search(MemoryQuery(keywords=("subject",), limit=2))
        assert len(hits) == 2

    @pytest.mark.asyncio
    async def test_search_respects_tier_filter(self) -> None:
        engine = MemoryEngine()
        await engine.remember("secret project", tier=MemoryTier.LONG_TERM)
        await engine.remember("secret project", tier=MemoryTier.SHORT_TERM)
        hits = await engine.search(
            MemoryQuery(keywords=("secret",), tier=MemoryTier.LONG_TERM)
        )
        assert len(hits) == 1
        assert hits[0].record.tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_search_with_no_keywords_returns_empty(self) -> None:
        engine = MemoryEngine()
        await engine.remember("something")
        hits = await engine.search(MemoryQuery())
        assert hits == []

    @pytest.mark.asyncio
    async def test_search_importance_breaks_keyword_tie(self) -> None:
        engine = MemoryEngine()
        await engine.remember("alpha beta", importance=0.2)
        await engine.remember("alpha beta", importance=0.9)
        hits = await engine.search(MemoryQuery(keywords=("alpha", "beta")))
        assert hits[0].record.importance == 0.9


class TestMemoryEngineSummary:
    """Tests for the summarize operation."""

    @pytest.mark.asyncio
    async def test_summarize_empty(self) -> None:
        engine = MemoryEngine()
        summary = await engine.summarize()
        assert summary.total_records == 0
        assert summary.short_term_count == 0
        assert summary.long_term_count == 0
        assert summary.average_importance == 0.0
        assert summary.tag_counts == {}
        assert summary.recent_records == ()

    @pytest.mark.asyncio
    async def test_summarize_counts(self) -> None:
        engine = MemoryEngine()
        await engine.remember("short one")
        await engine.remember("short two")
        await engine.remember("long one", tier=MemoryTier.LONG_TERM)
        summary = await engine.summarize()
        assert summary.total_records == 3
        assert summary.short_term_count == 2
        assert summary.long_term_count == 1

    @pytest.mark.asyncio
    async def test_summarize_average_importance(self) -> None:
        engine = MemoryEngine()
        await engine.remember("a", importance=0.2)
        await engine.remember("b", importance=0.8)
        summary = await engine.summarize()
        assert summary.average_importance == 0.5

    @pytest.mark.asyncio
    async def test_summarize_tag_counts(self) -> None:
        engine = MemoryEngine()
        await engine.remember("a", tags=("x", "y"))
        await engine.remember("b", tags=("x",))
        summary = await engine.summarize()
        assert summary.tag_counts == {"x": 2, "y": 1}

    @pytest.mark.asyncio
    async def test_summarize_recent_records_newest_first(self) -> None:
        store = InMemoryMemoryStore()
        engine = MemoryEngine(store=store)
        await store.save(
            _make_record(
                content="old",
            ).model_copy(update={"timestamp": datetime.utcnow() - timedelta(hours=1)})
        )
        await store.save(_make_record(content="new"))
        summary = await engine.summarize()
        assert [r.content for r in summary.recent_records] == ["new", "old"]


# ──────────────────────────────────────────────
# MemoryEngine — maintenance
# ──────────────────────────────────────────────


class TestMemoryEngineConsolidate:
    """Tests for short-term → long-term consolidation."""

    @pytest.mark.asyncio
    async def test_consolidates_high_importance(self) -> None:
        engine = MemoryEngine(consolidation_importance=0.7)
        await engine.remember("important fact", importance=0.8)
        promoted = await engine.consolidate()
        assert promoted == 1
        records = await engine.retrieve(MemoryQuery(limit=10))
        assert records[0].tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_consolidates_frequently_accessed(self) -> None:
        engine = MemoryEngine(consolidation_access_threshold=3)
        record_id = await engine.remember("popular fact")
        await engine.touch(record_id)
        await engine.touch(record_id)
        await engine.touch(record_id)
        promoted = await engine.consolidate()
        assert promoted == 1
        record = await engine.get(record_id)
        assert record is not None
        assert record.tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_does_not_consolidate_below_thresholds(self) -> None:
        engine = MemoryEngine(
            consolidation_importance=0.9,
            consolidation_access_threshold=10,
        )
        await engine.remember("plain fact", importance=0.2)
        promoted = await engine.consolidate()
        assert promoted == 0
        records = await engine.retrieve(MemoryQuery(limit=10))
        assert records[0].tier == MemoryTier.SHORT_TERM

    @pytest.mark.asyncio
    async def test_skips_already_long_term(self) -> None:
        engine = MemoryEngine()
        await engine.remember("long fact", tier=MemoryTier.LONG_TERM, importance=1.0)
        promoted = await engine.consolidate()
        assert promoted == 0

    @pytest.mark.asyncio
    async def test_consolidate_reports_count(self) -> None:
        engine = MemoryEngine(consolidation_importance=0.0)
        await engine.remember("one")
        await engine.remember("two")
        promoted = await engine.consolidate()
        assert promoted == 2


class TestMemoryEnginePrune:
    """Tests for short-term capacity pruning."""

    @pytest.mark.asyncio
    async def test_prune_removes_oldest_beyond_capacity(self) -> None:
        engine = MemoryEngine(short_term_capacity=2)
        await engine.remember("oldest", metadata={"order": 1})
        await engine.remember("middle", metadata={"order": 2})
        await engine.remember("newest", metadata={"order": 3})
        pruned = await engine.prune()
        assert pruned == 1
        remaining = await engine.retrieve(MemoryQuery(limit=10))
        contents = {r.content for r in remaining}
        assert contents == {"middle", "newest"}

    @pytest.mark.asyncio
    async def test_prune_keeps_long_term(self) -> None:
        engine = MemoryEngine(short_term_capacity=1)
        await engine.remember("short", tier=MemoryTier.SHORT_TERM)
        await engine.remember("long", tier=MemoryTier.LONG_TERM)
        await engine.remember("short2", tier=MemoryTier.SHORT_TERM)
        pruned = await engine.prune()
        assert pruned == 1
        remaining = await engine.retrieve(MemoryQuery(limit=10))
        contents = {r.content for r in remaining}
        assert contents == {"long", "short2"}

    @pytest.mark.asyncio
    async def test_prune_noop_within_capacity(self) -> None:
        engine = MemoryEngine(short_term_capacity=10)
        await engine.remember("one")
        await engine.remember("two")
        pruned = await engine.prune()
        assert pruned == 0
        assert await engine.count() == 2


# ──────────────────────────────────────────────
# MemoryEngine — misc
# ──────────────────────────────────────────────


class TestMemoryEngineMisc:
    """Miscellaneous engine behaviour."""

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        engine = MemoryEngine()
        assert await engine.count() == 0
        await engine.remember("a")
        await engine.remember("b")
        assert await engine.count() == 2

    @pytest.mark.asyncio
    async def test_engine_accepts_external_store(self) -> None:
        store = InMemoryMemoryStore()
        engine = MemoryEngine(store=store)
        await engine.remember("via engine")
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_engine_defaults_use_internal_store(self) -> None:
        engine = MemoryEngine()
        await engine.remember("fact")
        assert await engine.count() == 1


# ──────────────────────────────────────────────
# MemoryStore interface
# ──────────────────────────────────────────────


class TestMemoryStoreInterface:
    """Tests for the abstract MemoryStore contract."""

    @pytest.mark.asyncio
    async def test_abstract_methods_raise(self) -> None:
        store = MemoryStore()
        with pytest.raises(NotImplementedError):
            await store.save(_make_record())
        with pytest.raises(NotImplementedError):
            await store.get("id")
        with pytest.raises(NotImplementedError):
            await store.delete("id")
        with pytest.raises(NotImplementedError):
            await store.query(MemoryQuery())
        with pytest.raises(NotImplementedError):
            await store.all()
        with pytest.raises(NotImplementedError):
            await store.count()
        with pytest.raises(NotImplementedError):
            await store.clear()
