"""Memory engine — stage 13 of the cognitive pipeline.

Deterministic, rule-based memory subsystem.  Provides short-term and
long-term memory tiers, keyword retrieval, relevance-ranked search,
summarisation, consolidation of short-term memories into long-term
storage, and pruning of stale short-term records.

All scoring and consolidation logic is a pure function of the stored
records, so the engine is fully deterministic for testing.
"""

from __future__ import annotations

import logging
from typing import Any

from app.kernel.memory.models import (
    MemoryQuery,
    MemoryRecord,
    MemorySearchHit,
    MemorySummary,
    MemoryTier,
)

log = logging.getLogger(__name__)


def _extract_keywords(content: str) -> set[str]:
    """Split *content* into a set of lower-cased keywords."""
    return {token.strip().lower() for token in content.split() if token.strip()}


def _keyword_overlap(content: str, keywords: tuple[str, ...]) -> float:
    """Return the fraction of *keywords* present in *content* (0..1)."""
    if not keywords:
        return 0.0
    lowered = content.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return matched / len(keywords)


def _tag_overlap(tags: tuple[str, ...], query_tags: tuple[str, ...]) -> float:
    """Return the fraction of *query_tags* present in *tags* (0..1)."""
    if not query_tags:
        return 0.0
    tag_set = set(tags)
    matched = sum(1 for tag in query_tags if tag in tag_set)
    return matched / len(query_tags)


def _score_record(
    record: MemoryRecord,
    keywords: tuple[str, ...],
    query_tags: tuple[str, ...],
) -> float:
    """Compute a deterministic relevance score in the range 0..1.

    Score is a weighted blend of keyword overlap (70%), tag overlap
    (20%), and record importance (10%).
    """
    keyword_score = _keyword_overlap(record.content, keywords)
    tag_score = _tag_overlap(record.tags, query_tags)
    if keyword_score == 0.0 and tag_score == 0.0:
        return 0.0
    return round(
        0.7 * keyword_score + 0.2 * tag_score + 0.1 * record.importance,
        4,
    )


class MemoryStore:
    """Interface for memory persistence.

    Implementations may use in-memory storage, a database, or a
    remote service.
    """

    async def save(self, record: MemoryRecord) -> str:
        """Persist a memory record.

        Args:
            record: The record to persist.

        Returns:
            The record ID.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.save not yet implemented")

    async def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch a record by ID.

        Args:
            record_id: Record identifier.

        Returns:
            The record, or ``None`` if not found.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.get not yet implemented")

    async def delete(self, record_id: str) -> bool:
        """Delete a record by ID.

        Args:
            record_id: Record identifier.

        Returns:
            ``True`` if a record was removed.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.delete not yet implemented")

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """List records matching a query.

        Args:
            query: Query parameters.

        Returns:
            Matching records, newest first.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.query not yet implemented")

    async def all(self) -> list[MemoryRecord]:
        """Return every stored record.

        Returns:
            All records, newest first.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.all not yet implemented")

    async def count(self) -> int:
        """Return the total number of stored records.

        Returns:
            Total record count.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.count not yet implemented")

    async def clear(self) -> None:
        """Remove all records.

        Raises:
            NotImplementedError: Subclasses must override.
        """
        raise NotImplementedError("MemoryStore.clear not yet implemented")


class InMemoryMemoryStore(MemoryStore):
    """Deterministic in-memory memory storage."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._counter: int = 0

    async def save(self, record: MemoryRecord) -> str:
        if record.id:
            self._records[record.id] = record
            return record.id
        self._counter += 1
        record_id = f"mem_{self._counter}"
        self._records[record_id] = record.model_copy(update={"id": record_id})
        return record_id

    async def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    async def delete(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        records = list(self._records.values())

        if query.tier is not None:
            records = [r for r in records if r.tier == query.tier]
        if query.session_id is not None:
            records = [r for r in records if r.session_id == query.session_id]
        if query.tags is not None:
            tag_set = set(query.tags)
            records = [r for r in records if tag_set.issubset(set(r.tags))]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[: query.limit]

    async def all(self) -> list[MemoryRecord]:
        records = list(self._records.values())
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    async def count(self) -> int:
        return len(self._records)

    async def clear(self) -> None:
        self._records.clear()

    def raw_records(self) -> list[MemoryRecord]:
        """Return records without re-sorting (used for consolidation)."""
        return list(self._records.values())


class MemoryEngine:
    """Deterministic short-term / long-term memory subsystem.

    The engine owns a single backing store and applies rule-based
    consolidation (short-term → long-term) and pruning (capacity
    control) on demand.  Retrieval is keyword-based and deterministic.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        consolidation_importance: float = 0.7,
        consolidation_access_threshold: int = 3,
        short_term_capacity: int = 100,
    ) -> None:
        self._store = store or InMemoryMemoryStore()
        self._consolidation_importance = consolidation_importance
        self._consolidation_access_threshold = consolidation_access_threshold
        self._short_term_capacity = short_term_capacity

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    async def remember(
        self,
        content: str,
        *,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        session_id: str = "",
        tags: tuple[str, ...] = (),
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory record.

        Args:
            content: The fact to remember.
            tier: Memory tier to store in.
            session_id: Owning session identifier.
            tags: Retrieval tags.
            importance: Importance score (0..1).
            metadata: Arbitrary metadata.

        Returns:
            The new record ID.
        """
        record = MemoryRecord(
            content=content,
            tier=tier,
            session_id=session_id,
            tags=tags,
            importance=importance,
            metadata=metadata or {},
        )
        record_id = await self._store.save(record)
        log.info(
            "Memory remembered — id=%s, tier=%s, session=%s",
            record_id,
            tier.value,
            session_id,
        )
        return record_id

    async def forget(self, record_id: str) -> bool:
        """Remove a memory record.

        Args:
            record_id: Record identifier.

        Returns:
            ``True`` if a record was removed.
        """
        removed = await self._store.delete(record_id)
        if removed:
            log.info("Memory forgotten — id=%s", record_id)
        return removed

    async def clear(self) -> None:
        """Remove all memory records."""
        await self._store.clear()
        log.info("Memory cleared")

    async def touch(self, record_id: str) -> None:
        """Increment the access count of a record.

        Args:
            record_id: Record identifier.
        """
        record = await self._store.get(record_id)
        if record is None:
            return
        updated = record.model_copy(update={"access_count": record.access_count + 1})
        await self._store.save(updated)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch a single record by ID."""
        return await self._store.get(record_id)

    async def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Retrieve records matching a query.

        Args:
            query: Query parameters (tier, session, tags, limit).

        Returns:
            Matching records, newest first.
        """
        return await self._store.query(query)

    async def search(self, query: MemoryQuery) -> list[MemorySearchHit]:
        """Search memory by keywords, ranked by relevance.

        Args:
            query: Query parameters, including ``keywords``.

        Returns:
            Ranked search hits (highest score first).
        """
        keywords = query.keywords or ()
        query_tags = query.tags or ()

        records = await self._store.query(
            query.model_copy(update={"keywords": None, "tags": None, "limit": 100})
        )

        scored: list[MemorySearchHit] = []
        for record in records:
            score = _score_record(record, keywords, query_tags)
            if score > 0.0:
                scored.append(MemorySearchHit(record=record, score=score))

        scored.sort(key=lambda hit: (-hit.score, hit.record.timestamp))
        return scored[: query.limit]

    async def summarize(self) -> MemorySummary:
        """Produce aggregated statistics about stored memory.

        Returns:
            A summary with counts, tag distribution, average
            importance, and the newest records.
        """
        records = await self._store.all()
        total = len(records)
        if total == 0:
            return MemorySummary(
                total_records=0,
                short_term_count=0,
                long_term_count=0,
                average_importance=0.0,
                tag_counts={},
                recent_records=(),
            )

        short_term = sum(1 for r in records if r.tier == MemoryTier.SHORT_TERM)
        long_term = total - short_term
        avg_importance = round(sum(r.importance for r in records) / total, 4)

        tag_counts: dict[str, int] = {}
        for record in records:
            for tag in record.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return MemorySummary(
            total_records=total,
            short_term_count=short_term,
            long_term_count=long_term,
            average_importance=avg_importance,
            tag_counts=dict(sorted(tag_counts.items())),
            recent_records=tuple(records[:5]),
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def consolidate(self) -> int:
        """Promote important short-term records into long-term memory.

        A short-term record is promoted when its importance meets the
        configured threshold or its access count meets the configured
        threshold.

        Returns:
            The number of records promoted.
        """
        store = self._store
        if not isinstance(store, InMemoryMemoryStore):
            log.warning("Consolidation requires InMemoryMemoryStore")
            return 0

        promoted = 0
        for record in store.raw_records():
            if record.tier != MemoryTier.SHORT_TERM:
                continue
            if (
                record.importance >= self._consolidation_importance
                or record.access_count >= self._consolidation_access_threshold
            ):
                updated = record.model_copy(update={"tier": MemoryTier.LONG_TERM})
                await store.save(updated)
                promoted += 1

        if promoted:
            log.info("Memory consolidated — %d record(s) promoted", promoted)
        return promoted

    async def prune(self) -> int:
        """Remove the oldest short-term records beyond capacity.

        Returns:
            The number of records pruned.
        """
        store = self._store
        if not isinstance(store, InMemoryMemoryStore):
            log.warning("Pruning requires InMemoryMemoryStore")
            return 0

        short_term = [r for r in store.raw_records() if r.tier == MemoryTier.SHORT_TERM]
        short_term.sort(key=lambda r: r.timestamp)

        excess = len(short_term) - self._short_term_capacity
        if excess <= 0:
            return 0

        for record in short_term[:excess]:
            await store.delete(record.id)

        log.info("Memory pruned — removed %d short-term record(s)", excess)
        return excess

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return the total number of stored records."""
        return await self._store.count()
