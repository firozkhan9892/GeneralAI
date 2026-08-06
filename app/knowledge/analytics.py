"""Retrieval and indexing analytics.

Thread-safe recorder that tracks embeddings created, cache hit rate,
indexing latency, search latency, retrieval query patterns, and index
size.  Provides summary aggregates for observability.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnalyticsEntry:
    """A single recorded analytics event."""

    event_type: str
    timestamp: float
    latency_ms: float = 0.0
    value: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsSummary:
    """Aggregated analytics snapshot."""

    embeddings_created: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    total_indexing_latency_ms: float = 0.0
    total_search_latency_ms: float = 0.0
    indexing_operations: int = 0
    search_operations: int = 0
    avg_indexing_latency_ms: float = 0.0
    avg_search_latency_ms: float = 0.0
    index_size: int = 0
    # Retrieval-specific fields
    total_queries: int = 0
    avg_query_latency_ms: float = 0.0
    avg_hit_count: float = 0.0
    collections: dict[str, int] = field(default_factory=dict)
    top_queries: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    recent_retrievals: tuple[AnalyticsEntry, ...] = field(default_factory=tuple)


class KnowledgeAnalytics:
    """Thread-safe analytics recorder for the knowledge subsystem.

    Records events (embeddings created, indexing latency, search latency,
    cache hits/misses, retrieval queries) and provides aggregated summaries.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._lock = threading.RLock()
        self._embeddings_created = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_indexing_latency_ms = 0.0
        self._total_search_latency_ms = 0.0
        self._indexing_operations = 0
        self._search_operations = 0
        self._index_size = 0
        self._entries: list[AnalyticsEntry] = []
        self._max_entries = max_entries
        # Retrieval-specific fields
        self._total_queries = 0
        self._total_query_latency_ms = 0.0
        self._total_hit_count = 0
        self._collection_queries: dict[str, int] = defaultdict(int)
        self._query_counts: dict[str, int] = defaultdict(int)
        self._retrieval_entries: list[AnalyticsEntry] = []

    def record_embedding_created(self, count: int = 1) -> None:
        """Record that *count* embeddings were created."""
        with self._lock:
            self._embeddings_created += count

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1

    def record_indexing_latency(self, latency_ms: float) -> None:
        """Record an indexing operation's latency."""
        with self._lock:
            self._total_indexing_latency_ms += latency_ms
            self._indexing_operations += 1
            self._entries.append(
                AnalyticsEntry(
                    event_type="indexing",
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                )
            )
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def record_search_latency(self, latency_ms: float) -> None:
        """Record a search operation's latency."""
        with self._lock:
            self._total_search_latency_ms += latency_ms
            self._search_operations += 1
            self._entries.append(
                AnalyticsEntry(
                    event_type="search",
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                )
            )
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def set_index_size(self, size: int) -> None:
        """Set the current index size."""
        with self._lock:
            self._index_size = size

    def record_retrieval(
        self,
        query: str,
        collection_id: str = "",
        namespace: str = "",
        latency_ms: float = 0.0,
        hit_count: int = 0,
        top_score: float = 0.0,
        avg_score: float = 0.0,
        strategy: str = "hybrid",
        reranked: bool = False,
    ) -> None:
        """Record a retrieval query event.

        Args:
            query: The retrieval query string.
            collection_id: Target collection.
            namespace: Isolating namespace.
            latency_ms: Query latency.
            hit_count: Number of hits returned.
            top_score: Highest relevance score.
            avg_score: Average relevance score.
            strategy: Retrieval strategy used.
            reranked: Whether reranking was applied.
        """
        entry = AnalyticsEntry(
            event_type="retrieval",
            timestamp=time.time(),
            latency_ms=latency_ms,
            value=float(hit_count),
            metadata={
                "query": query,
                "collection_id": collection_id,
                "namespace": namespace,
                "top_score": top_score,
                "avg_score": avg_score,
                "strategy": strategy,
                "reranked": reranked,
            },
        )
        with self._lock:
            self._total_queries += 1
            self._total_query_latency_ms += latency_ms
            self._total_hit_count += hit_count
            self._collection_queries[collection_id] += 1
            self._query_counts[query] += 1
            self._retrieval_entries.append(entry)
            if len(self._retrieval_entries) > self._max_entries:
                self._retrieval_entries = self._retrieval_entries[-self._max_entries :]

    def retrieval_summary(self) -> dict[str, Any]:
        """Return retrieval-specific analytics.

        Returns a dictionary with total_queries, avg_query_latency_ms,
        avg_hit_count, collections, top_queries, and recent entries.
        """
        with self._lock:
            total = self._total_queries
            avg_latency = self._total_query_latency_ms / total if total > 0 else 0.0
            avg_hits = self._total_hit_count / total if total > 0 else 0.0
            top_queries = sorted(
                self._query_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
            recent = tuple(self._retrieval_entries[-20:])
            return {
                "total_queries": total,
                "avg_query_latency_ms": avg_latency,
                "avg_hit_count": avg_hits,
                "collections": dict(self._collection_queries),
                "top_queries": tuple(top_queries),
                "recent": recent,
            }

    def summary(self) -> AnalyticsSummary:
        """Return an aggregated analytics snapshot."""
        with self._lock:
            total_cache = self._cache_hits + self._cache_misses
            total = self._total_queries
            return AnalyticsSummary(
                embeddings_created=self._embeddings_created,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                cache_hit_rate=(
                    self._cache_hits / total_cache if total_cache > 0 else 0.0
                ),
                total_indexing_latency_ms=self._total_indexing_latency_ms,
                total_search_latency_ms=self._total_search_latency_ms,
                indexing_operations=self._indexing_operations,
                search_operations=self._search_operations,
                avg_indexing_latency_ms=(
                    self._total_indexing_latency_ms / self._indexing_operations
                    if self._indexing_operations > 0
                    else 0.0
                ),
                avg_search_latency_ms=(
                    self._total_search_latency_ms / self._search_operations
                    if self._search_operations > 0
                    else 0.0
                ),
                index_size=self._index_size,
                total_queries=total,
                avg_query_latency_ms=(
                    self._total_query_latency_ms / total if total > 0 else 0.0
                ),
                avg_hit_count=(self._total_hit_count / total if total > 0 else 0.0),
                collections=dict(self._collection_queries),
                top_queries=tuple(
                    sorted(
                        self._query_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
                recent_retrievals=tuple(self._retrieval_entries[-20:]),
            )

    def clear(self) -> None:
        """Reset all analytics counters."""
        with self._lock:
            self._embeddings_created = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_indexing_latency_ms = 0.0
            self._total_search_latency_ms = 0.0
            self._indexing_operations = 0
            self._search_operations = 0
            self._index_size = 0
            self._entries.clear()
            self._total_queries = 0
            self._total_query_latency_ms = 0.0
            self._total_hit_count = 0
            self._collection_queries.clear()
            self._query_counts.clear()
            self._retrieval_entries.clear()
