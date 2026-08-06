"""Retrieval and indexing analytics.

Thread-safe recorder that tracks embeddings created, cache hit rate,
indexing latency, search latency, and index size.  Provides summary
aggregates for observability.
"""

from __future__ import annotations

import threading
import time
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


class KnowledgeAnalytics:
    """Thread-safe analytics recorder for the knowledge subsystem.

    Records events (embeddings created, indexing latency, search latency,
    cache hits/misses) and provides aggregated summaries.
    """

    def __init__(self) -> None:
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
        self._max_entries = 1000

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

    def summary(self) -> AnalyticsSummary:
        """Return an aggregated analytics snapshot."""
        with self._lock:
            total_cache = self._cache_hits + self._cache_misses
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
