"""Thread-safe LRU embedding cache.

Caches embedding vectors keyed by ``(provider_name, model, sha256(text))``
so repeated embeddings are free.  The cache is an ``OrderedDict`` protected
by an ``RLock`` — no external dependencies beyond the stdlib.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


def _cache_key(provider_name: str, model: str, text: str) -> str:
    """Return a SHA-256 hex-digest cache key."""
    raw = f"{provider_name}\0{model}\0{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate (0.0–1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors.

    Parameters
    ----------
    max_size:
        Maximum number of cached entries.  Older entries are evicted
        when the limit is exceeded.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size < 0:
            raise ValueError("max_size must be >= 0")
        self._max_size = max_size
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, provider_name: str, model: str, text: str) -> list[float] | None:
        """Look up a cached vector.

        Returns the cached vector on hit, ``None`` on miss.  Moves the
        entry to the end (most-recently-used) on hit.
        """
        key = _cache_key(provider_name, model, text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(
        self, provider_name: str, model: str, text: str, vector: list[float]
    ) -> None:
        """Store a vector in the cache.

        Evicts the oldest entry if the cache is full.
        """
        if self._max_size == 0:
            return
        key = _cache_key(provider_name, model, text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = vector
                return
            self._cache[key] = vector
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def get_many(
        self,
        provider_name: str,
        model: str,
        texts: list[str],
    ) -> dict[int, list[float]]:
        """Batch lookup.  Returns ``{index: vector}`` for cache hits."""
        result: dict[int, list[float]] = {}
        for idx, text in enumerate(texts):
            vector = self.get(provider_name, model, text)
            if vector is not None:
                result[idx] = vector
        return result

    def put_many(
        self,
        provider_name: str,
        model: str,
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        """Batch store."""
        for text, vector in zip(texts, vectors, strict=True):
            self.put(provider_name, model, text, vector)

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._cache),
            )

    def clear(self) -> None:
        """Remove all cached entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache
