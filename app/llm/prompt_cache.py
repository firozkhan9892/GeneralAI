"""Prompt caching abstraction for LLM responses.

In-memory only for now; the :class:`PromptCache` API is designed to
support a future Redis backend with no changes to callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any

from app.llm.router_models import CacheKey, RouterStrategy
from app.llm.models import ChatRequest

log = logging.getLogger(__name__)


class PromptCache:
    """Thread-safe in-memory cache for LLM responses.

    Cache keys are SHA-256 hashes of the request's messages and
    parameters.  Values are :class:`CacheEntry` objects containing
    the response and metadata.

    Attributes:
        _cache: Maps cache key string → :class:`CacheEntry`.
        _lock: Protects ``_cache`` access.
        _max_size: Maximum number of entries before LRU eviction.
        _ttl: Time-to-live in seconds (None = no expiry).
        _enabled: Whether caching is active.
        _hit_count: Total cache hits.
        _miss_count: Total cache misses.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: float | None = 3600.0,
        enabled: bool = True,
    ) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl
        self._enabled = enabled
        self._hit_count = 0
        self._miss_count = 0
        self._clock = 0

    @property
    def enabled(self) -> bool:
        """Return whether the cache is active."""
        return self._enabled

    def enable(self) -> None:
        """Enable caching."""
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable caching."""
        with self._lock:
            self._enabled = False

    def _generate_hash(self, request: ChatRequest) -> str:
        """Generate a SHA-256 hash key from a :class:`ChatRequest`.

        The hash includes messages, model, temperature, max_tokens,
        tools, response_format, and stream flags to ensure cache
        correctness.
        """
        key_data = {
            "messages": [m.model_dump() for m in request.messages],
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stop": list(request.stop) if request.stop else [],
            "tools": (
                [t.model_dump() for t in request.tools] if request.tools else None
            ),
            "response_format": (
                request.response_format.model_dump()
                if request.response_format
                else None
            ),
            "stream": request.stream,
        }
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def build_key(self, request: ChatRequest, provider_id: str) -> CacheKey:
        """Build a :class:`CacheKey` from a request.

        Args:
            request: The chat request.
            provider_id: Provider that would handle the request.

        Returns:
            A :class:`CacheKey` with hash and metadata.
        """
        hash_val = self._generate_hash(request)
        return CacheKey(
            hash=hash_val,
            provider_id=provider_id,
            model=request.model,
            strategy=RouterStrategy.SMART,
        )

    def get(self, key: CacheKey) -> Any | None:
        """Retrieve a cached response by key.

        Returns ``None`` on miss or if cache is disabled.
        Updates hit/miss counters.
        """
        if not self._enabled:
            return None

        with self._lock:
            entry = self._cache.get(key.hash)
            if entry is None:
                self._miss_count += 1
                return None

            if entry.is_expired():
                del self._cache[key.hash]
                self._miss_count += 1
                return None

            self._hit_count += 1
            entry.access_count += 1
            self._clock += 1
            entry.last_accessed = self._clock
            return entry.value

    def put(self, key: CacheKey, value: Any, ttl: float | None = None) -> None:
        """Store a response in the cache.

        Args:
            key: Cache key.
            value: Response to cache.
            ttl: Optional per-entry TTL (defaults to cache-wide TTL).
        """
        if not self._enabled:
            return

        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            self._clock += 1
            self._cache[key.hash] = CacheEntry(
                key=key,
                value=value,
                created_at=time.monotonic(),
                ttl=ttl if ttl is not None else self._ttl,
            )
            self._cache[key.hash].last_accessed = self._clock

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry."""
        if not self._cache:
            return

        lru_hash = min(
            self._cache.keys(),
            key=lambda h: self._cache[h].last_accessed,
        )
        del self._cache[lru_hash]

    def invalidate(self, key: CacheKey) -> bool:
        """Remove an entry from the cache.

        Returns ``True`` if the key was found and removed.
        """
        with self._lock:
            return self._cache.pop(key.hash, None) is not None

    def invalidate_provider(self, provider_id: str) -> int:
        """Invalidate all entries for a provider.

        Returns the number of entries removed.
        """
        with self._lock:
            to_remove = [
                h for h, e in self._cache.items() if e.key.provider_id == provider_id
            ]
            for h in to_remove:
                del self._cache[h]
            return len(to_remove)

    def clear(self) -> int:
        """Clear all entries. Returns the number removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            return {
                "enabled": self._enabled,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": hit_rate,
            }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        with self._lock:
            self._hit_count = 0
            self._miss_count = 0


class CacheEntry:
    """A single entry in the :class:`PromptCache`.

    Attributes:
        key: The :class:`CacheKey` for this entry.
        value: The cached response.
        created_at: When the entry was created.
        last_accessed: Monotonic sequence number of the last access (LRU).
        access_count: Number of times accessed.
        ttl: Time-to-live in seconds.
    """

    __slots__ = (
        "key",
        "value",
        "created_at",
        "last_accessed",
        "access_count",
        "ttl",
    )

    def __init__(
        self,
        key: CacheKey,
        value: Any,
        created_at: float,
        ttl: float | None,
    ) -> None:
        self.key = key
        self.value = value
        self.created_at = created_at
        self.last_accessed = created_at
        self.access_count = 0
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Return ``True`` if the entry has exceeded its TTL."""
        if self.ttl is None:
            return False
        return time.monotonic() - self.created_at > self.ttl
