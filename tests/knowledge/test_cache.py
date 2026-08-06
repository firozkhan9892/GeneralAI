"""Tests for embedding cache."""

import pytest

from app.knowledge.embeddings.cache import CacheStats, EmbeddingCache, _cache_key


def test_cache_key_deterministic() -> None:
    k1 = _cache_key("prov", "model", "hello")
    k2 = _cache_key("prov", "model", "hello")
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex


def test_cache_key_differs_for_different_text() -> None:
    k1 = _cache_key("prov", "model", "hello")
    k2 = _cache_key("prov", "model", "world")
    assert k1 != k2


def test_cache_put_and_get() -> None:
    cache = EmbeddingCache(max_size=10)
    cache.put("p", "m", "text", [1.0, 2.0])
    assert cache.get("p", "m", "text") == [1.0, 2.0]


def test_cache_miss_returns_none() -> None:
    cache = EmbeddingCache(max_size=10)
    assert cache.get("p", "m", "missing") is None


def test_cache_stats() -> None:
    cache = EmbeddingCache(max_size=10)
    cache.put("p", "m", "a", [1.0])
    cache.get("p", "m", "a")  # hit
    cache.get("p", "m", "b")  # miss
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_rate == 0.5
    assert stats.size == 1


def test_cache_lru_eviction() -> None:
    cache = EmbeddingCache(max_size=2)
    cache.put("p", "m", "a", [1.0])
    cache.put("p", "m", "b", [2.0])
    cache.put("p", "m", "c", [3.0])  # evicts "a"
    assert cache.get("p", "m", "a") is None
    assert cache.get("p", "m", "b") == [2.0]
    assert cache.get("p", "m", "c") == [3.0]
    stats = cache.stats()
    assert stats.evictions == 1


def test_cache_lru_moves_to_end() -> None:
    cache = EmbeddingCache(max_size=2)
    cache.put("p", "m", "a", [1.0])
    cache.put("p", "m", "b", [2.0])
    cache.get("p", "m", "a")  # move "a" to end
    cache.put("p", "m", "c", [3.0])  # should evict "b" (not "a")
    assert cache.get("p", "m", "a") == [1.0]
    assert cache.get("p", "m", "b") is None


def test_cache_batch_put_and_get() -> None:
    cache = EmbeddingCache(max_size=10)
    texts = ["a", "b", "c"]
    vectors = [[1.0], [2.0], [3.0]]
    cache.put_many("p", "m", texts, vectors)
    hits = cache.get_many("p", "m", texts)
    assert len(hits) == 3
    assert hits[0] == [1.0]
    assert hits[1] == [2.0]
    assert hits[2] == [3.0]


def test_cache_clear() -> None:
    cache = EmbeddingCache(max_size=10)
    cache.put("p", "m", "a", [1.0])
    cache.clear()
    stats = cache.stats()
    assert stats.size == 0
    assert stats.hits == 0
    assert stats.misses == 0


def test_cache_zero_max_size() -> None:
    cache = EmbeddingCache(max_size=0)
    cache.put("p", "m", "a", [1.0])
    assert cache.get("p", "m", "a") is None


def test_cache_len() -> None:
    cache = EmbeddingCache(max_size=10)
    assert len(cache) == 0
    cache.put("p", "m", "a", [1.0])
    assert len(cache) == 1


def test_cache_overwrite_existing() -> None:
    cache = EmbeddingCache(max_size=10)
    cache.put("p", "m", "a", [1.0])
    cache.put("p", "m", "a", [2.0])
    assert cache.get("p", "m", "a") == [2.0]
    assert len(cache) == 1


def test_cache_stats_hit_rate_zero_total() -> None:
    stats = CacheStats(hits=0, misses=0)
    assert stats.hit_rate == 0.0


def test_cache_negative_max_size() -> None:
    with pytest.raises(ValueError):
        EmbeddingCache(max_size=-1)
