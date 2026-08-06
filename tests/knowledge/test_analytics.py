"""Tests for knowledge analytics."""

from app.knowledge.analytics import AnalyticsSummary, KnowledgeAnalytics


def test_record_embedding_created() -> None:
    a = KnowledgeAnalytics()
    a.record_embedding_created(5)
    s = a.summary()
    assert s.embeddings_created == 5


def test_record_cache_hit() -> None:
    a = KnowledgeAnalytics()
    a.record_cache_hit()
    a.record_cache_hit()
    a.record_cache_miss()
    s = a.summary()
    assert s.cache_hits == 2
    assert s.cache_misses == 1
    assert abs(s.cache_hit_rate - 2 / 3) < 0.01


def test_record_indexing_latency() -> None:
    a = KnowledgeAnalytics()
    a.record_indexing_latency(10.0)
    a.record_indexing_latency(20.0)
    s = a.summary()
    assert s.indexing_operations == 2
    assert s.total_indexing_latency_ms == 30.0
    assert s.avg_indexing_latency_ms == 15.0


def test_record_search_latency() -> None:
    a = KnowledgeAnalytics()
    a.record_search_latency(5.0)
    a.record_search_latency(15.0)
    s = a.summary()
    assert s.search_operations == 2
    assert s.total_search_latency_ms == 20.0
    assert s.avg_search_latency_ms == 10.0


def test_set_index_size() -> None:
    a = KnowledgeAnalytics()
    a.set_index_size(42)
    s = a.summary()
    assert s.index_size == 42


def test_clear() -> None:
    a = KnowledgeAnalytics()
    a.record_embedding_created(10)
    a.record_cache_hit()
    a.record_indexing_latency(5.0)
    a.clear()
    s = a.summary()
    assert s.embeddings_created == 0
    assert s.cache_hits == 0
    assert s.indexing_operations == 0


def test_summary_defaults() -> None:
    s = AnalyticsSummary()
    assert s.embeddings_created == 0
    assert s.cache_hit_rate == 0.0
    assert s.avg_indexing_latency_ms == 0.0
    assert s.avg_search_latency_ms == 0.0


def test_cache_hit_rate_zero_total() -> None:
    a = KnowledgeAnalytics()
    s = a.summary()
    assert s.cache_hit_rate == 0.0
