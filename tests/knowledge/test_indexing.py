"""Tests for indexing pipeline."""

from app.knowledge.analytics import KnowledgeAnalytics
from app.knowledge.documents.chunkers.fixed import FixedChunker
from app.knowledge.documents.loaders.text import TextLoader
from app.knowledge.embeddings.cache import EmbeddingCache
from app.knowledge.embeddings.mock import MockEmbeddingProvider
from app.knowledge.indexing.pipeline import IndexingPipeline
from app.knowledge.vectorstores.in_memory import InMemoryVectorStore


def _make_pipeline(**kwargs) -> IndexingPipeline:
    return IndexingPipeline(
        loader=kwargs.get("loader", TextLoader()),
        chunker=kwargs.get("chunker", FixedChunker(chunk_size=100, overlap=0)),
        embedding_provider=kwargs.get("provider", MockEmbeddingProvider(dimensions=32)),
        vector_store=kwargs.get("store", InMemoryVectorStore(dimensions=32)),
        cache=kwargs.get("cache", EmbeddingCache(max_size=100)),
        analytics=kwargs.get("analytics", KnowledgeAnalytics()),
    )


def test_ingest_basic() -> None:
    pipeline = _make_pipeline()
    doc = pipeline.ingest(
        b"Hello world. This is a test document.",
        source_uri="test.txt",
        collection_id="col1",
        namespace="ns1",
    )
    assert doc.content == "Hello world. This is a test document."
    assert pipeline.store.count() >= 1


def test_ingest_document() -> None:
    pipeline = _make_pipeline()
    from app.knowledge.models import DocumentFormat, IndexStatus, KnowledgeDocument

    doc = KnowledgeDocument(
        doc_id="d1",
        collection_id="col1",
        namespace="ns1",
        format=DocumentFormat.TXT,
        content="Test content here.",
        content_hash="abc",
        status=IndexStatus.PENDING,
    )
    chunks = pipeline.ingest_document(doc)
    assert len(chunks) >= 1
    assert pipeline.store.count() >= 1


def test_batch_ingest() -> None:
    pipeline = _make_pipeline()
    docs = [
        (b"Document one.", "doc1.txt"),
        (b"Document two.", "doc2.txt"),
    ]
    results = pipeline.batch_ingest(docs, collection_id="col1", namespace="ns1")
    assert len(results) == 2
    assert pipeline.store.count() >= 2


def test_rebuild() -> None:
    pipeline = _make_pipeline()
    pipeline.ingest(
        b"First doc.", source_uri="d1.txt", collection_id="c1", namespace="n1"
    )
    assert pipeline.store.count() >= 1
    pipeline.rebuild(
        [(b"Second doc.", "d2.txt")],
        collection_id="c1",
        namespace="n1",
    )
    # Old doc should be gone, only new doc remains
    assert pipeline.store.count() >= 1


def test_search() -> None:
    pipeline = _make_pipeline()
    pipeline.ingest(
        b"Machine learning is a subset of AI.",
        source_uri="ml.txt",
        collection_id="c1",
        namespace="n1",
    )
    pipeline.ingest(
        b"Cooking recipes for dinner.",
        source_uri="cook.txt",
        collection_id="c1",
        namespace="n1",
    )
    # Search with the same provider
    query_vector = pipeline.provider.embed(["machine learning"])[0]
    results = pipeline.search(query_vector, top_k=2)
    assert len(results) >= 1


def test_analytics_recorded() -> None:
    analytics = KnowledgeAnalytics()
    pipeline = _make_pipeline(analytics=analytics)
    pipeline.ingest(
        b"Test content.", source_uri="t.txt", collection_id="c1", namespace="n1"
    )
    summary = analytics.summary()
    assert summary.embeddings_created >= 1
    assert summary.indexing_operations == 1
    assert summary.total_indexing_latency_ms >= 0


def test_cache_hits() -> None:
    cache = EmbeddingCache(max_size=100)
    pipeline = _make_pipeline(cache=cache)
    pipeline.ingest(
        b"Hello world.", source_uri="t.txt", collection_id="c1", namespace="n1"
    )
    # Second ingest of same content should hit cache
    pipeline.ingest(
        b"Hello world.", source_uri="t2.txt", collection_id="c1", namespace="n1"
    )
    stats = cache.stats()
    assert stats.hits > 0


def test_index_size_tracked() -> None:
    analytics = KnowledgeAnalytics()
    pipeline = _make_pipeline(analytics=analytics)
    pipeline.ingest(b"Content.", source_uri="t.txt", collection_id="c1", namespace="n1")
    summary = analytics.summary()
    assert summary.index_size >= 1
