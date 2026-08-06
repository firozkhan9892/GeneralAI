"""Tests for hybrid retriever with RRF and pipeline stages."""

import asyncio

from app.knowledge.analytics import KnowledgeAnalytics
from app.knowledge.base import RetrievalContext
from app.knowledge.models import RetrievalHit, RetrievalQuery
from app.knowledge.retrieval.bm25 import BM25Index, BM25Retriever
from app.knowledge.retrieval.citations import DefaultCitationBuilder
from app.knowledge.retrieval.compress import IdentityCompressor
from app.knowledge.retrieval.hybrid import HybridRetriever, _rrf_fuse
from app.knowledge.retrieval.multiquery import MultiQueryRetriever, _generate_variants
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.knowledge.retrieval.rerank import IdentityReranker
from app.knowledge.retrieval.rewrite import IdentityQueryRewriter


def _make_hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id="c1",
            doc_id="d1",
            collection_id="col1",
            namespace="ns",
            content="hello world",
            score=0.9,
            ranks={"vector": 0.9},
        ),
        RetrievalHit(
            chunk_id="c2",
            doc_id="d2",
            collection_id="col1",
            namespace="ns",
            content="foo bar",
            score=0.7,
            ranks={"vector": 0.7},
        ),
        RetrievalHit(
            chunk_id="c3",
            doc_id="d3",
            collection_id="col1",
            namespace="ns",
            content="baz qux",
            score=0.5,
            ranks={"vector": 0.5},
        ),
    ]


# ── RRF fusion ────────────────────────────────────────────────────────


class TestRRFFuse:
    def test_basic_fusion(self) -> None:
        vector_hits = [
            RetrievalHit(
                chunk_id="c1",
                doc_id="d1",
                collection_id="col1",
                namespace="ns",
                content="",
                score=0.9,
                ranks={"vector": 0.9},
            ),
            RetrievalHit(
                chunk_id="c2",
                doc_id="d2",
                collection_id="col1",
                namespace="ns",
                content="",
                score=0.7,
                ranks={"vector": 0.7},
            ),
        ]
        bm25_hits = [
            RetrievalHit(
                chunk_id="c2",
                doc_id="d2",
                collection_id="col1",
                namespace="ns",
                content="",
                score=0.8,
                ranks={"bm25": 0.8},
            ),
            RetrievalHit(
                chunk_id="c3",
                doc_id="d3",
                collection_id="col1",
                namespace="ns",
                content="",
                score=0.6,
                ranks={"bm25": 0.6},
            ),
        ]
        fused = _rrf_fuse(vector_hits, bm25_hits, top_k=10)
        ids = [h.chunk_id for h in fused]
        # c2 appears in both lists, should rank high
        assert "c2" in ids
        assert "c1" in ids
        assert "c3" in ids

    def test_deduplication(self) -> None:
        vector_hits = [
            RetrievalHit(
                chunk_id="c1",
                doc_id="d1",
                collection_id="col1",
                namespace="ns",
                content="hello",
                score=0.9,
                ranks={"vector": 0.9},
            ),
        ]
        bm25_hits = [
            RetrievalHit(
                chunk_id="c1",
                doc_id="d1",
                collection_id="col1",
                namespace="ns",
                content="hello",
                score=0.8,
                ranks={"bm25": 0.8},
            ),
        ]
        fused = _rrf_fuse(vector_hits, bm25_hits, top_k=10)
        assert len(fused) == 1
        assert "vector" in fused[0].ranks
        assert "bm25" in fused[0].ranks

    def test_weights(self) -> None:
        vector_hits = [
            RetrievalHit(
                chunk_id="c1",
                doc_id="d1",
                collection_id="col1",
                namespace="ns",
                content="",
                score=0.9,
                ranks={"vector": 0.9},
            ),
        ]
        bm25_hits = [
            RetrievalHit(
                chunk_id="c2",
                doc_id="d2",
                collection_id="col1",
                namespace="ns",
                content="",
                score=0.8,
                ranks={"bm25": 0.8},
            ),
        ]
        # All weight to vector
        fused = _rrf_fuse(
            vector_hits, bm25_hits, vector_weight=1.0, bm25_weight=0.0, top_k=10
        )
        assert fused[0].chunk_id == "c1"

        # All weight to BM25
        fused = _rrf_fuse(
            vector_hits, bm25_hits, vector_weight=0.0, bm25_weight=1.0, top_k=10
        )
        assert fused[0].chunk_id == "c2"

    def test_empty_inputs(self) -> None:
        fused = _rrf_fuse([], [], top_k=10)
        assert fused == []

    def test_top_k_limit(self) -> None:
        hits = [
            RetrievalHit(
                chunk_id=f"c{i}",
                doc_id=f"d{i}",
                collection_id="col1",
                namespace="ns",
                content="",
                score=float(i),
                ranks={"vector": float(i)},
            )
            for i in range(20)
        ]
        fused = _rrf_fuse(hits, [], top_k=5)
        assert len(fused) == 5


# ── HybridRetriever ───────────────────────────────────────────────────


class TestHybridRetriever:
    def test_hybrid_with_bm25_only(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1", namespace="ns")
        idx.add(
            "c2", "goodbye world", doc_id="d2", collection_id="col1", namespace="ns"
        )
        bm25 = BM25Retriever(index=idx)
        hybrid = HybridRetriever(bm25_retriever=bm25)

        query = RetrievalQuery(query="hello", top_k=5)
        context = RetrievalContext(namespace="ns", collection_id="col1")
        hits = asyncio.run(hybrid.retrieve(query, context=context))
        assert len(hits) > 0
        assert hits[0].chunk_id == "c1"

    def test_hybrid_no_retrievers(self) -> None:
        hybrid = HybridRetriever()
        query = RetrievalQuery(query="test", top_k=5)
        context = RetrievalContext()
        hits = asyncio.run(hybrid.retrieve(query, context=context))
        assert hits == []

    def test_name(self) -> None:
        assert HybridRetriever().name == "hybrid"


# ── Identity stages ──────────────────────────────────────────────────


class TestIdentityQueryRewriter:
    def test_rewrite_passthrough(self) -> None:
        rewriter = IdentityQueryRewriter()
        context = RetrievalContext()
        result = asyncio.run(rewriter.rewrite("hello world", context=context))
        assert result == "hello world"

    def test_name(self) -> None:
        assert IdentityQueryRewriter().name == "identity"


class TestIdentityCompressor:
    def test_compress_passthrough(self) -> None:
        compressor = IdentityCompressor()
        hits = _make_hits()
        result = asyncio.run(compressor.compress(hits, query="test"))
        assert result == hits

    def test_name(self) -> None:
        assert IdentityCompressor().name == "identity"


class TestIdentityReranker:
    def test_rerank_passthrough(self) -> None:
        reranker = IdentityReranker()
        hits = _make_hits()
        result = asyncio.run(reranker.rerank("test", hits))
        assert result == hits

    def test_name(self) -> None:
        assert IdentityReranker().name == "identity"


# ── DefaultCitationBuilder ────────────────────────────────────────────


class TestDefaultCitationBuilder:
    def test_build_citations(self) -> None:
        builder = DefaultCitationBuilder()
        hits = _make_hits()
        result = builder.build(hits)
        assert len(result.citations) == 3
        assert len(result.sources) == 3  # d1, d2, d3 are different doc_ids
        for citation in result.citations:
            assert len(citation.citation_id) == 16

    def test_build_empty(self) -> None:
        builder = DefaultCitationBuilder()
        result = builder.build([])
        assert len(result.citations) == 0
        assert len(result.sources) == 0

    def test_dedup_sources(self) -> None:
        builder = DefaultCitationBuilder()
        hits = [
            RetrievalHit(
                chunk_id="c1",
                doc_id="d1",
                collection_id="col1",
                namespace="ns",
                content="a",
                score=0.9,
                ranks={"vector": 0.9},
                metadata={"title": "Doc1"},
            ),
            RetrievalHit(
                chunk_id="c2",
                doc_id="d1",
                collection_id="col1",
                namespace="ns",
                content="b",
                score=0.7,
                ranks={"vector": 0.7},
                metadata={"title": "Doc1"},
            ),
        ]
        result = builder.build(hits)
        assert len(result.sources) == 1  # same doc_id
        assert result.sources[0].confidence == 0.9  # max score

    def test_name(self) -> None:
        assert DefaultCitationBuilder().name == "default"


# ── Multi-query variants ─────────────────────────────────────────────


class TestGenerateVariants:
    def test_basic_variants(self) -> None:
        variants = _generate_variants("what is python programming?", 3)
        assert len(variants) == 3
        assert variants[0] == "what is python programming?"

    def test_single_variant(self) -> None:
        variants = _generate_variants("hello", 1)
        assert variants == ["hello"]

    def test_keyword_extraction(self) -> None:
        variants = _generate_variants("what is the best python library?", 3)
        # Should extract keywords
        assert any("python" in v for v in variants)


# ── MultiQueryRetriever ──────────────────────────────────────────────


class TestMultiQueryRetriever:
    def test_multi_query(self) -> None:
        idx = BM25Index()
        idx.add("c1", "python programming language", doc_id="d1", collection_id="col1")
        idx.add("c2", "java programming language", doc_id="d2", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        multi = MultiQueryRetriever(base_retriever=bm25, n_queries=2)

        query = RetrievalQuery(query="what is python?", top_k=5)
        context = RetrievalContext(collection_id="col1")
        hits = asyncio.run(multi.retrieve(query, context=context))
        assert len(hits) > 0

    def test_no_base_retriever(self) -> None:
        multi = MultiQueryRetriever(base_retriever=None)
        query = RetrievalQuery(query="test", top_k=5)
        context = RetrievalContext()
        hits = asyncio.run(multi.retrieve(query, context=context))
        assert hits == []

    def test_name(self) -> None:
        assert MultiQueryRetriever().name == "multi_query"


# ── RetrievalPipeline ────────────────────────────────────────────────


class TestRetrievalPipeline:
    def test_pipeline_with_bm25(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1", namespace="ns")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(
            query="hello", top_k=5, collection_id="col1", namespace="ns"
        )
        result = asyncio.run(pipeline.retrieve(query))
        assert result.total > 0
        assert result.latency_ms >= 0
        assert result.strategy == "hybrid"

    def test_pipeline_with_analytics(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        analytics = KnowledgeAnalytics()
        pipeline = RetrievalPipeline(retriever=bm25, analytics=analytics)

        query = RetrievalQuery(query="hello", top_k=5, collection_id="col1")
        asyncio.run(pipeline.retrieve(query))

        summary = analytics.summary()
        assert summary.total_queries == 1

    def test_pipeline_citations(self) -> None:
        idx = BM25Index()
        idx.add(
            "c1",
            "hello world",
            doc_id="d1",
            collection_id="col1",
            metadata={"title": "Test Doc"},
        )
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(
            query="hello", top_k=5, collection_id="col1", include_sources=True
        )
        result = asyncio.run(pipeline.retrieve(query))
        assert len(result.citations) > 0
        assert len(result.sources) > 0

    def test_pipeline_no_sources(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(
            query="hello", top_k=5, collection_id="col1", include_sources=False
        )
        result = asyncio.run(pipeline.retrieve(query))
        assert len(result.citations) == 0
        assert len(result.sources) == 0

    def test_pipeline_multi_query(self) -> None:
        idx = BM25Index()
        idx.add("c1", "python programming", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(
            query="what is python?", top_k=5, collection_id="col1", multi_query=True
        )
        result = asyncio.run(pipeline.retrieve(query))
        assert result.total > 0

    def test_pipeline_compression(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(
            query="hello", top_k=5, collection_id="col1", compression=True
        )
        result = asyncio.run(pipeline.retrieve(query))
        # Identity compressor returns all hits unchanged
        assert result.total > 0

    def test_pipeline_rerank(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(
            query="hello", top_k=5, collection_id="col1", rerank=True
        )
        result = asyncio.run(pipeline.retrieve(query))
        # Identity reranker returns all hits unchanged
        assert result.total > 0

    def test_pipeline_empty_query(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(query="", top_k=5, collection_id="col1")
        result = asyncio.run(pipeline.retrieve(query))
        assert result.total == 0

    def test_pipeline_rewritten_query(self) -> None:
        idx = BM25Index()
        idx.add("c1", "hello world", doc_id="d1", collection_id="col1")
        bm25 = BM25Retriever(index=idx)
        pipeline = RetrievalPipeline(retriever=bm25)

        query = RetrievalQuery(query="hello", top_k=5, collection_id="col1")
        result = asyncio.run(pipeline.retrieve(query))
        # Identity rewriter returns raw query unchanged
        assert result.rewritten_query == "hello"
