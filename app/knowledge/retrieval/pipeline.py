"""Retrieval pipeline orchestrator.

Coordinates the full retrieval flow: query rewriting → multi-query
expansion → retrieval (vector/BM25/hybrid) → context compression →
reranking → citation building → analytics recording.

The pipeline is composable: each stage is optional and falls back to
its identity (no-op) variant when not configured.
"""

from __future__ import annotations

import logging
import time

from app.knowledge.analytics import KnowledgeAnalytics
from app.knowledge.base import (
    CitationBuilder,
    ContextCompressor,
    EmbeddingProvider,
    QueryRewriter,
    Reranker,
    Retriever,
    RetrievalContext,
    VectorStore,
)
from app.knowledge.models import (
    CitationResult,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)
from app.knowledge.retrieval.citations import DefaultCitationBuilder
from app.knowledge.retrieval.compress import IdentityCompressor
from app.knowledge.retrieval.hybrid import HybridRetriever
from app.knowledge.retrieval.multiquery import MultiQueryRetriever
from app.knowledge.retrieval.rerank import IdentityReranker
from app.knowledge.retrieval.rewrite import IdentityQueryRewriter
from app.knowledge.retrieval.vector import VectorRetriever

log = logging.getLogger(__name__)


class RetrievalPipeline:
    """Orchestrates the full retrieval flow.

    Composes query rewriting, multi-query expansion, hybrid retrieval,
    compression, reranking, and citation building.  Each stage is
    optional and defaults to identity (no-op).
    """

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        query_rewriter: QueryRewriter | None = None,
        compressor: ContextCompressor | None = None,
        reranker: Reranker | None = None,
        citation_builder: CitationBuilder | None = None,
        analytics: KnowledgeAnalytics | None = None,
    ) -> None:
        self._query_rewriter = query_rewriter or IdentityQueryRewriter()
        self._compressor = compressor or IdentityCompressor()
        self._reranker = reranker or IdentityReranker()
        self._citation_builder = citation_builder or DefaultCitationBuilder()
        self._analytics = analytics

        # Build the retriever chain
        if retriever is not None:
            self._retriever = retriever
        else:
            vector_ret = VectorRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            self._retriever = HybridRetriever(vector_retriever=vector_ret)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Execute the full retrieval pipeline.

        Args:
            query: The retrieval query with all options.

        Returns:
            A complete retrieval result with hits, citations, and sources.
        """
        start = time.time()
        rewritten = ""
        hits: list[RetrievalHit] = []

        try:
            # 1. Query rewriting
            context = RetrievalContext(
                namespace=query.namespace,
                collection_id=query.collection_id,
            )
            rewritten = await self._query_rewriter.rewrite(query.query, context=context)
            active_query = RetrievalQuery(
                query=query.query,
                rewritten_query=rewritten,
                namespace=query.namespace,
                collection_id=query.collection_id,
                filters=query.filters,
                strategy=query.strategy,
                top_k=query.top_k,
                vector_weight=query.vector_weight,
                bm25_weight=query.bm25_weight,
            )

            # 2. Multi-query expansion or direct retrieval
            if query.multi_query:
                multi = MultiQueryRetriever(
                    base_retriever=self._retriever,
                    query_rewriter=self._query_rewriter,
                )
                hits = await multi.retrieve(active_query, context=context)
            else:
                hits = await self._retriever.retrieve(active_query, context=context)

            # 3. Context compression
            if query.compression and hits:
                hits = await self._compressor.compress(hits, query=rewritten)

            # 4. Reranking
            if query.rerank and hits:
                hits = await self._reranker.rerank(rewritten, hits)

            # 5. Citation building
            citation_result: CitationResult | None = None
            if query.include_sources and hits:
                citation_result = self._citation_builder.build(hits)

        except Exception:
            log.exception("RetrievalPipeline: retrieval failed")
            hits = []

        elapsed_ms = (time.time() - start) * 1000

        # 6. Analytics
        if self._analytics:
            top_score = hits[0].score if hits else 0.0
            avg_score = sum(h.score for h in hits) / len(hits) if hits else 0.0
            self._analytics.record_retrieval(
                query=query.query,
                collection_id=query.collection_id,
                namespace=query.namespace,
                latency_ms=elapsed_ms,
                hit_count=len(hits),
                top_score=top_score,
                avg_score=avg_score,
                strategy=query.strategy,
                reranked=query.rerank,
            )

        # Assemble result
        citations = citation_result.citations if citation_result else ()
        sources = citation_result.sources if citation_result else ()

        return RetrievalResult(
            query=query.query,
            rewritten_query=rewritten,
            total=len(hits),
            hits=tuple(hits),
            sources=sources,
            latency_ms=elapsed_ms,
            strategy=query.strategy,
            citations=citations,
        )
