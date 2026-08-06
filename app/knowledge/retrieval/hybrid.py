"""Hybrid retriever with Reciprocal Rank Fusion (RRF).

Runs both vector and BM25 retrieval in parallel and fuses the results
using RRF.  The fused score for each document is::

    score = w_v * rrf_v + w_b * rrf_b

where ``rrf = Σ 1/(k + rank)`` over both lists and *k* is a smoothing
constant (default 60, from the original RRF paper).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.knowledge.base import Retriever, RetrievalContext
from app.knowledge.constants import DEFAULT_TOP_K, RRF_K
from app.knowledge.models import RetrievalHit, RetrievalQuery

log = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Fuses vector and BM25 retrieval via Reciprocal Rank Fusion.

    Wraps two sub-retrievers (vector and BM25) and merges their
    ranked result lists.  Weights from the query control the blend.
    """

    name: str = "hybrid"

    def __init__(
        self,
        vector_retriever: Retriever | None = None,
        bm25_retriever: Retriever | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._rrf_k = rrf_k

    async def retrieve(
        self, query: RetrievalQuery, *, context: RetrievalContext
    ) -> list[RetrievalHit]:
        """Run vector + BM25 and fuse via RRF.

        Args:
            query: The retrieval query.
            context: Collection / store / filter context.

        Returns:
            Fused hits ranked by weighted RRF score.
        """
        vector_hits: list[RetrievalHit] = []
        bm25_hits: list[RetrievalHit] = []

        if self._vector_retriever is not None:
            try:
                vector_hits = await self._vector_retriever.retrieve(
                    query, context=context
                )
            except Exception:
                log.exception("HybridRetriever: vector retrieval failed")

        if self._bm25_retriever is not None:
            try:
                bm25_hits = await self._bm25_retriever.retrieve(query, context=context)
            except Exception:
                log.exception("HybridRetriever: BM25 retrieval failed")

        if not vector_hits and not bm25_hits:
            return []

        return _rrf_fuse(
            vector_hits,
            bm25_hits,
            vector_weight=query.vector_weight,
            bm25_weight=query.bm25_weight,
            k=self._rrf_k,
            top_k=query.top_k,
        )


def _rrf_fuse(
    vector_hits: list[RetrievalHit],
    bm25_hits: list[RetrievalHit],
    *,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalHit]:
    """Fuse two ranked hit lists using Reciprocal Rank Fusion.

    Duplicates (same ``chunk_id``) are merged: the RRF scores are
    summed and per-strategy ranks are combined.
    """
    # Build RRF scores per chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, dict] = {}

    # Vector list contributions
    for rank, hit in enumerate(vector_hits, start=1):
        rrf_v = 1.0 / (k + rank)
        rrf_scores[hit.chunk_id] += vector_weight * rrf_v
        if hit.chunk_id not in chunk_data:
            chunk_data[hit.chunk_id] = {
                "hit": hit,
                "ranks": dict(hit.ranks),
            }
        else:
            chunk_data[hit.chunk_id]["ranks"].update(hit.ranks)

    # BM25 list contributions
    for rank, hit in enumerate(bm25_hits, start=1):
        rrf_b = 1.0 / (k + rank)
        rrf_scores[hit.chunk_id] += bm25_weight * rrf_b
        if hit.chunk_id not in chunk_data:
            chunk_data[hit.chunk_id] = {
                "hit": hit,
                "ranks": dict(hit.ranks),
            }
        else:
            chunk_data[hit.chunk_id]["ranks"].update(hit.ranks)

    # Sort by fused score
    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )

    results: list[RetrievalHit] = []
    for chunk_id in sorted_ids[:top_k]:
        data = chunk_data[chunk_id]
        original = data["hit"]
        results.append(
            RetrievalHit(
                chunk_id=chunk_id,
                doc_id=original.doc_id,
                collection_id=original.collection_id,
                namespace=original.namespace,
                content=original.content,
                score=rrf_scores[chunk_id],
                ranks=data["ranks"],
                metadata=dict(original.metadata),
            )
        )

    return results
