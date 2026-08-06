"""Dense vector retriever.

Embeds the query using the collection's embedding provider and
searches the vector store for nearest-neighbour matches.  Results
carry ``ranks={"vector": score}`` for downstream fusion.
"""

from __future__ import annotations

import logging

from app.knowledge.base import (
    EmbeddingProvider,
    RetrievalContext,
    Retriever,
    VectorStore,
)
from app.knowledge.models import RetrievalHit, RetrievalQuery

log = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """Dense vector retrieval using an embedding provider and vector store.

    The retriever embeds the (possibly rewritten) query via the
    provided :class:`EmbeddingProvider`, then searches the
    :class:`VectorStore` for the nearest neighbours.
    """

    name: str = "vector"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(
        self, query: RetrievalQuery, *, context: RetrievalContext
    ) -> list[RetrievalHit]:
        """Embed the query and search the vector store.

        Args:
            query: The retrieval query (uses ``rewritten_query`` or
                ``query``).
            context: Collection context including the vector store.

        Returns:
            Hits ranked by vector similarity score.
        """
        provider = self._embedding_provider
        store = self._vector_store or context.vector_store

        if provider is None or store is None:
            log.debug("VectorRetriever: missing provider or store, returning empty")
            return []

        q = query.rewritten_query or query.query

        try:
            vectors = await provider.embed_async([q])
            if not vectors:
                return []
            query_vector = vectors[0]
        except Exception:
            log.exception("VectorRetriever: embedding failed")
            return []

        try:
            search_hits = await store.search_async(
                query_vector,
                top_k=query.top_k,
                filters=query.filters,
            )
        except Exception:
            log.exception("VectorRetriever: vector search failed")
            return []

        results: list[RetrievalHit] = []
        for hit in search_hits:
            results.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    collection_id=hit.collection_id,
                    namespace=hit.namespace,
                    content="",  # populated by pipeline from chunk store
                    score=hit.score,
                    ranks={"vector": hit.score},
                    metadata=dict(hit.metadata),
                )
            )
        return results
