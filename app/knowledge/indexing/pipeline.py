"""Indexing pipeline.

Orchestrates the document → chunk → embed → store flow.  Supports
batch indexing, incremental indexing (content-hash change detection),
and rebuild operations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.knowledge.analytics import KnowledgeAnalytics
from app.knowledge.base import Chunker, DocumentLoader, EmbeddingProvider, VectorStore
from app.knowledge.embeddings.cache import EmbeddingCache
from app.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    MetadataFilter,
    VectorSearchHit,
)

log = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestrates document ingestion, chunking, embedding, and storage.

    Parameters
    ----------
    loader:
        The document loader to use for parsing.
    chunker:
        The chunker to use for splitting documents.
    embedding_provider:
        The embedding provider to use for vectorising chunks.
    vector_store:
        The vector store to persist chunks and vectors.
    cache:
        Optional embedding cache for deduplication.
    analytics:
        Optional analytics recorder.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        cache: EmbeddingCache | None = None,
        analytics: KnowledgeAnalytics | None = None,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._provider = embedding_provider
        self._store = vector_store
        self._cache = cache
        self._analytics = analytics

    def ingest(
        self,
        content: bytes,
        *,
        source_uri: str,
        collection_id: str = "",
        namespace: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Parse, chunk, embed, and store a document.

        Returns the parsed :class:`KnowledgeDocument`.
        """
        start = time.time()

        # 1. Parse
        doc = self._loader.load(
            content,
            source_uri=source_uri,
            metadata={
                **(metadata or {}),
                "collection_id": collection_id,
                "namespace": namespace,
            },
        )

        # 2. Chunk
        chunks = self._chunker.chunk(doc)
        # Update document with chunk IDs
        doc = doc.model_copy(update={"chunk_ids": tuple(c.chunk_id for c in chunks)})

        # 3. Embed (with caching)
        texts = [c.content for c in chunks]
        vectors = self._embed_with_cache(texts)

        # 4. Store
        self._store.add(chunks, vectors)

        elapsed_ms = (time.time() - start) * 1000
        if self._analytics:
            self._analytics.record_embedding_created(len(vectors))
            self._analytics.record_indexing_latency(elapsed_ms)
            self._analytics.set_index_size(self._store.count())

        log.info(
            "Ingested %s (%d chunks, %.1fms)",
            source_uri,
            len(chunks),
            elapsed_ms,
        )
        return doc

    def ingest_document(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Chunk, embed, and store an already-parsed document.

        Returns the list of created chunks.
        """
        start = time.time()
        chunks = self._chunker.chunk(document)
        texts = [c.content for c in chunks]
        vectors = self._embed_with_cache(texts)
        self._store.add(chunks, vectors)

        elapsed_ms = (time.time() - start) * 1000
        if self._analytics:
            self._analytics.record_embedding_created(len(vectors))
            self._analytics.record_indexing_latency(elapsed_ms)
            self._analytics.set_index_size(self._store.count())

        return chunks

    def batch_ingest(
        self,
        documents: list[tuple[bytes, str]],
        *,
        collection_id: str = "",
        namespace: str = "",
    ) -> list[KnowledgeDocument]:
        """Ingest multiple documents.

        *documents* is a list of ``(content_bytes, source_uri)`` tuples.
        """
        return [
            self.ingest(
                content,
                source_uri=uri,
                collection_id=collection_id,
                namespace=namespace,
            )
            for content, uri in documents
        ]

    def rebuild(
        self,
        documents: list[tuple[bytes, str]],
        *,
        collection_id: str = "",
        namespace: str = "",
    ) -> list[KnowledgeDocument]:
        """Clear the store and re-index all documents."""
        self._store.clear()
        return self.batch_ingest(
            documents,
            collection_id=collection_id,
            namespace=namespace,
        )

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: tuple[MetadataFilter, ...] = (),
    ) -> list[VectorSearchHit]:
        """Search the vector store for similar chunks.

        Applies metadata filtering and returns ranked results.
        """
        start = time.time()
        results = self._store.search(query_vector, top_k=top_k, filters=filters)
        elapsed_ms = (time.time() - start) * 1000
        if self._analytics:
            self._analytics.record_search_latency(elapsed_ms)
        return results

    def _embed_with_cache(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, using the cache when available."""
        if not texts:
            return []

        provider_name = self._provider.name
        model = getattr(self._provider, "model_name", "")

        # Check cache
        cached: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if self._cache is not None:
            cached = self._cache.get_many(provider_name, model, texts)
            for idx, text in enumerate(texts):
                if idx not in cached:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
                    if self._analytics:
                        self._analytics.record_cache_miss()
                else:
                    if self._analytics:
                        self._analytics.record_cache_hit()
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Embed uncached texts
        new_vectors: list[list[float]] = []
        if uncached_texts:
            new_vectors = self._provider.embed(uncached_texts)

        # Populate cache
        if self._cache is not None and new_vectors:
            self._cache.put_many(provider_name, model, uncached_texts, new_vectors)

        # Assemble result in original order
        result: list[list[float]] = [[] for _ in texts]

        # First fill from cache
        for idx, vector in cached.items():
            result[idx] = vector

        # Then fill from newly computed
        for local_idx, global_idx in enumerate(uncached_indices):
            result[global_idx] = new_vectors[local_idx]

        return result

    @property
    def store(self) -> VectorStore:
        """Return the underlying vector store."""
        return self._store

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the underlying embedding provider."""
        return self._provider
