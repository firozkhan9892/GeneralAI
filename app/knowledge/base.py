"""Abstract base classes for the knowledge pipeline stages.

Each stage of the RAG pipeline is defined here as an abstract
contract: loaders parse raw bytes into documents, chunkers split
documents, embedding providers produce vectors, vector stores persist
and search them, and retrievers / rewriters / compressors / rerankers /
citation builders assemble the final retrieval result.

Providers follow the :class:`BaseLLMProvider` philosophy: a synchronous
core method is abstract, and the asynchronous variant is provided on
top via ``asyncio.to_thread`` unless overridden.  Retrieval stages are
naturally asynchronous and declare their ``async`` methods directly.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.knowledge.models import (
    CitationResult,
    DocumentFormat,
    EmbeddingModelInfo,
    KnowledgeChunk,
    KnowledgeDocument,
    MetadataFilter,
    RetrievalHit,
    RetrievalQuery,
    VectorSearchHit,
)


class DocumentLoader(ABC):
    """Parses raw bytes into a :class:`KnowledgeDocument`.

    Loaders are format-specific: one instance per format.  Heavy
    dependencies (e.g. ``pypdf``, ``python-docx``) should be imported
    lazily inside :meth:`load` so importing this module never breaks an
    environment that lacks them.
    """

    format: DocumentFormat = DocumentFormat.TXT

    @abstractmethod
    def load(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Parse *content* into a :class:`KnowledgeDocument`.

        Args:
            content: Raw bytes of the source file.
            source_uri: Origin URI recorded on the document.
            metadata: Optional document metadata to attach.

        Returns:
            The parsed document.

        Raises:
            KnowledgeIngestionError: If the content cannot be parsed.
        """

    async def load_async(
        self,
        content: bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Asynchronously parse *content* (thread-offloaded)."""
        return await asyncio.to_thread(
            self.load, content, source_uri=source_uri, metadata=metadata
        )


class Chunker(ABC):
    """Splits a :class:`KnowledgeDocument` into :class:`KnowledgeChunk` records."""

    name: str = ""

    @abstractmethod
    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Split *document* into an ordered list of chunks.

        Args:
            document: The document to chunk.

        Returns:
            Chunks with monotonically increasing ``chunk_index`` values.
        """

    async def chunk_async(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Asynchronously chunk *document* (thread-offloaded)."""
        return await asyncio.to_thread(self.chunk, document)


class EmbeddingProvider(ABC):
    """Abstract contract for embedding providers.

    Mirrors :class:`BaseLLMProvider`: a synchronous ``embed`` core with
    an ``embed_async`` offload.  Providers also describe their output
    dimensionality so vector stores can be configured accordingly.
    """

    name: str = ""
    dimensions: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* into a list of vectors.

        Args:
            texts: Input texts to embed.

        Returns:
            One vector per input text, each of length
            ``self.dimensions``.

        Raises:
            KnowledgeError: On any embedding failure.
        """

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed *texts* (thread-offloaded)."""
        return await asyncio.to_thread(self.embed, texts)

    @abstractmethod
    def model_info(self) -> EmbeddingModelInfo:
        """Describe the provider and its default model."""


class VectorStore(ABC):
    """Abstract contract for vector persistence backends.

    Stores map chunks to their embedding vectors and answer
    approximate nearest-neighbour searches.  Keys always include the
    namespace and collection so multiple collections never collide.
    """

    name: str = ""
    dimensions: int = 0

    @abstractmethod
    def add(self, chunks: list[KnowledgeChunk], vectors: list[list[float]]) -> None:
        """Store *vectors* for the given *chunks*.

        Args:
            chunks: Chunks whose vectors are stored.
            vectors: One vector per chunk, length ``self.dimensions``.
        """

    @abstractmethod
    def delete(self, chunk_ids: Iterable[str]) -> None:
        """Remove the vectors for the given *chunk_ids*."""

    @abstractmethod
    def delete_by_document(self, doc_id: str, namespace: str) -> None:
        """Remove all vectors belonging to *doc_id* in *namespace*."""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: tuple[MetadataFilter, ...] = (),
    ) -> list[VectorSearchHit]:
        """Return the *top_k* nearest vectors, optionally filtered.

        Args:
            vector: Query vector.
            top_k: Maximum number of hits to return.
            filters: Conjunctive metadata filters applied post-search.

        Returns:
            Matches ranked by descending similarity.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vectors."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored vectors."""

    async def add_async(
        self, chunks: list[KnowledgeChunk], vectors: list[list[float]]
    ) -> None:
        """Asynchronously store vectors (thread-offloaded)."""
        await asyncio.to_thread(self.add, chunks, vectors)

    async def delete_async(self, chunk_ids: Iterable[str]) -> None:
        """Asynchronously delete vectors (thread-offloaded)."""
        await asyncio.to_thread(self.delete, chunk_ids)

    async def search_async(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: tuple[MetadataFilter, ...] = (),
    ) -> list[VectorSearchHit]:
        """Asynchronously search (thread-offloaded)."""
        return await asyncio.to_thread(
            self.search, vector, top_k=top_k, filters=filters
        )


@dataclass(frozen=True)
class RetrievalContext:
    """Runtime bundle handed to retrieval stages.

    Retrievers and the stages around them receive a context rather than
    reaching into the container: the owning collection, an optional
    vector store, and the active metadata filters.  Later phases extend
    this with the BM25 index and an injectable clock.
    """

    namespace: str = ""
    collection_id: str = ""
    vector_store: VectorStore | None = None
    filters: tuple[MetadataFilter, ...] = field(default_factory=tuple)


class Retriever(ABC):
    """Retrieves candidate chunks for a query within a context."""

    name: str = ""

    @abstractmethod
    async def retrieve(
        self, query: RetrievalQuery, *, context: RetrievalContext
    ) -> list[RetrievalHit]:
        """Return ranked hits for *query* within *context*.

        Args:
            query: The (possibly rewritten) retrieval query.
            context: Collection / store / filter bundle.

        Returns:
            Hits ranked by descending relevance.
        """


class QueryRewriter(ABC):
    """Rewrites a raw query into a normalized, search-optimized form."""

    name: str = ""

    @abstractmethod
    async def rewrite(self, raw: str, *, context: RetrievalContext) -> str:
        """Return a rewritten query for *raw*.

        Args:
            raw: The raw user query.
            context: Retrieval context for conditioning.

        Returns:
            The rewritten query string.
        """


class ContextCompressor(ABC):
    """Compresses a hit set before it reaches the prompt."""

    name: str = ""

    @abstractmethod
    async def compress(
        self, hits: list[RetrievalHit], *, query: str
    ) -> list[RetrievalHit]:
        """Return a reduced hit set for *query*.

        Args:
            hits: The candidate hits.
            query: The query to compress against.

        Returns:
            The reduced hit set.
        """


class Reranker(ABC):
    """Reranks a hit set, overriding the fused relevance scores."""

    name: str = ""

    @abstractmethod
    async def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Return *hits* reranked for *query*.

        Args:
            query: The retrieval query.
            hits: The hits to rerank.

        Returns:
            The reranked hit list.
        """


class CitationBuilder(ABC):
    """Builds citations and source references from retrieval hits.

    A pure function of the hits: deterministic and unit-testable.
    """

    name: str = ""

    @abstractmethod
    def build(self, hits: list[RetrievalHit]) -> CitationResult:
        """Build citation and source metadata for *hits*.

        Args:
            hits: The final hit set (post rerank).

        Returns:
            Citations and deduplicated source references.
        """
