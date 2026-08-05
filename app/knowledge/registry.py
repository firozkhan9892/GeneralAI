"""Typed registries for the knowledge pipeline stages.

Each stage registry is a thin, thread-safe wrapper around
:class:`BaseRegistry` restricted to the relevant ABC type.  Registries
provide duplicate-protected registration, safe unregistration,
enumeration, counts, and immutable snapshots for safe cross-thread
sharing.

The ``KnowledgeRegistry`` generic base adds snapshot helpers on top of
:class:`BaseRegistry`; the concrete registries bind it to a stage type
so callers get static type checking.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from app.core.registry.base_registry import BaseRegistry
from app.knowledge.base import (
    Chunker,
    CitationBuilder,
    ContextCompressor,
    DocumentLoader,
    EmbeddingProvider,
    QueryRewriter,
    Reranker,
    Retriever,
    VectorStore,
)

T = TypeVar("T")


class KnowledgeRegistry(BaseRegistry[T], Generic[T]):
    """Thread-safe generic registry for knowledge components.

    Adds immutable snapshot helpers to :class:`BaseRegistry`.
    """

    def items(self) -> dict[str, T]:
        """Return an immutable snapshot of ``key -> item`` entries.

        The returned dictionary is a copy; mutating it never affects
        the registry.
        """
        return dict(zip(self.keys(), self.values(), strict=True))

    def snapshot(self) -> tuple[tuple[str, T], ...]:
        """Return an immutable ``(key, item)`` snapshot of all entries."""
        return tuple(self.items().items())


class LoaderRegistry(KnowledgeRegistry[DocumentLoader]):
    """Registry of document loaders, keyed by format name."""


class ChunkerRegistry(KnowledgeRegistry[Chunker]):
    """Registry of chunking strategies, keyed by strategy name."""


class EmbeddingProviderRegistry(KnowledgeRegistry[EmbeddingProvider]):
    """Registry of embedding providers, keyed by provider name."""


class VectorStoreRegistry(KnowledgeRegistry[VectorStore]):
    """Registry of vector store instances, keyed by store name."""


class RetrieverRegistry(KnowledgeRegistry[Retriever]):
    """Registry of retrievers, keyed by retriever name."""


class QueryRewriterRegistry(KnowledgeRegistry[QueryRewriter]):
    """Registry of query rewriters, keyed by rewriter name."""


class ContextCompressorRegistry(KnowledgeRegistry[ContextCompressor]):
    """Registry of context compressors, keyed by compressor name."""


class RerankerRegistry(KnowledgeRegistry[Reranker]):
    """Registry of rerankers, keyed by reranker name."""


class CitationBuilderRegistry(KnowledgeRegistry[CitationBuilder]):
    """Registry of citation builders, keyed by builder name."""
