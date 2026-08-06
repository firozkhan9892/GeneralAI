"""Enterprise Knowledge & RAG subsystem.

Provides the abstract pipeline-stage contracts, domain models, typed
registries, exceptions, events, and idempotent DI wiring for the
knowledge subsystem.  Phase 13b adds concrete loaders, chunkers,
the format parser, collection/namespace registries, and knowledge
settings.
"""

from __future__ import annotations

from app.knowledge.base import (
    Chunker,
    CitationBuilder,
    ContextCompressor,
    DocumentLoader,
    EmbeddingProvider,
    QueryRewriter,
    Reranker,
    Retriever,
    RetrievalContext,
    VectorStore,
)
from app.knowledge.bootstrap import register_knowledge_components
from app.knowledge.config import KnowledgeSettings
from app.knowledge.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
)
from app.knowledge.events import (
    EVENT_KNOWLEDGE_COLLECTION_CREATED,
    EVENT_KNOWLEDGE_COLLECTION_DELETED,
    EVENT_KNOWLEDGE_COLLECTION_UPDATED,
    EVENT_KNOWLEDGE_DOCUMENT_DELETED,
    EVENT_KNOWLEDGE_DOCUMENT_INGESTED,
    EVENT_KNOWLEDGE_DOCUMENT_UPDATED,
    EVENT_KNOWLEDGE_INDEX_COMPLETED,
    EVENT_KNOWLEDGE_INDEX_FAILED,
    EVENT_KNOWLEDGE_INDEX_STARTED,
    EVENT_KNOWLEDGE_NAMESPACE_CREATED,
    EVENT_KNOWLEDGE_NAMESPACE_DELETED,
    EVENT_KNOWLEDGE_RETRIEVED,
)
from app.knowledge.exceptions import (
    KnowledgeChunkNotFoundError,
    KnowledgeCollectionNotFoundError,
    KnowledgeDocumentNotFoundError,
    KnowledgeDuplicateError,
    KnowledgeError,
    KnowledgeIndexError,
    KnowledgeIngestionError,
    KnowledgeNamespaceNotFoundError,
    KnowledgeUnsupportedFormatError,
    KnowledgeValidationError,
    KnowledgeVersionError,
)
from app.knowledge.models import (
    Citation,
    CitationResult,
    CollectionMetadata,
    CollectionStatus,
    DocumentFormat,
    EmbeddingModelInfo,
    EmbeddingVector,
    IndexStatus,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEvent,
    MetadataFilter,
    NamespaceMetadata,
    RetrievalHit,
    RetrievalQuery,
    SourceReference,
    VectorSearchHit,
)
from app.knowledge.registry import (
    ChunkerRegistry,
    CitationBuilderRegistry,
    ContextCompressorRegistry,
    EmbeddingProviderRegistry,
    KnowledgeRegistry,
    LoaderRegistry,
    QueryRewriterRegistry,
    RerankerRegistry,
    RetrieverRegistry,
    VectorStoreRegistry,
)
from app.knowledge.utils import compute_content_hash, estimate_token_count
from app.knowledge.collection_registry import CollectionRegistry
from app.knowledge.namespace_registry import NamespaceRegistry

__all__ = [
    # Base abstractions
    "DocumentLoader",
    "Chunker",
    "EmbeddingProvider",
    "VectorStore",
    "Retriever",
    "QueryRewriter",
    "ContextCompressor",
    "Reranker",
    "CitationBuilder",
    "RetrievalContext",
    # Bootstrapping
    "register_knowledge_components",
    # Config / constants
    "KnowledgeSettings",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_NAMESPACE",
    "CHARS_PER_TOKEN",
    # Utils
    "compute_content_hash",
    "estimate_token_count",
    # Events
    "EVENT_KNOWLEDGE_NAMESPACE_CREATED",
    "EVENT_KNOWLEDGE_NAMESPACE_DELETED",
    "EVENT_KNOWLEDGE_COLLECTION_CREATED",
    "EVENT_KNOWLEDGE_COLLECTION_UPDATED",
    "EVENT_KNOWLEDGE_COLLECTION_DELETED",
    "EVENT_KNOWLEDGE_DOCUMENT_INGESTED",
    "EVENT_KNOWLEDGE_DOCUMENT_UPDATED",
    "EVENT_KNOWLEDGE_DOCUMENT_DELETED",
    "EVENT_KNOWLEDGE_INDEX_STARTED",
    "EVENT_KNOWLEDGE_INDEX_COMPLETED",
    "EVENT_KNOWLEDGE_INDEX_FAILED",
    "EVENT_KNOWLEDGE_RETRIEVED",
    # Exceptions
    "KnowledgeError",
    "KnowledgeValidationError",
    "KnowledgeNamespaceNotFoundError",
    "KnowledgeCollectionNotFoundError",
    "KnowledgeDocumentNotFoundError",
    "KnowledgeChunkNotFoundError",
    "KnowledgeDuplicateError",
    "KnowledgeUnsupportedFormatError",
    "KnowledgeIngestionError",
    "KnowledgeIndexError",
    "KnowledgeVersionError",
    # Models
    "DocumentFormat",
    "IndexStatus",
    "CollectionStatus",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "CollectionMetadata",
    "NamespaceMetadata",
    "EmbeddingVector",
    "KnowledgeEvent",
    "EmbeddingModelInfo",
    "MetadataFilter",
    "VectorSearchHit",
    "RetrievalQuery",
    "RetrievalHit",
    "Citation",
    "SourceReference",
    "CitationResult",
    # Registries
    "KnowledgeRegistry",
    "LoaderRegistry",
    "ChunkerRegistry",
    "EmbeddingProviderRegistry",
    "VectorStoreRegistry",
    "RetrieverRegistry",
    "QueryRewriterRegistry",
    "ContextCompressorRegistry",
    "RerankerRegistry",
    "CitationBuilderRegistry",
    "CollectionRegistry",
    "NamespaceRegistry",
]
