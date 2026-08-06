"""Enterprise Knowledge & RAG subsystem.

Provides the abstract pipeline-stage contracts, domain models, typed
registries, exceptions, events, and idempotent DI wiring for the
knowledge subsystem.  Phase 13d adds the retrieval engine: hybrid
retrieval (vector + BM25 + RRF), metadata filtering, query rewriting,
multi-query expansion, context compression, reranking, citation
building, and a unified retrieval pipeline.
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
    KnowledgeCacheError,
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
    RetrievalAnalyticsEntry,
    RetrievalAnalyticsSummary,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
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
from app.knowledge.analytics import KnowledgeAnalytics, AnalyticsSummary
from app.knowledge.embeddings.cache import EmbeddingCache, CacheStats
from app.knowledge.embeddings.mock import MockEmbeddingProvider
from app.knowledge.indexing.pipeline import IndexingPipeline
from app.knowledge.vectorstores.in_memory import InMemoryVectorStore
from app.knowledge.retrieval.bm25 import BM25Index, BM25Retriever
from app.knowledge.retrieval.citations import DefaultCitationBuilder
from app.knowledge.retrieval.compress import IdentityCompressor
from app.knowledge.retrieval.filter import evaluate_filter, evaluate_filters
from app.knowledge.retrieval.hybrid import HybridRetriever
from app.knowledge.retrieval.multiquery import MultiQueryRetriever
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.knowledge.retrieval.rerank import IdentityReranker
from app.knowledge.retrieval.rewrite import IdentityQueryRewriter
from app.knowledge.retrieval.vector import VectorRetriever

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
    "KnowledgeCacheError",
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
    # Embeddings & vector stores (13c)
    "KnowledgeAnalytics",
    "AnalyticsSummary",
    "EmbeddingCache",
    "CacheStats",
    "MockEmbeddingProvider",
    "IndexingPipeline",
    "InMemoryVectorStore",
    # Retrieval engine (13d)
    "BM25Index",
    "BM25Retriever",
    "VectorRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "IdentityQueryRewriter",
    "IdentityCompressor",
    "IdentityReranker",
    "DefaultCitationBuilder",
    "RetrievalPipeline",
    "evaluate_filter",
    "evaluate_filters",
    "RetrievalResult",
    "RetrievalAnalyticsEntry",
    "RetrievalAnalyticsSummary",
]
