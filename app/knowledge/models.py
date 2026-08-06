"""Knowledge subsystem domain models.

The knowledge module provides enterprise document ingestion and
retrieval.  These models describe the documents, chunks, collections,
namespaces and embedding metadata that later phases index and search.

All models are frozen so records can be safely shared, cached, and
diffed — matching the memory and automation domain-model conventions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return the current aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _utcnow_factory() -> datetime:
    """Factory returning the current aware UTC timestamp."""
    return _utcnow()


class DocumentFormat(str, Enum):
    """Supported document formats for ingestion."""

    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    JSON = "json"


class IndexStatus(str, Enum):
    """Indexing state of a document."""

    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class CollectionStatus(str, Enum):
    """Lifecycle state of a knowledge collection."""

    ACTIVE = "active"
    INDEXING = "indexing"
    ARCHIVED = "archived"
    DELETED = "deleted"
    FAILED = "failed"


class KnowledgeDocument(BaseModel):
    """A single ingested document in a knowledge collection.

    Documents are immutable; updates create a new version.  The
    ``content_hash`` enables incremental indexing by detecting
    unchanged content.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Unique document identifier")
    collection_id: str = Field(..., description="Owning collection identifier")
    namespace: str = Field(default="", description="Isolating namespace")
    title: str = Field(default="", description="Document title")
    source_uri: str = Field(default="", description="Origin URI (file/path/url)")
    format: DocumentFormat = Field(..., description="Source format")
    content: str = Field(default="", description="Extracted raw text")
    content_hash: str = Field(default="", description="SHA-256 of normalized content")
    version: int = Field(default=1, ge=1, description="Monotonic document version")
    status: IndexStatus = Field(
        default=IndexStatus.PENDING, description="Indexing state"
    )
    created_at: datetime = Field(default_factory=_utcnow_factory)
    updated_at: datetime = Field(default_factory=_utcnow_factory)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary, filterable metadata"
    )
    chunk_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Chunks belonging to this document"
    )


class KnowledgeChunk(BaseModel):
    """A single chunk of a :class:`KnowledgeDocument`.

    Chunks are the atomic unit of indexing, embedding, and retrieval.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Source document identifier")
    collection_id: str = Field(..., description="Owning collection identifier")
    namespace: str = Field(default="", description="Isolating namespace")
    content: str = Field(..., description="Chunk text")
    chunk_index: int = Field(default=0, ge=0, description="Position within document")
    token_count: int = Field(default=0, ge=0, description="Approximate token count")
    content_hash: str = Field(default="", description="SHA-256 of chunk content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Inherited document metadata"
    )


class CollectionMetadata(BaseModel):
    """Metadata describing a knowledge collection.

    Collections group documents under a namespace and carry the
    indexing configuration (embedding model, vector store) used for
    their contents.
    """

    model_config = ConfigDict(frozen=True)

    collection_id: str = Field(..., description="Unique collection identifier")
    name: str = Field(..., description="Human-readable collection name")
    namespace: str = Field(default="", description="Isolating namespace")
    description: str = Field(default="", description="Optional description")
    status: CollectionStatus = Field(
        default=CollectionStatus.ACTIVE, description="Collection lifecycle state"
    )
    embedding_model: str = Field(
        default="", description="Embedding model used to index this collection"
    )
    vector_store: str = Field(
        default="", description="Vector store backend name for this collection"
    )
    created_at: datetime = Field(default_factory=_utcnow_factory)
    updated_at: datetime = Field(default_factory=_utcnow_factory)
    document_count: int = Field(default=0, ge=0, description="Number of documents")
    chunk_count: int = Field(default=0, ge=0, description="Number of indexed chunks")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary, filterable metadata"
    )


class NamespaceMetadata(BaseModel):
    """Metadata describing a knowledge namespace.

    Namespaces isolate collections (e.g. ``prod``, ``staging``,
    ``team-a``) so retrieval never crosses isolation boundaries.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique namespace name")
    description: str = Field(default="", description="Optional description")
    created_at: datetime = Field(default_factory=_utcnow_factory)
    collection_count: int = Field(
        default=0, ge=0, description="Number of collections in this namespace"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary namespace metadata"
    )


class EmbeddingVector(BaseModel):
    """Metadata record describing an embedded chunk.

    Phase 13 defines the metadata only — no embedding computation or
    vector payload is stored yet.  Later phases persist actual vectors
    in a :class:`VectorStore`; this record ties a chunk to its
    embedding provenance (model, dimensions, timestamps).
    """

    model_config = ConfigDict(frozen=True)

    vector_id: str = Field(..., description="Unique embedding record identifier")
    chunk_id: str = Field(..., description="Embedded chunk identifier")
    doc_id: str = Field(..., description="Source document identifier")
    collection_id: str = Field(..., description="Owning collection identifier")
    namespace: str = Field(default="", description="Isolating namespace")
    model: str = Field(default="", description="Embedding model used")
    dimensions: int = Field(default=0, ge=0, description="Embedding dimensionality")
    created_at: datetime = Field(default_factory=_utcnow_factory)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary embedding metadata"
    )


class KnowledgeEvent(BaseModel):
    """Event payload recorded against the knowledge subsystem.

    Mirrors the automation :class:`WorkflowEvent` convention: a frozen
    payload with the event type, routing identifiers (namespace,
    collection, document), an optional error, and a free-form data bag
    published on the application :class:`EventBus`.
    """

    model_config = ConfigDict(frozen=True)

    event_type: str = Field(..., description="Dotted event type (e.g. knowledge.*)")
    namespace: str = Field(default="", description="Affected namespace")
    collection_id: str = Field(default="", description="Affected collection")
    doc_id: str = Field(default="", description="Affected document")
    status: str = Field(default="", description="Status value (e.g. index status)")
    error: str | None = Field(default=None, description="Error message on failure")
    timestamp: datetime = Field(default_factory=_utcnow_factory)
    data: dict[str, Any] = Field(default_factory=dict, description="Free-form payload")


class EmbeddingModelInfo(BaseModel):
    """Description of an embedding model exposed by a provider."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Provider name")
    provider: str = Field(..., description="Provider identifier")
    dimensions: int = Field(..., ge=0, description="Embedding dimensionality")
    model: str = Field(..., description="Model identifier")
    max_input_tokens: int | None = Field(
        default=None, ge=0, description="Maximum input token count"
    )


class MetadataFilter(BaseModel):
    """A single field-operator-value filter over chunk metadata."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(..., description="Metadata field name")
    op: str = Field(
        ...,
        description=(
            "Comparison operator: eq, neq, in, not_in, gt, gte, lt, lte, "
            "exists, contains"
        ),
    )
    value: Any = Field(default=None, description="Operand value")


class VectorSearchHit(BaseModel):
    """A single match returned by a :class:`VectorStore` search."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Matched chunk identifier")
    doc_id: str = Field(..., description="Source document identifier")
    namespace: str = Field(default="", description="Isolating namespace")
    collection_id: str = Field(..., description="Owning collection identifier")
    score: float = Field(..., description="Similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class RetrievalQuery(BaseModel):
    """A retrieval request against a knowledge collection."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Raw user query")
    rewritten_query: str = Field(default="", description="Rewritten/normalized query")
    namespace: str = Field(default="", description="Isolating namespace")
    collection_id: str = Field(default="", description="Target collection")
    filters: tuple[MetadataFilter, ...] = Field(
        default_factory=tuple, description="Conjunctive metadata filters"
    )
    strategy: str = Field(default="hybrid", description="vector|bm25|hybrid")
    top_k: int = Field(default=10, ge=1, description="Number of hits to return")
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Vector weight"
    )
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 weight")
    rerank: bool = Field(default=False, description="Apply reranker")
    compression: bool = Field(default=False, description="Apply context compression")
    multi_query: bool = Field(default=False, description="Expand into sub-queries")
    include_sources: bool = Field(default=True, description="Attach source references")


class RetrievalHit(BaseModel):
    """A single retrieved chunk with its relevance scores."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Retrieved chunk identifier")
    doc_id: str = Field(..., description="Source document identifier")
    collection_id: str = Field(..., description="Owning collection identifier")
    namespace: str = Field(default="", description="Isolating namespace")
    content: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Fused relevance score")
    ranks: dict[str, float] = Field(
        default_factory=dict, description="Per-strategy scores"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class Citation(BaseModel):
    """Attribution linking retrieved chunks back to their source."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(..., description="Stable citation identifier")
    doc_id: str = Field(..., description="Source document identifier")
    doc_title: str = Field(default="", description="Document title")
    chunk_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Covered chunk identifiers"
    )
    page: int | None = Field(default=None, description="Page number when known")
    source_uri: str = Field(default="", description="Origin URI")
    snippet: str = Field(default="", description="Short content excerpt")
    namespace: str = Field(default="", description="Isolating namespace")
    collection_id: str = Field(default="", description="Owning collection identifier")


class SourceReference(BaseModel):
    """Deduplicated source document attribution."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Source document identifier")
    doc_title: str = Field(default="", description="Document title")
    source_uri: str = Field(default="", description="Origin URI")
    namespace: str = Field(default="", description="Isolating namespace")
    collection_id: str = Field(default="", description="Owning collection identifier")
    version: int = Field(default=1, ge=1, description="Document version")
    confidence: float = Field(default=0.0, description="Max hit score")


class CitationResult(BaseModel):
    """Output of the citation building stage."""

    model_config = ConfigDict(frozen=True)

    citations: tuple[Citation, ...] = Field(
        default_factory=tuple, description="One citation per retrieved hit"
    )
    sources: tuple[SourceReference, ...] = Field(
        default_factory=tuple, description="Deduplicated source references"
    )


class RetrievalResult(BaseModel):
    """Final output of a retrieval pipeline execution."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(default="", description="Rewritten/normalized query")
    total: int = Field(default=0, ge=0, description="Total hits returned")
    hits: tuple[RetrievalHit, ...] = Field(
        default_factory=tuple, description="Ranked retrieval hits"
    )
    sources: tuple[SourceReference, ...] = Field(
        default_factory=tuple, description="Deduplicated source references"
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="Total retrieval latency in milliseconds"
    )
    strategy: str = Field(default="hybrid", description="Retrieval strategy used")
    citations: tuple[Citation, ...] = Field(
        default_factory=tuple, description="Citations for retrieved chunks"
    )
    analytics: dict[str, Any] | None = Field(
        default=None, description="Optional analytics snapshot"
    )


class RetrievalAnalyticsEntry(BaseModel):
    """A single recorded retrieval analytics event."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="The retrieval query")
    collection_id: str = Field(default="", description="Target collection")
    namespace: str = Field(default="", description="Isolating namespace")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Retrieval latency")
    hit_count: int = Field(default=0, ge=0, description="Number of hits returned")
    top_score: float = Field(default=0.0, description="Highest relevance score")
    avg_score: float = Field(default=0.0, description="Average relevance score")
    strategy: str = Field(default="hybrid", description="Strategy used")
    reranked: bool = Field(default=False, description="Whether reranking was applied")
    timestamp: datetime = Field(default_factory=_utcnow_factory)


class RetrievalAnalyticsSummary(BaseModel):
    """Aggregated retrieval analytics snapshot."""

    model_config = ConfigDict(frozen=True)

    total_queries: int = Field(default=0, ge=0, description="Total queries executed")
    avg_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Average query latency"
    )
    avg_hit_count: float = Field(
        default=0.0, ge=0.0, description="Average hits per query"
    )
    collections: dict[str, int] = Field(
        default_factory=dict, description="Query count per collection"
    )
    top_queries: tuple[tuple[str, int], ...] = Field(
        default_factory=tuple, description="Most frequent queries with counts"
    )
    recent: tuple[RetrievalAnalyticsEntry, ...] = Field(
        default_factory=tuple, description="Most recent analytics entries"
    )
