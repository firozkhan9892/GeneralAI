"""Knowledge subsystem configuration.

Frozen Pydantic settings that control default chunking parameters,
namespace policies, and cache sizes.  Loaded once at startup and
shared via the DI container.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.constants import (
    BM25_B,
    BM25_K1,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    FILTER_OVERSAMPLE,
    RRF_K,
)


class KnowledgeSettings(BaseModel):
    """Immutable runtime configuration for the knowledge subsystem."""

    model_config = ConfigDict(frozen=True)

    default_namespace: str = Field(
        default=DEFAULT_NAMESPACE,
        description="Namespace used when none is specified",
    )
    default_chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        ge=100,
        description="Default maximum characters per chunk",
    )
    default_chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        ge=0,
        description="Default overlap between consecutive chunks",
    )
    embedding_cache_size: int = Field(
        default=10_000,
        ge=0,
        description="Maximum entries in the embedding LRU cache",
    )
    index_workers: int = Field(
        default=2,
        ge=1,
        description="Number of concurrent background indexing workers",
    )
    keep_versions: int = Field(
        default=3,
        ge=1,
        description="Number of document versions to retain",
    )
    bm25_k1: float = Field(
        default=BM25_K1,
        ge=0.0,
        description="BM25 term frequency saturation parameter",
    )
    bm25_b: float = Field(
        default=BM25_B,
        ge=0.0,
        le=1.0,
        description="BM25 length normalization parameter",
    )
    rrf_k: int = Field(
        default=RRF_K,
        ge=1,
        description="Reciprocal Rank Fusion smoothing constant",
    )
    filter_oversample: int = Field(
        default=FILTER_OVERSAMPLE,
        ge=1,
        description="Candidate multiplier before metadata filtering",
    )
