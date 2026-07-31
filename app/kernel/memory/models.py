"""Memory domain models — stage 13 of the cognitive pipeline.

The Memory Engine provides both short-term and long-term memory for
the system.  Short-term memory holds recent, session-scoped facts
that decay over time; long-term memory holds consolidated knowledge
that persists across sessions.  All models are frozen so records can
be safely shared, cached, and diffed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MemoryTier(str, Enum):
    """Tier of a memory record."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class MemoryRecord(BaseModel):
    """A single stored memory item."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Unique memory identifier")
    content: str = Field(default="", description="The remembered fact")
    tier: MemoryTier = Field(
        default=MemoryTier.SHORT_TERM, description="Which memory tier this lives in"
    )
    session_id: str = Field(default="", description="Owning session identifier")
    tags: tuple[str, ...] = Field(
        default_factory=tuple, description="Categorical tags for retrieval"
    )
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Relative importance (0..1)"
    )
    access_count: int = Field(
        default=0, ge=0, description="How many times this record was retrieved"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the record was created"
    )
    metadata: dict[str, object] = Field(
        default_factory=dict, description="Arbitrary record metadata"
    )


class MemoryQuery(BaseModel):
    """Parameters for retrieving and searching memory records."""

    model_config = ConfigDict(frozen=True)

    tier: MemoryTier | None = Field(default=None, description="Filter by memory tier")
    session_id: str | None = Field(default=None, description="Filter by owning session")
    tags: tuple[str, ...] | None = Field(
        default=None, description="Records must carry all of these tags"
    )
    keywords: tuple[str, ...] | None = Field(
        default=None, description="Keyword search terms"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum results to return"
    )


class MemorySearchHit(BaseModel):
    """A memory record paired with its relevance score."""

    model_config = ConfigDict(frozen=True)

    record: MemoryRecord = Field(..., description="The matching record")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance score (0..1)"
    )


class MemorySummary(BaseModel):
    """Aggregated statistics about the memory store."""

    model_config = ConfigDict(frozen=True)

    total_records: int = Field(default=0, ge=0, description="Total record count")
    short_term_count: int = Field(default=0, ge=0, description="Short-term count")
    long_term_count: int = Field(default=0, ge=0, description="Long-term count")
    average_importance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean importance across records"
    )
    tag_counts: dict[str, int] = Field(
        default_factory=dict, description="Records per tag"
    )
    recent_records: tuple[MemoryRecord, ...] = Field(
        default_factory=tuple, description="Newest records, newest first"
    )
