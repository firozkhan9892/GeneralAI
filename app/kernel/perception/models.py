"""Perception domain models — stage 1 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModalityType(str, Enum):
    """Detected input modality for a message."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TOOL_RESULT = "tool_result"
    SYSTEM_EVENT = "system_event"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class Entity(BaseModel):
    """An extracted entity from raw input."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(..., description="Entity type (date, url, email, etc.)")
    value: str = Field(..., description="Extracted value")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    position: tuple[int, int] | None = Field(
        default=None, description="Start/end offset in source text"
    )


class QualityScore(BaseModel):
    """Input quality assessment scores."""

    model_config = ConfigDict(frozen=True)

    overall: float = Field(default=1.0, ge=0.0, le=1.0, description="Aggregate quality")
    length_sufficiency: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Is the input long enough"
    )
    completeness: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Does the input feel complete"
    )
    coherence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Is the input internally coherent"
    )


class RawMessage(BaseModel):
    """Raw, fully unprocessed input message as it arrives from the source."""

    model_config = ConfigDict(frozen=True)

    content: str | bytes = Field(..., description="Raw content")
    modality: ModalityType = Field(
        default=ModalityType.TEXT, description="Detected or declared modality"
    )
    source: str = Field(default="user", description="Source label")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Unfiltered source metadata"
    )


class Percept(BaseModel):
    """Structured, normalized output of the perception engine.

    This is the canonical representation passed to the intent engine.
    """

    model_config = ConfigDict(frozen=True)

    raw: RawMessage = Field(..., description="Original raw message")
    modality: ModalityType = Field(..., description="Final detected modality")
    normalized_content: str = Field(default="", description="Cleaned / normalised text")
    language: str = Field(default="en", description="Detected language code (BCP-47)")
    entities: tuple[Entity, ...] = Field(
        default_factory=tuple, description="Extracted entities"
    )
    quality: QualityScore = Field(
        default_factory=QualityScore, description="Input quality assessment"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the percept was built"
    )
