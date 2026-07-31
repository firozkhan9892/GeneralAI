"""Response domain models — stage 15 of the cognitive pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OutputMessage(BaseModel):
    """Final output message returned to the caller after pipeline completion."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="", description="Output text content")
    format: str = Field(
        default="text", description="Output format (text, json, markdown, etc.)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Output metadata"
    )
    session_id: str = Field(default="", description="Owning session identifier")
    success: bool = Field(
        default=True, description="Whether the overall pipeline succeeded"
    )
    error: str | None = Field(default=None, description="Error message if unsuccessful")


class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="", description="Chunk content")
    chunk_type: str = Field(
        default="text", description="Chunk type (text, tool_call, error, etc.)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    finished: bool = Field(
        default=False, description="Whether this is the terminal chunk"
    )
