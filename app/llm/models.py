"""Unified LLM provider models.

These are the provider-agnostic request/response types exchanged between
the rest of the application and any :class:`BaseLLMProvider`.  Every
provider translates its own wire format into these models at the
boundary, so provider-specific code never leaks beyond the provider
layer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Standardised message role for chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a chat conversation."""

    model_config = ConfigDict(frozen=True)

    role: Role = Field(..., description="Message role")
    content: str = Field(default="", description="Message content text")
    name: str | None = Field(default=None, description="Sender name for tool messages")
    tool_call_id: str | None = Field(
        default=None, description="Tool call id that produced this message"
    )


class ToolCall(BaseModel):
    """A function invocation requested by the model."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Unique tool call identifier")
    name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Parsed tool arguments"
    )


class ToolDefinition(BaseModel):
    """A tool definition presented to the model for function calling."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema of tool parameters"
    )


class Usage(BaseModel):
    """Token usage reported by a provider."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0, description="Prompt token count")
    completion_tokens: int = Field(
        default=0, ge=0, description="Completion token count"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total token count")


class ResponseFormat(BaseModel):
    """Structured output request — constrains the response to a shape."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(
        default="text",
        pattern="^(text|json|json_schema)$",
        description="Response format type",
    )
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for json_schema format"
    )


class ModelInfo(BaseModel):
    """Metadata about a specific model exposed by a provider."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider identifier")
    supports_streaming: bool = Field(
        default=False, description="Whether the model can stream"
    )
    supports_tools: bool = Field(
        default=False, description="Whether the model supports function calling"
    )
    supports_json: bool = Field(
        default=False, description="Whether the model supports structured JSON"
    )
    max_context_tokens: int = Field(
        default=4096, ge=1, description="Maximum context window in tokens"
    )
    max_output_tokens: int = Field(
        default=1024, ge=1, description="Maximum output tokens"
    )
    cost_per_1k_tokens: float = Field(
        default=0.0, ge=0.0, description="Approximate cost per 1k tokens (USD)"
    )


class ChatRequest(BaseModel):
    """A complete request sent to an LLM provider."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...] = Field(
        ..., description="Conversation messages in order"
    )
    model: str = Field(..., description="Model identifier to use")
    tools: tuple[ToolDefinition, ...] | None = Field(
        default=None, description="Available tools for function calling"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum output tokens"
    )
    stop: tuple[str, ...] = Field(default_factory=tuple, description="Stop sequences")
    response_format: ResponseFormat | None = Field(
        default=None, description="Requested output format"
    )
    stream: bool = Field(default=False, description="Whether to stream output")
    timeout_s: float = Field(
        default=60.0, gt=0, description="Request timeout in seconds"
    )


class ChatResponse(BaseModel):
    """A complete response received from an LLM provider."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="", description="Response text content")
    tool_calls: tuple[ToolCall, ...] = Field(
        default_factory=tuple, description="Tool calls requested by the model"
    )
    finish_reason: str = Field(default="stop", description="Why generation finished")
    usage: Usage = Field(
        default_factory=Usage, description="Token usage for this response"
    )
    model: str = Field(..., description="Model that produced the response")
    provider: str = Field(..., description="Provider that produced the response")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the response was created"
    )


class StreamChunk(BaseModel):
    """A single streaming chunk emitted by a provider."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="", description="Text delta for this chunk")
    tool_calls: tuple[ToolCall, ...] = Field(
        default_factory=tuple, description="Tool call deltas in this chunk"
    )
    finish_reason: str | None = Field(
        default=None, description="Finish reason if generation completed"
    )
    usage: Usage | None = Field(
        default=None, description="Final usage if this chunk completes generation"
    )
    model: str = Field(..., description="Model that produced the chunk")
    provider: str = Field(..., description="Provider that produced the chunk")
