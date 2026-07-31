"""Tools domain models — execution-level tool invocations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionType(str, Enum):
    """Where and how a tool is executed."""

    LOCAL = "local"
    REMOTE = "remote"
    SANDBOXED = "sandboxed"


class RateLimit(BaseModel):
    """Rate-limiting configuration for a tool."""

    model_config = ConfigDict(frozen=True)

    max_calls_per_minute: int = Field(
        default=60, ge=1, description="Maximum calls per minute"
    )
    max_calls_per_hour: int = Field(
        default=1000, ge=1, description="Maximum calls per hour"
    )


class ToolDescriptor(BaseModel):
    """Descriptor for a tool registered with the system."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique tool name")
    description: str = Field(default="", description="Human-readable description")
    version: str = Field(default="0.1.0", description="Semantic version")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for input parameters"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for output values"
    )
    execution_type: ExecutionType = Field(
        default=ExecutionType.LOCAL, description="How the tool runs"
    )
    timeout_s: int = Field(default=30, ge=1, description="Execution timeout in seconds")
    rate_limit: RateLimit | None = Field(
        default=None, description="Optional rate-limit configuration"
    )


class ToolRequest(BaseModel):
    """An invocation request for a specific tool."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Name of the tool to invoke")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution context"
    )


class ToolBinding(BaseModel):
    """A resolved tool binding — descriptor plus concrete parameters."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Resolved tool name")
    descriptor: ToolDescriptor = Field(
        ..., description="The registered tool descriptor"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Concrete invocation parameters"
    )


class ToolResult(BaseModel):
    """Result of a single tool execution."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Name of the tool that executed")
    output: Any = Field(default=None, description="Tool output value")
    duration_ms: int = Field(default=0, ge=0, description="Execution duration")
    token_cost: int = Field(default=0, ge=0, description="Tokens consumed")
    success: bool = Field(default=True, description="Whether execution succeeded")
    error: str | None = Field(default=None, description="Error message on failure")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When execution completed"
    )
