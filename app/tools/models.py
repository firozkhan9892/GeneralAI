"""Tool domain models.

These are the frozen, provider-agnostic value objects shared by every
part of the tool framework: the tool descriptor, its parameters, and
the result of an execution.  They deliberately carry no behaviour so
they can be safely serialised, cached, and diffed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCategory(str, Enum):
    """Logical grouping for tools."""

    BUILTIN = "builtin"
    FILE = "file"
    WEB = "web"
    SHELL = "shell"
    PYTHON = "python"
    HTTP = "http"


class ToolParameter(BaseModel):
    """A single declared parameter of a tool.

    ``param_type`` uses JSON-schema style names (``string``, ``integer``,
    ``number``, ``boolean``, ``object``, ``array``) so descriptors map
    cleanly onto LLM tool schemas and other external consumers.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Parameter name")
    description: str = Field(default="", description="Human-readable description")
    param_type: str = Field(
        default="string", description="JSON-schema style parameter type"
    )
    required: bool = Field(
        default=False, description="Whether the parameter is required"
    )
    default: Any = Field(default=None, description="Default value when absent")


class ToolMetadata(BaseModel):
    """Static description of a tool, suitable for introspection and LLM schemas."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique tool name")
    description: str = Field(default="", description="Human-readable description")
    category: ToolCategory = Field(
        default=ToolCategory.BUILTIN, description="Logical tool category"
    )
    version: str = Field(default="1.0.0", description="Semantic version")
    parameters: tuple[ToolParameter, ...] = Field(
        default_factory=tuple, description="Declared parameters"
    )
    timeout_s: float = Field(
        default=30.0, ge=0.1, description="Default execution timeout in seconds"
    )
    requires_confirmation: bool = Field(
        default=False, description="Whether invocation needs human confirmation"
    )
    sandboxable: bool = Field(
        default=False, description="Whether the tool can run inside a sandbox"
    )


class ToolResult(BaseModel):
    """Outcome of a single tool execution."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(default="", description="Name of the tool that ran")
    success: bool = Field(default=True, description="Whether execution succeeded")
    output: Any = Field(default=None, description="Tool output value")
    error: str | None = Field(default=None, description="Error message on failure")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary execution metadata"
    )
    execution_time: float = Field(
        default=0.0, ge=0.0, description="Wall-clock execution time in seconds"
    )
