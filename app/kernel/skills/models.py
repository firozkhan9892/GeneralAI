"""Skills domain models — stage 9-10 of the cognitive pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillRequirement(BaseModel):
    """A declared requirement that a specific skill must be available."""

    model_config = ConfigDict(frozen=True)

    skill_name: str = Field(..., description="Name of the required skill")
    optional: bool = Field(
        default=False, description="If True, execution may proceed without it"
    )
    min_version: str | None = Field(
        default=None, description="Minimum required version string"
    )


class ToolRequirement(BaseModel):
    """A tool required by a skill."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Name of the required tool")
    optional: bool = Field(
        default=False, description="If True, the skill can proceed without this tool"
    )


class SkillDescriptor(BaseModel):
    """Descriptor for a skill registered with the system."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique skill name")
    description: str = Field(default="", description="Human-readable description")
    version: str = Field(default="0.1.0", description="Semantic version")
    required_tools: tuple[ToolRequirement, ...] = Field(
        default_factory=tuple, description="Tools the skill depends on"
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for input parameters"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for output values"
    )
    estimated_cost: int = Field(default=0, ge=0, description="Estimated token cost")
    required_capabilities: tuple[str, ...] = Field(
        default_factory=tuple, description="Capabilities this skill requires"
    )


class Skill(BaseModel):
    """A fully resolved skill ready for execution, bound to its parameters."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Skill name")
    descriptor: SkillDescriptor = Field(..., description="The registered descriptor")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Resolved input parameters"
    )
    resolved_tools: tuple[str, ...] = Field(
        default_factory=tuple, description="Tools resolved for this invocation"
    )


class SkillResult(BaseModel):
    """Result of a skill execution."""

    model_config = ConfigDict(frozen=True)

    skill_name: str = Field(..., description="Name of the skill that executed")
    output: Any = Field(default=None, description="Execution output")
    duration_ms: int = Field(
        default=0, ge=0, description="Execution duration in milliseconds"
    )
    token_cost: int = Field(default=0, ge=0, description="Tokens consumed")
    success: bool = Field(default=True, description="Whether execution succeeded")
    error: str | None = Field(default=None, description="Error message on failure")
