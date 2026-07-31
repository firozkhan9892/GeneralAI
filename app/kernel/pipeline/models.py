"""Pipeline domain models — stage execution infrastructure."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorPolicy(str, Enum):
    """How the pipeline should behave when a stage errors."""

    ABORT = "abort"
    SKIP = "skip"
    RETRY = "retry"
    IGNORE = "ignore"


class PipelineStep(BaseModel):
    """Definition of a single step within a pipeline definition."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique step name within the pipeline")
    order: int = Field(..., ge=0, description="Execution order")
    description: str = Field(default="", description="What this step does")
    error_policy: ErrorPolicy = Field(
        default=ErrorPolicy.ABORT, description="Behaviour on error"
    )
    max_retries: int = Field(default=0, ge=0, description="Retry count for this step")
    timeout_s: int | None = Field(
        default=None, ge=1, description="Optional step-level timeout"
    )
    depends_on: tuple[str, ...] = Field(
        default_factory=tuple, description="Steps that must complete first"
    )


class PipelineDefinition(BaseModel):
    """A reusable pipeline definition — an ordered sequence of steps."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Pipeline name")
    version: str = Field(default="1.0.0", description="Semantic version")
    steps: tuple[PipelineStep, ...] = Field(..., description="Ordered step definitions")
    description: str = Field(default="", description="What this pipeline accomplishes")
    tags: tuple[str, ...] = Field(
        default_factory=tuple, description="Categorisation tags"
    )


class PipelineMetadata(BaseModel):
    """Mutable metadata accumulated during pipeline execution."""

    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="Execution start time"
    )
    completed_at: datetime | None = Field(
        default=None, description="Execution completion time"
    )
    stage_count: int = Field(default=0, ge=0, description="Total stages defined")
    current_stage: int = Field(
        default=0, ge=0, description="Index of the currently executing stage"
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered per stage"
    )


class PipelineContext(BaseModel):
    """The shared context object passed through every stage of the pipeline.

    This is the **only** data channel between stages (ADR-014).
    Each stage writes its output to the corresponding field.
    """

    session_id: str = Field(default="")
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)

    percept: Any = Field(default=None)
    intent: Any = Field(default=None)
    goal_hierarchy: Any = Field(default=None)
    plan: Any = Field(default=None)
    reasoning_trace: Any = Field(default=None)
    decision: Any = Field(default=None)
    capability: Any = Field(default=None)
    policy_verdict: Any = Field(default=None)
    skill_binding: Any = Field(default=None)
    tool_bindings: list[Any] = Field(default_factory=list)
    execution_results: list[Any] = Field(default_factory=list)
    reflection: Any = Field(default=None)
    experience: Any = Field(default=None)
    response: Any = Field(default=None)

    refinement_depth: int = Field(default=0, ge=0)
    max_refinement_depth: int = Field(default=3, ge=1)
    pending_clarification: Any | None = Field(default=None)
