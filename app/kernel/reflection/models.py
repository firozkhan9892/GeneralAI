"""Reflection domain models — stage 12 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorType(str, Enum):
    """Category of a detected error or quality issue."""

    HALLUCINATION = "hallucination"
    CONTRADICTION = "contradiction"
    INCOMPLETE = "incomplete"
    IRRELEVANT = "irrelevant"
    LOGICAL_GAP = "logical_gap"
    FACTUAL_ERROR = "factual_error"
    STYLE = "style"
    SAFETY = "safety"


class ReflectionScore(BaseModel):
    """A single scored dimension within a reflection evaluation."""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(..., description="Name of the evaluated dimension")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalised score")
    weight: float = Field(
        default=1.0, ge=0.0, description="Relative importance of this dimension"
    )


class ErrorDetail(BaseModel):
    """A single detected error or issue in the output."""

    model_config = ConfigDict(frozen=True)

    type: ErrorType = Field(..., description="Category of the error")
    severity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How severe the error is"
    )
    description: str = Field(default="", description="Human-readable description")
    location: str = Field(
        default="", description="Where in the trace the error occurred"
    )
    suggested_fix: str | None = Field(
        default=None, description="Optional suggested correction"
    )


class Refinement(BaseModel):
    """A suggested refinement to fix a detected problem."""

    model_config = ConfigDict(frozen=True)

    target_step_id: str | None = Field(
        default=None, description="Specific step to refine, if applicable"
    )
    modification: str = Field(
        default="", description="Description of the change to make"
    )
    priority: int = Field(
        default=0, ge=0, description="Priority (higher = more urgent)"
    )


class ReflectionRequest(BaseModel):
    """Input to the reflection engine specifying what to evaluate."""

    model_config = ConfigDict(frozen=True)

    output: Any = Field(..., description="The output to evaluate")
    trace: Any = Field(
        default=None, description="Associated trace from the producing engine"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Evaluation context"
    )
    mode: str = Field(
        default="standard", description="Reflection mode (standard, deep, fast)"
    )


class ReflectionReport(BaseModel):
    """Comprehensive report from the reflection engine.

    Includes both quantitative scores and qualitative error details.
    """

    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Aggregate quality score"
    )
    dimension_scores: tuple[ReflectionScore, ...] = Field(
        default_factory=tuple, description="Per-dimension scores"
    )
    errors: tuple[ErrorDetail, ...] = Field(
        default_factory=tuple, description="Detected errors"
    )
    refinements: tuple[Refinement, ...] = Field(
        default_factory=tuple, description="Suggested improvements"
    )
    verdict: str = Field(
        default="pass", description="Overall verdict (pass, fail, needs_review)"
    )
    token_cost: int = Field(
        default=0, ge=0, description="Tokens consumed for reflection"
    )
    duration_ms: int = Field(default=0, ge=0, description="Reflection duration")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the report was generated"
    )
