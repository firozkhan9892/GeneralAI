"""Experience domain models — cross-session learning system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.kernel.intent.models import IntentType


class LessonCategory(str, Enum):
    """Category of a learned lesson."""

    STRATEGY = "strategy"
    AVOIDANCE = "avoidance"
    PREFERENCE = "preference"
    OPTIMIZATION = "optimization"
    SAFETY = "safety"


class LessonLearned(BaseModel):
    """A single lesson extracted from a completed session.

    Lessons are the atomic unit of the experience system."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(..., description="What was learned")
    category: LessonCategory = Field(
        default=LessonCategory.STRATEGY, description="Category of lesson"
    )
    applicability: tuple[IntentType, ...] = Field(
        default_factory=tuple, description="Intent types this applies to"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in this lesson"
    )


class DecisionSummary(BaseModel):
    """A condensed summary of a decision made during a session."""

    model_config = ConfigDict(frozen=True)

    action_type: str = Field(..., description="Type of action chosen")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence at decision time"
    )
    success: bool | None = Field(
        default=None, description="Whether the action succeeded (None if unknown)"
    )


class Insight(BaseModel):
    """An aggregated insight derived from multiple experiences."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(..., description="Insight description")
    pattern: str = Field(default="", description="The recurring pattern observed")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Aggregate confidence"
    )
    supporting_experience_count: int = Field(
        default=0, ge=0, description="Number of experiences supporting this"
    )


class ExperienceQuery(BaseModel):
    """Query parameters for retrieving experience records."""

    model_config = ConfigDict(frozen=True)

    goal_types: tuple[IntentType, ...] | None = Field(
        default=None, description="Filter by goal intent types"
    )
    skills: tuple[str, ...] | None = Field(
        default=None, description="Filter by skills used"
    )
    success: bool | None = Field(default=None, description="Filter by success/failure")
    timeframe_hours: int | None = Field(
        default=None, ge=1, description="How far back to look in hours"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum results to return"
    )


class Experience(BaseModel):
    """A complete record of a single session experience.

    Captures goals, decisions, outcomes, and lessons for later
    retrieval and learning.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Unique experience identifier")
    session_id: str = Field(default="", description="Owning session identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the experience was recorded"
    )
    goal_type: IntentType = Field(
        default=IntentType.UNKNOWN, description="Primary goal intent type"
    )
    goal_description: str = Field(
        default="", description="What the session aimed to achieve"
    )
    plan_summary: str = Field(default="", description="Brief plan description")
    skills_used: tuple[str, ...] = Field(
        default_factory=tuple, description="Skills invoked during the session"
    )
    tools_used: tuple[str, ...] = Field(
        default_factory=tuple, description="Tools invoked during the session"
    )
    decisions: tuple[DecisionSummary, ...] = Field(
        default_factory=tuple, description="Key decisions made"
    )
    outcome_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall outcome quality"
    )
    success: bool = Field(default=True, description="Whether the session succeeded")
    failure_reason: str | None = Field(
        default=None, description="Why it failed if unsuccessful"
    )
    lessons: tuple[LessonLearned, ...] = Field(
        default_factory=tuple, description="Lessons extracted"
    )
    reflection_scores: dict[str, float] = Field(
        default_factory=dict, description="Reflection dimension scores"
    )
    token_cost: int = Field(default=0, ge=0, description="Total token cost")
    duration_ms: int = Field(default=0, ge=0, description="Total session duration")
