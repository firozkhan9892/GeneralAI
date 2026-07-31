"""Goal domain models — stage 3 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.kernel.intent.models import IntentType


class GoalStatus(str, Enum):
    """Lifecycle state of a goal."""

    PROPOSED = "proposed"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    SUPERSEDED = "superseded"
    PAUSED = "paused"


class GoalPriority(int, Enum):
    """Numeric priority for ordering goals (higher = more important)."""

    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 0


class GoalType(str, Enum):
    """Semantic category of a goal derived from the user's intent."""

    QUESTION = "question"
    TASK = "task"
    PROJECT = "project"
    LEARNING = "learning"
    EXPLORATION = "exploration"
    DEBUGGING = "debugging"
    SYSTEM = "system"


class Goal(BaseModel):
    """A single goal within a goal hierarchy.

    Goals are the system's interpretation of what must be achieved
    to satisfy the user's intent.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Unique goal identifier")
    parent_id: str | None = Field(
        default=None, description="Parent goal ID for hierarchy"
    )
    session_id: str = Field(default="", description="Owning session identifier")
    description: str = Field(..., description="Human-readable goal description")
    goal_type: GoalType = Field(default=GoalType.TASK, description="Semantic category")
    intent_type: IntentType = Field(
        default=IntentType.UNKNOWN, description="Originating intent type"
    )
    status: GoalStatus = Field(
        default=GoalStatus.PROPOSED, description="Current lifecycle state"
    )
    priority: GoalPriority = Field(
        default=GoalPriority.NORMAL, description="Execution priority"
    )
    sub_goal_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Ordered child goal IDs"
    )
    task_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Associated task IDs"
    )
    acceptance_criteria: tuple[str, ...] = Field(
        default_factory=tuple, description="Conditions for completion"
    )
    progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Completion fraction"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extensible metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Goal):
            return NotImplemented
        return self.model_dump(exclude={"created_at", "updated_at"}) == (
            other.model_dump(exclude={"created_at", "updated_at"})
        )


class GoalHierarchy(BaseModel):
    """Tree structure of goals for a single session.

    All goals are accessible both as a root/children tree and as
    a flat map for O(1) lookup by ID.
    """

    model_config = ConfigDict(frozen=True)

    root: Goal = Field(..., description="Root goal of the hierarchy")
    children: tuple[Goal, ...] = Field(
        default_factory=tuple, description="Direct sub-goals of the root"
    )
    all_goals: dict[str, Goal] = Field(
        default_factory=dict, description="Flat map of all goals by ID"
    )
