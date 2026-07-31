"""Tasks domain models — fine-grained execution units.

Tasks represent individual units of work spawned during plan execution.
They are smaller in scope than goals and map 1:1 to a single tool or
skill invocation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Lifecycle state of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class Task(BaseModel):
    """A single executable task spawned during plan execution."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Unique task identifier")
    session_id: str = Field(default="", description="Owning session identifier")
    goal_id: str = Field(default="", description="Parent goal identifier")
    plan_step_order: int = Field(
        default=0, ge=0, description="Corresponding step order in the plan"
    )
    action_type: str = Field(
        default="", description="Type of action (skill_execute, tool_call, etc.)"
    )
    action_name: str = Field(
        default="", description="Name of the skill or tool to invoke"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters"
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    depends_on: tuple[str, ...] = Field(
        default_factory=tuple, description="Task IDs that must complete first"
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, ge=0, description="Maximum allowed retries")
    token_budget: int = Field(default=0, ge=0, description="Token budget for this task")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary task metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


class TaskResult(BaseModel):
    """Outcome of a single task execution."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default="", description="The task that executed")
    status: TaskStatus = Field(..., description="Final status")
    output: Any = Field(default=None, description="Execution output value")
    duration_ms: int = Field(default=0, ge=0, description="Execution duration")
    token_cost: int = Field(default=0, ge=0, description="Tokens consumed")
    error: str | None = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, ge=0, description="Retries used")
    completed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Completion timestamp"
    )
