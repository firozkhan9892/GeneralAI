"""Agent runtime domain models.

The AgentRuntime is the execution brain of GeneralAI.  These frozen
models describe agent runs, per-step execution state, run summaries,
and final agent responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.kernel.decision.models import Decision
from app.kernel.experience.models import Experience
from app.kernel.goals.models import GoalHierarchy
from app.kernel.intent.models import Intent
from app.kernel.memory.models import MemorySummary
from app.kernel.planning.models import Plan
from app.kernel.policy.models import PolicyDecision
from app.kernel.reasoning.models import ReasoningTrace
from app.kernel.reflection.models import ReflectionReport
from app.kernel.response.models import OutputMessage


class AgentStatus(str, Enum):
    """Lifecycle state of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class AgentStepStatus(str, Enum):
    """Lifecycle state of a single plan step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentRunConfig(BaseModel):
    """Configuration controlling a single agent run."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(default="", description="Owning session identifier")
    max_iterations: int = Field(
        default=10, ge=1, le=100, description="Maximum loop iterations"
    )
    step_timeout_s: float = Field(
        default=30.0, ge=0.1, description="Per-step timeout in seconds"
    )
    overall_timeout_s: float = Field(
        default=120.0, ge=1.0, description="Overall run timeout in seconds"
    )
    max_retries: int = Field(
        default=2, ge=0, le=10, description="Retries per failed step"
    )
    fallback_tool: str = Field(
        default="echo", description="Tool used when no tool is selected"
    )
    memory_enabled: bool = Field(
        default=True, description="Record memory after each completed task"
    )
    reflection_enabled: bool = Field(
        default=True, description="Run reflection after plan execution"
    )
    experience_enabled: bool = Field(
        default=True, description="Record an experience after the run"
    )
    reasoning_enabled: bool = Field(
        default=True, description="Run the reasoning engine before planning"
    )


class AgentStep(BaseModel):
    """Outcome of executing a single plan step."""

    model_config = ConfigDict(frozen=True)

    order: int = Field(..., ge=0, description="Step order within the plan")
    skill_name: str = Field(..., description="Skill/step name from the plan")
    description: str = Field(default="", description="Human-readable description")
    status: AgentStepStatus = Field(
        default=AgentStepStatus.PENDING, description="Step outcome"
    )
    tool_name: str = Field(default="", description="Tool selected to run the step")
    tool_result: Any = Field(
        default=None, description="Raw output from the tool executor"
    )
    error: str | None = Field(default=None, description="Error if the step failed")
    retries: int = Field(default=0, ge=0, description="Retries consumed")
    memory_record_id: str | None = Field(
        default=None, description="Memory record written for this step"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary step metadata"
    )
    decision: Decision | None = Field(
        default=None, description="Decision that selected this step's action"
    )
    policy_verdict: PolicyDecision | None = Field(
        default=None, description="Policy verdict for the step action"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="When execution started"
    )
    completed_at: datetime | None = Field(
        default=None, description="When execution completed"
    )


class AgentRunSummary(BaseModel):
    """Aggregated statistics for an agent run."""

    model_config = ConfigDict(frozen=True)

    total_steps: int = Field(default=0, ge=0, description="Steps attempted")
    succeeded: int = Field(default=0, ge=0, description="Steps succeeded")
    failed: int = Field(default=0, ge=0, description="Steps failed")
    skipped: int = Field(default=0, ge=0, description="Steps skipped")
    retries: int = Field(default=0, ge=0, description="Total retries consumed")
    tools_invoked: tuple[str, ...] = Field(
        default_factory=tuple, description="Distinct tools invoked"
    )
    memory_records: int = Field(default=0, ge=0, description="Memory records written")
    duration_ms: int = Field(default=0, ge=0, description="Run duration (ms)")


class AgentRequest(BaseModel):
    """Input to the agent runtime."""

    model_config = ConfigDict(frozen=True)

    raw_input: str = Field(default="", description="Raw user input text")
    session_id: str = Field(default="", description="Owning session identifier")
    user_id: str | None = Field(default=None, description="Optional user identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary request metadata"
    )
    config: AgentRunConfig | None = Field(
        default=None, description="Optional per-request config override"
    )


class AgentResponse(BaseModel):
    """Final result of an agent run."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=False, description="Whether the run succeeded")
    status: AgentStatus = Field(default=AgentStatus.PENDING, description="Run status")
    output: OutputMessage = Field(
        default_factory=OutputMessage, description="Final response message"
    )
    session_id: str = Field(default="", description="Owning session identifier")
    intent: Intent | None = Field(default=None, description="Resolved intent")
    goal_hierarchy: GoalHierarchy | None = Field(
        default=None, description="Resolved goal hierarchy"
    )
    plan: Plan | None = Field(default=None, description="Executed plan")
    reasoning_trace: ReasoningTrace | None = Field(
        default=None, description="Reasoning trace produced before execution"
    )
    reflection_report: ReflectionReport | None = Field(
        default=None, description="Reflection report produced after execution"
    )
    memory_summary: MemorySummary | None = Field(
        default=None, description="Memory state after the run"
    )
    experience: Experience | None = Field(
        default=None, description="Experience recorded for the run"
    )
    steps: tuple[AgentStep, ...] = Field(
        default_factory=tuple, description="Per-step execution outcomes"
    )
    summary: AgentRunSummary = Field(
        default_factory=AgentRunSummary, description="Run statistics"
    )
    error: str | None = Field(default=None, description="Run-level error message")
