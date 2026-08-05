"""Workflow automation domain models.

Workflows are directed acyclic graphs (DAGs) of steps that can run in
sequence or parallel.  Definitions are immutable once validated and
**published versions are immutable** — editing always creates a new
draft version.  Every :class:`WorkflowRun` stores a complete execution
snapshot so historical runs remain reproducible even if later versions
change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.kernel.pipeline.models import ErrorPolicy


def _utcnow() -> datetime:
    """Return the current aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _utcnow_factory() -> datetime:
    """Factory returning the current aware UTC timestamp."""
    return _utcnow()


class WorkflowStatus(str, Enum):
    """Lifecycle state of a workflow definition."""

    DRAFT = "draft"
    VALIDATED = "validated"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


# ----------------------------------------------------------------------
# Step definitions
# ----------------------------------------------------------------------


class WorkflowStepType(str, Enum):
    """Discriminator for the supported step kinds."""

    TASK = "task"
    AGENT = "agent"
    LLM = "llm"
    SUBWORKFLOW = "subworkflow"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"
    DELAY = "delay"
    APPROVAL = "approval"
    CALLBACK = "callback"


class BackoffStrategy(str, Enum):
    """Backoff curve applied between retries."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class StepRetryPolicy(BaseModel):
    """Retry configuration for a single step."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(
        default=0, ge=0, le=10, description="Retries after first attempt"
    )
    backoff: BackoffStrategy = Field(default=BackoffStrategy.CONSTANT)
    base_delay_s: float = Field(
        default=1.0, ge=0.0, description="Base delay between retries"
    )
    max_delay_s: float = Field(
        default=60.0, ge=0.0, description="Upper bound on the retry delay"
    )
    jitter: bool = Field(default=False, description="Add random jitter to delays")


class ParallelJoinMode(str, Enum):
    """How a parallel step's results are merged after fan-out."""

    ALL = "all"
    ANY = "any"
    NONE = "none"


class ApprovalAutoDecision(str, Enum):
    """Automatic outcome applied when an approval step times out."""

    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    TIMEOUT_FAIL = "timeout_fail"


class Branch(BaseModel):
    """A named alternative inside a conditional or parallel step."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Branch identifier")
    when: str = Field(
        default="true", description="Condition expression for this branch"
    )
    steps: tuple[WorkflowStep, ...] = Field(
        default_factory=tuple, description="Steps executed in this branch"
    )


class WorkflowStep(BaseModel):
    """A single node in a workflow DAG.

    Step definitions are immutable.  Kind-specific fields are optional
    and only meaningful for the step's :attr:`type`; validation enforces
    that required kind-specific fields are present.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique step identifier within the workflow")
    type: WorkflowStepType = Field(..., description="Step kind")
    name: str = Field(default="", description="Human-readable step name")
    description: str = Field(default="", description="What this step does")
    depends_on: tuple[str, ...] = Field(
        default_factory=tuple, description="Step ids that must complete first"
    )
    timeout_s: float | None = Field(
        default=None, ge=0.1, description="Per-step timeout in seconds"
    )
    retry_policy: StepRetryPolicy | None = Field(
        default=None, description="Retry behaviour for this step"
    )
    error_policy: ErrorPolicy = Field(
        default=ErrorPolicy.ABORT, description="Behaviour when this step errors"
    )
    input_bindings: dict[str, str] = Field(
        default_factory=dict,
        description="Step input name -> template expression (e.g. ${inputs.user})",
    )
    output_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Output field -> expression producing it from the raw result",
    )
    condition: str | None = Field(
        default=None,
        description="Optional run-gate expression; step is skipped when false",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata"
    )

    # TASK
    tool_name: str = Field(default="", description="TASK: registered tool name")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="TASK: tool parameters"
    )

    # AGENT
    agent_name: str = Field(default="", description="AGENT: agent to run")
    agent_config: dict[str, Any] = Field(
        default_factory=dict, description="AGENT: AgentRunConfig overrides"
    )

    # LLM
    prompt_template: str = Field(default="", description="LLM: prompt template")
    system_prompt: str = Field(default="", description="LLM: system prompt")
    model_hint: str = Field(default="", description="LLM: preferred model hint")

    # SUBWORKFLOW
    workflow_id: str = Field(default="", description="SUBWORKFLOW: child workflow id")
    workflow_version: str = Field(
        default="", description="SUBWORKFLOW: child workflow version (empty = latest)"
    )

    # TRANSFORM
    expression: str = Field(default="", description="TRANSFORM: expression to evaluate")

    # CONDITIONAL / PARALLEL
    branches: tuple[Branch, ...] = Field(
        default_factory=tuple, description="CONDITIONAL/PARALLEL: alternatives"
    )
    join_mode: ParallelJoinMode = Field(
        default=ParallelJoinMode.ALL, description="PARALLEL: how to join results"
    )

    # LOOP
    iterable: str = Field(
        default="", description="LOOP: expression yielding the iterable"
    )
    loop_var: str = Field(default="item", description="LOOP: iteration variable name")
    loop_steps: tuple[WorkflowStep, ...] = Field(
        default_factory=tuple, description="LOOP: steps executed per iteration"
    )
    max_iterations: int = Field(
        default=100, ge=1, le=10000, description="LOOP: upper iteration bound"
    )
    break_condition: str = Field(
        default="", description="LOOP: optional break expression"
    )

    # DELAY
    delay_seconds: float = Field(
        default=0.0, ge=0.0, description="DELAY: seconds to wait"
    )

    # APPROVAL
    approvers: tuple[str, ...] = Field(
        default_factory=tuple, description="APPROVAL: users allowed to decide"
    )
    auto_decision: ApprovalAutoDecision = Field(
        default=ApprovalAutoDecision.TIMEOUT_FAIL,
        description="APPROVAL: outcome on timeout",
    )

    # CALLBACK
    callback_url: str = Field(default="", description="CALLBACK: webhook URL")
    callback_method: str = Field(default="POST", description="CALLBACK: HTTP method")
    callback_payload: str = Field(
        default="", description="CALLBACK: payload template expression"
    )


class WorkflowInput(BaseModel):
    """A declared workflow input."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Input name")
    type: str = Field(
        default="any", description="Expected type hint (str/int/float/bool/any)"
    )
    required: bool = Field(default=False, description="Whether the input is required")
    description: str = Field(default="")


class WorkflowOutput(BaseModel):
    """A declared workflow output."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Output name")
    source: str = Field(default="", description="Expression resolving the output value")
    description: str = Field(default="")


class WorkflowSettings(BaseModel):
    """Execution settings shared by every run of a workflow."""

    model_config = ConfigDict(frozen=True)

    default_timeout_s: float | None = Field(
        default=None, ge=0.1, description="Default per-step timeout"
    )
    overall_timeout_s: float | None = Field(
        default=None, ge=0.1, description="Overall run timeout"
    )
    max_concurrency: int = Field(
        default=4, ge=1, le=64, description="Max steps running concurrently"
    )
    max_iterations: int = Field(
        default=100, ge=1, le=10000, description="Max loop iterations"
    )
    timezone: str = Field(default="UTC", description="Schedule timezone")


class WorkflowDefinition(BaseModel):
    """An immutable workflow definition.

    Defines inputs, outputs, execution settings and the step DAG.
    Definitions are versioned; published versions are never mutated.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique workflow identifier (slug)")
    version: str = Field(default="1.0.0", description="Semantic version")
    status: WorkflowStatus = Field(
        default=WorkflowStatus.DRAFT, description="Definition lifecycle state"
    )
    name: str = Field(default="", description="Display name")
    description: str = Field(default="", description="What the workflow does")
    tags: tuple[str, ...] = Field(
        default_factory=tuple, description="Categorisation tags"
    )
    inputs: tuple[WorkflowInput, ...] = Field(
        default_factory=tuple, description="Declared inputs"
    )
    outputs: tuple[WorkflowOutput, ...] = Field(
        default_factory=tuple, description="Declared outputs"
    )
    settings: WorkflowSettings = Field(default_factory=WorkflowSettings)
    steps: tuple[WorkflowStep, ...] = Field(
        default_factory=tuple, description="Workflow step DAG"
    )
    created_at: datetime = Field(default_factory=_utcnow_factory)
    updated_at: datetime = Field(default_factory=_utcnow_factory)

    def with_status(self, status: WorkflowStatus) -> WorkflowDefinition:
        """Return a copy of this definition with a new status."""
        return self.model_copy(update={"status": status, "updated_at": _utcnow()})


# ----------------------------------------------------------------------
# Execution state
# ----------------------------------------------------------------------


class WorkflowRunStatus(str, Enum):
    """Lifecycle state of a workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_CALLBACK = "waiting_callback"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_RUN_STATUSES: frozenset[WorkflowRunStatus] = frozenset(
    {
        WorkflowRunStatus.SUCCEEDED,
        WorkflowRunStatus.FAILED,
        WorkflowRunStatus.CANCELLED,
        WorkflowRunStatus.TIMED_OUT,
    }
)

# Statuses that can be restored and resumed after a restart
RESUMABLE_RUN_STATUSES: frozenset[WorkflowRunStatus] = frozenset(
    {
        WorkflowRunStatus.PENDING,
        WorkflowRunStatus.RUNNING,
        WorkflowRunStatus.WAITING_APPROVAL,
        WorkflowRunStatus.WAITING_CALLBACK,
        WorkflowRunStatus.PAUSED,
    }
)


class StepExecutionStatus(str, Enum):
    """Lifecycle state of a single step execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_CALLBACK = "waiting_callback"


class RunTriggerKind(str, Enum):
    """How a run was started."""

    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    PARENT = "parent"


class RunTrigger(BaseModel):
    """Origin metadata for a workflow run."""

    model_config = ConfigDict(frozen=True)

    kind: RunTriggerKind = Field(default=RunTriggerKind.MANUAL)
    detail: str = Field(default="", description="Free-form trigger description")


class WorkflowSnapshot(BaseModel):
    """Complete execution snapshot captured when a run starts.

    Stores the workflow version, the serialised step definitions and
    the resolved configuration so historical runs remain reproducible
    even if later versions of the workflow change.
    """

    model_config = ConfigDict(frozen=True)

    workflow_id: str = Field(...)
    version: str = Field(...)
    step_definitions: tuple[WorkflowStep, ...] = Field(default_factory=tuple)
    settings: WorkflowSettings = Field(default_factory=WorkflowSettings)
    inputs: tuple[WorkflowInput, ...] = Field(default_factory=tuple)
    outputs: tuple[WorkflowOutput, ...] = Field(default_factory=tuple)
    captured_at: datetime = Field(default_factory=_utcnow_factory)


class ApprovalStatus(str, Enum):
    """Lifecycle state of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class ApprovalRequest(BaseModel):
    """A human-in-the-loop approval awaiting a decision."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(...)
    run_id: str = Field(...)
    step_id: str = Field(...)
    approvers: tuple[str, ...] = Field(default_factory=tuple)
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description="pending/approved/rejected/timed_out",
    )
    decided_by: str = Field(default="")
    decision_note: str = Field(default="")
    created_at: datetime = Field(default_factory=_utcnow_factory)
    expires_at: datetime | None = Field(default=None)
    decided_at: datetime | None = Field(default=None)


class WorkflowEvent(BaseModel):
    """An event recorded against a workflow run.

    Events are persisted with the run so the execution history can be
    replayed or rendered as a timeline.
    """

    model_config = ConfigDict(frozen=True)

    event_type: str = Field(...)
    run_id: str = Field(...)
    step_id: str | None = Field(default=None)
    timestamp: datetime = Field(default_factory=_utcnow_factory)
    data: dict[str, Any] = Field(default_factory=dict)


class StepExecution(BaseModel):
    """Outcome of executing a single workflow step."""

    model_config = ConfigDict(frozen=True)

    step_id: str = Field(...)
    status: StepExecutionStatus = Field(default=StepExecutionStatus.PENDING)
    inputs: dict[str, Any] = Field(default_factory=dict, description="Resolved inputs")
    output: Any = Field(default=None, description="JSON-safe step output")
    error: str | None = Field(default=None)
    retries_consumed: int = Field(default=0, ge=0)
    attempts: tuple[float, ...] = Field(
        default_factory=tuple, description="Retry delays used"
    )
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)


class WorkflowRun(BaseModel):
    """A single execution of a workflow definition.

    Immutable; updates are performed via :meth:`model_copy`.  Stores an
    :class:`WorkflowSnapshot` for reproducibility and a persisted event
    log for replay/debugging.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(...)
    workflow_id: str = Field(...)
    workflow_version: str = Field(...)
    status: WorkflowRunStatus = Field(default=WorkflowRunStatus.PENDING)
    snapshot: WorkflowSnapshot = Field(...)
    trigger: RunTrigger = Field(default_factory=RunTrigger)
    idempotency_key: str | None = Field(
        default=None, description="Deduplication key; duplicate runs are not created"
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    step_executions: tuple[StepExecution, ...] = Field(default_factory=tuple)
    approval_requests: tuple[ApprovalRequest, ...] = Field(default_factory=tuple)
    events: tuple[WorkflowEvent, ...] = Field(default_factory=tuple)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow_factory)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    paused_at: datetime | None = Field(default=None)

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` when the run has reached a terminal state."""
        return self.status in TERMINAL_RUN_STATUSES

    @property
    def is_resumable(self) -> bool:
        """Return ``True`` when the run can be restored after a restart."""
        return self.status in RESUMABLE_RUN_STATUSES

    def step(self, step_id: str) -> StepExecution | None:
        """Return the execution record for *step_id*, or ``None``."""
        for execution in self.step_executions:
            if execution.step_id == step_id:
                return execution
        return None


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------


class ScheduleTriggerType(str, Enum):
    """How a schedule fires."""

    CRON = "cron"
    INTERVAL = "interval"
    DATETIME = "datetime"


class ScheduleSpec(BaseModel):
    """A schedule that triggers a workflow at deterministic times."""

    model_config = ConfigDict(frozen=True)

    schedule_id: str = Field(...)
    workflow_id: str = Field(...)
    workflow_version: str = Field(default="", description="Empty = latest published")
    trigger_type: ScheduleTriggerType = Field(...)
    cron_expression: str = Field(
        default="", description="CRON: 5-field cron expression"
    )
    interval_seconds: float = Field(default=0.0, ge=0.0, description="INTERVAL: period")
    run_at: datetime | None = Field(default=None, description="DATETIME: one-shot time")
    timezone: str = Field(default="UTC")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Static run inputs"
    )
    enabled: bool = Field(default=True)
    max_concurrent_runs: int = Field(default=1, ge=1)
    next_run_at: datetime | None = Field(default=None)
    last_run_at: datetime | None = Field(default=None)
    last_status: str = Field(default="", description="Outcome of the last fired run")


# Forward reference resolution for recursive step structures
WorkflowStep.model_rebuild()
Branch.model_rebuild()
