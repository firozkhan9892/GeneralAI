"""Standard events for every stage of the cognitive pipeline.

Each stage fires events before and after processing, plus error/cancel events.
Consumers (logging, metrics, tracing, side-effects) subscribe via the EventBus.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class PipelineEvent(str, Enum):
    """Every event that can be emitted during pipeline execution.

    Naming convention: ``{engine}.{phase}`` where phase is one of:
    started, completed, failed, cancelled, skipped, timed_out.
    """

    # Perception
    PERCEPTION_STARTED = "perception.started"
    PERCEPTION_COMPLETED = "perception.completed"
    PERCEPTION_FAILED = "perception.failed"

    # Intent
    INTENT_STARTED = "intent.started"
    INTENT_COMPLETED = "intent.completed"
    INTENT_FAILED = "intent.failed"
    INTENT_CLARIFICATION_REQUESTED = "intent.clarification_requested"

    # Goal
    GOAL_STARTED = "goal.started"
    GOAL_CREATED = "goal.created"
    GOAL_COMPLETED = "goal.completed"
    GOAL_FAILED = "goal.failed"
    GOAL_DECOMPOSED = "goal.decomposed"

    # Planner
    PLANNER_STARTED = "planner.started"
    PLAN_GENERATED = "plan.generated"
    PLAN_REVISED = "plan.revised"
    PLANNER_FAILED = "planner.failed"

    # Reasoning
    REASONING_STARTED = "reasoning.started"
    REASONING_STEP = "reasoning.step"
    REASONING_COMPLETED = "reasoning.completed"
    REASONING_FAILED = "reasoning.failed"

    # Decision
    DECISION_STARTED = "decision.started"
    DECISION_SELECTED = "decision.selected"
    DECISION_FAILED = "decision.failed"

    # Capability
    CAPABILITY_STARTED = "capability.started"
    CAPABILITY_RESOLVED = "capability.resolved"
    CAPABILITY_UNAVAILABLE = "capability.unavailable"

    # Policy
    POLICY_STARTED = "policy.started"
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_DENIED = "policy.denied"
    POLICY_CONFIRMATION_REQUIRED = "policy.confirmation_required"

    # Task
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_SKIPPED = "task.skipped"

    # Tool
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Reflection
    REFLECTION_STARTED = "reflection.started"
    REFLECTION_COMPLETED = "reflection.completed"
    REFLECTION_PASSED = "reflection.passed"
    REFLECTION_FAILED = "reflection.failed"
    REFLECTION_REFINEMENT = "reflection.refinement"

    # Experience
    EXPERIENCE_STARTED = "experience.started"
    EXPERIENCE_STORED = "experience.stored"
    EXPERIENCE_RETRIEVED = "experience.retrieved"
    EXPERIENCE_FAILED = "experience.failed"

    # Response
    RESPONSE_BUILT = "response.built"
    RESPONSE_STREAM_CHUNK = "response.stream_chunk"
    RESPONSE_FAILED = "response.failed"

    # Cross-cutting
    PIPELINE_STAGE_STARTED = "pipeline.stage.started"
    PIPELINE_STAGE_COMPLETED = "pipeline.stage.completed"
    PIPELINE_STAGE_FAILED = "pipeline.stage.failed"
    PIPELINE_CANCELLED = "pipeline.cancelled"
    PIPELINE_TIMEOUT = "pipeline.timeout"
    STATE_CHANGED = "state.changed"
    STATE_ERROR = "state.error"


# ── Event severity / category helpers ────────────────────────────────────────


class EventSeverity(str, Enum):
    """Suggested severity level for a pipeline event."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Predefined severity mappings for common event categories
EVENT_SEVERITY: dict[PipelineEvent, EventSeverity] = {
    PipelineEvent.PERCEPTION_STARTED: EventSeverity.DEBUG,
    PipelineEvent.PERCEPTION_COMPLETED: EventSeverity.INFO,
    PipelineEvent.PERCEPTION_FAILED: EventSeverity.ERROR,
    PipelineEvent.INTENT_STARTED: EventSeverity.DEBUG,
    PipelineEvent.INTENT_COMPLETED: EventSeverity.INFO,
    PipelineEvent.INTENT_FAILED: EventSeverity.ERROR,
    PipelineEvent.INTENT_CLARIFICATION_REQUESTED: EventSeverity.WARNING,
    PipelineEvent.GOAL_CREATED: EventSeverity.INFO,
    PipelineEvent.GOAL_FAILED: EventSeverity.ERROR,
    PipelineEvent.PLAN_GENERATED: EventSeverity.INFO,
    PipelineEvent.PLAN_REVISED: EventSeverity.WARNING,
    PipelineEvent.REASONING_STARTED: EventSeverity.DEBUG,
    PipelineEvent.REASONING_COMPLETED: EventSeverity.INFO,
    PipelineEvent.REASONING_FAILED: EventSeverity.ERROR,
    PipelineEvent.DECISION_SELECTED: EventSeverity.INFO,
    PipelineEvent.DECISION_FAILED: EventSeverity.ERROR,
    PipelineEvent.CAPABILITY_RESOLVED: EventSeverity.INFO,
    PipelineEvent.CAPABILITY_UNAVAILABLE: EventSeverity.WARNING,
    PipelineEvent.POLICY_EVALUATED: EventSeverity.INFO,
    PipelineEvent.POLICY_DENIED: EventSeverity.WARNING,
    PipelineEvent.TASK_COMPLETED: EventSeverity.INFO,
    PipelineEvent.TASK_FAILED: EventSeverity.ERROR,
    PipelineEvent.TASK_CANCELLED: EventSeverity.WARNING,
    PipelineEvent.REFLECTION_COMPLETED: EventSeverity.INFO,
    PipelineEvent.REFLECTION_PASSED: EventSeverity.INFO,
    PipelineEvent.REFLECTION_FAILED: EventSeverity.WARNING,
    PipelineEvent.EXPERIENCE_STORED: EventSeverity.INFO,
    PipelineEvent.PIPELINE_CANCELLED: EventSeverity.WARNING,
    PipelineEvent.PIPELINE_TIMEOUT: EventSeverity.ERROR,
    PipelineEvent.STATE_ERROR: EventSeverity.ERROR,
}

# Backward-compatible string constants (for use with EventBus which takes str keys)
EVENT_PERCEPTION_STARTED: Final[str] = PipelineEvent.PERCEPTION_STARTED.value
EVENT_PERCEPTION_COMPLETED: Final[str] = PipelineEvent.PERCEPTION_COMPLETED.value
EVENT_INTENT_IDENTIFIED: Final[str] = PipelineEvent.INTENT_COMPLETED.value
EVENT_GOAL_CREATED: Final[str] = PipelineEvent.GOAL_CREATED.value
EVENT_GOAL_COMPLETED: Final[str] = PipelineEvent.GOAL_COMPLETED.value
EVENT_GOAL_DECOMPOSED: Final[str] = PipelineEvent.GOAL_DECOMPOSED.value
EVENT_PLAN_CREATED: Final[str] = PipelineEvent.PLAN_GENERATED.value
EVENT_PLAN_REVISED: Final[str] = PipelineEvent.PLAN_REVISED.value
EVENT_REASONING_STARTED: Final[str] = PipelineEvent.REASONING_STARTED.value
EVENT_REASONING_STEP: Final[str] = PipelineEvent.REASONING_STEP.value
EVENT_REASONING_COMPLETED: Final[str] = PipelineEvent.REASONING_COMPLETED.value
EVENT_DECISION_MADE: Final[str] = PipelineEvent.DECISION_SELECTED.value
EVENT_CAPABILITY_RESOLVED: Final[str] = PipelineEvent.CAPABILITY_RESOLVED.value
EVENT_CAPABILITY_UNAVAILABLE: Final[str] = PipelineEvent.CAPABILITY_UNAVAILABLE.value
EVENT_POLICY_EVALUATED: Final[str] = PipelineEvent.POLICY_EVALUATED.value
EVENT_POLICY_DENIED: Final[str] = PipelineEvent.POLICY_DENIED.value
EVENT_POLICY_CONFIRMATION_REQUIRED: Final[str] = (
    PipelineEvent.POLICY_CONFIRMATION_REQUIRED.value
)
EVENT_EXECUTION_STARTED: Final[str] = PipelineEvent.TASK_STARTED.value
EVENT_EXECUTION_COMPLETED: Final[str] = PipelineEvent.TASK_COMPLETED.value
EVENT_EXECUTION_FAILED: Final[str] = PipelineEvent.TASK_FAILED.value
EVENT_REFLECTION_STARTED: Final[str] = PipelineEvent.REFLECTION_STARTED.value
EVENT_REFLECTION_COMPLETED: Final[str] = PipelineEvent.REFLECTION_COMPLETED.value
EVENT_REFLECTION_REFINEMENT: Final[str] = PipelineEvent.REFLECTION_REFINEMENT.value
EVENT_EXPERIENCE_RECORDED: Final[str] = PipelineEvent.EXPERIENCE_STORED.value
EVENT_STATE_CHANGED: Final[str] = PipelineEvent.STATE_CHANGED.value
EVENT_STATE_ERROR: Final[str] = PipelineEvent.STATE_ERROR.value
