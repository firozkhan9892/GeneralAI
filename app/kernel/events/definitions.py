"""Kernel-specific event type definitions."""

from __future__ import annotations

from typing import Final

# Perception events
EVENT_PERCEPTION_STARTED: Final[str] = "kernel.perception.started"
EVENT_PERCEPTION_COMPLETED: Final[str] = "kernel.perception.completed"

# Intent events
EVENT_INTENT_IDENTIFIED: Final[str] = "kernel.intent.identified"
EVENT_INTENT_CLARIFICATION_REQUESTED: Final[str] = (
    "kernel.intent.clarification_requested"
)

# Goal events
EVENT_GOAL_CREATED: Final[str] = "kernel.goal.created"
EVENT_GOAL_COMPLETED: Final[str] = "kernel.goal.completed"
EVENT_GOAL_FAILED: Final[str] = "kernel.goal.failed"
EVENT_GOAL_DECOMPOSED: Final[str] = "kernel.goal.decomposed"

# Plan events
EVENT_PLAN_CREATED: Final[str] = "kernel.plan.created"
EVENT_PLAN_REVISED: Final[str] = "kernel.plan.revised"
EVENT_PLAN_FAILED: Final[str] = "kernel.plan.failed"

# Reasoning events
EVENT_REASONING_STARTED: Final[str] = "kernel.reasoning.started"
EVENT_REASONING_STEP: Final[str] = "kernel.reasoning.step"
EVENT_REASONING_COMPLETED: Final[str] = "kernel.reasoning.completed"

# Decision events
EVENT_DECISION_MADE: Final[str] = "kernel.decision.made"
EVENT_DECISION_FAILED: Final[str] = "kernel.decision.failed"

# Capability events
EVENT_CAPABILITY_RESOLVED: Final[str] = "kernel.capability.resolved"
EVENT_CAPABILITY_UNAVAILABLE: Final[str] = "kernel.capability.unavailable"

# Policy events
EVENT_POLICY_EVALUATED: Final[str] = "kernel.policy.evaluated"
EVENT_POLICY_DENIED: Final[str] = "kernel.policy.denied"
EVENT_POLICY_CONFIRMATION_REQUIRED: Final[str] = "kernel.policy.confirmation_required"

# Execution events
EVENT_EXECUTION_STARTED: Final[str] = "kernel.execution.started"
EVENT_EXECUTION_COMPLETED: Final[str] = "kernel.execution.completed"
EVENT_EXECUTION_FAILED: Final[str] = "kernel.execution.failed"

# Reflection events
EVENT_REFLECTION_STARTED: Final[str] = "kernel.reflection.started"
EVENT_REFLECTION_COMPLETED: Final[str] = "kernel.reflection.completed"
EVENT_REFLECTION_REFINEMENT: Final[str] = "kernel.reflection.refinement"

# Experience events
EVENT_EXPERIENCE_RECORDED: Final[str] = "kernel.experience.recorded"

# State events
EVENT_STATE_CHANGED: Final[str] = "kernel.state.changed"
EVENT_STATE_ERROR: Final[str] = "kernel.state.error"
