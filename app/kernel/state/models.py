"""State domain models — cognitive state machine definitions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CognitiveState(str, Enum):
    """Valid states in the cognitive state machine.

    Mirrors the 15-stage pipeline plus auxiliary states for
    error handling and lifecycle management.
    """

    IDLE = "idle"
    PERCEIVING = "perceiving"
    INTENT_ANALYZING = "intent_analyzing"
    CLARIFYING = "clarifying"
    GOAL_RESOLVING = "goal_resolving"
    PLAN_CREATING = "plan_creating"
    REASONING = "reasoning"
    DECIDING = "deciding"
    CAPABILITY_CHECKING = "capability_checking"
    POLICY_EVALUATING = "policy_evaluating"
    CONFIRMING = "confirming"
    SKILL_SELECTING = "skill_selecting"
    TOOL_RESOLVING = "tool_resolving"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    RECORDING = "recording"
    RESPONDING = "responding"
    ERROR = "error"
    TERMINATED = "terminated"


class Transition(BaseModel):
    """A recorded state transition in the cognitive state machine."""

    model_config = ConfigDict(frozen=True)

    from_state: CognitiveState = Field(..., description="Source state")
    to_state: CognitiveState = Field(..., description="Target state")
    reason: str = Field(default="", description="Why the transition occurred")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the transition occurred"
    )


class SessionState(BaseModel):
    """Full observable state of a cognitive session at a point in time.

    Unlike most models, this one is **mutable** because it is
    updated in-place during pipeline execution.
    """

    session_id: str = Field(default="", description="Session identifier")
    state: CognitiveState = Field(
        default=CognitiveState.IDLE, description="Current state"
    )
    previous_state: CognitiveState | None = Field(
        default=None, description="Immediately previous state"
    )
    transitions: list[Transition] = Field(
        default_factory=list, description="History of state transitions"
    )
    context_id: str = Field(default="", description="Associated context identifier")
    error_count: int = Field(default=0, ge=0, description="Cumulative error count")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Session creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extensible state metadata"
    )
