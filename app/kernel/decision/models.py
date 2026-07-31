"""Decision domain models — stage 6 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionScore(BaseModel):
    """A scored evaluation of a candidate along a single criterion."""

    model_config = ConfigDict(frozen=True)

    criterion_name: str = Field(..., description="Name of the criterion")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalised score")
    weight: float = Field(
        default=1.0, ge=0.0, description="Weight applied to this score"
    )
    rationale: str = Field(default="", description="Why this score was assigned")


class DecisionReason(BaseModel):
    """The reasoning behind a decision — why it was made."""

    model_config = ConfigDict(frozen=True)

    primary_rationale: str = Field(
        default="", description="Main reason for the decision"
    )
    criteria_scores: tuple[DecisionScore, ...] = Field(
        default_factory=tuple, description="Per-criterion breakdown"
    )
    trade_offs: tuple[str, ...] = Field(
        default_factory=tuple, description="Trade-offs considered"
    )


class ActionCandidate(BaseModel):
    """A single candidate action considered during decision-making."""

    model_config = ConfigDict(frozen=True)

    action_type: str = Field(
        ..., description="Type of action (tool_call, respond, wait, etc.)"
    )
    description: str = Field(default="", description="Human-readable description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass on execution"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence this is the right action"
    )
    estimated_cost: int = Field(default=0, ge=0, description="Estimated token cost")
    source: str = Field(
        default="reasoning", description="Which engine produced this candidate"
    )


class Decision(BaseModel):
    """The result of the decision engine — a single selected action."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(default="", description="Owning session identifier")
    selected_action: ActionCandidate = Field(..., description="The chosen action")
    candidates: tuple[ActionCandidate, ...] = Field(
        default_factory=tuple, description="All considered candidates"
    )
    reason: DecisionReason = Field(
        default_factory=DecisionReason, description="Why this action was selected"
    )
    strategy_used: str = Field(
        default="greedy", description="Strategy that selected the action"
    )
    status: str = Field(
        default="pending", description="Execution status (pending, approved, rejected)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the decision was made"
    )
