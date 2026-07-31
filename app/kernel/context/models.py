"""Context domain models — cross-cutting session context management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenBudget(BaseModel):
    """Token budget tracking for a session."""

    total: int = Field(default=4000, ge=1, description="Maximum token budget")
    used: int = Field(default=0, ge=0, description="Tokens consumed so far")
    reserved: int = Field(
        default=0, ge=0, description="Tokens reserved for upcoming operations"
    )
    threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Usage fraction that triggers pruning"
    )


class CognitiveContext(BaseModel):
    """The full cognitive context for a session.

    Unlike most models, this is **mutable** — it is updated
    incrementally as the pipeline executes.
    """

    id: str = Field(default="", description="Unique context identifier")
    session_id: str = Field(default="")
    messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Message history as dicts"
    )
    working_memory: dict[str, Any] = Field(
        default_factory=dict, description="Scratch-pad working memory"
    )
    active_goals: list[dict[str, Any]] = Field(
        default_factory=list, description="Currently active goals"
    )
    environment: dict[str, Any] = Field(
        default_factory=dict, description="Environment / runtime metadata"
    )
    token_budget: TokenBudget = Field(
        default_factory=TokenBudget, description="Token tracking"
    )
    version: int = Field(
        default=0, ge=0, description="Context version (incremented on each update)"
    )
    parent_id: str | None = Field(
        default=None, description="Parent context ID for branching"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


class ContextSource(BaseModel):
    """Source data for building a new context."""

    model_config = ConfigDict(frozen=True)

    percept: Any = Field(default=None)
    intent: Any = Field(default=None)
    session_history: list[dict[str, Any]] = Field(default_factory=list)
    memory_entries: list[dict[str, Any]] = Field(default_factory=list)


class ContextDelta(BaseModel):
    """An incremental update to apply to a context."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(..., description="Target field path")
    value: Any = Field(..., description="Value to set / add / remove")
    operation: str = Field(
        default="set", description="Operation: set, append, remove, clear"
    )


class ContextSnapshot(BaseModel):
    """A frozen snapshot of a context at a specific point in time."""

    model_config = ConfigDict(frozen=True)

    context: CognitiveContext = Field(..., description="The context at snapshot time")
    version: int = Field(..., ge=0, description="Context version at snapshot")
    captured_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the snapshot was taken"
    )
