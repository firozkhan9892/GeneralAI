"""Agent manager domain models.

The agent manager supervises multiple concurrent :class:`AgentRuntime`
sessions.  These frozen models describe the lifecycle of a single
managed session, its execution options, and the status vocabulary used
to reconcile runtime outcomes (``AgentStatus``) with session state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.kernel.agent.models import AgentRequest, AgentResponse, AgentRunConfig


class SessionStatus(str, Enum):
    """Lifecycle state of a managed agent session."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


#: Statuses from which a session can no longer transition.
TERMINAL_SESSION_STATUSES: frozenset[SessionStatus] = frozenset(
    {
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.CANCELLED,
        SessionStatus.TIMED_OUT,
    }
)


class AgentExecutionOptions(BaseModel):
    """Configuration controlling how a session is executed.

    Args:
        max_attempts: Total execution attempts before giving up (>= 1).
        retry_delay_s: Delay between failed attempts in seconds.
        persist: Whether session state transitions are persisted.
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=1, ge=1, le=10, description="Total attempts")
    retry_delay_s: float = Field(
        default=0.0, ge=0.0, description="Delay between retries (seconds)"
    )
    persist: bool = Field(default=True, description="Persist session state transitions")


class AgentSession(BaseModel):
    """A single managed agent session.

    Immutable value object: state transitions are expressed as copies
    produced by the transition helpers below, so the registry always
    stores a consistent snapshot.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Unique session identifier")
    request: AgentRequest = Field(..., description="The agent request")
    config: AgentRunConfig | None = Field(
        default=None, description="Run configuration for the request"
    )
    options: AgentExecutionOptions = Field(
        default_factory=AgentExecutionOptions, description="Execution options"
    )
    status: SessionStatus = Field(
        default=SessionStatus.PENDING, description="Session lifecycle state"
    )
    response: AgentResponse | None = Field(
        default=None, description="Final agent response, if completed"
    )
    attempts: int = Field(default=0, ge=0, description="Execution attempts made")
    error: str | None = Field(default=None, description="Session-level error")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary session metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the session was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last state change"
    )
    started_at: datetime | None = Field(
        default=None, description="When execution first started"
    )
    completed_at: datetime | None = Field(
        default=None, description="When the session reached a terminal state"
    )
    paused_at: datetime | None = Field(
        default=None, description="When the session was last paused"
    )

    # ------------------------------------------------------------------
    # Transition helpers (all return a new copy)
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` when the session reached a terminal state."""
        return self.status in TERMINAL_SESSION_STATUSES

    @property
    def is_running(self) -> bool:
        """Return ``True`` when the session is executing."""
        return self.status == SessionStatus.RUNNING

    def with_status(
        self,
        status: SessionStatus,
        *,
        error: str | None = None,
        response: AgentResponse | None = None,
        started: bool = False,
        completed: bool = False,
    ) -> AgentSession:
        """Return a copy with a new status and optional transition stamps.

        Args:
            status: The new session status.
            error: Optional session-level error message.
            response: Optional completed agent response.
            started: Whether to stamp ``started_at`` (first attempt).
            completed: Whether to stamp ``completed_at``.

        Returns:
            A new :class:`AgentSession` snapshot.
        """
        return self.model_copy(
            update={
                "status": status,
                "error": error,
                "response": response if response is not None else self.response,
                "updated_at": datetime.utcnow(),
                "started_at": self.started_at if not started else datetime.utcnow(),
                "completed_at": (
                    self.completed_at if not completed else datetime.utcnow()
                ),
                "paused_at": (
                    datetime.utcnow()
                    if status == SessionStatus.PAUSED
                    else self.paused_at
                ),
            }
        )
