"""Request and response models for the GeneralAI REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.models import AgentExecutionOptions, AgentSession
from app.kernel.agent.models import AgentRunConfig
from app.kernel.memory.models import MemorySearchHit


class ChatRequest(BaseModel):
    """Body of ``POST /chat``."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(default="", description="Optional session identifier")
    user_id: str | None = Field(default=None, description="Optional user identifier")
    config: AgentRunConfig | None = Field(
        default=None, description="Per-run configuration override"
    )
    options: AgentExecutionOptions | None = Field(
        default=None, description="Per-run execution options override"
    )


class ChatResponse(BaseModel):
    """Result of a non-streaming chat request."""

    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Terminal session status")
    content: str = Field(..., description="Final output content")
    success: bool = Field(..., description="Whether the run succeeded")
    error: str | None = Field(default=None, description="Error message on failure")


class AgentRunRequest(BaseModel):
    """Body of ``POST /agent/run``."""

    raw_input: str = Field(..., min_length=1, description="Raw user input")
    session_id: str = Field(default="", description="Optional session identifier")
    user_id: str | None = Field(default=None, description="Optional user identifier")
    config: AgentRunConfig | None = Field(
        default=None, description="Per-run configuration override"
    )
    options: AgentExecutionOptions | None = Field(
        default=None, description="Per-run execution options override"
    )
    wait: bool = Field(default=True, description="Await completion before responding")


class AgentCancelRequest(BaseModel):
    """Body of ``POST /agent/cancel``."""

    session_id: str = Field(..., min_length=1, description="Session to cancel")
    reason: str = Field(default="user_requested", description="Cancellation reason")


class AgentListResponse(BaseModel):
    """Result of ``GET /agents``."""

    total: int = Field(..., ge=0, description="Number of sessions returned")
    sessions: list[AgentSession] = Field(
        default_factory=list, description="Session snapshots, newest first"
    )


class ToolRunRequest(BaseModel):
    """Body of ``POST /tool/run``."""

    tool: str = Field(..., min_length=1, description="Registered tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool invocation arguments"
    )
    timeout_s: float | None = Field(default=None, ge=0.1, description="Timeout")
    max_retries: int = Field(default=0, ge=0, le=10, description="Retry budget")


class MemorySearchResponse(BaseModel):
    """Result of ``GET /memory/search``."""

    query: str = Field(..., description="Search query")
    total: int = Field(..., ge=0, description="Number of hits returned")
    hits: list[MemorySearchHit] = Field(
        default_factory=list, description="Ranked search hits"
    )


class HealthResponse(BaseModel):
    """Result of ``GET /health``."""

    status: str = Field(..., description="Liveness indicator ('ok')")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    sessions_active: int = Field(..., ge=0, description="Running sessions")
    sessions_total: int = Field(..., ge=0, description="Tracked sessions")


class MetricsResponse(BaseModel):
    """Result of ``GET /metrics``."""

    requests_total: int = Field(..., ge=0, description="Requests served")
    errors_total: int = Field(..., ge=0, description="Requests with 5xx status")
    requests_by_path: dict[str, int] = Field(
        default_factory=dict, description="Requests per URL path"
    )
    sessions_active: int = Field(..., ge=0, description="Running sessions")
    sessions_total: int = Field(..., ge=0, description="Tracked sessions")
    memory_records: int = Field(..., ge=0, description="Stored memory records")
    tools_count: int = Field(..., ge=0, description="Registered tools")
