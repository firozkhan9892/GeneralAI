"""Request and response models for the GeneralAI REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.agents.models import AgentExecutionOptions, AgentSession
from app.automation.models import (
    ScheduleSpec,
    ScheduleTriggerType,
    WorkflowDefinition,
    WorkflowRun,
)
from app.kernel.agent.models import AgentRunConfig
from app.kernel.memory.models import MemorySearchHit
from app.knowledge.models import (
    CollectionMetadata,
    NamespaceMetadata,
    RetrievalResult,
)


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


# ----------------------------------------------------------------------
# Workflow automation (Phase 12e)
# ----------------------------------------------------------------------


class WorkflowRunRequest(BaseModel):
    """Body of ``POST /workflows/{workflow_id}/run``."""

    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Input values for the run"
    )
    version: str | None = Field(default=None, description="Specific version to run")
    idempotency_key: str | None = Field(
        default=None, description="Deduplication key for the run"
    )


class WorkflowPublishRequest(BaseModel):
    """Body of ``POST /workflows/{workflow_id}/publish``."""

    version: str = Field(..., min_length=1, description="Version to publish")


class WorkflowVersionCreateRequest(BaseModel):
    """Body of ``POST /workflows/{workflow_id}/versions``."""

    version: str = Field(..., min_length=1, description="New version string")
    definition: WorkflowDefinition = Field(..., description="Definition to register")


class ApprovalDecisionRequest(BaseModel):
    """Body of approval decision endpoints."""

    decided_by: str = Field(default="", description="Who made the decision")
    decision_note: str = Field(default="", description="Optional note")


class ScheduleCreateRequest(BaseModel):
    """Body of ``POST /schedules``."""

    workflow_id: str = Field(..., min_length=1, description="Workflow to trigger")
    workflow_version: str = Field(default="", description="Empty = latest published")
    trigger_type: ScheduleTriggerType = Field(..., description="Trigger kind")
    cron_expression: str = Field(default="", description="CRON expression")
    interval_seconds: float = Field(default=0.0, ge=0.0, description="INTERVAL period")
    run_at: datetime | None = Field(default=None, description="DATETIME one-shot time")
    timezone: str = Field(default="UTC", description="Schedule timezone")
    payload: dict[str, Any] = Field(default_factory=dict, description="Static inputs")
    enabled: bool = Field(default=True, description="Start enabled")
    max_concurrent_runs: int = Field(
        default=1, ge=1, description="Per-schedule concurrency cap"
    )


class ScheduleUpdateRequest(BaseModel):
    """Body of ``PATCH /schedules/{schedule_id}`` (partial update)."""

    workflow_id: str | None = Field(default=None, description="Workflow to trigger")
    workflow_version: str | None = Field(default=None, description="Version override")
    trigger_type: ScheduleTriggerType | None = Field(
        default=None, description="Trigger kind"
    )
    cron_expression: str | None = Field(default=None, description="CRON expression")
    interval_seconds: float | None = Field(
        default=None, ge=0.0, description="INTERVAL period"
    )
    run_at: datetime | None = Field(default=None, description="DATETIME one-shot time")
    timezone: str | None = Field(default=None, description="Schedule timezone")
    payload: dict[str, Any] | None = Field(default=None, description="Static inputs")
    enabled: bool | None = Field(default=None, description="Enable/disable")
    max_concurrent_runs: int | None = Field(
        default=None, ge=1, description="Concurrency cap"
    )


class WorkflowListResponse(BaseModel):
    """Result of ``GET /workflows`` and ``GET /workflows/{id}/versions``."""

    total: int = Field(..., ge=0, description="Number of definitions returned")
    workflows: list[WorkflowDefinition] = Field(
        default_factory=list, description="Workflow definitions"
    )


class WorkflowRunListResponse(BaseModel):
    """Result of ``GET /workflows/runs``."""

    total: int = Field(..., ge=0, description="Number of runs returned")
    runs: list[WorkflowRun] = Field(default_factory=list, description="Run snapshots")


class ScheduleListResponse(BaseModel):
    """Result of ``GET /schedules``."""

    total: int = Field(..., ge=0, description="Number of schedules returned")
    schedules: list[ScheduleSpec] = Field(default_factory=list, description="Schedules")


# ── Knowledge / RAG Schemas ──────────────────────────────────────────


class KnowledgeIngestRequest(BaseModel):
    """Body of ``POST /knowledge/documents``."""

    content: str = Field(..., description="Document content (base64 for binary)")
    source_uri: str = Field(default="", description="Origin URI (file/path/url)")
    collection_id: str = Field(default="", description="Target collection ID")
    namespace: str = Field(default="", description="Isolating namespace")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary filterable metadata"
    )


class KnowledgeIngestResponse(BaseModel):
    """Result of document ingestion."""

    doc_id: str = Field(..., description="Ingested document identifier")
    source_uri: str = Field(..., description="Origin URI")
    chunk_count: int = Field(..., ge=0, description="Number of chunks created")
    collection_id: str = Field(..., description="Owning collection")


class KnowledgeTextIngestRequest(BaseModel):
    """Body of ``POST /knowledge/text``."""

    text: str = Field(..., min_length=1, description="Raw text content")
    source_uri: str = Field(default="text-input", description="Origin identifier")
    collection_id: str = Field(default="", description="Target collection ID")
    namespace: str = Field(default="", description="Isolating namespace")


class KnowledgeSearchRequest(BaseModel):
    """Body of ``POST /knowledge/search``."""

    query: str = Field(..., min_length=1, description="Search query")
    collection_id: str = Field(default="", description="Target collection")
    namespace: str = Field(default="", description="Isolating namespace")
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum hits")
    strategy: str = Field(default="hybrid", description="vector|bm25|hybrid")


class KnowledgeQueryRequest(BaseModel):
    """Body of ``POST /knowledge/query``."""

    query: str = Field(..., min_length=1, description="Natural language query")
    collection_id: str = Field(default="", description="Target collection")
    namespace: str = Field(default="", description="Isolating namespace")
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum hits")
    include_sources: bool = Field(default=True, description="Attach source references")


class KnowledgeSearchResponse(BaseModel):
    """Result of a knowledge search/query."""

    query: str = Field(..., description="Original query")
    total: int = Field(..., ge=0, description="Number of hits returned")
    hits: list[RetrievalResult] = Field(
        default_factory=list, description="Ranked retrieval results"
    )
    latency_ms: float = Field(..., ge=0.0, description="Query latency in milliseconds")


class KnowledgeCollectionsResponse(BaseModel):
    """Result of ``GET /knowledge/collections``."""

    total: int = Field(..., ge=0, description="Number of collections")
    collections: list[CollectionMetadata] = Field(
        default_factory=list, description="Collection metadata"
    )


class KnowledgeCollectionCreateRequest(BaseModel):
    """Body of ``POST /knowledge/collections``."""

    collection_id: str = Field(..., min_length=1, description="Unique collection ID")
    name: str = Field(default="", description="Human-readable name")
    namespace: str = Field(default="", description="Isolating namespace")
    description: str = Field(default="", description="Optional description")


class KnowledgeNamespacesResponse(BaseModel):
    """Result of ``GET /knowledge/namespaces``."""

    total: int = Field(..., ge=0, description="Number of namespaces")
    namespaces: list[NamespaceMetadata] = Field(
        default_factory=list, description="Namespace metadata"
    )


class KnowledgeNamespaceCreateRequest(BaseModel):
    """Body of ``POST /knowledge/namespaces``."""

    name: str = Field(..., min_length=1, description="Unique namespace name")
    description: str = Field(default="", description="Optional description")
