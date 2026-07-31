"""Contracts primitive types — shared foundation for every inter-engine message.

Why these contracts exist
-------------------------
Every engine in the Cognitive Kernel communicates through strongly typed
message contracts rather than raw Python calls. This provides:

1.  **Decoupling** — engines only depend on contract schemas, not on each
    other's implementation.
2.  **Distributed readiness** — contracts serialise to JSON/MessagePack,
    enabling future跨-process or跨-network dispatch without rewrites.
3.  **Observability** — every message carries correlation_id, session_id,
    timestamps, and tracing metadata.
4.  **Resilience** — standardised Result/Error types make error handling,
    retries, cancellations, and timeouts uniform across the system.
5.  **Versioning** — every contract carries a semver version string so
    schema changes are explicitly tracked.

How engines communicate
-----------------------
Engines never call each other directly. Instead:

    SourceEngine.send(TargetEngine, Request) → Envelope
        ↓
    Pipeline / Dispatcher unwraps and routes
        ↓
    TargetEngine.handle(Request) → Response
        ↓
    Result is returned inside the same Envelope

All communication is asynchronous (async/await). The pipeline executor
(see ``app/kernel/pipeline/``) owns the choreography.

Versioning strategy
-------------------
Each contract pair is versioned independently with a **semver** string
(e.g. ``"1.0.0"``). The major version increments when a field is removed
or changed in a breaking way. Minor/patch increments for additive changes.

Backward compatibility strategy
-------------------------------
- Consumers MUST ignore unknown fields (Pydantic ``extra="ignore"``).
- Producers MUST NOT remove fields without a major version bump.
- Optional fields should be preferred over required ones.
- Version negotiation is done via the envelope's ``version`` field.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# ── Engine identifiers ───────────────────────────────────────────────────────


class EngineType(str, Enum):
    """Canonical identifiers for every engine in the Cognitive Kernel."""

    PERCEPTION = "perception"
    INTENT = "intent"
    GOAL = "goal"
    PLANNER = "planner"
    REASONING = "reasoning"
    DECISION = "decision"
    CAPABILITY = "capability"
    POLICY = "policy"
    TASK = "task"
    TOOL = "tool"
    SKILL = "skill"
    REFLECTION = "reflection"
    EXPERIENCE = "experience"
    MEMORY = "memory"
    RESPONSE = "response"
    ORCHESTRATOR = "orchestrator"
    PIPELINE = "pipeline"
    UNKNOWN = "unknown"


# ── Result / Error primitives ────────────────────────────────────────────────


class ResultStatus(str, Enum):
    """Standardised outcome status for every inter-engine operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ErrorInfo(BaseModel):
    """Structured error payload carried in a failed contract response."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(
        ..., description="Machine-readable error code (e.g. 'capability.unavailable')"
    )
    message: str = Field(default="", description="Human-readable error description")
    details: dict[str, Any] | None = Field(
        default=None, description="Arbitrary error context"
    )
    retryable: bool = Field(default=False, description="Whether the caller may retry")
    source_engine: EngineType = Field(
        default=EngineType.UNKNOWN, description="Engine that raised the error"
    )


T = TypeVar("T", bound=BaseModel)


class ContractResult(BaseModel, Generic[T]):
    """Generic result wrapper returned by every engine handler.

    Supports success, failure, retry, cancellation, and timeout outcomes.
    """

    model_config = ConfigDict(frozen=True)

    status: ResultStatus = Field(
        default=ResultStatus.SUCCESS, description="Operation outcome"
    )
    error: ErrorInfo | None = Field(
        default=None, description="Error detail when status is not SUCCESS"
    )
    correlation_id: str = Field(
        default="", description="Links this result to the original request"
    )
    duration_ms: int = Field(default=0, ge=0, description="Handler execution duration")


# ── Envelope ─────────────────────────────────────────────────────────────────


class MessageEnvelope(BaseModel):
    """Universal message envelope wrapping every inter-engine communication.

    The envelope provides the infrastructure layer — routing, tracing,
    versioning — while the ``payload`` carries the domain-specific contract.
    """

    model_config = ConfigDict(frozen=True)

    correlation_id: str = Field(
        default="", description="Unique identifier tracing a request across engines"
    )
    session_id: str = Field(default="", description="Session this message belongs to")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the envelope was created"
    )
    source_engine: EngineType = Field(
        ..., description="Engine that created this message"
    )
    target_engine: EngineType = Field(
        ..., description="Engine this message is addressed to"
    )
    contract_version: str = Field(
        default="1.0.0", description="Semver of the contract schema"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary routing / tracing metadata"
    )
    context_ref: str | None = Field(
        default=None, description="Reference to the shared PipelineContext"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Serialised contract payload"
    )

    @classmethod
    def from_payload(
        cls,
        *,
        payload: BaseModel,
        source_engine: EngineType,
        target_engine: EngineType,
        session_id: str = "",
        correlation_id: str = "",
        contract_version: str = "1.0.0",
        context_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageEnvelope:
        """Build an envelope from a typed Pydantic payload model."""
        return cls(
            correlation_id=correlation_id,
            session_id=session_id,
            source_engine=source_engine,
            target_engine=target_engine,
            contract_version=contract_version,
            context_ref=context_ref,
            metadata=metadata or {},
            payload=payload.model_dump(),
        )


# ── Contract base classes ────────────────────────────────────────────────────


class ContractRequest(BaseModel):
    """Base class for all inter-engine request contracts."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    correlation_id: str = Field(default="", description="Links request to response")
    session_id: str = Field(default="", description="Owning session")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Request creation time"
    )
    source_engine: EngineType = Field(
        default=EngineType.UNKNOWN, description="Calling engine"
    )
    target_engine: EngineType = Field(
        default=EngineType.UNKNOWN, description="Called engine"
    )
    contract_version: str = Field(
        default="1.0.0", description="Contract schema version"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary routing metadata"
    )
    context_ref: str | None = Field(
        default=None, description="PipelineContext reference"
    )


class ContractResponse(BaseModel):
    """Base class for all inter-engine response contracts."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    correlation_id: str = Field(
        default="", description="Echoes the request correlation_id"
    )
    session_id: str = Field(default="", description="Owning session")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response creation time"
    )
    source_engine: EngineType = Field(
        default=EngineType.UNKNOWN, description="Responding engine"
    )
    target_engine: EngineType = Field(
        default=EngineType.UNKNOWN, description="Original caller"
    )
    contract_version: str = Field(
        default="1.0.0", description="Contract schema version"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary response metadata"
    )
    context_ref: str | None = Field(
        default=None, description="PipelineContext reference"
    )

    result: ContractResult[Any] = Field(
        default_factory=lambda: ContractResult(status=ResultStatus.SUCCESS)
    )
