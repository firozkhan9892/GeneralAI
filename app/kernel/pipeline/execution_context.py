"""Execution context — immutable context for a single pipeline run.

Carries tracing IDs, deadline, cancellation token, and metadata through
every stage of the cognitive pipeline.  The context is **frozen** —
stages produce updated copies via ``with_updates`` rather than mutating.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.kernel.contracts.base import EngineType


@dataclass(frozen=True)
class CancellationToken:
    """A lightweight, thread-safe cancellation signal.

    Once ``cancel()`` is called the token is permanently cancelled.
    All stages check ``is_cancelled`` before proceeding.
    """

    _cancelled: bool = field(default=False, init=False)
    _reason: str = field(default="", init=False)

    def cancel(self, reason: str = "") -> None:
        """Mark this token as cancelled.

        Subsequent calls are no-ops.
        """
        object.__setattr__(self, "_cancelled", True)
        if reason:
            object.__setattr__(self, "_reason", reason)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._reason


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable context for a single pipeline execution.

    Attributes:
        session_id: Owning session identifier.
        correlation_id: Links request/response across stages.
        trace_id: Distributed tracing trace identifier.
        span_id: Current span identifier.
        parent_span_id: Parent span (for nested calls).
        request_id: Unique request identifier.
        user_id: Optional user identifier.
        pipeline_id: Pipeline definition identifier.
        current_stage: The stage currently executing.
        deadline: Absolute deadline (``None`` = no deadline).
        timeout_s: Per-stage timeout in seconds.
        retry_count: Current retry attempt count.
        metadata: Arbitrary execution metadata.
        cancellation_token: Token for cancellation propagation.
        created_at: When this context was created.
    """

    session_id: str
    correlation_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    request_id: str
    user_id: str | None
    pipeline_id: str
    current_stage: EngineType
    deadline: datetime | None
    timeout_s: int
    retry_count: int
    metadata: dict[str, Any]
    cancellation_token: CancellationToken
    created_at: datetime

    @classmethod
    def create(
        cls,
        session_id: str = "",
        user_id: str | None = None,
        pipeline_id: str = "cognitive-v1",
        timeout_s: int = 300,
        ttl_s: int = 3600,
        metadata: dict[str, Any] | None = None,
        parent_trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> ExecutionContext:
        """Create a new execution context with generated IDs."""
        now = datetime.now(timezone.utc)
        trace_id = parent_trace_id or uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        return cls(
            session_id=session_id or uuid.uuid4().hex,
            correlation_id=uuid.uuid4().hex,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            request_id=uuid.uuid4().hex,
            user_id=user_id,
            pipeline_id=pipeline_id,
            current_stage=EngineType.UNKNOWN,
            deadline=now + timedelta(seconds=ttl_s),
            timeout_s=timeout_s,
            retry_count=0,
            metadata=metadata or {},
            cancellation_token=CancellationToken(),
            created_at=now,
        )

    def with_stage(self, stage: EngineType) -> ExecutionContext:
        """Return a copy with ``current_stage`` updated."""
        return ExecutionContext(
            session_id=self.session_id,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            request_id=self.request_id,
            user_id=self.user_id,
            pipeline_id=self.pipeline_id,
            current_stage=stage,
            deadline=self.deadline,
            timeout_s=self.timeout_s,
            retry_count=self.retry_count,
            metadata=self.metadata,
            cancellation_token=self.cancellation_token,
            created_at=self.created_at,
        )

    def with_retry(self) -> ExecutionContext:
        """Return a copy with ``retry_count`` incremented."""
        return ExecutionContext(
            session_id=self.session_id,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            request_id=self.request_id,
            user_id=self.user_id,
            pipeline_id=self.pipeline_id,
            current_stage=self.current_stage,
            deadline=self.deadline,
            timeout_s=self.timeout_s,
            retry_count=self.retry_count + 1,
            metadata=self.metadata,
            cancellation_token=self.cancellation_token,
            created_at=self.created_at,
        )

    def with_metadata(self, key: str, value: Any) -> ExecutionContext:
        """Return a copy with an additional metadata entry."""
        new_meta = dict(self.metadata)
        new_meta[key] = value
        return ExecutionContext(
            session_id=self.session_id,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            request_id=self.request_id,
            user_id=self.user_id,
            pipeline_id=self.pipeline_id,
            current_stage=self.current_stage,
            deadline=self.deadline,
            timeout_s=self.timeout_s,
            retry_count=self.retry_count,
            metadata=new_meta,
            cancellation_token=self.cancellation_token,
            created_at=self.created_at,
        )

    @property
    def is_deadline_expired(self) -> bool:
        """Whether the execution deadline has passed."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) >= self.deadline

    @property
    def is_cancelled(self) -> bool:
        """Whether this execution has been cancelled."""
        return self.cancellation_token.is_cancelled

    @property
    def remaining_seconds(self) -> float | None:
        """Seconds remaining until deadline (``None`` if no deadline)."""
        if self.deadline is None:
            return None
        delta = (self.deadline - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)
