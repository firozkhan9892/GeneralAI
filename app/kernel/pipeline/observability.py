"""Observability hooks for pipeline execution.

Provides:
    - ``StageMetrics``: timing and outcome tracking per stage.
    - ``MetricsCollector``: aggregates metrics across the pipeline.
    - ``EventPublisher``: publishes pipeline events via the EventBus.
    - ``TracingHook``: structured logging and trace span recording.

No external integrations (OpenTelemetry, Prometheus, etc.) are wired yet —
these are abstract hooks ready for future instrumentation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.kernel.contracts.base import EngineType
from app.kernel.contracts.events import PipelineEvent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageMetrics:
    """Metrics for a single stage execution."""

    stage: EngineType
    started_at: datetime
    ended_at: datetime | None
    duration_ms: int
    success: bool
    error: str | None
    retry_count: int
    exception: Exception | None = None

    @property
    def is_complete(self) -> bool:
        return self.ended_at is not None


@dataclass
class MetricsCollector:
    """Aggregates metrics across all stages of a pipeline run."""

    _stages: list[StageMetrics] = field(default_factory=list)
    _by_stage: dict[EngineType, list[StageMetrics]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _total_start: datetime | None = None
    _total_end: datetime | None = None

    def record_start(self, stage: EngineType) -> None:
        """Record that a stage has started."""
        if self._total_start is None:
            self._total_start = datetime.now(timezone.utc)

    def record_end(
        self,
        stage: EngineType,
        started_at: datetime,
        ended_at: datetime | None = None,
        success: bool = True,
        error: str | None = None,
        retry_count: int = 0,
        exception: Exception | None = None,
    ) -> StageMetrics:
        """Record that a stage has completed and return its metrics."""
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)
        metrics = StageMetrics(
            stage=stage,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            success=success,
            error=error,
            retry_count=retry_count,
            exception=exception,
        )
        self._stages.append(metrics)
        self._by_stage[stage].append(metrics)
        if self._total_end is None or ended_at > self._total_end:
            self._total_end = ended_at
        return metrics

    @property
    def total_duration_ms(self) -> int:
        """Total pipeline execution time in milliseconds."""
        if self._total_start is None or self._total_end is None:
            return 0
        return int((self._total_end - self._total_start).total_seconds() * 1000)

    @property
    def stage_count(self) -> int:
        """Number of stages that recorded metrics."""
        return len(self._stages)

    @property
    def success_count(self) -> int:
        """Number of stages that succeeded."""
        return sum(1 for s in self._stages if s.success)

    @property
    def failure_count(self) -> int:
        """Number of stages that failed."""
        return sum(1 for s in self._stages if not s.success)

    @property
    def total_retries(self) -> int:
        """Total retries across all stages."""
        return sum(s.retry_count for s in self._stages)

    def get_stage_metrics(self, stage: EngineType) -> list[StageMetrics]:
        """Return all metrics for a given stage."""
        return list(self._by_stage.get(stage, []))

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all metrics."""
        return {
            "total_duration_ms": self.total_duration_ms,
            "stage_count": self.stage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_retries": self.total_retries,
            "stages": [
                {
                    "stage": str(s.stage),
                    "duration_ms": s.duration_ms,
                    "success": s.success,
                    "error": s.error,
                    "retry_count": s.retry_count,
                }
                for s in self._stages
            ],
        }


class EventPublisher:
    """Publishes pipeline events to subscribers and (optionally) an EventBus.

    Supported event types
    ---------------------
    The publisher accepts two categories of events:

    1. **PipelineEvent** (canonical system events) — enum values such as
       ``PipelineEvent.PERCEPTION_STARTED`` used by all GeneralAI core
       components.
    2. **str** (custom application/plugin events) — arbitrary string identifiers
       for application-specific or plugin-defined events.

    Delivery guarantees
    -------------------
    Both event types are delivered identically to all registered subscribers.
    Subscribers receive exactly the event object that was published — no
    implicit conversion between ``PipelineEvent`` and ``str`` occurs.

    Compatibility
    -------------
    Existing string-event integrations remain supported.  Core framework
    components continue to prefer ``PipelineEvent`` for canonical system events.

    Versioning
    ----------
    This dual-type contract is part of the public API and must remain
    backward compatible unless explicitly deprecated through an ADR.
    """

    def __init__(self, event_bus: Any | None = None) -> None:
        self._event_bus = event_bus
        self._subscribers: list[
            Callable[[PipelineEvent | str, dict[str, Any]], None]
        ] = []

    def subscribe(
        self, callback: Callable[[PipelineEvent | str, dict[str, Any]], None]
    ) -> None:
        """Register a callback to receive all published events."""
        self._subscribers.append(callback)

    def publish(
        self, event: PipelineEvent | str, data: dict[str, Any] | None = None
    ) -> None:
        """Publish a pipeline event.

        If an EventBus is wired, the event is forwarded to it.
        All registered callbacks are also invoked.
        """
        payload = data or {}
        event_value = event.value if isinstance(event, PipelineEvent) else str(event)
        if self._event_bus is not None:
            try:
                self._event_bus.emit(event_value, payload)
            except Exception:
                log.debug("EventBus emit failed for %s", event_value, exc_info=True)
        for callback in self._subscribers:
            try:
                callback(event, payload)
            except Exception:
                log.debug(
                    "Event subscriber callback failed for %s",
                    event_value,
                    exc_info=True,
                )

    def publish_stage_start(
        self, stage: EngineType | str, context: dict[str, Any] | None = None
    ) -> None:
        """Publish a stage-started event."""
        stage_str = _stage_value(stage)
        event_map = {
            EngineType.PERCEPTION: PipelineEvent.PERCEPTION_STARTED,
            EngineType.INTENT: PipelineEvent.INTENT_STARTED,
            EngineType.GOAL: PipelineEvent.GOAL_CREATED,
            EngineType.PLANNER: PipelineEvent.PLANNER_STARTED,
            EngineType.REASONING: PipelineEvent.REASONING_STARTED,
            EngineType.DECISION: PipelineEvent.DECISION_STARTED,
            EngineType.CAPABILITY: PipelineEvent.CAPABILITY_STARTED,
            EngineType.POLICY: PipelineEvent.POLICY_STARTED,
            EngineType.TASK: PipelineEvent.TASK_STARTED,
            EngineType.TOOL: PipelineEvent.TOOL_STARTED,
            EngineType.REFLECTION: PipelineEvent.REFLECTION_STARTED,
            EngineType.EXPERIENCE: PipelineEvent.EXPERIENCE_STARTED,
            EngineType.MEMORY: PipelineEvent.PIPELINE_STAGE_STARTED,
            EngineType.RESPONSE: PipelineEvent.RESPONSE_BUILT,
        }
        event = event_map.get(stage, PipelineEvent.PIPELINE_STAGE_STARTED)  # type: ignore[arg-type]
        data = {"stage": stage_str, "timestamp": datetime.now(timezone.utc).isoformat()}
        if context:
            data.update(context)
        self.publish(event, data)

    def publish_stage_complete(
        self, stage: EngineType | str, context: dict[str, Any] | None = None
    ) -> None:
        """Publish a stage-completed event."""
        stage_str = _stage_value(stage)
        event_map = {
            EngineType.PERCEPTION: PipelineEvent.PERCEPTION_COMPLETED,
            EngineType.INTENT: PipelineEvent.INTENT_COMPLETED,
            EngineType.GOAL: PipelineEvent.GOAL_COMPLETED,
            EngineType.PLANNER: PipelineEvent.PLAN_GENERATED,
            EngineType.REASONING: PipelineEvent.REASONING_COMPLETED,
            EngineType.DECISION: PipelineEvent.DECISION_SELECTED,
            EngineType.CAPABILITY: PipelineEvent.CAPABILITY_RESOLVED,
            EngineType.POLICY: PipelineEvent.POLICY_EVALUATED,
            EngineType.TASK: PipelineEvent.TASK_COMPLETED,
            EngineType.TOOL: PipelineEvent.TOOL_COMPLETED,
            EngineType.REFLECTION: PipelineEvent.REFLECTION_COMPLETED,
            EngineType.EXPERIENCE: PipelineEvent.EXPERIENCE_STORED,
            EngineType.MEMORY: PipelineEvent.PIPELINE_STAGE_COMPLETED,
            EngineType.RESPONSE: PipelineEvent.RESPONSE_BUILT,
        }
        event = event_map.get(stage, PipelineEvent.PIPELINE_STAGE_COMPLETED)  # type: ignore[arg-type]
        data = {"stage": stage_str, "timestamp": datetime.now(timezone.utc).isoformat()}
        if context:
            data.update(context)
        self.publish(event, data)

    def publish_stage_error(
        self, stage: EngineType | str, error: str, context: dict[str, Any] | None = None
    ) -> None:
        """Publish a stage-error event."""
        stage_str = _stage_value(stage)
        event_map = {
            EngineType.PERCEPTION: PipelineEvent.PERCEPTION_FAILED,
            EngineType.INTENT: PipelineEvent.INTENT_FAILED,
            EngineType.GOAL: PipelineEvent.GOAL_FAILED,
            EngineType.PLANNER: PipelineEvent.PLANNER_FAILED,
            EngineType.REASONING: PipelineEvent.REASONING_FAILED,
            EngineType.DECISION: PipelineEvent.DECISION_FAILED,
            EngineType.CAPABILITY: PipelineEvent.CAPABILITY_UNAVAILABLE,
            EngineType.POLICY: PipelineEvent.POLICY_DENIED,
            EngineType.TASK: PipelineEvent.TASK_FAILED,
            EngineType.TOOL: PipelineEvent.TOOL_FAILED,
            EngineType.REFLECTION: PipelineEvent.REFLECTION_FAILED,
            EngineType.EXPERIENCE: PipelineEvent.EXPERIENCE_FAILED,
            EngineType.MEMORY: PipelineEvent.PIPELINE_STAGE_FAILED,
            EngineType.RESPONSE: PipelineEvent.RESPONSE_FAILED,
        }
        event = event_map.get(stage, PipelineEvent.PIPELINE_STAGE_FAILED)  # type: ignore[arg-type]
        data = {
            "stage": stage_str,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if context:
            data.update(context)
        self.publish(event, data)


def _stage_value(stage: EngineType | str) -> str:
    """Normalize a stage identifier to its string value."""
    return stage.value if isinstance(stage, EngineType) else str(stage)


class TracingHook:
    """Records structured trace information for pipeline execution.

    Produces log output in a structured format suitable for
    distributed tracing systems (OpenTelemetry, etc.) when wired.
    """

    def __init__(self, publisher: EventPublisher | None = None) -> None:
        self._publisher = publisher
        self._spans: list[dict[str, Any]] = []

    def start_span(
        self,
        stage: EngineType | str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Begin a new trace span for a stage."""
        stage_str = _stage_value(stage)
        span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "stage": stage_str,
            "start_time": time.monotonic(),
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._spans.append(span)
        log.debug(
            "span.start trace=%s span=%s stage=%s",
            trace_id,
            span_id,
            stage_str,
        )
        return span

    def end_span(
        self,
        span: dict[str, Any],
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """End a trace span and record its duration."""
        span["end_time"] = time.monotonic()
        span["duration_ms"] = int((span["end_time"] - span["start_time"]) * 1000)
        span["success"] = success
        span["error"] = error
        span["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        log.debug(
            "span.end trace=%s span=%s stage=%s duration_ms=%d success=%s",
            span["trace_id"],
            span["span_id"],
            span["stage"],
            span["duration_ms"],
            success,
        )

    @property
    def spans(self) -> list[dict[str, Any]]:
        """Return all recorded spans."""
        return list(self._spans)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all spans."""
        return {
            "span_count": len(self._spans),
            "total_duration_ms": sum(s.get("duration_ms", 0) for s in self._spans),
            "spans": [
                {
                    "trace_id": s["trace_id"],
                    "span_id": s["span_id"],
                    "stage": s["stage"],
                    "duration_ms": s.get("duration_ms", 0),
                    "success": s.get("success", True),
                    "error": s.get("error"),
                }
                for s in self._spans
            ],
        }
