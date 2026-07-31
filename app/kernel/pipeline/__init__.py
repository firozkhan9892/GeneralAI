"""Pipeline — stage execution infrastructure."""

from __future__ import annotations

from app.kernel.pipeline.dispatcher import EngineDispatcher, StageDefinition
from app.kernel.pipeline.execution_context import CancellationToken, ExecutionContext
from app.kernel.pipeline.executor import PipelineExecutor
from app.kernel.pipeline.models import PipelineContext, PipelineMetadata
from app.kernel.pipeline.observability import (
    EventPublisher,
    MetricsCollector,
    StageMetrics,
    TracingHook,
)
from app.kernel.pipeline.policies import FailurePolicy, PolicySet, StagePolicy

__all__ = [
    "CancellationToken",
    "EngineDispatcher",
    "EventPublisher",
    "ExecutionContext",
    "FailurePolicy",
    "MetricsCollector",
    "PipelineContext",
    "PipelineExecutor",
    "PipelineMetadata",
    "PolicySet",
    "StageDefinition",
    "StageMetrics",
    "StagePolicy",
    "TracingHook",
]
