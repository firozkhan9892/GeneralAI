"""Cognitive orchestrator — top-level coordinator for the kernel.

The orchestrator is the **single entry point** for the entire Cognitive
Kernel.  No engine may invoke another engine directly — only the
orchestrator controls execution order through the pipeline executor
and engine dispatcher.

Architecture rationale (ADR-016):
    - Centralized orchestration ensures deterministic execution ordering.
    - Engines communicate only through typed contract messages.
    - The dispatcher resolves engines from the DI container at runtime.
    - Failure policies, cancellation, and deadlines are enforced centrally.
    - Observability (metrics, events, tracing) is wired at the orchestrator level.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.container import DependencyContainer
from app.core.events import EventBus
from app.kernel.agent.models import (
    AgentRequest,
    AgentResponse,
    AgentRunConfig,
)
from app.kernel.agent.runtime import AgentRuntime
from app.kernel.pipeline.dispatcher import EngineDispatcher
from app.kernel.pipeline.execution_context import ExecutionContext
from app.kernel.pipeline.executor import PipelineExecutor
from app.kernel.pipeline.models import PipelineContext, PipelineMetadata
from app.kernel.pipeline.observability import (
    EventPublisher,
    MetricsCollector,
    TracingHook,
)
from app.kernel.pipeline.policies import PolicySet, resilient_policy
from app.kernel.pipeline.stages import build_stage_definitions
from app.kernel.response.models import OutputMessage
from app.tools.context import CancellationToken

log = logging.getLogger(__name__)


class CognitiveOrchestrator:
    """Top-level coordinator for the Cognitive Kernel.

    Owns the pipeline executor, engine dispatcher, and observability
    components.  Receives cognitive requests and drives them through
    the full pipeline.  Also exposes agent execution (``run_agent``,
    ``cancel_agent``) backed by the AgentRuntime, which is resolved
    from the shared DI container.

    Engines are resolved from the DI container — never hardcoded.
    """

    def __init__(
        self,
        container: DependencyContainer | None = None,
        pipeline: PipelineExecutor | None = None,
        dispatcher: EngineDispatcher | None = None,
        policy_set: PolicySet | None = None,
        metrics: MetricsCollector | None = None,
        publisher: EventPublisher | None = None,
        tracer: TracingHook | None = None,
        event_bus: EventBus | None = None,
        agent_runtime: AgentRuntime | None = None,
    ) -> None:
        self._container = container or DependencyContainer()
        self._policy_set = policy_set or resilient_policy()
        self._metrics = metrics or MetricsCollector()
        self._publisher = publisher or EventPublisher(event_bus)
        self._tracer = tracer or TracingHook(self._publisher)

        # Build stage definitions
        stages = build_stage_definitions()

        # Create or use provided dispatcher
        if dispatcher is not None:
            self._dispatcher = dispatcher
        else:
            self._dispatcher = EngineDispatcher(
                container=self._container,
                stages=stages,
                policy_set=self._policy_set,
                metrics=self._metrics,
                publisher=self._publisher,
                tracer=self._tracer,
            )

        # Create or use provided pipeline executor
        if pipeline is not None:
            self._pipeline = pipeline
            self._pipeline.set_dispatcher(self._dispatcher)
        else:
            self._pipeline = PipelineExecutor(
                stages=stages, dispatcher=self._dispatcher
            )

        # Track active sessions for cancellation
        self._active_sessions: dict[str, ExecutionContext] = {}

        # Agent runtime — resolved lazily from the container so the
        # orchestrator and agent share the same tool registry.
        self._agent_runtime = agent_runtime

    @property
    def container(self) -> DependencyContainer:
        return self._container

    @property
    def pipeline(self) -> PipelineExecutor:
        return self._pipeline

    @property
    def dispatcher(self) -> EngineDispatcher:
        return self._dispatcher

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def publisher(self) -> EventPublisher:
        return self._publisher

    @property
    def tracer(self) -> TracingHook:
        return self._tracer

    async def process(
        self,
        raw_input: Any,
        session_id: str | None = None,
        user_id: str | None = None,
        timeout_s: int = 300,
        ttl_s: int = 3600,
        metadata: dict[str, Any] | None = None,
    ) -> OutputMessage:
        """Process input through the full cognitive pipeline.

        Args:
            raw_input: Raw input (RawMessage or compatible).
            session_id: Optional session ID (generated if not provided).
            user_id: Optional user ID.
            timeout_s: Per-stage timeout in seconds.
            ttl_s: Total execution TTL in seconds.
            metadata: Additional execution metadata.

        Returns:
            Final output message from the response builder.

        Raises:
            Exception: If a stage fails with an ABORT policy.
            asyncio.CancelledError: If execution is cancelled.
            TimeoutError: If the deadline is exceeded.
        """
        # Create execution context with generated IDs
        exec_ctx = ExecutionContext.create(
            session_id=session_id or "",
            user_id=user_id,
            pipeline_id="cognitive-v1",
            timeout_s=timeout_s,
            ttl_s=ttl_s,
            metadata=metadata,
        )

        # Register session for cancellation tracking
        self._active_sessions[exec_ctx.session_id] = exec_ctx

        # Build pipeline context
        context = PipelineContext(
            session_id=exec_ctx.session_id,
            metadata=PipelineMetadata(started_at=exec_ctx.created_at),
        )

        # Store the raw input in the context (first stage reads it)
        context.percept = raw_input

        log.info(
            "Orchestrator.process: session=%s trace=%s correlation=%s",
            exec_ctx.session_id,
            exec_ctx.trace_id,
            exec_ctx.correlation_id,
        )

        try:
            # Execute the full pipeline
            await self._pipeline.execute(context, exec_ctx)

            # Extract the final response
            output = context.response
            if output is None:
                output = OutputMessage(
                    content="",
                    session_id=exec_ctx.session_id,
                    success=False,
                    error="Pipeline completed but no response was produced",
                )
            elif not isinstance(output, OutputMessage):
                output = OutputMessage(
                    content=str(output),
                    session_id=exec_ctx.session_id,
                    success=True,
                )

            return output

        except asyncio.CancelledError:
            log.info("Pipeline cancelled: session=%s", exec_ctx.session_id)
            return OutputMessage(
                content="",
                session_id=exec_ctx.session_id,
                success=False,
                error="Execution cancelled",
            )

        except TimeoutError as exc:
            log.warning(
                "Pipeline timeout: session=%s error=%s", exec_ctx.session_id, exc
            )
            return OutputMessage(
                content="",
                session_id=exec_ctx.session_id,
                success=False,
                error=f"Timeout: {exc}",
            )

        except Exception as exc:
            log.error(
                "Pipeline error: session=%s error=%s",
                exec_ctx.session_id,
                exc,
                exc_info=True,
            )
            return OutputMessage(
                content="",
                session_id=exec_ctx.session_id,
                success=False,
                error=str(exc),
            )

        finally:
            # Clean up session tracking
            self._active_sessions.pop(exec_ctx.session_id, None)

    async def cancel(self, session_id: str, reason: str = "user_requested") -> None:
        """Cancel an active session.

        If the session is currently executing:
            - The current stage stops (checked on next iteration).
            - Pending stages never execute.
            - Cleanup hooks run.

        Args:
            session_id: The session to cancel.
            reason: Why the session is being cancelled.
        """
        exec_ctx = self._active_sessions.get(session_id)
        if exec_ctx is not None:
            exec_ctx.cancellation_token.cancel(reason)
            log.info("Cancelled session %s: %s", session_id, reason)
        else:
            log.warning("Cancel requested for unknown session: %s", session_id)

        # Also notify the dispatcher
        self._dispatcher.cancel_session(session_id, reason)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Return a summary of all collected metrics."""
        return self._metrics.summary()

    def get_tracing_summary(self) -> dict[str, Any]:
        """Return a summary of all recorded trace spans."""
        return self._tracer.summary()

    def get_active_sessions(self) -> list[str]:
        """Return a list of currently active session IDs."""
        return list(self._active_sessions.keys())

    # ── Agent runtime ────────────────────────────────────────────────

    @property
    def agent_runtime(self) -> AgentRuntime:
        """Return the agent runtime, resolving it from the container if needed."""
        if self._agent_runtime is None:
            if self._container.has(AgentRuntime):
                self._agent_runtime = self._container.resolve(AgentRuntime)
            else:
                self._agent_runtime = AgentRuntime()
        return self._agent_runtime

    async def run_agent(
        self,
        request: AgentRequest,
        *,
        config: AgentRunConfig | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AgentResponse:
        """Execute a full agent run through the AgentRuntime.

        The runtime is resolved from the shared DI container so it uses
        the same tool registry as the rest of the kernel.

        Args:
            request: The agent request (raw input + session).
            config: Optional per-run config override.
            cancellation_token: Optional cooperative cancellation signal.

        Returns:
            The final agent response.
        """
        return await self.agent_runtime.run(
            request,
            config=config,
            cancellation_token=cancellation_token,
        )

    def cancel_agent(self, session_id: str, reason: str = "user_requested") -> None:
        """Request cancellation of an active agent session."""
        self.agent_runtime.cancel(session_id, reason)

    def get_active_agent_sessions(self) -> list[str]:
        """Return currently active agent session identifiers."""
        return self.agent_runtime.get_active_sessions()
