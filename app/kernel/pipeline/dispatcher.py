"""Engine dispatcher — routes execution requests to the correct engine.

The dispatcher is the single component that knows how to invoke each
engine.  It:

    - Locates engines from the DI container (no hardcoded instances).
    - Validates that the engine implements the expected interface.
    - Enforces stage ordering (engines never call each other directly).
    - Records execution timing.
    - Publishes pipeline events via the EventPublisher.

Engines communicate through typed contract request/response pairs.
The dispatcher translates between the pipeline context and these contracts.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from app.core.container import DependencyContainer
from app.kernel.contracts.base import EngineType
from app.kernel.pipeline.execution_context import ExecutionContext
from app.kernel.pipeline.observability import (
    EventPublisher,
    MetricsCollector,
    TracingHook,
)
from app.kernel.pipeline.policies import FailurePolicy, PolicySet, StagePolicy

log = logging.getLogger(__name__)


class StageDefinition:
    """Defines a single pipeline stage and how to dispatch to it.

    Attributes:
        engine_type: The EngineType this stage belongs to.
        name: Human-readable stage name.
        engine_attr: Attribute name on the orchestrator for the engine instance.
        method_name: Method name to call on the engine.
        request_field: Field on PipelineContext where the request input lives.
        response_field: Field on PipelineContext where the response output is stored.
        request_builder: Function (context, exec_ctx) -> ContractRequest.
        response_extractor: Function (response, context) -> Any (value to store).
    """

    def __init__(
        self,
        engine_type: EngineType,
        name: str,
        engine_attr: str,
        method_name: str,
        request_field: str,
        response_field: str,
        request_builder: Callable[[Any, ExecutionContext], Any],
        response_extractor: Callable[[Any, Any], Any],
        engine_class: type | None = None,
    ) -> None:
        self.engine_class = engine_class
        self.engine_type = engine_type
        self.name = name
        self.engine_attr = engine_attr
        self.method_name = method_name
        self.request_field = request_field
        self.response_field = response_field
        self.request_builder = request_builder
        self.response_extractor = response_extractor

    @property
    def stage_name(self) -> str:
        """Name used for policy lookups."""
        return self.name


class EngineDispatcher:
    """Dispatches execution requests to registered engines.

    The dispatcher holds a reference to the DI container and uses it
    to resolve engine instances at dispatch time.  This ensures
    engines are never hardcoded — they come from the container.
    """

    def __init__(
        self,
        container: DependencyContainer,
        stages: list[StageDefinition],
        policy_set: PolicySet | None = None,
        metrics: MetricsCollector | None = None,
        publisher: EventPublisher | None = None,
        tracer: TracingHook | None = None,
    ) -> None:
        self._container = container
        self._stages = stages
        self._stage_map: dict[EngineType, StageDefinition] = {
            s.engine_type: s for s in stages
        }
        self._ordered_engine_types: list[EngineType] = [s.engine_type for s in stages]
        self._policy_set = policy_set or PolicySet()
        self._metrics = metrics or MetricsCollector()
        self._publisher = publisher or EventPublisher()
        self._tracer = tracer or TracingHook()
        self._cancelled_sessions: set[str] = set()

    @property
    def ordered_stages(self) -> list[StageDefinition]:
        """Return stages in execution order."""
        return list(self._stages)

    @property
    def ordered_engine_types(self) -> list[EngineType]:
        """Return engine types in execution order."""
        return list(self._ordered_engine_types)

    def get_stage(self, engine_type: EngineType) -> StageDefinition | None:
        """Look up a stage definition by engine type."""
        return self._stage_map.get(engine_type)

    def get_policy_for_stage(self, stage_name: str) -> StagePolicy:
        """Return the failure policy for a stage."""
        return self._policy_set.get_policy_for_stage(stage_name)

    async def dispatch(
        self,
        stage: StageDefinition,
        context: Any,
        exec_ctx: ExecutionContext,
    ) -> Any:
        """Dispatch a single stage execution.

        Args:
            stage: The stage definition to execute.
            context: The PipelineContext (mutable).
            exec_ctx: The immutable execution context.

        Returns:
            The extracted result from the stage's response.

        Raises:
            Exception: If the stage fails and the policy is ABORT.
        """
        policy = self.get_policy_for_stage(stage.stage_name)
        engine = self._resolve_engine(stage)
        method = self._validate_engine_interface(engine, stage)

        # Check cancellation
        if exec_ctx.is_cancelled:
            log.debug("Stage %s skipped — execution cancelled", stage.name)
            self._publisher.publish_stage_error(
                stage.engine_type,
                "cancelled",
                {"reason": exec_ctx.cancellation_token.reason},
            )
            raise asyncio.CancelledError(
                f"Stage {stage.name} cancelled: {exec_ctx.cancellation_token.reason}"
            )

        # Check deadline
        if exec_ctx.is_deadline_expired:
            log.warning("Stage %s skipped — deadline expired", stage.name)
            self._publisher.publish_stage_error(stage.engine_type, "deadline_expired")
            raise TimeoutError(f"Stage {stage.name} deadline expired")

        # Build request
        request = stage.request_builder(context, exec_ctx)

        # Execute with retry
        last_exception: Exception | None = None
        attempts = 0
        max_attempts = (
            policy.max_retries + 1 if policy.policy == FailurePolicy.RETRY else 1
        )

        while attempts < max_attempts:
            attempts += 1
            started_at = datetime.now(timezone.utc)

            # Publish stage start
            self._publisher.publish_stage_start(
                stage.engine_type,
                {"attempt": attempts, "correlation_id": exec_ctx.correlation_id},
            )

            # Start trace span
            span = self._tracer.start_span(
                stage.engine_type,
                exec_ctx.trace_id,
                exec_ctx.span_id,
                exec_ctx.parent_span_id,
                {"attempt": attempts, "stage": stage.name},
            )

            try:
                result = await self._execute_with_timeout(
                    method,
                    request,
                    exec_ctx,
                    stage,
                )
                ended_at = datetime.now(timezone.utc)
                self._tracer.end_span(span, success=True)
                self._metrics.record_end(
                    stage.engine_type,
                    started_at,
                    ended_at,
                    success=True,
                    retry_count=attempts - 1,
                )
                self._publisher.publish_stage_complete(
                    stage.engine_type,
                    {
                        "duration_ms": int(
                            (ended_at - started_at).total_seconds() * 1000
                        )
                    },
                )
                return stage.response_extractor(result, context)

            except asyncio.CancelledError:
                ended_at = datetime.now(timezone.utc)
                self._tracer.end_span(span, success=False, error="cancelled")
                self._metrics.record_end(
                    stage.engine_type,
                    started_at,
                    ended_at,
                    success=False,
                    error="cancelled",
                    retry_count=attempts - 1,
                )
                raise

            except Exception as exc:
                ended_at = datetime.now(timezone.utc)
                self._tracer.end_span(span, success=False, error=str(exc))
                self._metrics.record_end(
                    stage.engine_type,
                    started_at,
                    ended_at,
                    success=False,
                    error=str(exc),
                    retry_count=attempts - 1,
                    exception=exc,
                )
                self._publisher.publish_stage_error(
                    stage.engine_type,
                    str(exc),
                    {"attempt": attempts, "error_type": type(exc).__name__},
                )
                last_exception = exc

                if policy.policy == FailurePolicy.RETRY and attempts < max_attempts:
                    await asyncio.sleep(policy.retry_delay_s)
                    continue

                # Handle policy
                if policy.policy == FailurePolicy.ABORT:
                    raise
                elif policy.policy == FailurePolicy.CONTINUE:
                    log.warning("Stage %s failed — continuing: %s", stage.name, exc)
                    return None
                elif policy.policy == FailurePolicy.FALLBACK:
                    fallback = policy.get_fallback()
                    log.warning("Stage %s failed — using fallback: %s", stage.name, exc)
                    return fallback
                else:
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        return None

    def _resolve_engine(self, stage: StageDefinition) -> Any:
        """Resolve the engine instance from the DI container."""
        # Try resolving by class type if engine_class is set
        if stage.engine_class is not None:
            try:
                return self._container.resolve(stage.engine_class)
            except Exception:
                pass
        # Fallback: raise descriptive error
        raise RuntimeError(
            f"Engine '{stage.name}' ({stage.engine_type.value}) not registered "
            f"in the DI container. Register the engine class with "
            f"container.register_singleton(<EngineClass>) or provide "
            f"an engine_class on the StageDefinition."
        )

    def _validate_engine_interface(
        self, engine: Any, stage: StageDefinition
    ) -> Callable[..., Awaitable[Any]]:
        """Validate that the engine implements the expected method.

        Returns:
            The bound async method ready to call.
        """
        method = getattr(engine, stage.method_name, None)
        if method is None:
            raise RuntimeError(
                f"Engine '{type(engine).__name__}' does not implement '{stage.method_name}'"
            )
        if not callable(method):
            raise RuntimeError(
                f"'{type(engine).__name__}.{stage.method_name}' is not callable"
            )
        return method

    async def _execute_with_timeout(
        self,
        method: Callable[..., Awaitable[Any]],
        request: Any,
        exec_ctx: ExecutionContext,
        stage: StageDefinition,
    ) -> Any:
        """Execute the engine method with a timeout.

        The timeout is the minimum of the stage's timeout and the
        remaining time until the execution deadline.
        """
        timeout = exec_ctx.timeout_s

        # Adjust timeout if deadline is closer
        remaining = exec_ctx.remaining_seconds
        if remaining is not None:
            timeout = min(timeout, int(remaining))

        if timeout <= 0:
            raise TimeoutError(f"No time remaining for stage {stage.name}")

        try:
            return await asyncio.wait_for(method(request), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Stage {stage.name} timed out after {timeout}s")

    def cancel_session(self, session_id: str, reason: str = "user_requested") -> None:
        """Mark a session as cancelled.

        This does not interrupt running stages directly — they
        check the cancellation token on their next iteration.
        """
        self._cancelled_sessions.add(session_id)
        log.info("Session %s cancelled: %s", session_id, reason)

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def publisher(self) -> EventPublisher:
        return self._publisher

    @property
    def tracer(self) -> TracingHook:
        return self._tracer
