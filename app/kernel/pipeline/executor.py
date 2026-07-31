"""Pipeline executor — orchestrates stage execution through the dispatcher.

The executor owns the ordered sequence of stages and drives the
pipeline context through each one.  It delegates to the
``EngineDispatcher`` for actual engine invocation, retry, and
observability.
"""

from __future__ import annotations

import logging

from app.kernel.pipeline.dispatcher import EngineDispatcher, StageDefinition
from app.kernel.pipeline.execution_context import ExecutionContext
from app.kernel.pipeline.models import PipelineContext

log = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes the cognitive pipeline through all registered stages.

    The executor is configured with a list of ``StageDefinition``
    objects that describe the pipeline ordering.  It uses an
    ``EngineDispatcher`` to invoke each stage's engine.
    """

    def __init__(
        self,
        stages: list[StageDefinition] | None = None,
        dispatcher: EngineDispatcher | None = None,
    ) -> None:
        self._stages = stages or []
        self._dispatcher = dispatcher
        self._stage_map: dict[str, StageDefinition] = {s.name: s for s in self._stages}

    @property
    def stages(self) -> list[StageDefinition]:
        """Return the ordered stage definitions."""
        return list(self._stages)

    @property
    def dispatcher(self) -> EngineDispatcher | None:
        """Return the engine dispatcher."""
        return self._dispatcher

    def set_dispatcher(self, dispatcher: EngineDispatcher) -> None:
        """Set the engine dispatcher."""
        self._dispatcher = dispatcher

    def register_stage(self, stage: StageDefinition) -> None:
        """Register a pipeline stage.

        Stages must be registered in execution order.
        """
        self._stages.append(stage)
        self._stage_map[stage.name] = stage
        log.debug(
            "Registered pipeline stage '%s' (order=%d)",
            stage.name,
            len(self._stages) - 1,
        )

    def get_stage(self, name: str) -> StageDefinition | None:
        """Look up a stage by name."""
        return self._stage_map.get(name)

    async def execute(
        self,
        context: PipelineContext,
        exec_ctx: ExecutionContext,
    ) -> PipelineContext:
        """Execute the full pipeline through all registered stages.

        Args:
            context: Initial pipeline context (will be mutated in place).
            exec_ctx: Immutable execution context with tracing IDs and deadline.

        Returns:
            The pipeline context after all stages have executed.

        Raises:
            Exception: If a stage fails with an ABORT policy.
            asyncio.CancelledError: If execution is cancelled.
            TimeoutError: If the deadline is exceeded.
        """
        if self._dispatcher is None:
            raise RuntimeError("PipelineExecutor has no dispatcher configured")

        log.info(
            "Starting pipeline execution: session=%s trace=%s stages=%d",
            exec_ctx.session_id,
            exec_ctx.trace_id,
            len(self._stages),
        )

        for stage in self._stages:
            # Update execution context with current stage
            stage_exec_ctx = exec_ctx.with_stage(stage.engine_type)

            # Update pipeline context metadata
            context.metadata.current_stage = len(self._dispatcher.metrics._stages)

            # Dispatch the stage
            result = await self._dispatcher.dispatch(stage, context, stage_exec_ctx)

            # Store the result in the pipeline context
            if result is not None:
                setattr(context, stage.response_field, result)

            # Check for cancellation after each stage
            if exec_ctx.is_cancelled:
                log.info("Pipeline execution cancelled at stage '%s'", stage.name)
                break

            # Check deadline
            if exec_ctx.is_deadline_expired:
                log.warning(
                    "Pipeline execution deadline expired at stage '%s'", stage.name
                )
                break

        log.info(
            "Pipeline execution complete: session=%s stages_executed=%d",
            exec_ctx.session_id,
            len(self._dispatcher.metrics._stages),
        )

        return context

    async def execute_from_stage(
        self,
        context: PipelineContext,
        exec_ctx: ExecutionContext,
        start_stage: str,
    ) -> PipelineContext:
        """Execute the pipeline starting from a specific stage.

        Args:
            context: Initial pipeline context.
            exec_ctx: Execution context.
            start_stage: Name of the stage to start from.

        Returns:
            The pipeline context after execution.
        """
        if start_stage not in self._stage_map:
            raise ValueError(f"Unknown stage: {start_stage}")

        start_idx = self._stages.index(self._stage_map[start_stage])
        stages_to_run = self._stages[start_idx:]

        original_stages = self._stages
        self._stages = stages_to_run
        try:
            return await self.execute(context, exec_ctx)
        finally:
            self._stages = original_stages
