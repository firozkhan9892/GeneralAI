"""Workflow execution engine.

The :class:`WorkflowExecutor` walks a workflow's step DAG, dispatching
every step through a :class:`StepTypeRegistry`.  The engine contains no
step-kind-specific logic — step executors (built-in or plugin-defined)
are looked up purely by step type, so plugins extend the engine without
modifying it.

Execution guarantees:

* **Step isolation** — every step gets its own :class:`StepExecutionContext`;
  parallel branches never share mutable state; data merges only through
  explicit ``input_bindings`` / ``output_mapping`` expressions.
* **Deterministic ordering** — ready steps run in declaration order,
  bounded by ``max_concurrency``.
* **Retry & timeout** — per-step retry policies and timeouts are enforced
  before the error policy decides the final outcome.
* **Pause/resume** — approval steps pause the run; the run can be resumed
  from its persisted snapshot (completed steps are restored, not re-run).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable

from app.automation.context import (
    OutputStore,
    ScopedRunContext,
    StepExecutionContext,
    WorkflowRunContext,
)
from app.automation.events import (
    EVENT_WORKFLOW_APPROVAL_REQUESTED,
    EVENT_WORKFLOW_RUN_COMPLETED,
    EVENT_WORKFLOW_RUN_FAILED,
    EVENT_WORKFLOW_RUN_STARTED,
    EVENT_WORKFLOW_RUN_TIMED_OUT,
)
from app.automation.exceptions import (
    WorkflowExecutionError,
    WorkflowStepError,
)
from app.automation.graph import WorkflowGraph
from app.automation.models import (
    ApprovalRequest,
    ApprovalStatus,
    ErrorPolicy,
    StepExecution,
    StepExecutionStatus,
    StepRetryPolicy,
    WorkflowEvent,
    WorkflowOutput,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowSnapshot,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.registries import StepExecutor, StepTypeRegistry
from app.automation.template import (
    evaluate_condition,
    resolve_bindings,
    resolve_template,
)
from app.automation.time import Clock, SystemClock
from app.automation.utils import json_safe

log = logging.getLogger(__name__)


@dataclass
class _EngineState:
    """Mutable state shared across a single run's execution."""

    outputs: OutputStore = field(default_factory=OutputStore)
    completed: set[str] = field(default_factory=set)
    step_executions: list[StepExecution] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)
    events: list[WorkflowEvent] = field(default_factory=list)
    run_id: str = ""


class WorkflowExecutor:
    """Executes workflow runs by dispatching steps through a registry.

    The executor is intentionally free of step-kind-specific logic: it
    looks up the registered executor for a step's type and awaits it.
    Built-in executors are registered via :meth:`register_builtins`;
    plugins register additional executors through the same registry.
    """

    def __init__(
        self,
        step_types: StepTypeRegistry,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._step_types = step_types
        self._clock: Clock = clock or SystemClock()

    # ------------------------------------------------------------------
    # Step type registration
    # ------------------------------------------------------------------

    def register_builtins(
        self,
        tool_executor: Any | None = None,
        agent_manager: Any | None = None,
        llm_router: Any | None = None,
        subworkflow_runner: Callable[..., Awaitable[Any]] | None = None,
        callback_sender: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        """Register built-in executors for the core step types.

        Every core step type is wired to a handler that delegates to the
        optional collaborator when provided, and fails with a clear
        error otherwise.  This is additive — existing registrations are
        left untouched so plugins can override.
        """
        handlers: dict[WorkflowStepType, StepExecutor] = {
            WorkflowStepType.TASK: _make_task_executor(tool_executor),
            WorkflowStepType.AGENT: _make_agent_executor(agent_manager),
            WorkflowStepType.LLM: _make_llm_executor(llm_router),
            WorkflowStepType.SUBWORKFLOW: _make_subworkflow_executor(
                subworkflow_runner
            ),
            WorkflowStepType.TRANSFORM: self._transform_executor,
            WorkflowStepType.DELAY: self._delay_executor,
            WorkflowStepType.APPROVAL: self._approval_executor,
            WorkflowStepType.CALLBACK: _make_callback_executor(callback_sender),
            WorkflowStepType.CONDITIONAL: self._conditional_executor,
            WorkflowStepType.LOOP: self._loop_executor,
            WorkflowStepType.PARALLEL: self._parallel_executor,
        }
        for step_type, executor in handlers.items():
            if not self._step_types.has(step_type):
                self._step_types.register(step_type, executor)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, run: WorkflowRun) -> WorkflowRun:
        """Execute (or resume) a workflow run to a terminal or paused state.

        Completed steps are restored from the run's persisted
        ``step_executions`` so a resumed run continues where it left
        off.  Returns a new immutable run reflecting the outcome.
        """
        snapshot = run.snapshot
        graph = WorkflowGraph(snapshot.step_definitions)
        state = _EngineState(run_id=run.run_id)
        state.approvals = list(run.approval_requests)
        state.events = list(run.events)

        log.debug(
            "Workflow run started",
            extra={"workflow_id": run.workflow_id, "run_id": run.run_id},
        )

        # Restore completed step outputs for resume support.
        for execution in run.step_executions:
            if execution.status == StepExecutionStatus.SUCCEEDED:
                if execution.output is not None:
                    state.outputs.put(execution.step_id, execution.output)
                state.completed.add(execution.step_id)
            elif execution.status == StepExecutionStatus.SKIPPED:
                state.completed.add(execution.step_id)
            state.step_executions.append(execution)

        start_clock = self._clock.time()
        overall_deadline = (
            start_clock + snapshot.settings.overall_timeout_s
            if snapshot.settings.overall_timeout_s is not None
            else None
        )

        current = run.model_copy(
            update={
                "status": WorkflowRunStatus.RUNNING,
                "started_at": run.started_at or self._clock.utcnow(),
            }
        )
        current = self._record_event(
            current, state, EVENT_WORKFLOW_RUN_STARTED, None, {"run_id": current.run_id}
        )

        try:
            while True:
                ready = graph.ready_steps(state.completed)
                if not ready:
                    break

                if (
                    overall_deadline is not None
                    and self._clock.time() > overall_deadline
                ):
                    current = current.model_copy(
                        update={
                            "status": WorkflowRunStatus.TIMED_OUT,
                            "completed_at": self._clock.utcnow(),
                            "error": "Workflow exceeded its overall timeout",
                        }
                    )
                    current = self._record_event(
                        current, state, EVENT_WORKFLOW_RUN_TIMED_OUT, None, {}
                    )
                    break

                semaphore = asyncio.Semaphore(snapshot.settings.max_concurrency)

                remaining_budget = None
                if overall_deadline is not None:
                    remaining_budget = max(overall_deadline - self._clock.time(), 0.0)

                async def _run_ready(step_id: str) -> StepExecution:
                    step = graph.get(step_id)
                    assert step is not None
                    async with semaphore:
                        return await self._execute_step(
                            step,
                            state,
                            run,
                            snapshot,
                            timeout_cap=remaining_budget,
                        )

                results = await asyncio.gather(
                    *(_run_ready(step_id) for step_id in ready),
                    return_exceptions=True,
                )

                paused: ApprovalRequest | None = None
                for result in results:
                    if isinstance(result, WorkflowPause):
                        paused = result.approval_request
                        continue
                    if isinstance(result, BaseException):
                        raise result

                if paused is not None:
                    if all(a.request_id != paused.request_id for a in state.approvals):
                        state.approvals.append(paused)
                    current = current.model_copy(
                        update={
                            "status": WorkflowRunStatus.WAITING_APPROVAL,
                            "approval_requests": tuple(state.approvals),
                            "paused_at": self._clock.utcnow(),
                        }
                    )
                    current = self._record_event(
                        current,
                        state,
                        EVENT_WORKFLOW_APPROVAL_REQUESTED,
                        paused.step_id,
                        {"request_id": paused.request_id},
                    )
                    return self._finalize(current, state)

            # Resolve declared outputs.
            run_context = WorkflowRunContext(current.inputs, state.outputs)
            resolved_outputs = {
                output.name: self._resolve_output(output, run_context)
                for output in snapshot.outputs
            }
            current = current.model_copy(
                update={
                    "status": WorkflowRunStatus.SUCCEEDED,
                    "completed_at": self._clock.utcnow(),
                    "outputs": json_safe(resolved_outputs),
                }
            )
            current = self._record_event(
                current, state, EVENT_WORKFLOW_RUN_COMPLETED, None, {}
            )
        except _OverallTimeoutBudget as exc:
            current = current.model_copy(
                update={
                    "status": WorkflowRunStatus.TIMED_OUT,
                    "completed_at": self._clock.utcnow(),
                    "error": f"Workflow exceeded its overall timeout ({exc.step_id})",
                }
            )
            current = self._record_event(
                current, state, EVENT_WORKFLOW_RUN_TIMED_OUT, exc.step_id, {}
            )
        except WorkflowStepError as exc:
            current = current.model_copy(
                update={
                    "status": WorkflowRunStatus.FAILED,
                    "completed_at": self._clock.utcnow(),
                    "error": str(exc),
                }
            )
            current = self._record_event(
                current,
                state,
                EVENT_WORKFLOW_RUN_FAILED,
                exc.step_id,
                {"error": str(exc)},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            current = current.model_copy(
                update={
                    "status": WorkflowRunStatus.FAILED,
                    "completed_at": self._clock.utcnow(),
                    "error": str(exc),
                }
            )
            current = self._record_event(
                current, state, EVENT_WORKFLOW_RUN_FAILED, None, {}
            )

        log.debug(
            "Workflow run finished",
            extra={
                "workflow_id": run.workflow_id,
                "run_id": run.run_id,
                "status": current.status.value,
            },
        )
        return self._finalize(current, state)

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        step: WorkflowStep,
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext | None = None,
        step_ctx: StepExecutionContext | None = None,
        timeout_cap: float | None = None,
    ) -> StepExecution:
        """Execute a single step, returning its execution record.

        Raises :class:`WorkflowPause` when the step requires approval.
        Idempotent for resume: steps already completed are skipped.
        """
        if step.id in state.completed:
            return (
                state.step_executions[-1]
                if state.step_executions
                else _stub_execution(step)
            )

        if shared is None:
            shared = WorkflowRunContext(run.inputs, state.outputs)
        if step_ctx is None:
            step_ctx = StepExecutionContext(step.id, {}, shared)

        # Evaluate run-gate condition; skip when false.
        if step.condition:
            try:
                should_run = evaluate_condition(step.condition, shared)
            except WorkflowExecutionError:
                should_run = True
            if not should_run:
                execution = StepExecution(
                    step_id=step.id,
                    status=StepExecutionStatus.SKIPPED,
                    completed_at=self._clock.utcnow(),
                )
                state.completed.add(step.id)
                state.step_executions.append(execution)
                return execution

        inputs = resolve_bindings(step.input_bindings, shared)
        scoped_ctx = StepExecutionContext(step.id, inputs, shared)
        executor = self._step_types.get(step.type)
        if executor is None:
            raise WorkflowStepError(
                f"No executor registered for step type '{step.type.value}'",
                step_id=step.id,
            )

        log.debug(
            "Executing step",
            extra={
                "workflow_id": run.workflow_id,
                "run_id": run.run_id,
                "step_id": step.id,
                "step_type": step.type.value,
            },
        )

        started_at = self._clock.utcnow()
        retry_policy = step.retry_policy
        effective_retries = retry_policy.max_retries if retry_policy else 0
        if step.error_policy == ErrorPolicy.RETRY and effective_retries == 0:
            effective_retries = 1
        delays: list[float] = []
        last_error: str | None = None

        for attempt in range(effective_retries + 1):
            try:
                timeout = step.timeout_s or snapshot.settings.default_timeout_s
                capped_by_budget = timeout_cap is not None and (
                    timeout is None or timeout_cap <= timeout
                )
                if timeout_cap is not None:
                    timeout = (
                        min(timeout_cap, timeout)
                        if timeout is not None
                        else timeout_cap
                    )
                if timeout is not None:
                    result = await asyncio.wait_for(
                        executor(step, scoped_ctx, self, state, run, snapshot, shared),
                        timeout=timeout,
                    )
                else:
                    result = await executor(
                        step, scoped_ctx, self, state, run, snapshot, shared
                    )
                state.outputs.put(step.id, json_safe(result))
                state.completed.add(step.id)
                execution = StepExecution(
                    step_id=step.id,
                    status=StepExecutionStatus.SUCCEEDED,
                    inputs=inputs,
                    output=json_safe(result),
                    retries_consumed=attempt,
                    attempts=tuple(delays),
                    started_at=started_at,
                    completed_at=self._clock.utcnow(),
                )
                state.step_executions.append(execution)
                return execution
            except WorkflowPause:
                raise
            except asyncio.TimeoutError:
                last_error = f"Step '{step.id}' timed out"
                if capped_by_budget:
                    raise _OverallTimeoutBudget(step.id) from None
                if attempt >= effective_retries:
                    break
                delay = _backoff_delay(retry_policy, attempt + 1)
                delays.append(delay)
                await asyncio.sleep(delay)
            except Exception as exc:  # noqa: BLE001
                last_error = f"Step '{step.id}' failed: {exc}"
                if attempt >= effective_retries:
                    break
                delay = _backoff_delay(retry_policy, attempt + 1)
                delays.append(delay)
                await asyncio.sleep(delay)

        # Retries exhausted — apply the step's error policy.
        if step.error_policy == ErrorPolicy.SKIP:
            execution = StepExecution(
                step_id=step.id,
                status=StepExecutionStatus.SKIPPED,
                inputs=inputs,
                error=last_error,
                retries_consumed=effective_retries,
                attempts=tuple(delays),
                started_at=started_at,
                completed_at=self._clock.utcnow(),
            )
            state.completed.add(step.id)
            state.step_executions.append(execution)
            return execution

        if step.error_policy == ErrorPolicy.IGNORE:
            state.outputs.put(step.id, None)
            state.completed.add(step.id)
            execution = StepExecution(
                step_id=step.id,
                status=StepExecutionStatus.SUCCEEDED,
                inputs=inputs,
                output=None,
                error=last_error,
                retries_consumed=effective_retries,
                attempts=tuple(delays),
                started_at=started_at,
                completed_at=self._clock.utcnow(),
            )
            state.step_executions.append(execution)
            return execution

        log.warning(
            "Step failed after retries",
            extra={
                "workflow_id": run.workflow_id,
                "run_id": run.run_id,
                "step_id": step.id,
                "error": last_error,
            },
        )
        raise WorkflowStepError(
            last_error or f"Step '{step.id}' failed",
            step_id=step.id,
        )

    # ------------------------------------------------------------------
    # Structural step executors (conditional / loop / parallel)
    # ------------------------------------------------------------------

    async def _conditional_executor(
        self,
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: "WorkflowExecutor",
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Select the first matching branch and run its steps."""
        selected: str | None = None
        for branch in step.branches:
            try:
                matched = evaluate_condition(branch.when, shared)
            except WorkflowExecutionError:
                matched = False
            if matched:
                selected = branch.name
                break

        if selected is None:
            return {"branch": None, "outputs": {}}

        branch = next(b for b in step.branches if b.name == selected)
        outputs = await self._run_nested_steps(
            branch.steps, state, run, snapshot, shared, step_ctx.inputs
        )
        return {"branch": selected, "outputs": outputs}

    async def _loop_executor(
        self,
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: "WorkflowExecutor",
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Iterate over a resolved iterable, running loop steps per item."""
        iterable = resolve_template(step.iterable, shared)
        if iterable is None:
            iterable = []
        if isinstance(iterable, dict):
            iterable = list(iterable.values())
        if not isinstance(iterable, (list, tuple)):
            iterable = [iterable]

        results: list[Any] = []
        for item in list(iterable)[: step.max_iterations]:
            scope_inputs = dict(step_ctx.inputs)
            scope_inputs[step.loop_var] = item
            scope = ScopedRunContext(shared, state.outputs, scope_inputs)
            iteration_outputs = await self._run_nested_steps(
                step.loop_steps, state, run, snapshot, scope, scope_inputs
            )
            results.append(iteration_outputs)
            if step.break_condition:
                try:
                    if evaluate_condition(step.break_condition, scope):
                        break
                except WorkflowExecutionError:
                    pass
        return {"iterations": results, "count": len(results)}

    async def _parallel_executor(
        self,
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: "WorkflowExecutor",
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Run branch steps concurrently with fully isolated contexts.

        Each branch executes through its own :class:`ScopedRunContext` so
        parallel branches never share mutable state; results are merged
        only through the returned branch outputs.
        """
        if step.join_mode.value == "none":
            # Fire branches concurrently, merge every branch's outputs.
            branches_outputs = await asyncio.gather(
                *(
                    self._run_branch_isolated(
                        branch.steps, step_ctx.inputs, state, run, snapshot, shared
                    )
                    for branch in step.branches
                ),
                return_exceptions=True,
            )
            merged: dict[str, Any] = {}
            for branch, output in zip(step.branches, branches_outputs):
                merged[branch.name] = (
                    output
                    if not isinstance(output, BaseException)
                    else {"error": str(output)}
                )
            return merged

        # ALL / ANY: sequential branch evaluation until satisfied.
        branch_outputs: dict[str, Any] = {}
        for branch in step.branches:
            output = await self._run_branch_isolated(
                branch.steps, step_ctx.inputs, state, run, snapshot, shared
            )
            branch_outputs[branch.name] = output
            if step.join_mode.value == "any":
                break
        return branch_outputs

    async def _run_branch_isolated(
        self,
        steps: tuple[WorkflowStep, ...],
        base_inputs: dict[str, Any],
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Run a parallel branch's steps under an isolated scope."""
        branch_outputs = OutputStore()
        scope = ScopedRunContext(shared, branch_outputs, base_inputs)
        results = await self._run_nested_steps(
            steps, state, run, snapshot, scope, base_inputs
        )
        return results

    async def _run_nested_steps(
        self,
        steps: tuple[WorkflowStep, ...],
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
        base_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run nested steps in order, returning a dict of step_id -> output.

        Nested steps run sequentially; outputs are read back from the
        shared context so later nested steps can reference earlier ones
        via ``${step.<id>.output.<path>}`` expressions.
        """
        outputs: dict[str, Any] = {}
        for sub_step in steps:
            sub_ctx = StepExecutionContext(sub_step.id, base_inputs, shared)
            execution = await self._execute_step(
                sub_step, state, run, snapshot, shared, sub_ctx
            )
            if execution.status == StepExecutionStatus.SUCCEEDED:
                outputs[sub_step.id] = execution.output
        return outputs

    # ------------------------------------------------------------------
    # Built-in leaf step executors
    # ------------------------------------------------------------------

    async def _transform_executor(
        self,
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: "WorkflowExecutor",
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Evaluate a transform expression against the run context."""
        try:
            return resolve_template(step.expression, shared)
        except WorkflowExecutionError as exc:
            raise WorkflowStepError(str(exc), step_id=step.id) from exc

    async def _delay_executor(
        self,
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: "WorkflowExecutor",
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Wait for the configured delay."""
        await asyncio.sleep(step.delay_seconds)
        return {"delayed_seconds": step.delay_seconds}

    async def _approval_executor(
        self,
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: "WorkflowExecutor",
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        """Pause the run and raise a pending approval request.

        If a decision already exists for this run/step (resume path), it
        is applied; otherwise the run pauses.
        """
        existing = _find_approval(state.approvals, run.run_id, step.id)
        if existing is not None and existing.status != ApprovalStatus.PENDING:
            return {
                "approved": existing.status == ApprovalStatus.APPROVED,
                "decided_by": existing.decided_by,
                "decision_note": existing.decision_note,
            }

        if (
            existing is not None
            and existing.status == ApprovalStatus.PENDING
            and existing.expires_at is not None
            and self._clock.utcnow() >= existing.expires_at
        ):
            # Expired pending approval -> apply the auto decision.
            if step.auto_decision.value == "auto_approve":
                return {"approved": True, "auto": True}
            if step.auto_decision.value == "auto_reject":
                raise WorkflowStepError(
                    f"Approval for step '{step.id}' auto-rejected after timeout",
                    step_id=step.id,
                )
            raise WorkflowStepError(
                f"Approval for step '{step.id}' timed out",
                step_id=step.id,
            )

        request = ApprovalRequest(
            request_id=uuid.uuid4().hex,
            run_id=run.run_id,
            step_id=step.id,
            approvers=step.approvers,
            created_at=self._clock.utcnow(),
            expires_at=(
                self._clock.utcnow() + timedelta(seconds=step.timeout_s)
                if step.timeout_s is not None
                else None
            ),
        )
        raise WorkflowPause(request)

    def _resolve_output(
        self, output: WorkflowOutput, context: WorkflowRunContext
    ) -> Any:
        """Resolve a declared workflow output from its source expression."""
        source = output.source
        if not source:
            return None
        return json_safe(resolve_template(source, context))

    # ------------------------------------------------------------------
    # Finalisation helpers
    # ------------------------------------------------------------------

    def _finalize(self, run: WorkflowRun, state: _EngineState) -> WorkflowRun:
        """Build the final immutable run with persisted state."""
        return run.model_copy(
            update={
                "step_executions": tuple(state.step_executions),
                "events": tuple(state.events),
                "approval_requests": tuple(state.approvals),
            }
        )

    def _record_event(
        self,
        run: WorkflowRun,
        state: _EngineState,
        event_type: str,
        step_id: str | None,
        data: dict[str, Any],
    ) -> WorkflowRun:
        """Append an event to the run's persisted log."""
        state.events.append(
            WorkflowEvent(
                event_type=event_type,
                run_id=run.run_id,
                step_id=step_id,
                timestamp=self._clock.utcnow(),
                data=json_safe(data),
            )
        )
        return run.model_copy(update={"events": tuple(state.events)})


class WorkflowPause(Exception):
    """Internal control-flow marker raised when a run pauses.

    Raised when an approval step is waiting for a decision; the run is
    persisted and resumed later.
    """

    def __init__(self, approval_request: ApprovalRequest) -> None:
        super().__init__(
            f"Workflow paused awaiting approval '{approval_request.request_id}'"
        )
        self.approval_request = approval_request


class _OverallTimeoutBudget(Exception):
    """Internal signal that a step exceeded the run's overall budget."""

    def __init__(self, step_id: str) -> None:
        super().__init__(f"Overall timeout budget exceeded by step '{step_id}'")
        self.step_id = step_id


# ----------------------------------------------------------------------
# Collaborator-backed executors
# ----------------------------------------------------------------------


def _make_task_executor(tool_executor: Any | None) -> StepExecutor:
    """Build the TASK executor that runs a registered tool."""

    async def execute(
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: WorkflowExecutor,
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        if tool_executor is None:
            raise WorkflowStepError(
                f"Tool executor is not available for step '{step.id}'",
                step_id=step.id,
            )
        params = dict(step.parameters)
        bindings = resolve_bindings(step.input_bindings, shared)
        params.update(bindings)
        result = await tool_executor.execute_async(
            step.tool_name,
            params,
            timeout_s=step.timeout_s,
        )
        if result is None:
            raise WorkflowStepError(
                f"Tool '{step.tool_name}' returned no result for step '{step.id}'",
                step_id=step.id,
            )
        if getattr(result, "success", True) is False:
            raise WorkflowStepError(
                getattr(result, "error", None) or f"Tool '{step.tool_name}' failed",
                step_id=step.id,
            )
        return json_safe(getattr(result, "output", result))

    return execute


def _make_agent_executor(agent_manager: Any | None) -> StepExecutor:
    """Build the AGENT executor that runs a managed agent session."""

    async def execute(
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: WorkflowExecutor,
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        if agent_manager is None:
            raise WorkflowStepError(
                f"Agent manager is not available for step '{step.id}'",
                step_id=step.id,
            )
        from app.agents.models import AgentRequest  # local import to avoid cycles

        bindings = resolve_bindings(step.input_bindings, shared)
        request = AgentRequest(
            raw_input=bindings.get("raw_input", "")
            or bindings.get("input", "")
            or step.name,
            session_id="",
            metadata={"workflow_run_id": run.run_id, "step_id": step.id},
        )
        config = None
        if step.agent_config:
            from app.kernel.agent.models import AgentRunConfig

            config = AgentRunConfig(**step.agent_config)
        session = await agent_manager.execute(request, config=config)
        completed = await agent_manager.await_completion(session.session_id)
        response = getattr(completed, "response", None)
        return json_safe(response)

    return execute


def _make_llm_executor(llm_router: Any | None) -> StepExecutor:
    """Build the LLM executor that calls the multi-LLM router."""

    async def execute(
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: WorkflowExecutor,
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        if llm_router is None:
            raise WorkflowStepError(
                f"LLM router is not available for step '{step.id}'",
                step_id=step.id,
            )
        from app.llm.models import ChatRequest, Message, Role

        bindings = resolve_bindings(step.input_bindings, shared)
        prompt = step.prompt_template
        for key, value in bindings.items():
            prompt = prompt.replace("${" + key + "}", str(value))
        messages = []
        if step.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=step.system_prompt))
        messages.append(Message(role=Role.USER, content=prompt))
        request = ChatRequest(
            messages=tuple(messages), model=step.model_hint or "default"
        )
        response = await llm_router.generate_async(request)
        return json_safe(
            getattr(response, "content", None) or getattr(response, "text", "")
        )

    return execute


def _make_subworkflow_executor(
    subworkflow_runner: Callable[..., Awaitable[Any]] | None,
) -> StepExecutor:
    """Build the SUBWORKFLOW executor that runs a child workflow."""

    async def execute(
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: WorkflowExecutor,
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        if subworkflow_runner is None:
            raise WorkflowStepError(
                f"Subworkflow runner is not available for step '{step.id}'",
                step_id=step.id,
            )
        bindings = resolve_bindings(step.input_bindings, shared)
        result = await subworkflow_runner(
            step.workflow_id,
            step.workflow_version or None,
            bindings,
            run.run_id,
        )
        return json_safe(result)

    return execute


def _make_callback_executor(
    callback_sender: Callable[..., Awaitable[Any]] | None,
) -> StepExecutor:
    """Build the CALLBACK executor that sends an external webhook."""

    async def execute(
        step: WorkflowStep,
        step_ctx: StepExecutionContext,
        engine: WorkflowExecutor,
        state: _EngineState,
        run: WorkflowRun,
        snapshot: WorkflowSnapshot,
        shared: WorkflowRunContext,
    ) -> Any:
        if callback_sender is None:
            raise WorkflowStepError(
                f"Callback sender is not available for step '{step.id}'",
                step_id=step.id,
            )
        payload = {}
        if step.callback_payload:
            payload = resolve_bindings({"payload": step.callback_payload}, shared)[
                "payload"
            ]
        result = await callback_sender(
            step.callback_url,
            method=step.callback_method,
            payload=payload,
            timeout_s=step.timeout_s,
        )
        return json_safe(result)

    return execute


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _backoff_delay(policy: StepRetryPolicy | None, attempt: int) -> float:
    """Compute the backoff delay for a retry attempt."""
    if policy is None:
        return 0.0
    base = policy.base_delay_s
    if policy.backoff.value == "linear":
        delay = base * attempt
    elif policy.backoff.value == "exponential":
        delay = base * (2 ** (attempt - 1))
    else:
        delay = base
    return min(delay, policy.max_delay_s)


def _find_approval(
    approvals: list[ApprovalRequest], run_id: str, step_id: str
) -> ApprovalRequest | None:
    """Return the most recent approval request for a run/step pair."""
    matches = [a for a in approvals if a.run_id == run_id and a.step_id == step_id]
    return max(matches, key=lambda a: a.created_at) if matches else None


def _stub_execution(step: WorkflowStep) -> StepExecution:
    """Return a placeholder execution for an already-completed step."""
    return StepExecution(step_id=step.id, status=StepExecutionStatus.SUCCEEDED)
