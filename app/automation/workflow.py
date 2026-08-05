"""Workflow service — the application-level façade for automation.

The :class:`WorkflowService` coordinates the registries, executor,
scheduler, validator, graph exporter and durable stores behind a single
public API used by the server layer and plugins.  It owns the run id,
snapshot and idempotency semantics and wires the scheduler's ``run
starter`` and the executor's ``subworkflow runner`` back into itself so
scheduled and nested runs flow through the same code path.

The service is thread-safe (an :class:`RLock` guards registry/store
mutations) and async-first for execution.  Definitions, runs and
schedules are persisted through the injected stores on every mutation.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.automation.exceptions import (
    WorkflowApprovalError,
    WorkflowNotFoundError,
    WorkflowValidationError,
)
from app.automation.executor import WorkflowExecutor
from app.automation.graph import WorkflowGraph
from app.automation.models import (
    ApprovalStatus,
    RunTrigger,
    RunTriggerKind,
    ScheduleSpec,
    ScheduleTriggerType,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowSnapshot,
    WorkflowStatus,
)
from app.automation.registries import (
    WorkflowRegistry,
    WorkflowRunRegistry,
)
from app.automation.scheduler import WorkflowScheduler
from app.automation.stores import (
    EventStore,
    ScheduleStore,
    WorkflowRunStore,
    WorkflowStore,
)
from app.automation.time import Clock, SystemClock
from app.automation.validation import WorkflowValidator

log = logging.getLogger(__name__)

RunIdFactory = Callable[[], str]


def _default_id_factory() -> str:
    """Return a random hex identifier."""
    return uuid.uuid4().hex


class WorkflowGraphExporter:
    """Export a workflow definition to a JSON-safe graph representation.

    Produces ``nodes`` (step id/type/name), ``edges`` (dependency pairs)
    and the deterministic topological ordering for rendering or analysis.
    """

    def export(self, definition: WorkflowDefinition) -> dict[str, Any]:
        """Return a JSON-safe graph description of *definition*."""
        graph = WorkflowGraph(definition.steps)
        nodes = [
            {
                "id": step.id,
                "type": step.type.value,
                "name": step.name,
                "description": step.description,
            }
            for step in definition.steps
        ]
        edges = [
            {"source": dep, "target": step.id}
            for step in definition.steps
            for dep in step.depends_on
        ]
        return {
            "workflow_id": definition.id,
            "version": definition.version,
            "status": definition.status.value,
            "nodes": nodes,
            "edges": edges,
            "topological_order": graph.topological_order(),
        }


class WorkflowService:
    """Coordinates workflow definitions, runs and schedules.

    Usage::

        service = WorkflowService(
            registry=..., run_registry=..., executor=..., validator=...,
            exporter=..., definition_store=..., run_store=...,
            schedule_store=..., event_store=...,
        )
        definition = service.publish_definition("wf", "1.0.0")
        run = await service.execute("wf", {"user": "alice"})
    """

    def __init__(
        self,
        *,
        registry: WorkflowRegistry,
        run_registry: WorkflowRunRegistry,
        executor: WorkflowExecutor,
        validator: WorkflowValidator,
        exporter: WorkflowGraphExporter,
        definition_store: WorkflowStore,
        run_store: WorkflowRunStore,
        schedule_store: ScheduleStore,
        event_store: EventStore,
        clock: Clock | None = None,
        run_id_factory: RunIdFactory | None = None,
        tool_executor: Any | None = None,
        agent_manager: Any | None = None,
        llm_router: Any | None = None,
        callback_sender: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._registry = registry
        self._run_registry = run_registry
        self._executor = executor
        self._validator = validator
        self._exporter = exporter
        self._definition_store = definition_store
        self._run_store = run_store
        self._schedule_store = schedule_store
        self._event_store = event_store
        self._clock: Clock = clock or SystemClock()
        self._run_id_factory: RunIdFactory = run_id_factory or _default_id_factory
        self._lock = threading.RLock()

        # Event cursor per run so durable timeline appends are idempotent.
        self._synced_events: dict[str, int] = {}

        # Wire the executor's subworkflow runner back into the service so
        # nested runs use the same execute() semantics.  Additive — plugin
        # registered executors are never overwritten.
        executor.register_builtins(
            tool_executor=tool_executor,
            agent_manager=agent_manager,
            llm_router=llm_router,
            subworkflow_runner=self._run_subworkflow,
            callback_sender=callback_sender,
        )

        self._scheduler = WorkflowScheduler(
            schedule_store=schedule_store,
            run_store=run_store,
            run_starter=self._start_scheduled_run,
            event_sink=event_store.append_event,
            clock=self._clock,
        )

    # ------------------------------------------------------------------
    # Definition management
    # ------------------------------------------------------------------

    def create_definition(self, definition: WorkflowDefinition) -> WorkflowDefinition:
        """Register a workflow definition and persist it.

        Raises:
            WorkflowVersionError: If a published version with the same
                id/version already exists.
        """
        with self._lock:
            self._registry.register(definition)
            self._definition_store.save_definition(definition)
            return definition

    def publish_definition(self, workflow_id: str, version: str) -> WorkflowDefinition:
        """Validate then publish a definition.

        Raises:
            WorkflowNotFoundError: If the definition does not exist.
            WorkflowValidationError: If validation fails.
        """
        with self._lock:
            definition = self._registry.get(workflow_id, version)
            if definition is None:
                raise WorkflowNotFoundError(workflow_id, version=version)
            report = self._validator.validate(definition)
            if not report.valid:
                raise WorkflowValidationError(
                    f"Workflow '{workflow_id}' version '{version}' failed validation",
                    violations=[v.message for v in report.errors],
                )
            published = self._registry.publish(workflow_id, version)
            self._definition_store.save_definition(published)
            return published

    def delete_definition(self, workflow_id: str, version: str | None = None) -> bool:
        """Remove a definition (draft versions only).

        Raises:
            WorkflowVersionError: If the definition is published.
        """
        with self._lock:
            removed = self._registry.unregister(workflow_id, version)
            if removed:
                self._definition_store.delete_definition(workflow_id, version)
            return removed

    def get_definition(
        self, workflow_id: str, version: str | None = None
    ) -> WorkflowDefinition | None:
        """Return a definition by id (and optional version)."""
        with self._lock:
            definition = self._registry.get(workflow_id, version)
            if definition is None:
                definition = self._definition_store.get_definition(workflow_id, version)
            return definition

    def list_definitions(
        self, status: WorkflowStatus | None = None
    ) -> list[WorkflowDefinition]:
        """Return all definitions, optionally filtered by status."""
        with self._lock:
            return self._registry.list_all(status)

    def export_graph(
        self, workflow_id: str, version: str | None = None
    ) -> dict[str, Any]:
        """Return a JSON-safe graph description of a definition."""
        definition = self.get_definition(workflow_id, version)
        if definition is None:
            raise WorkflowNotFoundError(workflow_id, version=version)
        return self._exporter.export(definition)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        workflow_id: str,
        inputs: dict[str, Any] | None = None,
        *,
        version: str | None = None,
        idempotency_key: str | None = None,
        trigger: RunTrigger | None = None,
    ) -> WorkflowRun:
        """Start (or deduplicate) a workflow run.

        When *version* is omitted the currently published definition is
        used.  A non-``None`` *idempotency_key* resolves duplicate
        requests to the existing run.
        """
        with self._lock:
            definition = self._resolve_executable(workflow_id, version)
            resolved_version = definition.version

            if idempotency_key is not None:
                existing = self._find_idempotent_run(
                    workflow_id, resolved_version, idempotency_key
                )
                if existing is not None:
                    return existing

            run = self._build_run(
                definition,
                inputs,
                idempotency_key=idempotency_key,
                trigger=trigger,
            )
            self._persist_run(run)

        return await self._execute_run(run)

    async def execute_version(
        self,
        workflow_id: str,
        version: str,
        inputs: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> WorkflowRun:
        """Execute a specific (published) version of a workflow."""
        return await self.execute(
            workflow_id,
            inputs,
            version=version,
            idempotency_key=idempotency_key,
        )

    async def resume(self, run_id: str) -> WorkflowRun:
        """Resume a paused run (approval / delay / after restart).

        Terminal runs are returned unchanged.
        """
        with self._lock:
            run = self._get_run(run_id)
            if run is None:
                raise WorkflowNotFoundError(run_id)
            if run.is_terminal:
                return run
        return await self._execute_run(run)

    def cancel(self, run_id: str) -> WorkflowRun:
        """Cancel a non-terminal run.

        Raises:
            WorkflowNotFoundError: If the run does not exist.
        """
        with self._lock:
            run = self._get_run(run_id)
            if run is None:
                raise WorkflowNotFoundError(run_id)
            if run.is_terminal:
                return run
            cancelled = run.model_copy(
                update={
                    "status": WorkflowRunStatus.CANCELLED,
                    "completed_at": self._clock.utcnow(),
                }
            )
            self._persist_run(cancelled)
            return cancelled

    def approve(
        self,
        run_id: str,
        request_id: str,
        *,
        decided_by: str = "",
        decision_note: str = "",
    ) -> WorkflowRun:
        """Record an approval decision for a pending request."""
        return self._decide(
            run_id,
            request_id,
            ApprovalStatus.APPROVED,
            decided_by=decided_by,
            decision_note=decision_note,
        )

    def reject(
        self,
        run_id: str,
        request_id: str,
        *,
        decided_by: str = "",
        decision_note: str = "",
    ) -> WorkflowRun:
        """Record a rejection decision for a pending request."""
        return self._decide(
            run_id,
            request_id,
            ApprovalStatus.REJECTED,
            decided_by=decided_by,
            decision_note=decision_note,
        )

    def get_run(self, run_id: str) -> WorkflowRun | None:
        """Return a run by id, or ``None``."""
        with self._lock:
            return self._get_run(run_id)

    def list_runs(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowRun]:
        """Return runs, newest first, optionally filtered."""
        with self._lock:
            return self._run_registry.list_all(workflow_id, status)

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    def create_schedule(
        self,
        *,
        workflow_id: str,
        trigger_type: ScheduleTriggerType,
        cron_expression: str = "",
        interval_seconds: float = 0.0,
        run_at: Any = None,
        timezone: str = "UTC",
        payload: dict[str, Any] | None = None,
        enabled: bool = True,
        max_concurrent_runs: int = 1,
        schedule_id: str | None = None,
    ) -> ScheduleSpec:
        """Create a schedule that fires a workflow on a trigger."""
        spec = ScheduleSpec(
            schedule_id=schedule_id or _default_id_factory(),
            workflow_id=workflow_id,
            trigger_type=trigger_type,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
            run_at=run_at,
            timezone=timezone,
            payload=dict(payload or {}),
            enabled=enabled,
            max_concurrent_runs=max_concurrent_runs,
        )
        with self._lock:
            return self._scheduler.add_schedule(spec)

    def update_schedule(self, spec: ScheduleSpec) -> ScheduleSpec:
        """Replace an existing schedule."""
        with self._lock:
            return self._scheduler.update_schedule(spec)

    def delete_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule, returning whether it existed."""
        with self._lock:
            return self._scheduler.remove_schedule(schedule_id)

    def enable_schedule(self, schedule_id: str) -> ScheduleSpec:
        """Mark a schedule enabled."""
        with self._lock:
            spec = self._scheduler.get_schedule(schedule_id)
            if spec is None:
                raise WorkflowNotFoundError(schedule_id)
            return self._scheduler.update_schedule(
                spec.model_copy(update={"enabled": True})
            )

    def disable_schedule(self, schedule_id: str) -> ScheduleSpec:
        """Mark a schedule disabled."""
        with self._lock:
            spec = self._scheduler.get_schedule(schedule_id)
            if spec is None:
                raise WorkflowNotFoundError(schedule_id)
            return self._scheduler.update_schedule(
                spec.model_copy(update={"enabled": False})
            )

    def get_schedule(self, schedule_id: str) -> ScheduleSpec | None:
        """Return a schedule by id, or ``None``."""
        with self._lock:
            return self._scheduler.get_schedule(schedule_id)

    def list_schedules(self, enabled: bool | None = None) -> list[ScheduleSpec]:
        """Return stored schedules, optionally filtered by enabled state."""
        with self._lock:
            return self._scheduler.list_schedules(enabled)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Restore persisted state and begin scheduling.

        Loads definitions and runs from their stores into the in-memory
        registries, recomputes schedule timers, resumes runs left waiting
        after a restart, and starts the scheduler's poll loop.
        """
        with self._lock:
            for definition in self._definition_store.list_definitions():
                if not self._registry.has(definition.id, definition.version):
                    self._registry.register(definition)
            for run in self._run_store.list_runs():
                if self._run_registry.get(run.run_id) is None:
                    self._run_registry.save(run)

        self._scheduler.restore()
        await self._scheduler.resume_pending_runs(self._resume_cb)
        await self._scheduler.start()
        log.info("Workflow service started")

    async def shutdown(self) -> None:
        """Gracefully stop the scheduler and drain in-flight runs."""
        await self._scheduler.shutdown()
        log.info("Workflow service shut down")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_executable(
        self, workflow_id: str, version: str | None
    ) -> WorkflowDefinition:
        if version is not None:
            definition = self._registry.get(workflow_id, version)
            if definition is None:
                raise WorkflowNotFoundError(workflow_id, version=version)
            return definition
        definition = self._registry.get_published(workflow_id)
        if definition is None:
            raise WorkflowNotFoundError(workflow_id, version="(latest published)")
        return definition

    def _find_idempotent_run(
        self, workflow_id: str, version: str, key: str
    ) -> WorkflowRun | None:
        run = self._run_registry.find_by_idempotency(workflow_id, version, key)
        if run is not None:
            return run
        return self._run_store.find_by_idempotency(workflow_id, version, key)

    def _build_run(
        self,
        definition: WorkflowDefinition,
        inputs: dict[str, Any] | None,
        *,
        idempotency_key: str | None,
        trigger: RunTrigger | None,
    ) -> WorkflowRun:
        snapshot = WorkflowSnapshot(
            workflow_id=definition.id,
            version=definition.version,
            step_definitions=definition.steps,
            settings=definition.settings,
            inputs=definition.inputs,
            outputs=definition.outputs,
            captured_at=self._clock.utcnow(),
        )
        return WorkflowRun(
            run_id=self._run_id_factory(),
            workflow_id=definition.id,
            workflow_version=definition.version,
            snapshot=snapshot,
            trigger=trigger or RunTrigger(),
            idempotency_key=idempotency_key,
            inputs=dict(inputs or {}),
            created_at=self._clock.utcnow(),
        )

    async def _execute_run(self, run: WorkflowRun) -> WorkflowRun:
        """Run *run* through the executor and persist the outcome."""
        result = await self._executor.run(run)
        with self._lock:
            self._persist_run(result)
        log.debug(
            "Run executed",
            extra={
                "workflow_id": result.workflow_id,
                "run_id": result.run_id,
                "status": result.status.value,
            },
        )
        return result

    def _persist_run(self, run: WorkflowRun) -> None:
        self._run_registry.save(run)
        self._run_store.save_run(run)
        cursor = self._synced_events.get(run.run_id, 0)
        for event in run.events[cursor:]:
            self._event_store.append_event(event)
        self._synced_events[run.run_id] = len(run.events)

    def _get_run(self, run_id: str) -> WorkflowRun | None:
        run = self._run_registry.get(run_id)
        if run is None:
            run = self._run_store.get_run(run_id)
        return run

    def _decide(
        self,
        run_id: str,
        request_id: str,
        decision: ApprovalStatus,
        *,
        decided_by: str,
        decision_note: str,
    ) -> WorkflowRun:
        with self._lock:
            run = self._get_run(run_id)
            if run is None:
                raise WorkflowNotFoundError(run_id)

            request = next(
                (a for a in run.approval_requests if a.request_id == request_id),
                None,
            )
            if request is None:
                raise WorkflowApprovalError(
                    f"Approval request '{request_id}' not found for run '{run_id}'"
                )
            if request.status != ApprovalStatus.PENDING:
                raise WorkflowApprovalError(
                    f"Approval request '{request_id}' for run '{run_id}' "
                    f"was already decided ({request.status.value})"
                )

            updated_request = request.model_copy(
                update={
                    "status": decision,
                    "decided_by": decided_by,
                    "decision_note": decision_note,
                    "decided_at": self._clock.utcnow(),
                }
            )
            approvals = tuple(
                updated_request if a.request_id == request_id else a
                for a in run.approval_requests
            )
            updated = run.model_copy(update={"approval_requests": approvals})
            self._persist_run(updated)
            return updated

    async def _start_scheduled_run(
        self, spec: ScheduleSpec, idempotency_key: str
    ) -> WorkflowRun:
        """Scheduler ``run_starter``: fire a schedule's workflow."""
        return await self.execute(
            spec.workflow_id,
            dict(spec.payload),
            version=spec.workflow_version or None,
            idempotency_key=idempotency_key,
            trigger=RunTrigger(kind=RunTriggerKind.SCHEDULE, detail=spec.schedule_id),
        )

    async def _run_subworkflow(
        self,
        workflow_id: str,
        version: str | None,
        inputs: dict[str, Any],
        parent_run_id: str,
    ) -> Any:
        """Executor ``subworkflow_runner``: run a child workflow."""
        run = await self.execute(
            workflow_id,
            dict(inputs),
            version=version,
            trigger=RunTrigger(kind=RunTriggerKind.PARENT, detail=parent_run_id),
        )
        return run.outputs

    async def _resume_cb(self, run: WorkflowRun) -> WorkflowRun:
        """Resume callback used by the scheduler after a restart."""
        return await self.resume(run.run_id)

    # ------------------------------------------------------------------
    # Collaborator access
    # ------------------------------------------------------------------

    @property
    def registry(self) -> WorkflowRegistry:
        """Return the workflow definition registry."""
        return self._registry

    @property
    def run_registry(self) -> WorkflowRunRegistry:
        """Return the workflow run registry."""
        return self._run_registry

    @property
    def scheduler(self) -> WorkflowScheduler:
        """Return the workflow scheduler."""
        return self._scheduler

    @property
    def step_type_registry(self):
        """Return the executor's step type registry."""
        return self._executor._step_types  # noqa: SLF001

    @property
    def event_store(self) -> EventStore:
        """Return the durable event store."""
        return self._event_store

    @property
    def clock(self) -> Clock:
        """Return the injected clock."""
        return self._clock
