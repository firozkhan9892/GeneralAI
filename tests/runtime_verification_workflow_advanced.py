"""Advanced runtime verification of workflow engine.

Covers:
  1. Workflow approval flow (pause → approve → complete)
  2. Workflow timeout (overall timeout kills a slow run)
  3. Workflow scheduler (create / list / enable / disable / delete)
  4. Workflow validation (missing steps, circular deps)

Run:  python -m pytest tests/runtime_verification_workflow_advanced.py -v 2>&1
"""

from __future__ import annotations


import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service():
    """Construct a fresh WorkflowService wired to in-memory stores."""
    from app.automation.executor import WorkflowExecutor
    from app.automation.registries import (
        StepTypeRegistry,
        WorkflowRegistry,
        WorkflowRunRegistry,
    )
    from app.automation.stores import (
        InMemoryEventStore,
        InMemoryScheduleStore,
        InMemoryWorkflowRunStore,
        InMemoryWorkflowStore,
    )
    from app.automation.validation import WorkflowValidator
    from app.automation.workflow import WorkflowGraphExporter, WorkflowService

    step_types = StepTypeRegistry()
    executor = WorkflowExecutor(step_types=step_types)

    service = WorkflowService(
        registry=WorkflowRegistry(),
        run_registry=WorkflowRunRegistry(),
        executor=executor,
        validator=WorkflowValidator(),
        exporter=WorkflowGraphExporter(),
        definition_store=InMemoryWorkflowStore(),
        run_store=InMemoryWorkflowRunStore(),
        schedule_store=InMemoryScheduleStore(),
        event_store=InMemoryEventStore(),
    )
    return service


# ===========================================================================
# 1. WORKFLOW APPROVAL FLOW
# ===========================================================================


class TestWorkflowApprovalFlow:
    """Workflow with an APPROVAL step pauses, then completes on approve."""

    @pytest.mark.asyncio
    async def test_approval_flow(self) -> None:
        from app.automation.models import (
            ApprovalStatus,
            WorkflowDefinition,
            WorkflowRunStatus,
            WorkflowStep,
            WorkflowStepType,
        )

        service = _build_service()

        # Build a simple workflow: approval → transform (finish)
        definition = WorkflowDefinition(
            id="approval-flow",
            version="1.0.0",
            name="Approval Flow Test",
            steps=(
                WorkflowStep(
                    id="ask",
                    type=WorkflowStepType.APPROVAL,
                    name="Ask for approval",
                    approvers=("admin",),
                ),
                WorkflowStep(
                    id="finish",
                    type=WorkflowStepType.TRANSFORM,
                    name="Finish",
                    expression='"done"',
                    depends_on=("ask",),
                ),
            ),
        )

        # Register + publish
        service.create_definition(definition)
        published = service.publish_definition("approval-flow", "1.0.0")
        assert published.status.value == "published"

        # Execute – should pause at the approval step
        run = await service.execute("approval-flow", {"user": "alice"})
        assert run.status == WorkflowRunStatus.WAITING_APPROVAL, (
            f"Expected WAITING_APPROVAL, got {run.status}"
        )
        assert len(run.approval_requests) >= 1, "No approval request created"

        request_id = run.approval_requests[0].request_id
        assert run.approval_requests[0].status == ApprovalStatus.PENDING

        # Approve
        run_after = service.approve(
            run.run_id,
            request_id,
            decided_by="admin",
            decision_note="Looks good",
        )
        assert run_after.approval_requests[0].status == ApprovalStatus.APPROVED

        # Resume – should complete the remaining transform step
        resumed = await service.resume(run.run_id)
        assert resumed.status == WorkflowRunStatus.SUCCEEDED, (
            f"Expected SUCCEEDED after resume, got {resumed.status}"
        )

    @pytest.mark.asyncio
    async def test_reject_flow(self) -> None:
        from app.automation.models import (
            ApprovalStatus,
            WorkflowDefinition,
            WorkflowRunStatus,
            WorkflowStep,
            WorkflowStepType,
        )

        service = _build_service()

        definition = WorkflowDefinition(
            id="reject-flow",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="ask",
                    type=WorkflowStepType.APPROVAL,
                    name="Ask",
                    approvers=("admin",),
                ),
                WorkflowStep(
                    id="after",
                    type=WorkflowStepType.TRANSFORM,
                    name="After",
                    expression='"ok"',
                    depends_on=("ask",),
                ),
            ),
        )

        service.create_definition(definition)
        service.publish_definition("reject-flow", "1.0.0")

        run = await service.execute("reject-flow")
        assert run.status == WorkflowRunStatus.WAITING_APPROVAL

        request_id = run.approval_requests[0].request_id
        service.reject(run.run_id, request_id, decided_by="admin")

        # After rejection the run stays in WAITING_APPROVAL until explicitly handled.
        # Verify the request is now REJECTED.
        refreshed = service.get_run(run.run_id)
        assert refreshed.approval_requests[0].status == ApprovalStatus.REJECTED


# ===========================================================================
# 2. WORKFLOW TIMEOUT
# ===========================================================================


class TestWorkflowTimeout:
    """A workflow exceeding its overall timeout is marked TIMED_OUT."""

    @pytest.mark.asyncio
    async def test_overall_timeout(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowRunStatus,
            WorkflowSettings,
            WorkflowStep,
            WorkflowStepType,
        )

        service = _build_service()

        # A DELAY step that takes 10 seconds, with an overall timeout of 0.5s
        definition = WorkflowDefinition(
            id="timeout-wf",
            version="1.0.0",
            settings=WorkflowSettings(overall_timeout_s=0.5),
            steps=(
                WorkflowStep(
                    id="slow",
                    type=WorkflowStepType.DELAY,
                    name="Slow delay",
                    delay_seconds=10.0,
                ),
            ),
        )

        service.create_definition(definition)
        service.publish_definition("timeout-wf", "1.0.0")

        run = await service.execute("timeout-wf")
        assert run.status == WorkflowRunStatus.TIMED_OUT, (
            f"Expected TIMED_OUT, got {run.status}"
        )
        assert run.error is not None
        assert "timeout" in run.error.lower()

    @pytest.mark.asyncio
    async def test_step_timeout(self) -> None:
        from app.automation.models import (
            ErrorPolicy,
            WorkflowDefinition,
            WorkflowRunStatus,
            WorkflowStep,
            WorkflowStepType,
        )

        service = _build_service()

        # A DELAY of 5s with a per-step timeout of 0.5s and ABORT policy
        definition = WorkflowDefinition(
            id="step-timeout-wf",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="slow",
                    type=WorkflowStepType.DELAY,
                    name="Slow",
                    delay_seconds=5.0,
                    timeout_s=0.5,
                    error_policy=ErrorPolicy.ABORT,
                ),
            ),
        )

        service.create_definition(definition)
        service.publish_definition("step-timeout-wf", "1.0.0")

        run = await service.execute("step-timeout-wf")
        assert run.status == WorkflowRunStatus.FAILED, (
            f"Expected FAILED from step timeout, got {run.status}"
        )
        assert run.error is not None
        assert "timed out" in run.error.lower()


# ===========================================================================
# 3. WORKFLOW SCHEDULER
# ===========================================================================


class TestWorkflowScheduler:
    """Verify schedule CRUD operations and tick behaviour."""

    @pytest.mark.asyncio
    async def test_create_list_delete_schedule(self) -> None:
        from app.automation.models import ScheduleTriggerType

        service = _build_service()

        # Create an interval schedule
        spec = service.create_schedule(
            workflow_id="dummy-wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=300,
            payload={"key": "val"},
        )
        assert spec.schedule_id
        assert spec.enabled is True

        # List – should contain it
        all_schedules = service.list_schedules()
        assert any(s.schedule_id == spec.schedule_id for s in all_schedules)

        # Get by id
        fetched = service.get_schedule(spec.schedule_id)
        assert fetched is not None
        assert fetched.workflow_id == "dummy-wf"

        # Disable
        disabled = service.disable_schedule(spec.schedule_id)
        assert disabled.enabled is False

        # List enabled only – should be empty
        enabled_only = service.list_schedules(enabled=True)
        assert not any(s.schedule_id == spec.schedule_id for s in enabled_only)

        # Re-enable
        re_enabled = service.enable_schedule(spec.schedule_id)
        assert re_enabled.enabled is True

        # Delete
        deleted = service.delete_schedule(spec.schedule_id)
        assert deleted is True
        assert service.get_schedule(spec.schedule_id) is None

        # Double delete returns False
        deleted_again = service.delete_schedule(spec.schedule_id)
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_cron_schedule(self) -> None:
        from app.automation.models import ScheduleTriggerType

        service = _build_service()

        spec = service.create_schedule(
            workflow_id="cron-wf",
            trigger_type=ScheduleTriggerType.CRON,
            cron_expression="*/5 * * * *",
        )
        assert spec.next_run_at is not None
        assert spec.schedule_id

    @pytest.mark.asyncio
    async def test_datetime_schedule(self) -> None:
        from datetime import datetime, timezone

        from app.automation.models import ScheduleTriggerType

        service = _build_service()

        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        spec = service.create_schedule(
            workflow_id="shot-wf",
            trigger_type=ScheduleTriggerType.DATETIME,
            run_at=future,
        )
        assert spec.next_run_at == future

    @pytest.mark.asyncio
    async def test_tick_fires_due_schedule(self) -> None:
        """Advance a clock so a schedule becomes due and verify tick fires it."""
        from datetime import datetime, timedelta, timezone

        from app.automation.models import (
            ScheduleTriggerType,
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.time import Clock

        class ManualClock(Clock):
            def __init__(self, now: datetime) -> None:
                self._now = now

            def utcnow(self) -> datetime:
                return self._now

            def now(self) -> datetime:
                return self._now

            def time(self) -> float:
                return self._now.timestamp()

            def advance(self, delta: timedelta) -> None:
                self._now += delta

        base = datetime(2030, 1, 1, tzinfo=timezone.utc)
        clock = ManualClock(base)

        # Build a service with a manual clock
        from app.automation.executor import WorkflowExecutor
        from app.automation.registries import (
            StepTypeRegistry,
            WorkflowRegistry,
            WorkflowRunRegistry,
        )
        from app.automation.stores import (
            InMemoryEventStore,
            InMemoryScheduleStore,
            InMemoryWorkflowRunStore,
            InMemoryWorkflowStore,
        )
        from app.automation.validation import WorkflowValidator
        from app.automation.workflow import WorkflowGraphExporter, WorkflowService

        step_types = StepTypeRegistry()
        executor = WorkflowExecutor(step_types=step_types, clock=clock)
        service = WorkflowService(
            registry=WorkflowRegistry(),
            run_registry=WorkflowRunRegistry(),
            executor=executor,
            validator=WorkflowValidator(),
            exporter=WorkflowGraphExporter(),
            definition_store=InMemoryWorkflowStore(),
            run_store=InMemoryWorkflowRunStore(),
            schedule_store=InMemoryScheduleStore(),
            event_store=InMemoryEventStore(),
            clock=clock,
        )

        # Register a trivial workflow
        defn = WorkflowDefinition(
            id="tick-wf",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="t",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"ok"',
                ),
            ),
        )
        service.create_definition(defn)
        service.publish_definition("tick-wf", "1.0.0")

        # Create an interval schedule that fires every 60s
        spec = service.create_schedule(
            workflow_id="tick-wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60,
        )

        # Tick immediately – schedule is 60s in the future, not due yet
        fired = await service.scheduler.tick()
        assert len(fired) == 0

        # Advance clock past the due time
        clock.advance(timedelta(seconds=61))

        # Tick again – should fire the schedule
        fired = await service.scheduler.tick()
        assert len(fired) == 1
        assert fired[0].schedule_id == spec.schedule_id


# ===========================================================================
# 4. WORKFLOW VALIDATION
# ===========================================================================


class TestWorkflowValidation:
    """Verify the validator catches common definition errors."""

    def test_no_steps(self) -> None:
        from app.automation.models import WorkflowDefinition
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(id="empty", version="1.0.0", steps=())
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "no_steps" in codes

    def test_empty_id(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="s",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"x"',
                ),
            ),
        )
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "empty_id" in codes

    def test_invalid_version(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="wf",
            version="not-semver",
            steps=(
                WorkflowStep(
                    id="s",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"x"',
                ),
            ),
        )
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "invalid_version" in codes

    def test_circular_dependency(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="cyclic",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="a",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"a"',
                    depends_on=("b",),
                ),
                WorkflowStep(
                    id="b",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"b"',
                    depends_on=("a",),
                ),
            ),
        )
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "cycle" in codes

    def test_invalid_dependency_reference(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="bad-ref",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="x",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"x"',
                    depends_on=("nonexistent",),
                ),
            ),
        )
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "invalid_dependency" in codes

    def test_duplicate_step_ids(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="dup",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="same",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"a"',
                ),
                WorkflowStep(
                    id="same",
                    type=WorkflowStepType.TRANSFORM,
                    expression='"b"',
                ),
            ),
        )
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "duplicate_step_id" in codes

    def test_task_step_missing_tool_name(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="missing-tool",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="t1",
                    type=WorkflowStepType.TASK,
                    tool_name="",  # missing required field
                ),
            ),
        )
        report = validator.validate(definition)
        assert not report.valid
        codes = [v.code for v in report.errors]
        assert "missing_required_field" in codes

    def test_valid_definition_passes(self) -> None:
        from app.automation.models import (
            WorkflowDefinition,
            WorkflowStep,
            WorkflowStepType,
        )
        from app.automation.validation import WorkflowValidator

        validator = WorkflowValidator()
        definition = WorkflowDefinition(
            id="good",
            version="1.0.0",
            steps=(
                WorkflowStep(
                    id="s1",
                    type=WorkflowStepType.TRANSFORM,
                    name="Step 1",
                    expression='"hello"',
                ),
                WorkflowStep(
                    id="s2",
                    type=WorkflowStepType.TRANSFORM,
                    name="Step 2",
                    expression='"world"',
                    depends_on=("s1",),
                ),
            ),
        )
        report = validator.validate(definition)
        assert report.valid, f"Expected valid, got violations: {report.violations}"
