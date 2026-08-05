"""Unit tests for automation persistence stores (in-memory)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.automation.models import (
    ScheduleSpec,
    ScheduleTriggerType,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowSnapshot,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.stores import (
    InMemoryEventStore,
    InMemoryScheduleStore,
    InMemoryWorkflowRunStore,
    InMemoryWorkflowStore,
)


def _definition(workflow_id: str = "wf", version: str = "1.0.0") -> WorkflowDefinition:
    return WorkflowDefinition(
        id=workflow_id,
        version=version,
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK),),
    )


def _run(run_id: str = "run1", status: WorkflowRunStatus = WorkflowRunStatus.PENDING):
    definition = _definition()
    snapshot = WorkflowSnapshot(
        workflow_id=definition.id,
        version=definition.version,
        step_definitions=definition.steps,
    )
    return WorkflowRun(
        run_id=run_id,
        workflow_id=definition.id,
        workflow_version=definition.version,
        status=status,
        snapshot=snapshot,
    )


def test_in_memory_workflow_store_roundtrip() -> None:
    store = InMemoryWorkflowStore()
    definition = _definition()
    store.save_definition(definition)

    assert store.get_definition("wf", "1.0.0") == definition
    assert store.get_definition("wf") == definition
    assert store.has("wf", "1.0.0") is True
    assert store.has("ghost") is False
    assert len(store) == 1

    assert store.delete_definition("ghost") is False
    assert store.delete_definition("wf", "1.0.0") is True
    assert store.get_definition("wf") is None


def test_in_memory_workflow_store_prefers_published() -> None:
    store = InMemoryWorkflowStore()
    store.save_definition(_definition(version="1.0.0"))
    published = _definition(version="2.0.0").with_status(WorkflowStatus.PUBLISHED)
    store.save_definition(published)

    assert store.get_definition("wf") == published
    draft = store.get_definition("wf", "1.0.0")
    assert draft is not None
    assert draft.status == WorkflowStatus.DRAFT
    assert len(store.list_definitions()) == 2
    assert len(store.list_definitions(status=WorkflowStatus.PUBLISHED)) == 1


def test_in_memory_run_store_roundtrip() -> None:
    store = InMemoryWorkflowRunStore()
    run = _run()
    store.save_run(run)

    assert store.get_run("run1") == run
    assert store.get_run("missing") is None
    assert len(store.list_runs()) == 1
    assert len(store.list_runs(workflow_id="wf")) == 1
    assert len(store.list_runs(workflow_id="other")) == 0
    assert store.delete_run("run1") is True
    assert store.delete_run("run1") is False


def test_in_memory_run_store_find_by_idempotency() -> None:
    store = InMemoryWorkflowRunStore()
    first = _run("run-a").model_copy(
        update={"idempotency_key": "key-1", "status": WorkflowRunStatus.FAILED}
    )
    second = _run("run-b").model_copy(
        update={"idempotency_key": "key-1", "status": WorkflowRunStatus.SUCCEEDED}
    )
    store.save_run(first)
    store.save_run(second)

    found = store.find_by_idempotency("wf", "1.0.0", "key-1")
    assert found is not None
    assert found.run_id == "run-b"
    assert store.find_by_idempotency("wf", "1.0.0", "missing") is None


def test_in_memory_run_store_list_resumable() -> None:
    store = InMemoryWorkflowRunStore()
    store.save_run(_run("pending", WorkflowRunStatus.PENDING))
    store.save_run(_run("approval", WorkflowRunStatus.WAITING_APPROVAL))
    store.save_run(_run("done", WorkflowRunStatus.SUCCEEDED))

    resumable = store.list_resumable()
    assert {run.run_id for run in resumable} == {"pending", "approval"}


def test_in_memory_schedule_store_roundtrip() -> None:
    store = InMemoryScheduleStore()
    spec = ScheduleSpec(
        schedule_id="s1",
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.INTERVAL,
        interval_seconds=60.0,
    )
    store.save_schedule(spec)

    assert store.get_schedule("s1") == spec
    assert len(store.list_schedules()) == 1
    assert len(store.list_schedules(enabled=False)) == 0
    assert store.delete_schedule("s1") is True
    assert store.get_schedule("s1") is None


def test_in_memory_event_store() -> None:
    store = InMemoryEventStore()
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.append_event(
        WorkflowEvent(event_type="workflow.run.started", run_id="r1", timestamp=stamp)
    )
    store.append_event(
        WorkflowEvent(
            event_type="workflow.step.completed", run_id="r1", timestamp=stamp
        )
    )
    store.append_event(
        WorkflowEvent(event_type="workflow.run.started", run_id="r2", timestamp=stamp)
    )

    assert len(store.list_events(run_id="r1")) == 2
    assert len(store.list_events(event_type="workflow.run.started")) == 2
    assert len(store.events_for_run("r1")) == 2
    assert len(store.events_for_run("r2")) == 1
    assert store.delete_events("r2") is True
    assert store.delete_events("r2") is False
    assert len(store.events_for_run("r2")) == 0
