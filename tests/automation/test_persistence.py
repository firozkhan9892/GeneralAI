"""Unit tests for JSON-backed automation persistence stores."""

from __future__ import annotations

from datetime import datetime, timezone

from app.automation.models import (
    ApprovalRequest,
    ApprovalStatus,
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
from app.automation.persistence import (
    JsonEventStore,
    JsonScheduleStore,
    JsonWorkflowRunStore,
    JsonWorkflowStore,
)


def _definition(workflow_id: str = "wf", version: str = "1.0.0") -> WorkflowDefinition:
    return WorkflowDefinition(
        id=workflow_id,
        version=version,
        name="JSON workflow",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                tool_name="echo",
                input_bindings={"message": "${inputs.message}"},
            ),
        ),
    )


def _run(
    run_id: str = "run1",
    status: WorkflowRunStatus = WorkflowRunStatus.PENDING,
    with_approval: bool = False,
) -> WorkflowRun:
    definition = _definition()
    snapshot = WorkflowSnapshot(
        workflow_id=definition.id,
        version=definition.version,
        step_definitions=definition.steps,
        settings=definition.settings,
        inputs=definition.inputs,
        outputs=definition.outputs,
    )
    approval = (
        (
            ApprovalRequest(
                request_id="req1",
                run_id=run_id,
                step_id="a",
                approvers=("alice",),
                status=ApprovalStatus.PENDING,
            ),
        )
        if with_approval
        else ()
    )
    return WorkflowRun(
        run_id=run_id,
        workflow_id=definition.id,
        workflow_version=definition.version,
        status=status,
        snapshot=snapshot,
        idempotency_key=f"idem-{run_id}",
        inputs={"message": "hello"},
        approval_requests=approval,
    )


def test_json_workflow_store_roundtrip(tmp_path) -> None:
    store = JsonWorkflowStore(tmp_path / "workflows.json")
    store.save_definition(_definition())

    fresh = JsonWorkflowStore(tmp_path / "workflows.json")
    loaded = fresh.get_definition("wf", "1.0.0")
    assert loaded is not None
    assert loaded.name == "JSON workflow"
    assert loaded.steps[0].tool_name == "echo"


def test_json_workflow_store_prefers_published(tmp_path) -> None:
    store = JsonWorkflowStore(tmp_path / "workflows.json")
    store.save_definition(_definition(version="1.0.0"))
    store.save_definition(
        _definition(version="2.0.0").with_status(WorkflowStatus.PUBLISHED)
    )

    fresh = JsonWorkflowStore(tmp_path / "workflows.json")
    latest = fresh.get_definition("wf")
    assert latest is not None
    assert latest.version == "2.0.0"
    draft = fresh.get_definition("wf", "1.0.0")
    assert draft is not None
    assert draft.status == WorkflowStatus.DRAFT
    assert len(fresh.list_definitions()) == 2


def test_json_workflow_store_delete_and_has(tmp_path) -> None:
    store = JsonWorkflowStore(tmp_path / "workflows.json")
    store.save_definition(_definition())
    assert store.has("wf") is True

    assert store.delete_definition("wf", "1.0.0") is True
    assert store.has("wf", "1.0.0") is False


def test_json_run_store_roundtrip_preserves_idempotency(tmp_path) -> None:
    store = JsonWorkflowRunStore(tmp_path / "runs.json")
    store.save_run(_run("run1"))

    fresh = JsonWorkflowRunStore(tmp_path / "runs.json")
    loaded = fresh.get_run("run1")
    assert loaded is not None
    assert loaded.idempotency_key == "idem-run1"
    assert loaded.inputs == {"message": "hello"}
    assert loaded.snapshot.step_definitions[0].tool_name == "echo"


def test_json_run_store_find_by_idempotency(tmp_path) -> None:
    store = JsonWorkflowRunStore(tmp_path / "runs.json")
    store.save_run(_run("run1").model_copy(update={"status": WorkflowRunStatus.FAILED}))
    store.save_run(
        _run("run2").model_copy(update={"status": WorkflowRunStatus.SUCCEEDED})
    )

    fresh = JsonWorkflowRunStore(tmp_path / "runs.json")
    found = fresh.find_by_idempotency("wf", "1.0.0", "idem-run2")
    assert found is not None
    assert found.run_id == "run2"
    assert fresh.find_by_idempotency("wf", "1.0.0", "missing") is None


def test_json_run_store_resume_roundtrip(tmp_path) -> None:
    """A WAITING_APPROVAL run survives a restart and stays resumable."""
    store = JsonWorkflowRunStore(tmp_path / "runs.json")
    waiting = _run("waiting", WorkflowRunStatus.WAITING_APPROVAL, with_approval=True)
    completed = _run("done", WorkflowRunStatus.SUCCEEDED)
    store.save_run(waiting)
    store.save_run(completed)

    fresh = JsonWorkflowRunStore(tmp_path / "runs.json")
    restored = fresh.list_resumable()
    assert [run.run_id for run in restored] == ["waiting"]
    assert restored[0].approval_requests[0].request_id == "req1"
    assert restored[0].approval_requests[0].status == ApprovalStatus.PENDING


def test_json_run_store_list_resumable_excludes_completed(tmp_path) -> None:
    store = JsonWorkflowRunStore(tmp_path / "runs.json")
    store.save_run(_run("pending"))
    store.save_run(_run("done", WorkflowRunStatus.SUCCEEDED))

    fresh = JsonWorkflowRunStore(tmp_path / "runs.json")
    assert [run.run_id for run in fresh.list_resumable()] == ["pending"]


def test_json_schedule_store_roundtrip(tmp_path) -> None:
    store = JsonScheduleStore(tmp_path / "schedules.json")
    spec = ScheduleSpec(
        schedule_id="s1",
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.CRON,
        cron_expression="0 9 * * 1",
        payload={"channel": "slack"},
    )
    store.save_schedule(spec)

    fresh = JsonScheduleStore(tmp_path / "schedules.json")
    loaded = fresh.get_schedule("s1")
    assert loaded is not None
    assert loaded.cron_expression == "0 9 * * 1"
    assert loaded.payload == {"channel": "slack"}
    assert len(fresh.list_schedules()) == 1


def test_json_event_store_replay(tmp_path) -> None:
    store = JsonEventStore(tmp_path / "events.json")
    stamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    store.append_event(
        WorkflowEvent(event_type="workflow.run.started", run_id="r1", timestamp=stamp)
    )
    store.append_event(
        WorkflowEvent(
            event_type="workflow.step.completed",
            run_id="r1",
            timestamp=stamp,
            data={"step_id": "a"},
        )
    )

    fresh = JsonEventStore(tmp_path / "events.json")
    replayed = fresh.events_for_run("r1")
    assert [event.event_type for event in replayed] == [
        "workflow.run.started",
        "workflow.step.completed",
    ]
    assert replayed[1].data == {"step_id": "a"}
    assert replayed[0].timestamp == stamp
    assert fresh.delete_events("r1") is True
    assert fresh.events_for_run("r1") == []


def test_json_stores_isolate_documents(tmp_path) -> None:
    """Different store types never collide on the same directory."""
    directory = tmp_path
    JsonWorkflowStore(directory / "workflows.json").save_definition(_definition())
    JsonScheduleStore(directory / "schedules.json").save_schedule(
        ScheduleSpec(
            schedule_id="s1",
            workflow_id="wf",
            trigger_type=ScheduleTriggerType.INTERVAL,
            interval_seconds=60.0,
        )
    )

    assert len(JsonWorkflowStore(directory / "workflows.json").list_definitions()) == 1
    assert len(JsonScheduleStore(directory / "schedules.json").list_schedules()) == 1


def test_corrupt_document_recovers_empty(tmp_path) -> None:
    path = tmp_path / "runs.json"
    path.write_text("{not json", encoding="utf-8")
    store = JsonWorkflowRunStore(path)
    assert store.get_run("anything") is None
    assert store.list_runs() == []
