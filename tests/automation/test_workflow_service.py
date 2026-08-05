"""Tests for the workflow service façade."""

from __future__ import annotations

import asyncio

import pytest

from app.automation.exceptions import (
    WorkflowApprovalError,
    WorkflowNotFoundError,
    WorkflowValidationError,
    WorkflowVersionError,
)
from app.automation.executor import WorkflowExecutor
from app.automation.models import (
    ApprovalStatus,
    ScheduleTriggerType,
    WorkflowDefinition,
    WorkflowInput,
    WorkflowOutput,
    WorkflowRunStatus,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepType,
)
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
from app.automation.time import FakeClock
from app.automation.validation import WorkflowValidator
from app.automation.workflow import WorkflowGraphExporter, WorkflowService


def _echo_task_executor():
    async def execute(step, step_ctx, engine, state, run, snapshot, shared):
        return {"echo": dict(step_ctx.inputs)}

    return execute


def _transform_definition(
    workflow_id: str = "wf", version: str = "1.0.0"
) -> WorkflowDefinition:
    return WorkflowDefinition(
        id=workflow_id,
        version=version,
        inputs=(WorkflowInput(name="message", type="str", required=True),),
        steps=(
            WorkflowStep(
                id="t1",
                type=WorkflowStepType.TRANSFORM,
                expression="${inputs.message}",
            ),
        ),
        outputs=(WorkflowOutput(name="out", source="${step.t1}"),),
    )


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def service(fake_clock) -> WorkflowService:
    run_registry = WorkflowRunRegistry()
    step_types = StepTypeRegistry()
    step_types.register(WorkflowStepType.TASK, _echo_task_executor())
    executor = WorkflowExecutor(step_types, clock=fake_clock)
    counter = {"n": 0}

    def _run_id() -> str:
        counter["n"] += 1
        return f"run-{counter['n']}"

    return WorkflowService(
        registry=WorkflowRegistry(),
        run_registry=run_registry,
        executor=executor,
        validator=WorkflowValidator(),
        exporter=WorkflowGraphExporter(),
        definition_store=InMemoryWorkflowStore(),
        run_store=InMemoryWorkflowRunStore(),
        schedule_store=InMemoryScheduleStore(),
        event_store=InMemoryEventStore(),
        clock=fake_clock,
        run_id_factory=_run_id,
    )


# ----------------------------------------------------------------------
# Definition management
# ----------------------------------------------------------------------


def test_create_and_get_definition(service) -> None:
    definition = _transform_definition()
    created = service.create_definition(definition)
    assert created.id == "wf"
    got = service.get_definition("wf", "1.0.0")
    assert got is not None and got.id == "wf"
    assert service.get_definition("missing") is None


def test_list_definitions_filters_by_status(service) -> None:
    service.create_definition(_transform_definition("a"))
    service.create_definition(_transform_definition("b"))
    published = service.publish_definition("a", "1.0.0")
    assert published.status == WorkflowStatus.PUBLISHED
    drafts = service.list_definitions(WorkflowStatus.DRAFT)
    assert {d.id for d in drafts} == {"b"}
    published_list = service.list_definitions(WorkflowStatus.PUBLISHED)
    assert {d.id for d in published_list} == {"a"}


def test_publish_validates_definition(service) -> None:
    invalid = WorkflowDefinition(
        id="bad",
        version="1.0.0",
        steps=(),
    )
    service.create_definition(invalid)
    with pytest.raises(WorkflowValidationError):
        service.publish_definition("bad", "1.0.0")


def test_publish_unknown_raises(service) -> None:
    with pytest.raises(WorkflowNotFoundError):
        service.publish_definition("nope", "1.0.0")


def test_published_definition_is_immutable(service) -> None:
    service.create_definition(_transform_definition("wf", "1.0.0"))
    service.publish_definition("wf", "1.0.0")
    duplicate = _transform_definition("wf", "1.0.0")
    with pytest.raises(WorkflowVersionError):
        service.create_definition(duplicate)


def test_delete_draft_only(service) -> None:
    service.create_definition(_transform_definition("wf", "1.0.0"))
    assert service.delete_definition("wf", "1.0.0") is True
    assert service.get_definition("wf", "1.0.0") is None
    assert service.delete_definition("wf", "1.0.0") is False


def test_delete_published_raises(service) -> None:
    service.create_definition(_transform_definition("wf", "1.0.0"))
    service.publish_definition("wf", "1.0.0")
    with pytest.raises(WorkflowVersionError):
        service.delete_definition("wf", "1.0.0")


def test_export_graph(service) -> None:
    definition = WorkflowDefinition(
        id="wf",
        version="1.0.0",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK),
            WorkflowStep(id="b", type=WorkflowStepType.TASK, depends_on=("a",)),
        ),
    )
    service.create_definition(definition)
    graph = service.export_graph("wf")
    assert graph["workflow_id"] == "wf"
    assert {node["id"] for node in graph["nodes"]} == {"a", "b"}
    assert {"source": "a", "target": "b"} in graph["edges"]
    assert graph["topological_order"] == ["a", "b"]


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------


def test_execute_published_workflow_succeeds(service, fake_clock) -> None:
    service.create_definition(_transform_definition())
    service.publish_definition("wf", "1.0.0")
    run = asyncio.run(service.execute("wf", {"message": "hello"}))
    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert run.outputs == {"out": "hello"}
    assert run.snapshot.workflow_id == "wf"
    assert run.snapshot.step_definitions == _transform_definition().steps


def test_execute_requires_published_when_no_version(service) -> None:
    service.create_definition(_transform_definition())
    with pytest.raises(WorkflowNotFoundError):
        asyncio.run(service.execute("wf", {"message": "hi"}))


def test_execute_version_runs_specific_version(service) -> None:
    service.create_definition(_transform_definition("wf", "1.0.0"))
    service.publish_definition("wf", "1.0.0")
    run = asyncio.run(service.execute_version("wf", "1.0.0", {"message": "v1"}))
    assert run.workflow_version == "1.0.0"
    assert run.status == WorkflowRunStatus.SUCCEEDED


def test_execute_unknown_workflow_raises(service) -> None:
    with pytest.raises(WorkflowNotFoundError):
        asyncio.run(service.execute("missing", {}))


def test_execute_idempotency_returns_existing_run(service) -> None:
    service.create_definition(_transform_definition())
    service.publish_definition("wf", "1.0.0")
    first = asyncio.run(service.execute("wf", {"message": "a"}, idempotency_key="k1"))
    second = asyncio.run(service.execute("wf", {"message": "b"}, idempotency_key="k1"))
    assert first.run_id == second.run_id
    assert second.inputs == {"message": "a"}
    assert service.list_runs() == [first]


def test_approval_flow_approve_then_resume(service) -> None:
    definition = WorkflowDefinition(
        id="approval",
        version="1.0.0",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.APPROVAL,
                name="Review",
                timeout_s=60.0,
            ),
            WorkflowStep(
                id="b",
                type=WorkflowStepType.TRANSFORM,
                expression="${step.a.output.approved}",
                depends_on=("a",),
            ),
        ),
    )
    service.create_definition(definition)
    service.publish_definition("approval", "1.0.0")
    run = asyncio.run(service.execute("approval", {}))
    assert run.status == WorkflowRunStatus.WAITING_APPROVAL
    request = run.approval_requests[0]
    assert request.status == ApprovalStatus.PENDING

    decided = service.approve(run.run_id, request.request_id, decided_by="bob")
    assert decided.approval_requests[0].status == ApprovalStatus.APPROVED

    resumed = asyncio.run(service.resume(run.run_id))
    assert resumed.status == WorkflowRunStatus.SUCCEEDED


def test_approval_flow_reject_then_resume(service) -> None:
    definition = WorkflowDefinition(
        id="approval",
        version="1.0.0",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.APPROVAL,
                name="Review",
                timeout_s=60.0,
            ),
        ),
    )
    service.create_definition(definition)
    service.publish_definition("approval", "1.0.0")
    run = asyncio.run(service.execute("approval", {}))
    request = run.approval_requests[0]
    service.reject(run.run_id, request.request_id, decided_by="bob")
    resumed = asyncio.run(service.resume(run.run_id))
    assert resumed.approval_requests[0].status == ApprovalStatus.REJECTED


def test_approve_unknown_request_raises(service) -> None:
    definition = WorkflowDefinition(
        id="approval",
        version="1.0.0",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.APPROVAL,
                timeout_s=60.0,
            ),
        ),
    )
    service.create_definition(definition)
    service.publish_definition("approval", "1.0.0")
    run = asyncio.run(service.execute("approval", {}))
    with pytest.raises(WorkflowApprovalError):
        service.approve(run.run_id, "not-a-request")


def test_approve_twice_raises(service) -> None:
    definition = WorkflowDefinition(
        id="approval",
        version="1.0.0",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.APPROVAL,
                timeout_s=60.0,
            ),
        ),
    )
    service.create_definition(definition)
    service.publish_definition("approval", "1.0.0")
    run = asyncio.run(service.execute("approval", {}))
    request = run.approval_requests[0]
    service.approve(run.run_id, request.request_id)
    with pytest.raises(WorkflowApprovalError):
        service.approve(run.run_id, request.request_id)


def test_cancel_paused_run(service) -> None:
    definition = WorkflowDefinition(
        id="approval",
        version="1.0.0",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.APPROVAL,
                timeout_s=60.0,
            ),
        ),
    )
    service.create_definition(definition)
    service.publish_definition("approval", "1.0.0")
    run = asyncio.run(service.execute("approval", {}))
    assert run.status == WorkflowRunStatus.WAITING_APPROVAL
    cancelled = service.cancel(run.run_id)
    assert cancelled.status == WorkflowRunStatus.CANCELLED
    assert cancelled.completed_at is not None


def test_cancel_unknown_run_raises(service) -> None:
    with pytest.raises(WorkflowNotFoundError):
        service.cancel("ghost")


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------


def test_create_and_disable_schedule(service) -> None:
    service.create_definition(_transform_definition())
    service.publish_definition("wf", "1.0.0")
    spec = service.create_schedule(
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.INTERVAL,
        interval_seconds=60.0,
    )
    assert service.get_schedule(spec.schedule_id) is spec
    assert service.list_schedules(enabled=True) == [spec]

    disabled = service.disable_schedule(spec.schedule_id)
    assert disabled.enabled is False
    assert service.list_schedules(enabled=True) == []

    reenabled = service.enable_schedule(spec.schedule_id)
    assert reenabled.enabled is True


def test_update_and_delete_schedule(service) -> None:
    service.create_definition(_transform_definition())
    service.publish_definition("wf", "1.0.0")
    spec = service.create_schedule(
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.INTERVAL,
        interval_seconds=60.0,
    )
    updated = service.update_schedule(
        spec.model_copy(update={"interval_seconds": 120.0})
    )
    assert updated.interval_seconds == 120.0
    assert service.delete_schedule(spec.schedule_id) is True
    assert service.delete_schedule(spec.schedule_id) is False


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_startup_restores_and_shuts_down(service, fake_clock) -> None:
    service.create_definition(_transform_definition())
    service.publish_definition("wf", "1.0.0")
    service.create_schedule(
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.INTERVAL,
        interval_seconds=60.0,
    )

    async def lifecycle() -> None:
        await service.startup()
        assert service.scheduler is not None
        await service.shutdown()

    asyncio.run(lifecycle())


def test_resume_terminal_run_is_noop(service) -> None:
    service.create_definition(_transform_definition())
    service.publish_definition("wf", "1.0.0")
    run = asyncio.run(service.execute("wf", {"message": "x"}))
    resumed = asyncio.run(service.resume(run.run_id))
    assert resumed.run_id == run.run_id
    assert resumed.status == WorkflowRunStatus.SUCCEEDED
