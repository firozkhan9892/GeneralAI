"""Tests for workflow registries."""

from __future__ import annotations

import pytest

from app.automation.exceptions import (
    WorkflowNotFoundError,
    WorkflowVersionError,
)
from app.automation.models import (
    ScheduleSpec,
    ScheduleTriggerType,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowSnapshot,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.registries import (
    ScheduleRegistry,
    StepTypeRegistry,
    WorkflowRegistry,
    WorkflowRunRegistry,
    _version_key,
)


@pytest.fixture
def draft(linear_definition: WorkflowDefinition) -> WorkflowDefinition:
    return linear_definition


@pytest.fixture
def registry() -> WorkflowRegistry:
    return WorkflowRegistry()


def test_register_and_get(registry, draft) -> None:
    registry.register(draft)
    assert registry.get("linear") is not None
    assert registry.get("linear", version="1.0.0") is not None
    assert registry.get("missing") is None


def test_publish_sets_status_and_is_immutable(registry, draft) -> None:
    registry.register(draft)
    published = registry.publish("linear", "1.0.0")
    assert published.status == WorkflowStatus.PUBLISHED
    assert registry.get_published("linear").version == "1.0.0"

    with pytest.raises(WorkflowVersionError):
        registry.register(
            WorkflowDefinition(
                id="linear",
                version="1.0.0",
                steps=(
                    WorkflowStep(id="x", type=WorkflowStepType.TASK, tool_name="echo"),
                ),
            )
        )


def test_unregister_published_raises(registry, draft) -> None:
    registry.register(draft)
    registry.publish("linear", "1.0.0")
    with pytest.raises(WorkflowVersionError):
        registry.unregister("linear", "1.0.0")


def test_unregister_draft_succeeds(registry, draft) -> None:
    registry.register(draft)
    assert registry.unregister("linear", "1.0.0") is True
    assert registry.get("linear") is None


def test_get_latest_returns_published_preference(registry) -> None:
    registry.register(
        WorkflowDefinition(
            id="wf",
            version="1.0.0",
            steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
        )
    )
    registry.publish("wf", "1.0.0")
    registry.register(
        WorkflowDefinition(
            id="wf",
            version="2.0.0",
            steps=(WorkflowStep(id="b", type=WorkflowStepType.TASK, tool_name="echo"),),
        )
    )
    latest = registry.get("wf")
    assert latest.version == "1.0.0"  # published wins over draft 2.0.0


def test_get_versions_newest_first(registry, draft) -> None:
    registry.register(draft)
    registry.register(
        WorkflowDefinition(
            id="linear",
            version="2.1.0",
            steps=(WorkflowStep(id="z", type=WorkflowStepType.TASK, tool_name="echo"),),
        )
    )
    assert registry.list_versions("linear") == ["2.1.0", "1.0.0"]


def test_list_filters_by_status(registry, draft) -> None:
    registry.register(draft)
    assert len(registry.list_all(status=WorkflowStatus.DRAFT)) == 1
    assert len(registry.list_all(status=WorkflowStatus.PUBLISHED)) == 0


def test_version_key_sorting() -> None:
    versions = ["1.10.0", "1.2.0", "1.2.1"]
    assert sorted(versions, key=_version_key) == ["1.2.0", "1.2.1", "1.10.0"]


def test_run_registry_save_and_retrieve() -> None:
    runs = WorkflowRunRegistry()
    run = WorkflowRun(
        run_id="r1",
        workflow_id="wf",
        workflow_version="1.0.0",
        snapshot=WorkflowSnapshot(workflow_id="wf", version="1.0.0"),
    )
    runs.save(run)
    assert runs.get("r1") is run
    assert runs.get("missing") is None
    assert len(runs) == 1


def test_run_registry_delete() -> None:
    runs = WorkflowRunRegistry()
    run = WorkflowRun(
        run_id="r1",
        workflow_id="wf",
        workflow_version="1.0.0",
        snapshot=WorkflowSnapshot(workflow_id="wf", version="1.0.0"),
    )
    runs.save(run)
    assert runs.delete("r1") is True
    assert runs.delete("r1") is False


def test_run_registry_find_by_idempotency() -> None:
    runs = WorkflowRunRegistry()
    first = WorkflowRun(
        run_id="r1",
        workflow_id="wf",
        workflow_version="1.0.0",
        snapshot=WorkflowSnapshot(workflow_id="wf", version="1.0.0"),
        idempotency_key="dup-key",
    )
    second = first.model_copy(update={"run_id": "r2"})
    runs.save(first)
    runs.save(second)
    found = runs.find_by_idempotency("wf", "1.0.0", "dup-key")
    assert found is not None
    assert found.run_id == "r2"
    assert runs.find_by_idempotency("wf", "1.0.0", "other") is None


def test_schedule_registry() -> None:
    schedules = ScheduleRegistry()
    spec = ScheduleSpec(
        schedule_id="s1",
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.INTERVAL,
        interval_seconds=60,
    )
    schedules.save(spec)
    assert schedules.get("s1") is spec
    assert len(schedules.list_all()) == 1
    schedules.save(spec.model_copy(update={"enabled": False}))
    assert schedules.list_all(enabled=False) == [
        spec.model_copy(update={"enabled": False})
    ]
    assert schedules.delete("s1") is True


def test_step_type_registry() -> None:
    step_types = StepTypeRegistry()

    async def executor(step, step_ctx, engine, state, run, snapshot, shared):
        return {"ok": True}

    step_types.register(WorkflowStepType.TASK, executor)
    assert step_types.has(WorkflowStepType.TASK)
    assert step_types.get(WorkflowStepType.TASK) is executor
    assert step_types.list() == ["task"]

    with pytest.raises(ValueError):
        step_types.register(WorkflowStepType.TASK, executor)

    step_types.unregister(WorkflowStepType.TASK)
    assert not step_types.has(WorkflowStepType.TASK)


def test_publish_missing_raises(registry) -> None:
    with pytest.raises(WorkflowNotFoundError):
        registry.publish("ghost", "1.0.0")
