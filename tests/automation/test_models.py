"""Tests for workflow domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.automation.models import (
    Branch,
    RESUMABLE_RUN_STATUSES,
    ScheduleSpec,
    ScheduleTriggerType,
    StepExecution,
    StepExecutionStatus,
    StepRetryPolicy,
    TERMINAL_RUN_STATUSES,
    WorkflowDefinition,
    WorkflowInput,
    WorkflowOutput,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowSnapshot,
    WorkflowStep,
    WorkflowStepType,
    WorkflowStatus,
)
from app.kernel.pipeline.models import ErrorPolicy


def test_definition_is_frozen() -> None:
    definition = WorkflowDefinition(id="wf", steps=())
    with pytest.raises(ValidationError):
        definition.id = "other"  # type: ignore[misc]


def test_step_is_frozen() -> None:
    step = WorkflowStep(id="s", type=WorkflowStepType.TASK, tool_name="echo")
    with pytest.raises(ValidationError):
        step.name = "other"  # type: ignore[misc]


def test_default_error_policy_is_abort() -> None:
    step = WorkflowStep(id="s", type=WorkflowStepType.TASK, tool_name="echo")
    assert step.error_policy == ErrorPolicy.ABORT


def test_with_status_returns_copy() -> None:
    definition = WorkflowDefinition(id="wf", steps=())
    published = definition.with_status(WorkflowStatus.PUBLISHED)
    assert published is not definition
    assert published.status == WorkflowStatus.PUBLISHED
    assert definition.status == WorkflowStatus.DRAFT


def test_nested_loop_steps_round_trip() -> None:
    step = WorkflowStep(
        id="loop",
        type=WorkflowStepType.LOOP,
        iterable="${inputs.items}",
        loop_steps=(
            WorkflowStep(id="inner", type=WorkflowStepType.TASK, tool_name="echo"),
        ),
    )
    dumped = step.model_dump(mode="python")
    reloaded = WorkflowStep.model_validate(dumped)
    assert reloaded.loop_steps[0].id == "inner"
    assert reloaded.loop_steps[0].type == WorkflowStepType.TASK


def test_nested_branch_steps_round_trip() -> None:
    step = WorkflowStep(
        id="cond",
        type=WorkflowStepType.CONDITIONAL,
        branches=(
            Branch(
                name="yes",
                when="true",
                steps=(
                    WorkflowStep(
                        id="sub", type=WorkflowStepType.DELAY, delay_seconds=1
                    ),
                ),
            ),
        ),
    )
    dumped = step.model_dump(mode="python")
    reloaded = WorkflowStep.model_validate(dumped)
    assert reloaded.branches[0].steps[0].id == "sub"
    assert reloaded.branches[0].when == "true"


def test_retry_policy_validation() -> None:
    with pytest.raises(ValidationError):
        StepRetryPolicy(max_retries=50)


def test_definition_json_serialisable() -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(WorkflowStep(id="s", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    dumped = definition.model_dump(mode="json")
    reloaded = WorkflowDefinition.model_validate(dumped)
    assert reloaded == definition


def test_run_is_frozen_and_tracks_terminal_state() -> None:
    run = WorkflowRun(
        run_id="r1",
        workflow_id="wf",
        workflow_version="1.0.0",
        snapshot=WorkflowSnapshot(workflow_id="wf", version="1.0.0"),
    )
    assert run.status == WorkflowRunStatus.PENDING
    assert not run.is_terminal
    assert run.is_resumable

    completed = run.model_copy(update={"status": WorkflowRunStatus.SUCCEEDED})
    assert completed.is_terminal
    assert not completed.is_resumable


def test_run_step_lookup() -> None:
    run = WorkflowRun(
        run_id="r1",
        workflow_id="wf",
        workflow_version="1.0.0",
        snapshot=WorkflowSnapshot(workflow_id="wf", version="1.0.0"),
        step_executions=(
            StepExecution(step_id="s1", status=StepExecutionStatus.SUCCEEDED),
        ),
    )
    step = run.step("s1")
    assert step is not None
    assert step.status == StepExecutionStatus.SUCCEEDED
    assert run.step("missing") is None


def test_terminal_and_resumable_sets_are_disjoint() -> None:
    assert TERMINAL_RUN_STATUSES.isdisjoint(RESUMABLE_RUN_STATUSES)


def test_schedule_spec_defaults() -> None:
    schedule = ScheduleSpec(
        schedule_id="s1",
        workflow_id="wf",
        trigger_type=ScheduleTriggerType.INTERVAL,
        interval_seconds=60,
    )
    assert schedule.enabled is True
    assert schedule.max_concurrent_runs == 1
    assert schedule.timezone == "UTC"


def test_inputs_outputs_declared() -> None:
    definition = WorkflowDefinition(
        id="wf",
        inputs=(WorkflowInput(name="x", required=True),),
        outputs=(WorkflowOutput(name="y", source="${step.a.output}"),),
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    assert definition.inputs[0].name == "x"
    assert definition.outputs[0].source == "${step.a.output}"
