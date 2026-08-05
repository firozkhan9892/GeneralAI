"""Tests for the workflow executor."""

from __future__ import annotations

import asyncio

import pytest

from app.automation.executor import WorkflowExecutor
from app.automation.models import (
    Branch,
    ErrorPolicy,
    ParallelJoinMode,
    StepRetryPolicy,
    WorkflowDefinition,
    WorkflowOutput,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowSettings,
    WorkflowSnapshot,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.registries import StepTypeRegistry


def _make_run(
    definition: WorkflowDefinition,
    inputs: dict | None = None,
    run_id: str = "run1",
    idempotency_key: str | None = None,
) -> WorkflowRun:
    snapshot = WorkflowSnapshot(
        workflow_id=definition.id,
        version=definition.version,
        step_definitions=definition.steps,
        settings=definition.settings,
        inputs=definition.inputs,
        outputs=definition.outputs,
    )
    return WorkflowRun(
        run_id=run_id,
        workflow_id=definition.id,
        workflow_version=definition.version,
        snapshot=snapshot,
        inputs=inputs or {},
        idempotency_key=idempotency_key,
    )


@pytest.fixture
def executor() -> WorkflowExecutor:
    registry = StepTypeRegistry()
    ex = WorkflowExecutor(registry)
    ex.register_builtins()
    return ex


def _set_task_executor(executor: WorkflowExecutor, fn) -> None:
    """Register a TASK executor, replacing any existing one."""
    executor._step_types.unregister(WorkflowStepType.TASK)
    executor._step_types.register(WorkflowStepType.TASK, fn)


def _echo_step_executor() -> object:
    """A TASK executor that echoes its resolved inputs."""

    async def execute(step, step_ctx, engine, state, run, snapshot, shared):
        return {"echo": step_ctx.inputs}

    return execute


def _fail_step_executor() -> object:
    async def execute(step, step_ctx, engine, state, run, snapshot, shared):
        raise RuntimeError("boom")

    return execute


def test_linear_execution_echoes(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, name="A"),
            WorkflowStep(
                id="b", type=WorkflowStepType.TASK, name="B", depends_on=("a",)
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    a = result.step("a")
    b = result.step("b")
    assert a is not None and a.status.value == "succeeded"
    assert b is not None and b.status.value == "succeeded"
    assert a.output == {"echo": {}}


def test_skipped_when_condition_false(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                name="A",
                condition="${inputs.enabled}",
            ),
        ),
    )
    run = _make_run(definition, {"enabled": False})
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    a = result.step("a")
    assert a is not None and a.status.value == "skipped"


def test_step_failure_aborts_run(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                name="A",
                error_policy=ErrorPolicy.ABORT,
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _fail_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.FAILED
    assert result.error is not None


def test_step_failure_skip_continues(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                name="A",
                error_policy=ErrorPolicy.SKIP,
            ),
            WorkflowStep(
                id="b", type=WorkflowStepType.TASK, name="B", depends_on=("a",)
            ),
        ),
    )
    run = _make_run(definition)

    async def fail_a_echo_b(step, step_ctx, engine, state, run, snapshot, shared):
        if step.id == "a":
            raise RuntimeError("boom")
        return {"ok": True}

    _set_task_executor(executor, fail_a_echo_b)
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert result.step("a").status.value == "skipped"
    assert result.step("b").status.value == "succeeded"


def test_step_failure_ignore_marks_succeeded(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                name="A",
                error_policy=ErrorPolicy.IGNORE,
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _fail_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    a = result.step("a")
    assert a is not None and a.status.value == "succeeded"
    assert a.output is None


def test_retry_then_success(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                name="A",
                retry_policy=StepRetryPolicy(max_retries=2, base_delay_s=0),
            ),
        ),
    )
    run = _make_run(definition)
    attempts = {"count": 0}

    async def flaky(step, step_ctx, engine, state, run, snapshot, shared):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return {"ok": True}

    _set_task_executor(executor, flaky)
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert attempts["count"] == 3
    assert result.step("a").retries_consumed == 2


def test_step_timeout(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, name="A", timeout_s=0.1),
        ),
    )
    run = _make_run(definition)

    async def slow(step, step_ctx, engine, state, run, snapshot, shared):
        import asyncio

        await asyncio.sleep(10)
        return {"ok": True}

    _set_task_executor(executor, slow)
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.FAILED
    assert "timed out" in (result.error or "")


def test_conditional_selects_matching_branch(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="cond",
                type=WorkflowStepType.CONDITIONAL,
                name="Cond",
                branches=(
                    Branch(
                        name="a",
                        when="${inputs.which} == a",
                        steps=(
                            WorkflowStep(
                                id="sa", type=WorkflowStepType.TASK, name="SA"
                            ),
                        ),
                    ),
                    Branch(
                        name="b",
                        when="${inputs.which} == b",
                        steps=(
                            WorkflowStep(
                                id="sb", type=WorkflowStepType.TASK, name="SB"
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    run = _make_run(definition, {"which": "b"})
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    cond = result.step("cond")
    assert cond is not None and cond.status.value == "succeeded"
    assert cond.output == {"branch": "b", "outputs": {"sb": {"echo": {}}}}


def test_conditional_no_match_returns_none(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="cond",
                type=WorkflowStepType.CONDITIONAL,
                name="Cond",
                branches=(
                    Branch(
                        name="a",
                        when="${inputs.which} == a",
                        steps=(
                            WorkflowStep(
                                id="sa", type=WorkflowStepType.TASK, name="SA"
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    run = _make_run(definition, {"which": "z"})
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert result.step("cond").output == {"branch": None, "outputs": {}}


def test_loop_iterates(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="loop",
                type=WorkflowStepType.LOOP,
                name="Loop",
                iterable="${inputs.items}",
                loop_var="item",
                loop_steps=(
                    WorkflowStep(id="inner", type=WorkflowStepType.TASK, name="Inner"),
                ),
            ),
        ),
    )
    run = _make_run(definition, {"items": [1, 2, 3]})
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    loop = result.step("loop")
    assert loop is not None and loop.output["count"] == 3


def test_loop_respects_max_iterations(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="loop",
                type=WorkflowStepType.LOOP,
                name="Loop",
                iterable="${inputs.items}",
                loop_var="item",
                max_iterations=2,
                loop_steps=(
                    WorkflowStep(id="inner", type=WorkflowStepType.TASK, name="Inner"),
                ),
            ),
        ),
    )
    run = _make_run(definition, {"items": [1, 2, 3, 4, 5]})
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    loop = result.step("loop")
    assert loop is not None and loop.output["count"] == 2


def test_parallel_join_all(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="par",
                type=WorkflowStepType.PARALLEL,
                name="Par",
                join_mode=ParallelJoinMode.ALL,
                branches=(
                    Branch(
                        name="b1",
                        when="true",
                        steps=(
                            WorkflowStep(
                                id="p1", type=WorkflowStepType.TASK, name="P1"
                            ),
                        ),
                    ),
                    Branch(
                        name="b2",
                        when="true",
                        steps=(
                            WorkflowStep(
                                id="p2", type=WorkflowStepType.TASK, name="P2"
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert set(result.step("par").output.keys()) == {"b1", "b2"}


def test_parallel_join_any(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="par",
                type=WorkflowStepType.PARALLEL,
                name="Par",
                join_mode=ParallelJoinMode.ANY,
                branches=(
                    Branch(
                        name="b1",
                        when="true",
                        steps=(
                            WorkflowStep(
                                id="p1", type=WorkflowStepType.TASK, name="P1"
                            ),
                        ),
                    ),
                    Branch(
                        name="b2",
                        when="true",
                        steps=(
                            WorkflowStep(
                                id="p2", type=WorkflowStepType.TASK, name="P2"
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    par = result.step("par")
    assert par is not None and set(par.output.keys()) == {"b1"}


def test_approval_pauses_run(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, name="A"),
            WorkflowStep(
                id="gate",
                type=WorkflowStepType.APPROVAL,
                name="Gate",
                depends_on=("a",),
                approvers=("admin",),
            ),
            WorkflowStep(
                id="b", type=WorkflowStepType.TASK, name="B", depends_on=("gate",)
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _echo_step_executor())
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.WAITING_APPROVAL
    assert result.approval_requests
    request = result.approval_requests[0]
    assert request.step_id == "gate"
    assert request.status.value == "pending"


def test_approval_resume_after_approve(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, name="A"),
            WorkflowStep(
                id="gate",
                type=WorkflowStepType.APPROVAL,
                name="Gate",
                depends_on=("a",),
                approvers=("admin",),
            ),
            WorkflowStep(
                id="b", type=WorkflowStepType.TASK, name="B", depends_on=("gate",)
            ),
        ),
    )
    run = _make_run(definition)
    _set_task_executor(executor, _echo_step_executor())
    paused = asyncio.run(executor.run(run))
    request = paused.approval_requests[0]

    decided = request.model_copy(update={"status": "approved", "decided_by": "admin"})
    resumed_run = paused.model_copy(update={"approval_requests": (decided,)})
    result = asyncio.run(executor.run(resumed_run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert result.step("b") is not None
    assert result.step("b").status.value == "succeeded"


def test_resume_does_not_rerun_completed(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, name="A"),
            WorkflowStep(
                id="gate",
                type=WorkflowStepType.APPROVAL,
                name="Gate",
                depends_on=("a",),
                approvers=("admin",),
            ),
        ),
    )
    run = _make_run(definition)
    counter = {"calls": 0}

    async def counting(step, step_ctx, engine, state, run, snapshot, shared):
        counter["calls"] += 1
        return {"called": counter["calls"]}

    _set_task_executor(executor, counting)
    paused = asyncio.run(executor.run(run))
    assert counter["calls"] == 1
    decided = paused.approval_requests[0].model_copy(
        update={"status": "approved", "decided_by": "admin"}
    )
    resumed_run = paused.model_copy(update={"approval_requests": (decided,)})
    result = asyncio.run(executor.run(resumed_run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert counter["calls"] == 1


def test_overall_timeout(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        settings=WorkflowSettings(overall_timeout_s=0.1),
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, name="A"),),
    )
    run = _make_run(definition)

    async def slow(step, step_ctx, engine, state, run, snapshot, shared):
        import asyncio

        await asyncio.sleep(10)
        return {"ok": True}

    _set_task_executor(executor, slow)
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.TIMED_OUT


def test_workflow_outputs_resolved(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        outputs=(WorkflowOutput(name="result", source="${step.a}"),),
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TRANSFORM,
                name="T",
                expression="${inputs.value}",
            ),
        ),
    )
    run = _make_run(definition, {"value": 42})
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert result.outputs == {"result": 42}


def test_transform_executor(executor) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a", type=WorkflowStepType.TRANSFORM, expression="${inputs.x}"
            ),
        ),
    )
    run = _make_run(definition, {"x": "hello"})
    result = asyncio.run(executor.run(run))
    assert result.step("a").output == "hello"


def test_step_output_reference_resolves_through_expression(executor) -> None:
    """A transform can reference another step's output via ``output.X``."""
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TRANSFORM,
                expression="${inputs.value}",
            ),
            WorkflowStep(
                id="b",
                type=WorkflowStepType.TRANSFORM,
                expression="${step.a.output.message}",
                depends_on=("a",),
            ),
        ),
    )
    run = _make_run(definition, {"value": {"message": "hello"}})
    result = asyncio.run(executor.run(run))
    assert result.status == WorkflowRunStatus.SUCCEEDED
    assert result.step("a").output == {"message": "hello"}
    assert result.step("b").output == "hello"
