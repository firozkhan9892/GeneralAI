"""Tests for workflow definition validation."""

from __future__ import annotations

import pytest

from app.automation.models import (
    Branch,
    WorkflowDefinition,
    WorkflowInput,
    WorkflowOutput,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.validation import Violation, WorkflowValidator


@pytest.fixture
def validator() -> WorkflowValidator:
    return WorkflowValidator()


def test_valid_linear_workflow(validator, linear_definition) -> None:
    report = validator.validate(linear_definition)
    assert report.valid
    assert report.errors == []


def test_empty_id(validator) -> None:
    definition = WorkflowDefinition(
        id="",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "empty_id" for v in report.errors)


def test_invalid_semver(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        version="not-semver",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "invalid_version" for v in report.errors)


def test_no_steps(validator) -> None:
    definition = WorkflowDefinition(id="wf", steps=())
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "no_steps" for v in report.errors)


def test_duplicate_step_id(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),
            WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),
        ),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "duplicate_step_id" for v in report.errors)


def test_missing_required_field(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK),),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "missing_required_field" for v in report.errors)


def test_unknown_dependency(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.TASK,
                tool_name="echo",
                depends_on=("ghost",),
            ),
        ),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "invalid_dependency" for v in report.errors)


def test_cycle_detected(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("b",)
            ),
            WorkflowStep(
                id="b", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("a",)
            ),
        ),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "cycle" for v in report.errors)


def test_conditional_requires_branches(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.CONDITIONAL),),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "no_branches" for v in report.errors)


def test_conditional_branch_steps_validated(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.CONDITIONAL,
                branches=(
                    Branch(
                        name="x",
                        when="true",
                        steps=(WorkflowStep(id="a1", type=WorkflowStepType.TASK),),
                    ),
                ),
            ),
        ),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "missing_required_field" for v in report.errors)


def test_duplicate_branch_name(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.CONDITIONAL,
                branches=(
                    Branch(name="x", when="true", steps=()),
                    Branch(name="x", when="false", steps=()),
                ),
            ),
        ),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "duplicate_branch_name" for v in report.errors)


def test_loop_steps_validated(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(
            WorkflowStep(
                id="a",
                type=WorkflowStepType.LOOP,
                iterable="${inputs.items}",
                loop_steps=(WorkflowStep(id="a1", type=WorkflowStepType.TASK),),
            ),
        ),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "missing_required_field" for v in report.errors)


def test_duplicate_input(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        inputs=(
            WorkflowInput(name="x", required=True),
            WorkflowInput(name="x", required=False),
        ),
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "duplicate_input" for v in report.errors)


def test_invalid_output_source(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        outputs=(WorkflowOutput(name="y", source="${step.ghost.output}"),),
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    report = validator.validate(definition)
    assert not report.valid
    assert any(v.code == "invalid_output_source" for v in report.errors)


def test_missing_step_name_is_warning(validator) -> None:
    definition = WorkflowDefinition(
        id="wf",
        steps=(WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),),
    )
    report = validator.validate(definition)
    assert report.valid
    assert any(v.code == "missing_step_name" for v in report.warnings)


def test_violation_equality_and_serialisation() -> None:
    a = Violation("error", "code", "message", "s1")
    b = Violation("error", "code", "message", "s1")
    c = Violation("warning", "code", "message", "s1")
    assert a == b
    assert a != c
    assert a.to_dict() == {
        "severity": "error",
        "code": "code",
        "message": "message",
        "step_id": "s1",
    }


def test_report_helpers(validator, linear_definition) -> None:
    report = validator.validate(linear_definition)
    assert bool(report) is True
    assert report.warnings != [] or report.errors == []
    dumped = report.to_dict()
    assert dumped["valid"] is True
