"""Tests for the automation exception hierarchy."""

from __future__ import annotations

import pytest

from app.automation.exceptions import (
    ApprovalTimeoutError,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowNotFoundError,
    WorkflowStepError,
    WorkflowValidationError,
)
from app.core.exceptions.base import GeneralAIError


def test_workflow_error_derives_from_general_ai_error() -> None:
    assert issubclass(WorkflowError, GeneralAIError)


def test_workflow_error_sets_module() -> None:
    error = WorkflowError("boom")
    assert error.module == "automation"
    assert str(error) == "[automation] boom"


def test_workflow_error_accepts_context_and_cause() -> None:
    cause = ValueError("inner")
    error = WorkflowError("boom", cause=cause, context={"workflow_id": "x"})
    assert error.cause is cause
    assert error.context == {"workflow_id": "x"}


def test_not_found_error_includes_version() -> None:
    error = WorkflowNotFoundError("wf", version="1.2.0")
    assert error.message == "Workflow 'wf' (version '1.2.0') not found"
    assert error.context["workflow_id"] == "wf"
    assert error.context["version"] == "1.2.0"


def test_not_found_error_without_version() -> None:
    error = WorkflowNotFoundError("wf")
    assert error.message == "Workflow 'wf' not found"


def test_step_error_carries_step_id() -> None:
    error = WorkflowStepError("step blew up", step_id="s1")
    assert error.context["step_id"] == "s1"


def test_validation_error_carries_violations() -> None:
    error = WorkflowValidationError("bad", violations=["cycle"])
    assert error.context["violations"] == ["cycle"]


def test_execution_error_derives_from_workflow_error() -> None:
    assert issubclass(WorkflowExecutionError, WorkflowError)


def test_approval_timeout_derives_from_workflow_error() -> None:
    assert issubclass(ApprovalTimeoutError, WorkflowError)


def test_all_exceptions_are_catchable_as_general_ai_error() -> None:
    errors = [
        WorkflowNotFoundError("wf"),
        WorkflowStepError("boom", step_id="s"),
        WorkflowValidationError("boom"),
    ]
    for error in errors:
        assert isinstance(error, GeneralAIError)
        with pytest.raises(GeneralAIError):
            raise error
