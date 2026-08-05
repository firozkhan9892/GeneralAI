"""Shared fixtures for the automation module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.automation.models import (
    WorkflowDefinition,
    WorkflowInput,
    WorkflowStep,
    WorkflowStepType,
)
from app.automation.time import FakeClock, SystemClock


@pytest.fixture
def fake_clock() -> FakeClock:
    """A deterministic fake clock starting at 2026-01-01T00:00:00Z."""
    return FakeClock()


@pytest.fixture
def system_clock() -> SystemClock:
    """The real system clock."""
    return SystemClock()


@pytest.fixture
def linear_definition() -> WorkflowDefinition:
    """A simple three-step linear workflow."""
    return WorkflowDefinition(
        id="linear",
        version="1.0.0",
        name="Linear workflow",
        inputs=(WorkflowInput(name="message", type="str", required=True),),
        steps=(
            WorkflowStep(
                id="first",
                type=WorkflowStepType.TASK,
                name="First",
                tool_name="echo",
                input_bindings={"message": "${inputs.message}"},
            ),
            WorkflowStep(
                id="second",
                type=WorkflowStepType.TASK,
                name="Second",
                tool_name="echo",
                depends_on=("first",),
                input_bindings={"prefix": "${step.first.output.message}"},
            ),
            WorkflowStep(
                id="third",
                type=WorkflowStepType.TASK,
                name="Third",
                tool_name="echo",
                depends_on=("second",),
            ),
        ),
    )


@pytest.fixture
def diamond_definition() -> WorkflowDefinition:
    """A diamond-shaped DAG: a -> (b, c) -> d."""
    return WorkflowDefinition(
        id="diamond",
        version="1.0.0",
        name="Diamond workflow",
        steps=(
            WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),
            WorkflowStep(
                id="b", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("a",)
            ),
            WorkflowStep(
                id="c", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("a",)
            ),
            WorkflowStep(
                id="d",
                type=WorkflowStepType.TASK,
                tool_name="echo",
                depends_on=("b", "c"),
            ),
        ),
    )


@pytest.fixture
def utc_now() -> datetime:
    """A fixed reference timestamp."""
    return datetime(2026, 1, 1, tzinfo=timezone.utc)
