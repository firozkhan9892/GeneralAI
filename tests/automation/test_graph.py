"""Tests for the workflow dependency graph."""

from __future__ import annotations

import pytest

from app.automation.exceptions import WorkflowValidationError
from app.automation.graph import WorkflowGraph
from app.automation.models import (
    WorkflowStep,
    WorkflowStepType,
)


def _graph_with(*steps: WorkflowStep) -> WorkflowGraph:
    return WorkflowGraph(steps)


def test_topological_order_linear(linear_definition) -> None:
    graph = WorkflowGraph.from_definition(linear_definition)
    assert graph.topological_order() == ["first", "second", "third"]


def test_topological_order_diamond(diamond_definition) -> None:
    graph = WorkflowGraph.from_definition(diamond_definition)
    order = graph.topological_order()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_ready_steps_linear(linear_definition) -> None:
    graph = WorkflowGraph.from_definition(linear_definition)
    assert graph.ready_steps(set()) == ["first"]
    assert graph.ready_steps({"first"}) == ["second"]
    assert graph.ready_steps({"first", "second"}) == ["third"]
    assert graph.ready_steps({"first", "second", "third"}) == []


def test_ready_steps_diamond(diamond_definition) -> None:
    graph = WorkflowGraph.from_definition(diamond_definition)
    assert graph.ready_steps({"a"}) == ["b", "c"]
    assert graph.ready_steps({"a", "b", "c"}) == ["d"]


def test_dependencies_and_dependents(linear_definition) -> None:
    graph = WorkflowGraph.from_definition(linear_definition)
    assert graph.dependencies("second") == ["first"]
    assert graph.dependents("first") == ["second"]
    assert graph.dependencies("first") == []


def test_cycle_detection() -> None:
    graph = _graph_with(
        WorkflowStep(
            id="a", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("b",)
        ),
        WorkflowStep(
            id="b", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("a",)
        ),
    )
    assert graph.has_cycle()
    with pytest.raises(WorkflowValidationError):
        graph.topological_order()


def test_self_dependency_is_a_cycle() -> None:
    graph = _graph_with(
        WorkflowStep(
            id="a", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("a",)
        ),
    )
    assert graph.has_cycle()


def test_validate_references_catches_unknown_and_self() -> None:
    graph = _graph_with(
        WorkflowStep(
            id="a", type=WorkflowStepType.TASK, tool_name="echo", depends_on=("ghost",)
        ),
    )
    errors = graph.validate_references()
    assert len(errors) == 1
    assert "ghost" in errors[0]


def test_get_and_has() -> None:
    graph = _graph_with(
        WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),
    )
    assert graph.has("a")
    assert not graph.has("b")
    assert graph.get("a") is not None
    assert graph.get("b") is None


def test_step_count_and_node_ids(diamond_definition) -> None:
    graph = WorkflowGraph.from_definition(diamond_definition)
    assert graph.step_count == 4
    assert graph.node_ids == ["a", "b", "c", "d"]


def test_ready_steps_never_returns_completed() -> None:
    graph = _graph_with(
        WorkflowStep(id="a", type=WorkflowStepType.TASK, tool_name="echo"),
    )
    assert graph.ready_steps({"a"}) == []
