"""Tests for template expression resolution."""

from __future__ import annotations

import pytest

from app.automation.context import OutputStore, WorkflowRunContext
from app.automation.exceptions import WorkflowExecutionError
from app.automation.template import (
    ExpressionContext,
    evaluate_condition,
    resolve_bindings,
    resolve_expression,
)


class _Context(ExpressionContext):
    """Minimal test context."""

    def __init__(self, inputs: dict | None = None, steps: dict | None = None) -> None:
        self._inputs = inputs or {}
        self._steps = steps or {}

    def resolve_input(self, name: str):
        return self._inputs.get(name)

    def resolve_step(self, step_id: str, path: str):
        value = self._steps.get(step_id)
        for part in filter(None, path.split(".")):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value


def test_resolve_input_reference() -> None:
    ctx = _Context(inputs={"user": "alice"})
    assert resolve_expression("inputs.user", ctx) == "alice"


def test_resolve_step_output_reference() -> None:
    ctx = _Context(steps={"a": {"output": {"count": 5}}})
    assert resolve_expression("step.a.output.count", ctx) == 5


def test_resolve_unknown_path_returns_none() -> None:
    ctx = _Context(inputs={"user": "alice"})
    assert resolve_expression("inputs.missing", ctx) is None


def test_resolve_bindings_passthrough_literals() -> None:
    ctx = _Context(inputs={"user": "alice"})
    result = resolve_bindings({"message": "hello"}, ctx)
    assert result == {"message": "hello"}


def test_resolve_bindings_interpolates_templates() -> None:
    ctx = _Context(inputs={"user": "alice"})
    result = resolve_bindings({"greeting": "Hello ${inputs.user}!"}, ctx)
    assert result == {"greeting": "Hello alice!"}


def test_resolve_bindings_pure_reference_preserves_type() -> None:
    ctx = _Context(inputs={"count": 7})
    result = resolve_bindings({"n": "${inputs.count}"}, ctx)
    assert result == {"n": 7}
    assert isinstance(result["n"], int)


def test_resolve_bindings_multiple_references() -> None:
    ctx = _Context(inputs={"a": "x", "b": "y"})
    result = resolve_bindings({"combined": "${inputs.a}-${inputs.b}"}, ctx)
    assert result == {"combined": "x-y"}


def test_evaluate_condition_boolean_literal() -> None:
    ctx = _Context()
    assert evaluate_condition("true", ctx) is True
    assert evaluate_condition("false", ctx) is False


def test_evaluate_condition_reference() -> None:
    ctx = _Context(inputs={"enabled": True})
    assert evaluate_condition("${inputs.enabled}", ctx) is True


def test_evaluate_condition_comparison() -> None:
    ctx = _Context(steps={"a": {"output": {"count": 5}}})
    assert evaluate_condition("${step.a.output.count} > 3", ctx) is True
    assert evaluate_condition("${step.a.output.count} > 10", ctx) is False
    assert evaluate_condition("${step.a.output.count} == 5", ctx) is True


def test_evaluate_condition_empty_is_true() -> None:
    assert evaluate_condition("", _Context()) is True


def test_evaluate_condition_invalid_raises() -> None:
    with pytest.raises(WorkflowExecutionError):
        evaluate_condition("nonsense expression", _Context())


def test_evaluate_condition_string_value() -> None:
    ctx = _Context(inputs={"status": "ok"})
    assert evaluate_condition("${inputs.status} == ok", ctx) is True


def _run_context(outputs: dict) -> WorkflowRunContext:
    """A real run context seeded with the given step outputs."""
    store = OutputStore()
    for step_id, value in outputs.items():
        store.put(step_id, value)
    return WorkflowRunContext({}, store)


def test_resolve_step_output_legacy_raw_field() -> None:
    """Legacy: step output stored raw; ``step.a.output.field`` resolves it."""
    ctx = _run_context({"a": {"approved": True, "count": 5}})
    assert resolve_expression("step.a.output.approved", ctx) is True
    assert resolve_expression("step.a.output.count", ctx) == 5


def test_resolve_step_output_documented_wrapper() -> None:
    """Documented convention: step output wrapped in an ``output`` key."""
    ctx = _run_context({"a": {"output": {"count": 5}}})
    assert resolve_expression("step.a.output.count", ctx) == 5


def test_resolve_step_whole_output() -> None:
    ctx = _run_context({"a": {"approved": True}})
    assert resolve_expression("step.a.output", ctx) == {"approved": True}


def test_resolve_step_legacy_direct_path() -> None:
    """Legacy raw lookup without the ``output`` marker."""
    ctx = _run_context({"a": {"count": 5}})
    assert resolve_expression("step.a.count", ctx) == 5


def test_resolve_step_scalar_output() -> None:
    ctx = _run_context({"a": 42})
    assert resolve_expression("step.a.output", ctx) == 42
