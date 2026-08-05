"""Template expression resolution.

Workflow step ``input_bindings`` and conditions use a small template
language: expressions like ``${inputs.user}`` or ``${stepA.output.count}``
are resolved against a run context.  Resolution is a pure function of
the context mapping — there is **no** arbitrary code evaluation.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from app.automation.exceptions import WorkflowExecutionError

_EXPRESSION_RE = re.compile(r"\$\{([^}]+)\}")

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


class ExpressionContext:
    """Provides values for expression resolution.

    Implementations expose inputs and step outputs; see
    :class:`app.automation.context.WorkflowRunContext`.
    """

    def resolve_step(self, step_id: str, path: str) -> Any:
        """Resolve ``<step_id>.<path>`` against step outputs.

        Args:
            step_id: The step identifier.
            path: Dot-separated field path (may be empty).
        """
        raise NotImplementedError

    def resolve_input(self, name: str) -> Any:
        """Resolve an input value by name."""
        raise NotImplementedError


def _coerce_boolean(value: Any) -> bool:
    """Coerce *value* to a boolean for condition evaluation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        return len(lowered) > 0
    if value is None:
        return False
    return bool(value)


def _get_path(value: Any, path: str) -> Any:
    """Traverse a dot-separated path over nested mappings/lists.

    Indexing into lists is supported with ``[n]`` segments.
    """
    if not path:
        return value
    current = value
    for segment in _split_segments(path):
        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(segment)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _split_segments(path: str) -> list[str]:
    """Split a path into dot-separated segments supporting ``a.b[0].c``."""
    segments: list[str] = []
    current = ""
    for char in path:
        if char == ".":
            if current:
                segments.append(current)
                current = ""
        elif char == "[":
            if current:
                segments.append(current)
                current = ""
        elif char == "]":
            if current:
                segments.append(current)
                current = ""
        else:
            current += char
    if current:
        segments.append(current)
    return segments


def resolve_expression(expression: str, context: ExpressionContext) -> Any:
    """Resolve a single expression such as ``inputs.user``.

    The leading namespace distinguishes inputs from step outputs::

        inputs.name            -> workflow input
        stepA.output.field     -> output of step ``stepA``

    Args:
        expression: The expression body (without ``${}``).
        context: Value provider.

    Returns:
        The resolved value, or ``None`` when a path does not resolve.
    """
    stripped = expression.strip()
    if not stripped:
        return None
    parts = stripped.split(".")
    namespace = parts[0]

    if namespace == "inputs":
        input_name = parts[1] if len(parts) > 1 else ""
        value = context.resolve_input(input_name)
        if len(parts) > 2:
            value = _get_path(value, ".".join(parts[2:]))
        return value
    if namespace == "step":
        if len(parts) < 2:
            return None
        step_id = parts[1]
        path = ".".join(parts[2:])
        return context.resolve_step(step_id, path)
    return None


def resolve_template(expression: str, context: ExpressionContext) -> Any:
    """Resolve an expression that may be wrapped in ``${...}``.

    Args:
        expression: An expression body or a ``${...}`` template.
        context: Value provider.

    Returns:
        The resolved value, or ``None`` when the path does not resolve.
    """
    stripped = expression.strip()
    if len(stripped) >= 2 and stripped.startswith("${") and stripped.endswith("}"):
        return resolve_expression(stripped[2:-1], context)
    return resolve_expression(stripped, context)


def resolve_bindings(
    bindings: Mapping[str, str], context: ExpressionContext
) -> dict[str, Any]:
    """Resolve a mapping of step input names to expressions.

    Literal values (strings not containing ``${...}``) are passed
    through unchanged.

    Args:
        bindings: Mapping of input name to template expression.
        context: Value provider.

    Returns:
        A mapping of input name to resolved value.
    """
    resolved: dict[str, Any] = {}
    for name, raw in bindings.items():
        if not isinstance(raw, str):
            resolved[name] = raw
            continue
        matches = _EXPRESSION_RE.findall(raw)
        if not matches:
            resolved[name] = raw
            continue
        value: Any = raw
        for match in matches:
            replacement = resolve_expression(match, context)
            value = value.replace("${" + match + "}", _stringify(replacement))
        if len(matches) == 1 and raw.strip() == "${" + matches[0] + "}":
            value = resolve_expression(matches[0], context)
        resolved[name] = value
    return resolved


def evaluate_condition(expression: str, context: ExpressionContext) -> bool:
    """Evaluate a condition expression to a boolean.

    Supports simple comparisons and boolean values::

        true
        ${inputs.enabled}
        ${stepA.output.count} > 3

    Args:
        expression: The condition to evaluate.
        context: Value provider.

    Raises:
        WorkflowExecutionError: If the condition cannot be parsed.
    """
    stripped = expression.strip()
    if not stripped:
        return True

    match = _EXPRESSION_RE.search(stripped)
    if match is not None:
        resolved = resolve_expression(match.group(1), context)
        left = stripped[match.end() :].strip()
        operator = _find_operator(left)
        if operator is None:
            return _coerce_boolean(resolved)
        op, right_raw = operator
        right_text = right_raw.strip()
        right: Any
        right_match = _EXPRESSION_RE.fullmatch(right_text)
        if right_match is not None:
            right = resolve_expression(right_match.group(1), context)
        else:
            right = _parse_literal(right_text)
        return _compare(resolved, op, right)

    lowered = stripped.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise WorkflowExecutionError(f"Cannot evaluate condition '{expression}'")


_OPERATORS = ("==", "!=", ">=", "<=", ">", "<")


def _find_operator(text: str) -> tuple[str, str] | None:
    """Return the first comparison operator and the right-hand text."""
    for op in _OPERATORS:
        index = text.find(op)
        if index != -1:
            return op, text[index + len(op) :]
    return None


def _parse_literal(text: str) -> Any:
    """Parse a literal number or quoted string."""
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
        return stripped[1:-1]
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def _compare(left: Any, operator: str, right: Any) -> bool:
    """Compare two values using *operator*."""
    try:
        if operator == "==":
            return left == right
        if operator == "!=":
            return left != right
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == ">":
            return left > right
        if operator == "<":
            return left < right
    except TypeError:
        return False
    raise WorkflowExecutionError(f"Unsupported comparison operator '{operator}'")


def _stringify(value: Any) -> str:
    """Convert a resolved value to a string for comparison/templates."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
