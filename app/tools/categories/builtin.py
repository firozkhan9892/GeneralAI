"""Built-in utility tools: calculator, echo, clock, and text utilities."""

from __future__ import annotations

import ast
import operator
import time
from collections.abc import Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolValidationError
from app.tools.models import ToolCategory, ToolParameter

_BIN_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
}


def _eval_expression(expression: str) -> Any:
    """Safely evaluate a restricted arithmetic expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolValidationError(
            f"Invalid expression: {expression}",
            module="tools.categories.builtin",
        ) from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolValidationError(
            "Unsupported constant literal",
            module="tools.categories.builtin",
        )
    if isinstance(node, ast.BinOp):
        op = _BIN_OPERATORS.get(type(node.op))
        if op is None:
            raise ToolValidationError(
                f"Unsupported operator: {type(node.op).__name__}",
                module="tools.categories.builtin",
            )
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ToolValidationError(
                f"Unsupported unary operator: {type(node.op).__name__}",
                module="tools.categories.builtin",
            )
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolValidationError(
                "Unsupported call target",
                module="tools.categories.builtin",
            )
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ToolValidationError(
                f"Unsupported function: {node.func.id}",
                module="tools.categories.builtin",
            )
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.List):
        return [_eval_node(element) for element in node.elts]
    raise ToolValidationError(
        f"Unsupported syntax: {type(node).__name__}",
        module="tools.categories.builtin",
    )


class CalculatorTool(Tool):
    """Evaluate a restricted arithmetic expression."""

    name = "calculator"
    description = "Evaluate a safe arithmetic expression"
    category = ToolCategory.BUILTIN
    parameters = (
        ToolParameter(
            name="expression",
            description="Arithmetic expression to evaluate",
            param_type="string",
            required=True,
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        expression = arguments["expression"].strip()
        if not expression:
            raise ToolValidationError(
                "expression parameter is required",
                module="tools.categories.builtin",
            )
        return _eval_expression(expression)


class EchoTool(Tool):
    """Echo back the supplied text."""

    name = "echo"
    description = "Return the supplied text unchanged"
    category = ToolCategory.BUILTIN
    parameters = (
        ToolParameter(
            name="text",
            description="Text to echo",
            param_type="string",
            required=True,
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        return arguments["text"]


class ClockTool(Tool):
    """Return the current time in a requested format."""

    name = "clock"
    description = "Return the current UTC time"
    category = ToolCategory.BUILTIN
    parameters = (
        ToolParameter(
            name="format",
            description="Output format: iso, unix, or human",
            param_type="string",
            default="iso",
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        fmt = arguments.get("format", "iso")
        now = time.time()
        if fmt == "unix":
            return now
        import datetime

        utc = datetime.datetime.now(datetime.timezone.utc)
        if fmt == "human":
            return utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        return utc.isoformat()


class TextUtilsTool(Tool):
    """Common text transformations."""

    name = "text_utils"
    description = "Apply a text transformation (uppercase, lowercase, strip, title)"
    category = ToolCategory.BUILTIN
    parameters = (
        ToolParameter(
            name="text",
            description="Text to transform",
            param_type="string",
            required=True,
        ),
        ToolParameter(
            name="operation",
            description="Operation: uppercase, lowercase, strip, or title",
            param_type="string",
            default="strip",
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        text = arguments["text"]
        operation = arguments.get("operation", "strip")
        if operation == "uppercase":
            return text.upper()
        if operation == "lowercase":
            return text.lower()
        if operation == "strip":
            return text.strip()
        if operation == "title":
            return text.title()
        raise ToolValidationError(
            f"Unknown operation: {operation}",
            module="tools.categories.builtin",
        )
