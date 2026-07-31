"""Python tools: evaluate restricted Python expressions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolExecutionError, ToolValidationError
from app.tools.models import ToolCategory, ToolParameter

# Namespace exposed to evaluated expressions.
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "abs": abs,
        "len": len,
        "min": min,
        "max": max,
        "round": round,
        "sum": sum,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "sorted": sorted,
    }
}


class PythonEvalTool(Tool):
    """Evaluate a restricted Python expression."""

    name = "python_eval"
    description = "Evaluate a sandboxed Python expression"
    category = ToolCategory.PYTHON
    requires_confirmation = True
    sandboxable = True
    parameters = (
        ToolParameter(
            name="expression",
            description="Python expression to evaluate",
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
                module="tools.categories.python",
            )
        try:
            return eval(expression, _SAFE_GLOBALS, {})  # noqa: S307
        except Exception as exc:
            raise ToolExecutionError(
                f"Evaluation failed: {exc}",
                module="tools.categories.python",
                cause=exc,
            ) from exc
