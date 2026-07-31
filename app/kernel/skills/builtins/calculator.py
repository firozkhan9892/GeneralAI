"""Calculator skill — evaluates mathematical expressions."""

from __future__ import annotations

import ast
import operator as op
from typing import Any

_SAFE_BIN_OPS: dict[type, Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_SAFE_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        handler = _SAFE_BIN_OPS.get(type(node.op))
        if handler is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return handler(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        handler = _SAFE_UNARY_OPS.get(type(node.op))
        if handler is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return handler(operand)
    if isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are supported")
        func = _SAFE_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unsupported function: {node.func.id}")
        args = [_safe_eval(a) for a in node.args]
        return float(func(*args))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


async def execute(parameters: dict[str, Any]) -> Any:
    """Evaluate a mathematical expression.

    Args:
        parameters: Must contain 'expression' key.

    Returns:
        The numeric result.
    """
    expression = parameters.get("expression", "")
    if not expression:
        raise ValueError("expression parameter is required")
    tree = ast.parse(str(expression), mode="eval")
    result = _safe_eval(tree.body)
    return result
