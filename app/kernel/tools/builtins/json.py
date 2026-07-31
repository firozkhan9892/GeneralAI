"""JSON tool — performs JSON operations."""

from __future__ import annotations

import json
from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Perform JSON operations.

    Args:
        parameters: Must contain 'operation' ('parse', 'stringify', 'validate').
                    For 'parse'/'validate': 'data' with JSON string.
                    For 'stringify': 'data' with Python object.

    Returns:
        Parsed object, stringified JSON, or validation result.
    """
    operation = parameters.get("operation", "parse")
    data = parameters.get("data", "")

    if operation == "parse":
        if isinstance(data, str):
            return json.loads(data)
        return data
    if operation == "stringify":
        return json.dumps(data, indent=2, default=str)
    if operation == "validate":
        if isinstance(data, str):
            try:
                json.loads(data)
                return True
            except (json.JSONDecodeError, TypeError):
                return False
        return False
    raise ValueError(f"Unknown operation: {operation}")
