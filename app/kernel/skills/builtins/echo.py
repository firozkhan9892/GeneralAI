"""Echo skill — echoes the input message."""

from __future__ import annotations

from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Echo the input message.

    Args:
        parameters: Must contain 'message' key.

    Returns:
        The echoed message.
    """
    message = parameters.get("message", "")
    return message
