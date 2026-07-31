"""Clock tool — returns current time and date information."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Return current time information.

    Args:
        parameters: Optional 'format' key ('iso', 'unix', 'human').

    Returns:
        Time information as string or dict.
    """
    fmt = parameters.get("format", "iso")
    now = datetime.now(timezone.utc)
    if fmt == "unix":
        return now.timestamp()
    if fmt == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    return now.isoformat()
