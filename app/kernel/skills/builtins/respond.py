"""Respond skill — builds a response message."""

from __future__ import annotations

from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Build a response message.

    Args:
        parameters: Must contain 'content' key.
                    Optional 'format' (default 'text').

    Returns:
        Response object with content and metadata.
    """
    content = parameters.get("content", "")
    fmt = parameters.get("format", "text")

    return {
        "content": str(content),
        "format": fmt,
        "metadata": {
            "skill": "respond",
            "content_length": len(str(content)),
        },
    }
