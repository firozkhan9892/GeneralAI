"""Search memory skill — searches experience records."""

from __future__ import annotations

from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Search memory/experience records.

    Args:
        parameters: Must contain 'query' key.
                    Optional 'limit' (default 5).

    Returns:
        List of matching experience records.
    """
    query = parameters.get("query", "")
    limit = parameters.get("limit", 5)

    # In a real implementation, this would query the ExperienceStore.
    # For deterministic testing, we return an empty list or a placeholder.
    # The skill handler receives the experience store via context if needed.
    return {
        "query": query,
        "limit": limit,
        "results": [],
        "total": 0,
    }
