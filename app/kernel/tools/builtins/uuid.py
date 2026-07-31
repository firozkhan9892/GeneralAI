"""UUID tool — generates unique identifiers."""

from __future__ import annotations

import uuid
from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Generate a UUID.

    Args:
        parameters: Optional 'version' key (4 for random, 1 for MAC-based).

    Returns:
        UUID string.
    """
    version = parameters.get("version", 4)
    if version == 1:
        return str(uuid.uuid1())
    return str(uuid.uuid4())
