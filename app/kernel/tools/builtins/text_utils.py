"""Text utilities tool — performs text manipulation operations."""

from __future__ import annotations

import re
from typing import Any


async def execute(parameters: dict[str, Any]) -> Any:
    """Perform text operations.

    Args:
        parameters: Must contain 'operation' and 'text'.
                    Operations: 'uppercase', 'lowercase', 'strip', 'truncate',
                    'word_count', 'char_count', 'replace', 'split_paragraphs'.

    Returns:
        Result of the text operation.
    """
    operation = parameters.get("operation", "strip")
    text = parameters.get("text", "")

    if operation == "uppercase":
        return text.upper()
    if operation == "lowercase":
        return text.lower()
    if operation == "strip":
        return text.strip()
    if operation == "truncate":
        max_len = parameters.get("max_length", 100)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
    if operation == "word_count":
        return len(text.split())
    if operation == "char_count":
        return len(text)
    if operation == "replace":
        old = parameters.get("old", "")
        new = parameters.get("new", "")
        return text.replace(old, new)
    if operation == "split_paragraphs":
        paragraphs = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in paragraphs if p.strip()]
    raise ValueError(f"Unknown operation: {operation}")
