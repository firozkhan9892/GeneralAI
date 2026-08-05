"""Shared helpers for the automation module."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


def json_safe(value: Any) -> Any:
    """Recursively convert *value* into a JSON-serialisable structure.

    Pydantic models are dumped to dictionaries, enums to their values,
    and any remaining non-serialisable object to its string form.  This
    guarantees step outputs and run events can be persisted and replayed.
    """
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
