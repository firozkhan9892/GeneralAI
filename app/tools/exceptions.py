"""Tool framework exception hierarchy.

All exceptions derive from :class:`ToolError` which in turn derives from
the platform-wide :class:`GeneralAIError`, so callers can catch and
report tool failures uniformly.
"""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class ToolError(GeneralAIError):
    """Base exception for the tool framework."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class ToolAlreadyRegisteredError(ToolError):
    """Raised when registering a tool that already exists."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation or coercion."""


class ToolExecutionError(ToolError):
    """Raised when a tool raises while running."""


class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds its execution time budget."""


class ToolCancelledError(ToolError):
    """Raised when a tool execution is cancelled."""


class PermissionDeniedError(ToolError):
    """Raised when the permission system rejects a tool invocation."""
