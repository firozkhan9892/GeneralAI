"""Base exception for the entire GeneralAI platform.

All custom exceptions inherit from :class:`GeneralAIError` so that
top-level handlers can catch and report them uniformly.
"""

from __future__ import annotations

from typing import Any


class GeneralAIError(Exception):
    """Base exception for all GeneralAI errors.

    Every subclass should pass a *message* and, where applicable,
    a *module* name and originating *cause*.
    """

    def __init__(
        self,
        message: str = "",
        *,
        module: str = "",
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialise GeneralAIError.

        Args:
            message: Human-readable description of the error.
            module: Name of the module that raised the error.
            cause: The original exception that caused this error.
            context: Arbitrary key-value pairs with additional context.
        """
        self.module = module
        self.cause = cause
        self.context = context or {}
        super().__init__(message)

    @property
    def message(self) -> str:
        """Return the exception message."""
        return str(self.args[0]) if self.args else ""

    def __str__(self) -> str:
        parts = [self.message]
        if self.module:
            parts.insert(0, f"[{self.module}]")
        return " ".join(parts)
