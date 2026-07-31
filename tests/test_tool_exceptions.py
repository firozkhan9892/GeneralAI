"""Tests for the tool exception hierarchy."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError
from app.tools.exceptions import (
    PermissionDeniedError,
    ToolAlreadyRegisteredError,
    ToolCancelledError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)


class TestToolExceptions:
    def test_hierarchy(self) -> None:
        for exc_type in (
            ToolError,
            ToolNotFoundError,
            ToolAlreadyRegisteredError,
            ToolValidationError,
            ToolExecutionError,
            ToolTimeoutError,
            ToolCancelledError,
            PermissionDeniedError,
        ):
            assert issubclass(exc_type, GeneralAIError)
            assert issubclass(exc_type, ToolError)

    def test_message_and_module(self) -> None:
        error = ToolNotFoundError("missing", module="tools.registry")
        assert error.message == "missing"
        assert error.module == "tools.registry"
        assert "tools.registry" in str(error)
