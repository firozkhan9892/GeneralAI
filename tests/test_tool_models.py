"""Tests for tool domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.tools.models import (
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)


class TestToolCategory:
    def test_members(self) -> None:
        assert ToolCategory.BUILTIN.value == "builtin"
        assert ToolCategory.FILE.value == "file"
        assert ToolCategory.WEB.value == "web"
        assert ToolCategory.SHELL.value == "shell"
        assert ToolCategory.PYTHON.value == "python"
        assert ToolCategory.HTTP.value == "http"


class TestToolParameter:
    def test_defaults(self) -> None:
        param = ToolParameter(name="foo")
        assert param.description == ""
        assert param.param_type == "string"
        assert param.required is False
        assert param.default is None

    def test_full(self) -> None:
        param = ToolParameter(
            name="count",
            description="How many",
            param_type="integer",
            required=True,
            default=3,
        )
        assert param.name == "count"
        assert param.param_type == "integer"
        assert param.required is True
        assert param.default == 3

    def test_requires_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolParameter()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        param = ToolParameter(name="foo")
        with pytest.raises(ValidationError):
            param.name = "bar"  # type: ignore[misc]


class TestToolMetadata:
    def test_defaults(self) -> None:
        meta = ToolMetadata(name="calc")
        assert meta.description == ""
        assert meta.category == ToolCategory.BUILTIN
        assert meta.version == "1.0.0"
        assert meta.parameters == ()
        assert meta.timeout_s == 30.0
        assert meta.requires_confirmation is False
        assert meta.sandboxable is False

    def test_full(self) -> None:
        param = ToolParameter(name="x")
        meta = ToolMetadata(
            name="calc",
            description="Adds",
            category=ToolCategory.PYTHON,
            version="2.0.0",
            parameters=(param,),
            timeout_s=5.0,
            requires_confirmation=True,
            sandboxable=True,
        )
        assert meta.category == ToolCategory.PYTHON
        assert meta.parameters[0] is param
        assert meta.requires_confirmation is True
        assert meta.sandboxable is True

    def test_frozen(self) -> None:
        meta = ToolMetadata(name="calc")
        with pytest.raises(ValidationError):
            meta.name = "other"  # type: ignore[misc]

    def test_timeout_positive(self) -> None:
        with pytest.raises(ValidationError):
            ToolMetadata(name="calc", timeout_s=0)


class TestToolResult:
    def test_defaults(self) -> None:
        result = ToolResult()
        assert result.tool_name == ""
        assert result.success is True
        assert result.output is None
        assert result.error is None
        assert result.metadata == {}
        assert result.execution_time == 0.0

    def test_success(self) -> None:
        result = ToolResult(tool_name="calc", output=5)
        assert result.success is True
        assert result.output == 5
        assert result.error is None

    def test_failure(self) -> None:
        result = ToolResult(tool_name="calc", success=False, error="boom")
        assert result.success is False
        assert result.error == "boom"

    def test_metadata(self) -> None:
        result = ToolResult(metadata={"attempts": 2, "timed_out": True})
        assert result.metadata["attempts"] == 2
        assert result.metadata["timed_out"] is True

    def test_execution_time_ge_zero(self) -> None:
        with pytest.raises(ValidationError):
            ToolResult(execution_time=-1)

    def test_frozen(self) -> None:
        result = ToolResult()
        with pytest.raises(ValidationError):
            result.success = False  # type: ignore[misc]
