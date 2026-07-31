"""Tests for the Tool base abstraction."""

from __future__ import annotations

import pytest

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolValidationError
from app.tools.models import ToolCategory, ToolMetadata, ToolParameter


class _SumTool(Tool):
    """Concrete tool for exercising the base class."""

    name = "sum"
    description = "Adds two numbers"
    category = ToolCategory.PYTHON
    version = "2.0.0"
    timeout_s = 5.0
    requires_confirmation = True
    sandboxable = True
    parameters = (
        ToolParameter(name="a", param_type="integer", required=True),
        ToolParameter(name="b", param_type="integer", required=True),
        ToolParameter(name="scale", param_type="number", default=1.0),
    )

    def run(self, arguments, context=None):
        return (arguments["a"] + arguments["b"]) * arguments["scale"]


class TestToolAbstract:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Tool()  # type: ignore[abstract]

    def test_metadata(self) -> None:
        meta = _SumTool().metadata
        assert isinstance(meta, ToolMetadata)
        assert meta.name == "sum"
        assert meta.description == "Adds two numbers"
        assert meta.category == ToolCategory.PYTHON
        assert meta.version == "2.0.0"
        assert meta.timeout_s == 5.0
        assert meta.requires_confirmation is True
        assert meta.sandboxable is True


class TestToolValidation:
    def test_requires_all_required(self) -> None:
        with pytest.raises(ToolValidationError, match="a"):
            _SumTool().validate({"b": 2})

    def test_coerces_types(self) -> None:
        cleaned = _SumTool().validate({"a": "1", "b": 2})
        assert cleaned["a"] == 1
        assert isinstance(cleaned["a"], int)

    def test_applies_defaults(self) -> None:
        cleaned = _SumTool().validate({"a": 1, "b": 2})
        assert cleaned["scale"] == 1.0

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ToolValidationError, match="Unknown"):
            _SumTool().validate({"a": 1, "b": 2, "c": 3})

    def test_coercion_failure(self) -> None:
        with pytest.raises(ToolValidationError, match="integer"):
            _SumTool().validate({"a": "nope", "b": 2})

    def test_empty_arguments_defaults(self) -> None:
        tool = _SumTool()
        with pytest.raises(ToolValidationError):
            tool.validate({})


class TestToolRun:
    def test_sync_run(self) -> None:
        assert _SumTool().run({"a": 2, "b": 3, "scale": 1.0}) == 5

    @pytest.mark.asyncio
    async def test_async_run_offloads(self) -> None:
        tool = _SumTool()
        output = await tool.arun({"a": 2, "b": 3, "scale": 2.0})
        assert output == 10.0

    @pytest.mark.asyncio
    async def test_async_run_accepts_context(self) -> None:
        context = ToolContext()
        output = await _SumTool().arun({"a": 1, "b": 1, "scale": 1.0}, context)
        assert output == 2.0
