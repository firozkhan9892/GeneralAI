"""Tests for the deterministic MockTool."""

from __future__ import annotations

import pytest

from app.tools.context import ToolContext
from app.tools.exceptions import ToolExecutionError
from app.tools.mock import MockTool
from app.tools.models import ToolCategory, ToolMetadata


class TestMockMetadata:
    def test_name(self) -> None:
        assert MockTool().name == "mock"

    def test_custom_name(self) -> None:
        assert MockTool(name="custom").name == "custom"

    def test_metadata(self) -> None:
        meta = MockTool(name="t").metadata
        assert isinstance(meta, ToolMetadata)
        assert meta.name == "t"
        assert meta.category == ToolCategory.BUILTIN
        assert meta.parameters[0].name == "input"


class TestMockOutput:
    def test_echo(self) -> None:
        tool = MockTool()
        assert tool.run({"input": "hello"}) == "Echo: hello"

    def test_fixed_result(self) -> None:
        tool = MockTool(echo_input=False, result=42)
        assert tool.run({}) == 42

    def test_deterministic(self) -> None:
        tool = MockTool()
        first = tool.run({"input": "same"})
        second = tool.run({"input": "same"})
        assert first == second

    def test_call_count(self) -> None:
        tool = MockTool()
        tool.run({})
        tool.run({})
        assert tool.call_count == 2

    def test_on_run_handler(self) -> None:
        tool = MockTool(on_run=lambda args, ctx: args["input"].upper())
        assert tool.run({"input": "hi"}) == "HI"


class TestMockFailure:
    def test_fail(self) -> None:
        tool = MockTool(fail="kaboom")
        with pytest.raises(ToolExecutionError, match="kaboom"):
            tool.run({})

    def test_fail_first_n(self) -> None:
        tool = MockTool(fail_first_n=2)
        with pytest.raises(ToolExecutionError):
            tool.run({})
        with pytest.raises(ToolExecutionError):
            tool.run({})
        assert tool.run({}) == "Echo: "


class TestMockContext:
    def test_context_received(self) -> None:
        captured: list[ToolContext | None] = []

        def handler(args, ctx):
            captured.append(ctx)
            return "done"

        tool = MockTool(on_run=handler)
        ctx = ToolContext()
        tool.run({"input": "x"}, ctx)
        assert captured == [ctx]

    @pytest.mark.asyncio
    async def test_async_run(self) -> None:
        tool = MockTool()
        output = await tool.arun({"input": "async"})
        assert output == "Echo: async"
