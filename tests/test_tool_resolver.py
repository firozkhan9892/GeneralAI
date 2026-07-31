"""Tests for ToolResolver."""

from __future__ import annotations

import pytest

from app.kernel.tools.executor import ToolResolver
from app.kernel.tools.models import ToolBinding, ToolDescriptor
from app.kernel.tools.builtins import register_builtin_tools


class TestToolResolverRegistration:
    """Tests for ToolResolver registration methods."""

    def test_register_tool(self) -> None:
        resolver = ToolResolver()
        desc = ToolDescriptor(name="test_tool", description="A test tool")
        resolver.register_tool(desc)
        assert resolver.has_tool("test_tool") is True

    def test_register_tool_overwrite(self) -> None:
        resolver = ToolResolver()
        desc1 = ToolDescriptor(name="test_tool", description="First")
        desc2 = ToolDescriptor(name="test_tool", description="Second")
        resolver.register_tool(desc1)
        resolver.register_tool(desc2)
        assert resolver.has_tool("test_tool") is True

    def test_has_tool_not_registered(self) -> None:
        resolver = ToolResolver()
        assert resolver.has_tool("nonexistent") is False

    def test_list_tools_empty(self) -> None:
        resolver = ToolResolver()
        assert resolver.list_tools() == []

    def test_list_tools_after_registration(self) -> None:
        resolver = ToolResolver()
        resolver.register_tool(ToolDescriptor(name="tool_a"))
        resolver.register_tool(ToolDescriptor(name="tool_b"))
        tools = resolver.list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert len(tools) == 2


class TestToolResolverResolve:
    """Tests for ToolResolver.resolve."""

    @pytest.mark.asyncio
    async def test_resolve_existing_tool(self) -> None:
        resolver = ToolResolver()
        resolver.register_tool(ToolDescriptor(name="calc", description="Calculator"))
        binding = await resolver.resolve("calc", {"expression": "1+1"})
        assert isinstance(binding, ToolBinding)
        assert binding.tool_name == "calc"
        assert binding.parameters == {"expression": "1+1"}
        assert binding.descriptor.name == "calc"

    @pytest.mark.asyncio
    async def test_resolve_with_no_parameters(self) -> None:
        resolver = ToolResolver()
        resolver.register_tool(ToolDescriptor(name="clock"))
        binding = await resolver.resolve("clock")
        assert isinstance(binding, ToolBinding)
        assert binding.tool_name == "clock"
        assert binding.parameters == {}

    @pytest.mark.asyncio
    async def test_resolve_unregistered_tool(self) -> None:
        resolver = ToolResolver()
        with pytest.raises(KeyError):
            await resolver.resolve("nonexistent")

    @pytest.mark.asyncio
    async def test_resolve_returns_tool_binding(self) -> None:
        resolver = ToolResolver()
        resolver.register_tool(
            ToolDescriptor(
                name="test",
                input_schema={"type": "object"},
                output_schema={"type": "string"},
            )
        )
        binding = await resolver.resolve("test", {"key": "value"})
        assert binding.tool_name == "test"
        assert binding.descriptor.input_schema == {"type": "object"}
        assert binding.descriptor.output_schema == {"type": "string"}


class TestToolResolverBuiltinRegistration:
    """Tests for registering built-in tools."""

    @pytest.mark.asyncio
    async def test_register_builtin_tools(self) -> None:
        resolver = ToolResolver()

        register_builtin_tools(resolver)
        assert resolver.has_tool("calculator")
        assert resolver.has_tool("clock")
        assert resolver.has_tool("uuid")
        assert resolver.has_tool("json")
        assert resolver.has_tool("text_utils")

    @pytest.mark.asyncio
    async def test_resolve_builtin_tool(self) -> None:
        resolver = ToolResolver()

        register_builtin_tools(resolver)
        binding = await resolver.resolve("calculator", {"expression": "2+2"})
        assert binding.tool_name == "calculator"
        assert binding.parameters == {"expression": "2+2"}
