"""Tests for the tool registry."""

from __future__ import annotations

import pytest

from app.tools.exceptions import ToolNotFoundError
from app.tools.mock import MockTool
from app.tools.models import ToolCategory
from app.tools.registry import ToolRegistry


def _tool(name: str = "alpha") -> MockTool:
    return MockTool(name=name)


class TestToolRegistryRegistration:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = _tool("alpha")
        registry.register(tool)
        assert registry.get("alpha") is tool

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_tool("alpha"))

    def test_register_overwrite(self) -> None:
        registry = ToolRegistry()
        first = _tool("alpha")
        second = _tool("alpha")
        registry.register(first)
        registry.register(second, overwrite=True)
        assert registry.get("alpha") is second

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        registry.unregister("alpha")
        assert not registry.has("alpha")

    def test_clear(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        registry.register(_tool("beta"))
        registry.clear()
        assert len(registry) == 0

    def test_has(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        assert registry.has("alpha")
        assert not registry.has("missing")


class TestToolRegistryQuery:
    def test_get_or_raise(self) -> None:
        registry = ToolRegistry()
        tool = _tool("alpha")
        registry.register(tool)
        assert registry.get_or_raise("alpha") is tool

    def test_get_or_raise_missing(self) -> None:
        with pytest.raises(ToolNotFoundError, match="not registered"):
            ToolRegistry().get_or_raise("missing")

    def test_names(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        registry.register(_tool("beta"))
        assert sorted(registry.names()) == ["alpha", "beta"]

    def test_tools(self) -> None:
        registry = ToolRegistry()
        alpha = _tool("alpha")
        beta = _tool("beta")
        registry.register(alpha)
        registry.register(beta)
        assert set(registry.tools()) == {alpha, beta}

    def test_contains_iter_len(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        assert "alpha" in registry
        assert "missing" not in registry
        assert list(registry) == registry.tools()
        assert len(registry) == 1


class TestToolRegistryDiscover:
    def test_discovers_defaults(self) -> None:
        registry = ToolRegistry()
        count = registry.discover()
        assert count == 11
        assert registry.has("calculator")
        assert registry.has("echo")
        assert registry.has("file_read")
        assert registry.has("web_fetch")
        assert registry.has("shell_run")
        assert registry.has("python_eval")
        assert registry.has("http_request")

    def test_discovers_all_categories(self) -> None:
        registry = ToolRegistry()
        registry.discover()
        categories = {tool.category for tool in registry.tools()}
        assert categories == {
            ToolCategory.BUILTIN,
            ToolCategory.FILE,
            ToolCategory.WEB,
            ToolCategory.SHELL,
            ToolCategory.PYTHON,
            ToolCategory.HTTP,
        }

    def test_discover_by_category(self) -> None:
        registry = ToolRegistry()
        count = registry.discover(category=ToolCategory.FILE)
        assert count == 3
        assert all(tool.category == ToolCategory.FILE for tool in registry.tools())

    def test_discover_custom_tools(self) -> None:
        registry = ToolRegistry()
        count = registry.discover([_tool("custom")])
        assert count == 1
        assert registry.has("custom")

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("alpha"))
        metadata = registry.list_tools()
        assert len(metadata) == 1
        assert metadata[0].name == "alpha"

    def test_list_tools_by_category(self) -> None:
        registry = ToolRegistry()
        registry.discover()
        file_tools = registry.list_tools(category=ToolCategory.FILE)
        assert file_tools
        assert all(t.category == ToolCategory.FILE for t in file_tools)
