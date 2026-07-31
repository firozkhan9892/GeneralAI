"""Tests for tool framework DI wiring and package exports."""

from __future__ import annotations

from app.core.container import DependencyContainer
from app.tools.bootstrap import register_tool_components
from app.tools.executor import ToolExecutor
from app.tools.permissions import PermissionSystem
from app.tools.registry import ToolRegistry


class TestBootstrap:
    def test_registers_registry_singleton(self) -> None:
        container = DependencyContainer()
        register_tool_components(container)
        registry = container.resolve(ToolRegistry)
        assert isinstance(registry, ToolRegistry)
        assert container.resolve(ToolRegistry) is registry

    def test_registers_permission_singleton(self) -> None:
        container = DependencyContainer()
        register_tool_components(container)
        permissions = container.resolve(PermissionSystem)
        assert isinstance(permissions, PermissionSystem)
        assert container.resolve(PermissionSystem) is permissions

    def test_registers_executor_singleton(self) -> None:
        container = DependencyContainer()
        register_tool_components(container)
        executor = container.resolve(ToolExecutor)
        assert isinstance(executor, ToolExecutor)
        assert container.resolve(ToolExecutor) is executor

    def test_executor_runs_discovered_tools(self) -> None:
        container = DependencyContainer()
        register_tool_components(container)
        registry = container.resolve(ToolRegistry)
        executor = container.resolve(ToolExecutor)
        registry.discover()
        result = executor.execute("echo", {"text": "hi"})
        assert result.success is True
        assert result.output == "hi"


class TestPackageExports:
    def test_top_level_imports(self) -> None:
        import app.tools as tools

        symbols = (
            "Tool",
            "ToolRegistry",
            "ToolExecutor",
            "PermissionSystem",
            "MockTool",
            "ToolContext",
            "ToolResult",
            "ToolMetadata",
            "ToolParameter",
            "ToolCategory",
            "CancellationToken",
            "register_tool_components",
        )
        assert all(getattr(tools, name, None) is not None for name in symbols)
