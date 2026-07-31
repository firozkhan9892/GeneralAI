"""Tests for BaseRegistry and typed registries."""

from __future__ import annotations

import pytest

from app.core.registry import (
    BaseRegistry,
    BrainRegistry,
    MemoryRegistry,
    ToolRegistry,
    AgentRegistry,
    PlannerRegistry,
    PluginRegistry,
    WorkflowRegistry,
)


class TestBaseRegistry:
    """Suite for the generic base registry."""

    @pytest.fixture
    def registry(self) -> BaseRegistry[str]:
        return BaseRegistry[str]()

    def test_register_and_get(self, registry: BaseRegistry[str]) -> None:
        registry.register("key1", "value1")
        assert registry.get("key1") == "value1"

    def test_get_nonexistent(self, registry: BaseRegistry[str]) -> None:
        assert registry.get("missing") is None

    def test_get_or_raise(self, registry: BaseRegistry[str]) -> None:
        registry.register("a", "b")
        assert registry.get_or_raise("a") == "b"
        with pytest.raises(KeyError):
            registry.get_or_raise("missing")

    def test_has(self, registry: BaseRegistry[str]) -> None:
        registry.register("x", "y")
        assert registry.has("x") is True
        assert registry.has("z") is False

    def test_unregister(self, registry: BaseRegistry[str]) -> None:
        registry.register("k", "v")
        registry.unregister("k")
        assert registry.has("k") is False

    def test_clear(self, registry: BaseRegistry[str]) -> None:
        registry.register("a", "1")
        registry.register("b", "2")
        registry.clear()
        assert registry.is_empty is True
        assert registry.count == 0

    def test_values(self, registry: BaseRegistry[str]) -> None:
        registry.register("a", "1")
        registry.register("b", "2")
        items = registry.values()
        assert sorted(items) == ["1", "2"]

    def test_keys(self, registry: BaseRegistry[str]) -> None:
        registry.register("x", "10")
        registry.register("y", "20")
        assert sorted(registry.keys()) == ["x", "y"]

    def test_duplicate_key_raises(self, registry: BaseRegistry[str]) -> None:
        registry.register("k", "v1")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("k", "v2")

    def test_overwrite_allowed(self, registry: BaseRegistry[str]) -> None:
        registry.register("k", "v1")
        registry.register("k", "v2", overwrite=True)
        assert registry.get("k") == "v2"

    def test_contains(self, registry: BaseRegistry[str]) -> None:
        registry.register("a", "b")
        assert "a" in registry
        assert "missing" not in registry

    def test_len(self, registry: BaseRegistry[str]) -> None:
        assert len(registry) == 0
        registry.register("a", "1")
        assert len(registry) == 1

    def test_iter(self, registry: BaseRegistry[str]) -> None:
        registry.register("a", "1")
        registry.register("b", "2")
        assert sorted(list(registry)) == ["1", "2"]


class TestTypedRegistries:
    """Verify all typed registries instantiate correctly."""

    def test_brain_registry(self) -> None:
        reg = BrainRegistry()
        assert reg.is_empty is True

    def test_memory_registry(self) -> None:
        reg = MemoryRegistry()
        assert reg.count == 0

    def test_tool_registry(self) -> None:
        reg = ToolRegistry()
        assert list(reg) == []

    def test_agent_registry(self) -> None:
        reg = AgentRegistry()
        assert reg.keys() == []

    def test_planner_registry(self) -> None:
        reg = PlannerRegistry()
        assert reg.is_empty is True

    def test_plugin_registry(self) -> None:
        reg = PluginRegistry()
        assert reg.count == 0

    def test_workflow_registry(self) -> None:
        reg = WorkflowRegistry()
        assert reg.is_empty is True
