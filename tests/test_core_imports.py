"""Smoke tests verifying core package top-level imports work."""

from __future__ import annotations

from app.core import (
    DependencyContainer,
    EventBus,
    LifecycleManager,
    PluginLoader,
    PluginMetadata,
    BaseRegistry,
    BrainRegistry,
    MemoryRegistry,
    ToolRegistry,
    AgentRegistry,
    PlannerRegistry,
    PluginRegistry,
    WorkflowRegistry,
    IModule,
    IBrain,
    IMemory,
    ITool,
    IAgent,
    IPlanner,
    IPlugin,
    IWorkflow,
    IEventBus,
    Event,
    GeneralAIError,
    ConfigurationError,
    ContainerError,
    EventError,
    LifecycleError,
    PluginError,
    BrainError,
    MemoryError,
    ToolError,
    PlannerError,
    AgentError,
    LifecycleStage,
    HOOK_AFTER_PLUGINS,
    HOOK_AFTER_START,
)


class TestCorePackageImports:
    """All public API symbols should be importable from ``app.core``."""

    def test_container_import(self) -> None:
        assert DependencyContainer is not None

    def test_event_bus_import(self) -> None:
        assert EventBus is not None

    def test_lifecycle_import(self) -> None:
        assert LifecycleManager is not None

    def test_plugin_imports(self) -> None:
        assert PluginLoader is not None
        assert PluginMetadata is not None

    def test_registry_imports(self) -> None:
        assert BaseRegistry is not None
        assert BrainRegistry is not None
        assert MemoryRegistry is not None
        assert ToolRegistry is not None
        assert AgentRegistry is not None
        assert PlannerRegistry is not None
        assert PluginRegistry is not None
        assert WorkflowRegistry is not None

    def test_interface_imports(self) -> None:
        assert IModule is not None
        assert IBrain is not None
        assert IMemory is not None
        assert ITool is not None
        assert IAgent is not None
        assert IPlanner is not None
        assert IPlugin is not None
        assert IWorkflow is not None
        assert IEventBus is not None
        assert Event is not None

    def test_exception_imports(self) -> None:
        assert GeneralAIError is not None
        assert ConfigurationError is not None
        assert ContainerError is not None
        assert EventError is not None
        assert LifecycleError is not None
        assert PluginError is not None
        assert BrainError is not None
        assert MemoryError is not None
        assert ToolError is not None
        assert PlannerError is not None
        assert AgentError is not None

    def test_lifecycle_stage_import(self) -> None:
        assert LifecycleStage is not None

    def test_hook_constants(self) -> None:
        assert HOOK_AFTER_PLUGINS is not None
        assert HOOK_AFTER_START is not None
