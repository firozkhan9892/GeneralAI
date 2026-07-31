"""Core framework package for GeneralAI.

Provides the internal infrastructure that every AI module depends on:
dependency injection, event bus, lifecycle management, plugin loading,
registries, abstract interfaces, and typed exceptions.

Design principles:
- Zero global state — everything is instantiated and wired explicitly.
- Everything is extensible via the plugin system.
- Modules communicate only through the event bus.

Usage::

    from app.core import DependencyContainer, EventBus, LifecycleManager

    container = DependencyContainer()
    bus = EventBus()
    lifecycle = LifecycleManager()
"""

from app.core.container import DependencyContainer
from app.core.events import EventBus
from app.core.lifecycle import LifecycleManager
from app.core.plugins import PluginLoader, PluginMetadata
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
from app.core.interfaces import (
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
    EventHandler,
)
from app.core.exceptions import (
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
)
from app.core.constants.lifecycle import (
    LifecycleStage,
    HOOK_AFTER_PLUGINS,
    HOOK_AFTER_START,
)

__all__ = [
    # Container
    "DependencyContainer",
    # Events
    "EventBus",
    "Event",
    "EventHandler",
    "IEventBus",
    # Lifecycle
    "LifecycleManager",
    "LifecycleStage",
    "HOOK_AFTER_PLUGINS",
    "HOOK_AFTER_START",
    # Plugins
    "PluginLoader",
    "PluginMetadata",
    # Registries
    "BaseRegistry",
    "BrainRegistry",
    "MemoryRegistry",
    "ToolRegistry",
    "AgentRegistry",
    "PlannerRegistry",
    "PluginRegistry",
    "WorkflowRegistry",
    # Interfaces
    "IModule",
    "IBrain",
    "IMemory",
    "ITool",
    "IAgent",
    "IPlanner",
    "IPlugin",
    "IWorkflow",
    # Exceptions
    "GeneralAIError",
    "ConfigurationError",
    "ContainerError",
    "EventError",
    "LifecycleError",
    "PluginError",
    "BrainError",
    "MemoryError",
    "ToolError",
    "PlannerError",
    "AgentError",
]
