"""Exception hierarchy — importable shortcuts."""

from app.core.exceptions.base import GeneralAIError
from app.core.exceptions.configuration import (
    ConfigurationError,
    ConfigValidationError,
    ConfigLoadError,
)
from app.core.exceptions.container import (
    ContainerError,
    RegistrationError,
    ResolutionError,
    CircularDependencyError,
    TypeNotRegisteredError,
)
from app.core.exceptions.event import (
    EventError,
    SubscriptionError,
    PublishError,
    HandlerError,
)
from app.core.exceptions.lifecycle import (
    LifecycleError,
    InvalidTransitionError,
    HookExecutionError,
    StageTimeoutError,
)
from app.core.exceptions.plugin import (
    PluginError,
    PluginDiscoveryError,
    PluginLoadError,
    PluginDependencyError,
    PluginValidationError,
    PluginDisabledError,
)
from app.core.exceptions.domain import (
    BrainError,
    MemoryError,
    ToolError,
    PlannerError,
    AgentError,
)

__all__ = [
    "GeneralAIError",
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigLoadError",
    "ContainerError",
    "RegistrationError",
    "ResolutionError",
    "CircularDependencyError",
    "TypeNotRegisteredError",
    "EventError",
    "SubscriptionError",
    "PublishError",
    "HandlerError",
    "LifecycleError",
    "InvalidTransitionError",
    "HookExecutionError",
    "StageTimeoutError",
    "PluginError",
    "PluginDiscoveryError",
    "PluginLoadError",
    "PluginDependencyError",
    "PluginValidationError",
    "PluginDisabledError",
    "BrainError",
    "MemoryError",
    "ToolError",
    "PlannerError",
    "AgentError",
]
