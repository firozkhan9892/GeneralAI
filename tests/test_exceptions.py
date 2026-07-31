"""Tests for the exceptions hierarchy."""

from __future__ import annotations

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
    RegistrationError,
    ResolutionError,
    CircularDependencyError,
    TypeNotRegisteredError,
    InvalidTransitionError,
    PluginDiscoveryError,
    PluginDependencyError,
)


class TestExceptions:
    """Suite for exception hierarchy."""

    def test_base_error_carries_context(self) -> None:
        exc = GeneralAIError("msg", module="test", context={"k": "v"})
        assert exc.message == "msg"
        assert exc.module == "test"
        assert exc.context == {"k": "v"}

    def test_base_error_with_cause(self) -> None:
        cause = ValueError("original")
        exc = GeneralAIError("wrapped", cause=cause)
        assert exc.cause is cause

    def test_base_error_str(self) -> None:
        exc = GeneralAIError("fail", module="core")
        assert "fail" in str(exc)
        assert "core" in str(exc)

    def test_all_exceptions_inherit_from_base(self) -> None:
        exceptions = [
            ConfigurationError("x"),
            ContainerError("x"),
            EventError("x"),
            LifecycleError("x"),
            PluginError("x"),
            BrainError("x"),
            MemoryError("x"),
            ToolError("x"),
            PlannerError("x"),
            AgentError("x"),
        ]
        for exc in exceptions:
            assert isinstance(exc, GeneralAIError)

    def test_container_specific_exceptions(self) -> None:
        assert issubclass(RegistrationError, ContainerError)
        assert issubclass(ResolutionError, ContainerError)
        assert issubclass(CircularDependencyError, ResolutionError)
        assert issubclass(TypeNotRegisteredError, ResolutionError)

    def test_invalid_transition_inheritance(self) -> None:
        assert issubclass(InvalidTransitionError, LifecycleError)

    def test_plugin_discovery_inheritance(self) -> None:
        assert issubclass(PluginDiscoveryError, PluginError)
        assert issubclass(PluginDependencyError, PluginError)

    def test_exception_can_be_raised_and_caught(self) -> None:
        try:
            raise BrainError("cognitive failure", module="brain")
        except GeneralAIError as exc:
            assert exc.module == "brain"
            assert "cognitive failure" in str(exc)
