"""Tests for DependencyContainer."""

from __future__ import annotations

import pytest

from app.core.container import DependencyContainer
from app.core.exceptions.container import (
    CircularDependencyError,
    RegistrationError,
    TypeNotRegisteredError,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
class _Engine:
    def __init__(self) -> None:
        self.started = False


class _Car:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine


class _NoAnnotation:
    def __init__(self, engine) -> None:  # type: ignore[no-untyped-def]
        self.engine = engine


class _CircularA:
    def __init__(self, b: "_CircularB") -> None:
        self.b = b


class _CircularB:
    def __init__(self, a: _CircularA) -> None:
        self.a = a


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestDependencyContainer:
    """Suite for dependency injection."""

    def setup_method(self) -> None:
        self.container = DependencyContainer()

    def test_register_and_has(self) -> None:
        self.container.register_singleton(_Engine)
        assert self.container.has(_Engine) is True
        assert self.container.has(_Car) is False

    def test_resolve_singleton(self) -> None:
        self.container.register_singleton(_Engine)
        engine = self.container.resolve(_Engine)
        assert isinstance(engine, _Engine)

    def test_singleton_returns_same_instance(self) -> None:
        self.container.register_singleton(_Engine)
        a = self.container.resolve(_Engine)
        b = self.container.resolve(_Engine)
        assert a is b

    def test_register_factory_returns_new_instance(self) -> None:
        self.container.register_factory(_Engine, lambda: _Engine())
        a = self.container.resolve(_Engine)
        b = self.container.resolve(_Engine)
        assert a is not b

    def test_constructor_injection(self) -> None:
        self.container.register_singleton(_Engine)
        self.container.register_singleton(_Car)
        car = self.container.resolve(_Car)
        assert isinstance(car, _Car)
        assert isinstance(car.engine, _Engine)

    def test_register_duplicate_raises(self) -> None:
        self.container.register_singleton(_Engine)
        with pytest.raises(RegistrationError):
            self.container.register_singleton(_Engine)

    def test_resolve_unregistered_raises(self) -> None:
        with pytest.raises(TypeNotRegisteredError):
            self.container.resolve(_Car)

    def test_circular_dependency_raises(self) -> None:
        self.container.register_singleton(_CircularA)
        self.container.register_singleton(_CircularB)
        with pytest.raises(CircularDependencyError):
            self.container.resolve(_CircularA)

    def test_unregister_removes_entry(self) -> None:
        self.container.register_singleton(_Engine)
        assert self.container.has(_Engine) is True
        self.container.unregister(_Engine)
        assert self.container.has(_Engine) is False

    def test_clear_removes_all(self) -> None:
        self.container.register_singleton(_Engine)
        self.container.register_singleton(_Car)
        self.container.clear()
        assert self.container.has(_Engine) is False
        assert self.container.has(_Car) is False

    def test_register_with_instance(self) -> None:
        engine = _Engine()
        self.container.register_singleton(_Engine, instance=engine)
        resolved = self.container.resolve(_Engine)
        assert resolved is engine

    def test_register_with_factory_singleton(self) -> None:
        self.container.register_singleton(_Engine, factory=lambda: _Engine())
        a = self.container.resolve(_Engine)
        b = self.container.resolve(_Engine)
        assert a is b
