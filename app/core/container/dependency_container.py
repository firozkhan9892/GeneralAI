"""Thread-safe dependency injection container.

Supports singleton and factory registrations with automatic
constructor injection based on type hints.
"""

from __future__ import annotations

import inspect
import threading
import typing
from typing import Any, Callable, TypeVar

from app.core.exceptions.container import (
    CircularDependencyError,
    RegistrationError,
    ResolutionError,
    TypeNotRegisteredError,
)

T = TypeVar("T")

# Sentinel for unregistered entries
_MISSING: Any = object()


class _Registration:
    """Internal container entry."""

    def __init__(
        self,
        *,
        instance: Any = None,
        factory: Callable | None = None,
        is_singleton: bool | None = None,
    ) -> None:
        self.instance = instance
        self.factory = factory
        self.is_singleton = (
            is_singleton if is_singleton is not None else (factory is None)
        )


class DependencyContainer:
    """Lightweight, thread-safe DI container.

    Typical usage::

        container = DependencyContainer()
        container.register_singleton(Config, config)
        container.register_factory(IService, Service)
        service = container.resolve(IService)
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Registration] = {}
        self._lock = threading.RLock()
        self._resolve_stack: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_singleton(
        self,
        interface: type,
        instance: Any = None,
        factory: Callable | None = None,
    ) -> None:
        """Register *interface* as a singleton.

        If *instance* is provided, it is used directly.  Otherwise
        *factory* is called once and the result is cached.  If neither
        is given, the *interface* type itself is instantiated.

        Args:
            interface: The type to register.
            instance: Pre-built instance to register.
            factory: Zero-argument callable that returns an instance.

        Raises:
            RegistrationError: If the type is already registered.
        """
        key = self._key(interface)
        with self._lock:
            if key in self._entries:
                raise RegistrationError(
                    f"Type '{key}' is already registered",
                    module="container",
                )

            if instance is not None:
                self._entries[key] = _Registration(instance=instance, is_singleton=True)
            elif factory is not None:
                self._entries[key] = _Registration(factory=factory, is_singleton=True)
            else:
                self._entries[key] = _Registration(
                    factory=lambda: self._build(interface),
                    is_singleton=True,
                )

    def register_factory(self, interface: type, factory: Callable) -> None:
        """Register *factory* as the provider for *interface*.

        The factory is called on every resolution (non-singleton).

        Args:
            interface: The type to register.
            factory: Callable that returns an instance.

        Raises:
            RegistrationError: If the type is already registered.
        """
        key = self._key(interface)
        with self._lock:
            if key in self._entries:
                raise RegistrationError(
                    f"Type '{key}' is already registered",
                    module="container",
                )
            self._entries[key] = _Registration(factory=factory, is_singleton=False)

    def resolve(self, interface: type[T]) -> T:
        """Resolve an instance of *interface*.

        Supports recursive constructor injection: if the registered
        factory is a class, its ``__init__`` parameters annotated
        with registered types are injected automatically.

        Args:
            interface: The type to resolve.

        Returns:
            An instance of the requested type.

        Raises:
            TypeNotRegisteredError: If the type has no registration.
            CircularDependencyError: If a circular dependency is detected.
            ResolutionError: If resolution fails for any other reason.
        """
        key = self._key(interface)
        with self._lock:
            if key not in self._entries:
                raise TypeNotRegisteredError(
                    f"Type '{key}' is not registered",
                    module="container",
                    context={"type": key},
                )

            if key in self._resolve_stack:
                raise CircularDependencyError(
                    f"Circular dependency detected for '{key}': "
                    f"{' -> '.join(self._resolve_stack + [key])}",
                    module="container",
                    context={"stack": list(self._resolve_stack), "type": key},
                )

            entry = self._entries[key]
            self._resolve_stack.append(key)

            try:
                if entry.instance is not _MISSING and entry.instance is not None:
                    return entry.instance  # type: ignore[return-value]

                instance = entry.factory() if entry.factory else self._build(interface)
                if entry.is_singleton:
                    entry.instance = instance
                return instance  # type: ignore[return-value]
            except CircularDependencyError:
                raise
            except Exception as exc:
                raise ResolutionError(
                    f"Failed to resolve '{key}': {exc}",
                    module="container",
                    cause=exc if isinstance(exc, Exception) else None,
                    context={"type": key},
                ) from exc
            finally:
                self._resolve_stack.pop()

    def unregister(self, interface: type) -> None:
        """Remove a registration.

        Args:
            interface: The type to unregister.
        """
        key = self._key(interface)
        with self._lock:
            self._entries.pop(key, None)

    def has(self, interface: type) -> bool:
        """Return ``True`` if *interface* has a registration."""
        return self._key(interface) in self._entries

    def clear(self) -> None:
        """Remove all registrations and reset the container."""
        with self._lock:
            self._entries.clear()
            self._resolve_stack.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(interface: type) -> str:
        """Derive a string key from a type."""
        return f"{interface.__module__}.{interface.__qualname__}"

    def _build(self, interface: type[T]) -> T:
        """Instantiate *interface* with constructor injection.

        Inspects ``__init__`` parameters, resolves each annotated
        type from the container, and passes them as arguments.
        Uses ``typing.get_type_hints`` to resolve forward references
        and ``from __future__ import annotations`` strings.
        """
        try:
            hints = typing.get_type_hints(interface.__init__)
        except Exception:
            hints = {}

        sig = inspect.signature(interface.__init__)
        kwargs: dict[str, Any] = {}
        missing: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.annotation is inspect.Parameter.empty and param_name not in hints:
                continue

            # Skip *args and **kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            annotation = hints.get(param_name, param.annotation)
            if isinstance(annotation, str):
                if param.default is inspect.Parameter.empty:
                    missing.append(param_name)
                continue

            if self.has(annotation):
                kwargs[param_name] = self.resolve(annotation)
            elif param.default is inspect.Parameter.empty:
                missing.append(param_name)

        if missing:
            raise ResolutionError(
                f"Cannot construct '{interface.__qualname__}': "
                f"unresolvable parameters {missing}",
                module="container",
                context={"type": self._key(interface), "missing": missing},
            )

        return interface(**kwargs)
