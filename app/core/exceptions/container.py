"""Dependency injection container exceptions."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class ContainerError(GeneralAIError):
    """Base for all DI container errors."""


class RegistrationError(ContainerError):
    """Raised when a type registration fails."""


class ResolutionError(ContainerError):
    """Raised when a dependency cannot be resolved."""


class CircularDependencyError(ResolutionError):
    """Raised when a circular dependency is detected during resolution."""


class TypeNotRegisteredError(ResolutionError):
    """Raised when the requested type has no registration."""
