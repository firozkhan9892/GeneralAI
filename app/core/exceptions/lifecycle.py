"""Lifecycle manager exceptions."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class LifecycleError(GeneralAIError):
    """Base for all lifecycle errors."""


class InvalidTransitionError(LifecycleError):
    """Raised when an illegal stage transition is attempted."""


class HookExecutionError(LifecycleError):
    """Raised when a lifecycle hook raises an exception."""


class StageTimeoutError(LifecycleError):
    """Raised when a lifecycle stage exceeds its time limit."""
