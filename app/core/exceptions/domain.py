"""Domain-specific module exceptions (stubs for future use)."""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class BrainError(GeneralAIError):
    """Raised by brain module implementations."""


class MemoryError(GeneralAIError):
    """Raised by memory module implementations."""


class ToolError(GeneralAIError):
    """Raised by tool implementations."""


class PlannerError(GeneralAIError):
    """Raised by planner implementations."""


class AgentError(GeneralAIError):
    """Raised by agent implementations."""
