"""Agent manager exception hierarchy.

All exceptions derive from :class:`AgentManagerError` which in turn
derives from the platform-wide :class:`GeneralAIError`, so callers can
catch and report agent session failures uniformly.
"""

from __future__ import annotations

from app.core.exceptions.base import GeneralAIError


class AgentManagerError(GeneralAIError):
    """Base exception for the agent manager layer."""


class SessionNotFoundError(AgentManagerError):
    """Raised when a session identifier does not exist."""


class SessionAlreadyExistsError(AgentManagerError):
    """Raised when registering a session whose id is already in use."""


class SessionNotRunnableError(AgentManagerError):
    """Raised when a session cannot transition into the requested state."""
