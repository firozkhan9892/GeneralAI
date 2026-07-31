"""Agent Manager — application-level multi-session agent orchestration.

The agent manager supervises multiple concurrent
:class:`~app.kernel.agent.runtime.AgentRuntime` sessions: it creates and
tracks one :class:`AgentSession` per request, drives each through the
runtime, and supports cancellation, pause/resume, retries, timeouts,
memory, tools, and session persistence.
"""

from __future__ import annotations

from app.agents.bootstrap import register_agent_manager_components
from app.agents.exceptions import (
    AgentManagerError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionNotRunnableError,
)
from app.agents.manager import AgentManager
from app.agents.models import (
    TERMINAL_SESSION_STATUSES,
    AgentExecutionOptions,
    AgentSession,
    SessionStatus,
)
from app.agents.persistence import (
    InMemorySessionStore,
    JsonSessionStore,
    SessionStore,
)
from app.agents.registry import SessionRegistry

__all__ = [
    "AgentExecutionOptions",
    "AgentManager",
    "AgentManagerError",
    "AgentSession",
    "InMemorySessionStore",
    "JsonSessionStore",
    "SessionAlreadyExistsError",
    "SessionNotFoundError",
    "SessionNotRunnableError",
    "SessionRegistry",
    "SessionStatus",
    "SessionStore",
    "TERMINAL_SESSION_STATUSES",
    "register_agent_manager_components",
]
