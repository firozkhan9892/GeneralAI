"""Session persistence for the agent manager.

A :class:`SessionStore` persists :class:`AgentSession` snapshots so
managed sessions survive process restarts.  The default is an in-memory
store; :class:`JsonSessionStore` writes one JSON file per session.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.agents.models import AgentSession

log = logging.getLogger(__name__)


def json_safe(value: Any) -> Any:
    """Recursively convert *value* into a JSON-serialisable structure.

    Pydantic models are dumped to dictionaries, enums to their values,
    and any remaining non-serialisable object to its string form.

    Args:
        value: Arbitrary value (e.g. an ``AgentResponse`` whose step
            results may be arbitrary objects).

    Returns:
        A JSON-safe equivalent of ``value``.
    """
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SessionStore(ABC):
    """Interface for session persistence.

    Implementations may use in-memory storage, a database, or a remote
    service.  All operations are asynchronous.
    """

    @abstractmethod
    async def save(self, session: AgentSession) -> None:
        """Persist a session snapshot.

        Args:
            session: The session to persist.
        """

    @abstractmethod
    async def load(self, session_id: str) -> AgentSession | None:
        """Load a session by identifier.

        Args:
            session_id: The session identifier.

        Returns:
            The session, or ``None`` if not found.
        """

    @abstractmethod
    async def list(self) -> list[AgentSession]:
        """Return all persisted sessions."""

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a persisted session.

        Args:
            session_id: The session identifier.

        Returns:
            ``True`` if a record was removed.
        """


class InMemorySessionStore(SessionStore):
    """Deterministic in-memory session storage (default)."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.RLock()

    async def save(self, session: AgentSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    async def load(self, session_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    async def list(self) -> list[AgentSession]:
        with self._lock:
            return list(self._sessions.values())

    async def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


class JsonSessionStore(SessionStore):
    """JSON file-per-session persistence.

    Sessions are written to ``directory/<session_id>.json``.  The store
    is thread-safe and creates the directory on demand.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._lock = threading.RLock()

    @property
    def directory(self) -> Path:
        """Return the directory holding session files."""
        return self._directory

    def _path_for(self, session_id: str) -> Path:
        return self._directory / f"{session_id}.json"

    async def save(self, session: AgentSession) -> None:
        payload = json_safe(session.model_dump(mode="python"))
        with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            path = self._path_for(session.session_id)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    async def load(self, session_id: str) -> AgentSession | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return AgentSession.model_validate(data)

    async def list(self) -> list[AgentSession]:
        with self._lock:
            if not self._directory.exists():
                return []
            sessions: list[AgentSession] = []
            for path in sorted(self._directory.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    sessions.append(AgentSession.model_validate(data))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Skipping malformed session file %s: %s", path, exc)
            return sessions

    async def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True
