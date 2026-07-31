"""Session registry for the agent manager.

A thread-safe registry of :class:`AgentSession` snapshots with optional
backing :class:`SessionStore` persistence.  The registry owns the
session records; the manager owns the running tasks and cancellation
tokens.
"""

from __future__ import annotations

import logging
import threading

from app.agents.exceptions import (
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from app.agents.models import AgentSession, SessionStatus
from app.agents.persistence import InMemorySessionStore, SessionStore

log = logging.getLogger(__name__)


class SessionRegistry:
    """Thread-safe store of managed agent sessions.

    Args:
        store: Optional backing :class:`SessionStore` used for
            persistence.  Defaults to in-memory storage.
    """

    def __init__(self, store: SessionStore | None = None) -> None:
        self._store = store or InMemorySessionStore()
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.RLock()

    @property
    def store(self) -> SessionStore:
        """Return the backing persistence store."""
        return self._store

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, session: AgentSession) -> AgentSession:
        """Register a new session, raising if the id is taken.

        Args:
            session: The session to register.

        Returns:
            The registered session.

        Raises:
            SessionAlreadyExistsError: If ``session.session_id`` exists.
        """
        with self._lock:
            if session.session_id in self._sessions:
                raise SessionAlreadyExistsError(
                    f"Session '{session.session_id}' already exists",
                    module="agents.registry",
                )
            self._sessions[session.session_id] = session
        log.debug("Registered session %s", session.session_id)
        return session

    def update(self, session: AgentSession) -> AgentSession:
        """Upsert a session snapshot.

        Args:
            session: The updated session.

        Returns:
            The stored session.
        """
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> AgentSession | None:
        """Return a session, or ``None`` if absent."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_raise(self, session_id: str) -> AgentSession:
        """Return a session, raising if absent.

        Args:
            session_id: The session identifier.

        Returns:
            The session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        session = self.get(session_id)
        if session is None:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found",
                module="agents.registry",
            )
        return session

    def has(self, session_id: str) -> bool:
        """Return ``True`` if *session_id* is registered."""
        with self._lock:
            return session_id in self._sessions

    def count(self) -> int:
        """Return the number of registered sessions."""
        with self._lock:
            return len(self._sessions)

    def all(self) -> list[AgentSession]:
        """Return all sessions, newest first."""
        with self._lock:
            sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def list(
        self,
        *,
        status: SessionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentSession]:
        """Return sessions filtered by status with pagination.

        Args:
            status: Optional status filter.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip (newest first).

        Returns:
            Matching sessions, newest first.
        """
        sessions = self.all()
        if status is not None:
            sessions = [s for s in sessions if s.status == status]
        return sessions[offset : offset + limit]

    def remove(self, session_id: str) -> bool:
        """Remove a session from the in-memory registry.

        Args:
            session_id: The session identifier.

        Returns:
            ``True`` if a session was removed.
        """
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        """Remove all sessions from memory (does not touch the store)."""
        with self._lock:
            self._sessions.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def persist(self, session: AgentSession) -> None:
        """Persist a session snapshot to the backing store.

        Args:
            session: The session to persist.
        """
        await self._store.save(session)

    async def load_all(self) -> int:
        """Load all persisted sessions into the registry.

        Returns:
            The number of sessions loaded.
        """
        stored = await self._store.list()
        with self._lock:
            for session in stored:
                if session.session_id not in self._sessions:
                    self._sessions[session.session_id] = session
        log.info("Loaded %d persisted session(s)", len(stored))
        return len(stored)

    async def delete(self, session_id: str) -> bool:
        """Remove a session from memory and the backing store.

        Args:
            session_id: The session identifier.

        Returns:
            ``True`` if a session was removed.
        """
        removed = self.remove(session_id)
        if removed:
            await self._store.delete(session_id)
        return removed
