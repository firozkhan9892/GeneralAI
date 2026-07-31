"""State manager — cross-cutting."""

from __future__ import annotations

import logging

from app.kernel.state.models import CognitiveState, SessionState

log = logging.getLogger(__name__)


class StateManager:
    """Manages the cognitive state machine per session."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    async def initialize(self, session_id: str) -> SessionState:
        """Initialize state for a new session.

        Args:
            session_id: The session identifier.

        Returns:
            Initial session state.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("StateManager.initialize not yet implemented")

    async def transition(
        self, session_id: str, new_state: CognitiveState
    ) -> SessionState:
        """Transition to a new state.

        Args:
            session_id: The session identifier.
            new_state: The target state.

        Returns:
            Updated session state.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("StateManager.transition not yet implemented")

    async def get_state(self, session_id: str) -> SessionState:
        """Get the current state of a session.

        Args:
            session_id: The session identifier.

        Returns:
            Current session state.
        """
        return self._sessions.get(
            session_id, SessionState(session_id=session_id, state=CognitiveState.IDLE)
        )
