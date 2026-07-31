"""Streaming helpers for SSE and WebSocket consumers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import asyncio
import json
from typing import Any

from app.agents.manager import AgentManager
from app.agents.models import SessionStatus

SSE_SEPARATOR = "\n"


def sse_format(event: str, data: object) -> str:
    """Format a single SSE frame."""
    return f"event: {event}{SSE_SEPARATOR}data: {json.dumps(data, default=str)}{SSE_SEPARATOR}{SSE_SEPARATOR}"


def session_to_payload(session) -> dict:
    """Return a JSON-safe snapshot of a session."""
    return session.model_dump(mode="json")


async def poll_session(
    manager: AgentManager,
    session_id: str,
    *,
    interval: float = 0.05,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Yield status-change and terminal events for a session until it reaches a terminal state.

    Each yielded value is a ``(event_name, payload)`` tuple suitable for SSE or WebSocket framing.
    """
    last_status: SessionStatus | None = None
    while True:
        session = manager.get_session(session_id)
        if session is None:
            yield ("session.error", {"detail": f"Session '{session_id}' not found"})
            return
        if session.status != last_status:
            last_status = session.status
            yield (
                "session.status",
                {"session_id": session_id, "status": session.status.value},
            )
        if session.is_terminal or session.status == SessionStatus.PAUSED:
            yield ("session.completed", session_to_payload(session))
            return
        await asyncio.sleep(interval)
