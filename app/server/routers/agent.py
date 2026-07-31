"""Agent lifecycle endpoints and WebSocket."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from app.agents.exceptions import SessionNotFoundError
from app.kernel.agent.models import AgentRequest
from app.server.dependencies import get_agent_manager
from app.server.schemas import (
    AgentCancelRequest,
    AgentListResponse,
    AgentRunRequest,
)
from app.server.streaming import poll_session

log = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run")
async def agent_run(
    body: AgentRunRequest,
    manager=Depends(get_agent_manager),
) -> dict:
    """Execute an agent request.

    When ``wait`` is ``True`` (default), the endpoint blocks until
    the session reaches a terminal state and returns the final
    snapshot.  When ``False``, it returns immediately with the
    ``RUNNING`` session so the caller can poll ``/agent/status/{id}``.
    """
    session = await manager.execute(
        AgentRequest(
            raw_input=body.raw_input,
            session_id=body.session_id,
            user_id=body.user_id,
        ),
        config=body.config,
        options=body.options,
    )
    if body.wait:
        session = await manager.await_completion(session.session_id)
    return session.model_dump(mode="json")


@router.post("/cancel")
async def agent_cancel(
    body: AgentCancelRequest,
    manager=Depends(get_agent_manager),
) -> dict:
    """Cancel a running or paused session."""
    try:
        session = await manager.cancel(body.session_id, reason=body.reason)
        return session.model_dump(mode="json")
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc


@router.get("/status/{session_id}", summary="Get session status")
async def agent_status(
    session_id: str,
    manager=Depends(get_agent_manager),
) -> dict:
    """Return the current snapshot of *session_id*."""
    try:
        return manager.get_status(session_id).model_dump(mode="json")
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc


@router.get("/agents", response_model=AgentListResponse, summary="List sessions")
async def list_sessions(
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    manager=Depends(get_agent_manager),
) -> AgentListResponse:
    """Return tracked sessions, newest first.

    *status* filters by lifecycle state (e.g. ``running``, ``completed``).
    """
    from app.agents.models import SessionStatus

    status_enum: SessionStatus | None = None
    if status is not None:
        try:
            status_enum = SessionStatus(status)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of {[s.value for s in SessionStatus]}.",
            ) from exc
    sessions = manager.list_sessions(status=status_enum, limit=limit, offset=offset)
    return AgentListResponse(total=len(sessions), sessions=sessions)


@router.websocket("/ws")
async def agent_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent interaction.

    Supported messages:
    - ``{"type": "run", "raw_input": "...", "session_id": "..."}``
    - ``{"type": "cancel", "session_id": "...", "reason": "..."}``
    - ``{"type": "status", "session_id": "..."}``
    - ``{"type": "ping"}``
    """
    # Check API key manually for WebSocket if needed
    settings = websocket.app.state.settings
    if settings.api_key:
        api_key = websocket.headers.get("x-api-key") or websocket.query_params.get(
            "api_key"
        )
        if api_key != settings.api_key:
            await websocket.close(code=4001, reason="Invalid API key")
            return

    manager = websocket.app.state.agent_manager
    await websocket.accept()
    try:
        await websocket.send_json({"type": "connected"})
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            mtype = raw.get("type")
            if mtype == "ping":
                await websocket.send_json({"type": "pong"})
            elif mtype == "run":
                raw_input = raw.get("raw_input", "")
                if not raw_input:
                    await websocket.send_json(
                        {"type": "error", "detail": "raw_input is required"}
                    )
                    continue
                session_id = raw.get("session_id", "")
                try:
                    session = await manager.execute(
                        AgentRequest(
                            raw_input=raw_input,
                            session_id=session_id,
                            user_id=raw.get("user_id"),
                        )
                    )
                    await websocket.send_json(
                        {
                            "type": "session.started",
                            "session_id": session.session_id,
                            "status": session.status.value,
                        }
                    )
                    async for event, payload in poll_session(
                        manager, session.session_id
                    ):
                        if event == "session.completed":
                            await websocket.send_json(
                                {"type": "session.completed", "session": payload}
                            )
                        else:
                            await websocket.send_json({"type": event, **payload})
                except Exception as exc:
                    log.exception("WebSocket run failed")
                    await websocket.send_json({"type": "error", "detail": str(exc)})
            elif mtype == "cancel":
                session_id = raw.get("session_id", "")
                if not session_id:
                    await websocket.send_json(
                        {"type": "error", "detail": "session_id is required"}
                    )
                    continue
                try:
                    session = await manager.cancel(
                        session_id, reason=raw.get("reason", "user_requested")
                    )
                    await websocket.send_json(
                        {
                            "type": "session.cancelled",
                            "session": session.model_dump(mode="json"),
                        }
                    )
                except SessionNotFoundError as exc:
                    await websocket.send_json({"type": "error", "detail": exc.message})
            elif mtype == "status":
                session_id = raw.get("session_id", "")
                if not session_id:
                    await websocket.send_json(
                        {"type": "error", "detail": "session_id is required"}
                    )
                    continue
                try:
                    session = manager.get_session(session_id)
                    await websocket.send_json(
                        {
                            "type": "session.status",
                            "session": session.model_dump(mode="json")
                            if session
                            else None,
                        }
                    )
                except SessionNotFoundError as exc:
                    await websocket.send_json({"type": "error", "detail": exc.message})
            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unknown message type '{mtype}'"}
                )
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
