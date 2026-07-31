"""Chat endpoint with optional SSE streaming."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.kernel.agent.models import AgentRequest, AgentRunConfig
from app.server.dependencies import get_agent_manager
from app.server.schemas import ChatRequest, ChatResponse
from app.server.streaming import poll_session, sse_format

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _build_request(
    body: ChatRequest, settings
) -> tuple[AgentRequest, AgentRunConfig | None]:
    session_id = body.session_id or ""
    request = AgentRequest(
        raw_input=body.message,
        session_id=session_id,
        user_id=body.user_id,
    )
    config = body.config
    if config is not None and not config.session_id and session_id:
        config = config.model_copy(update={"session_id": session_id})
    return request, config


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    manager=Depends(get_agent_manager),
) -> ChatResponse:
    """Execute a chat request and return the final response.

    The agent runs to completion before the response is returned.
    """
    session = await manager.execute(
        AgentRequest(
            raw_input=body.message, session_id=body.session_id, user_id=body.user_id
        ),
        config=body.config,
        options=body.options,
    )
    session = await manager.await_completion(session.session_id)
    content = session.response.output.content if session.response else ""
    return ChatResponse(
        session_id=session.session_id,
        status=session.status.value,
        content=content,
        success=session.status.value == "completed",
        error=session.error,
    )


@router.post("/stream", response_model=None)
async def chat_stream(
    body: ChatRequest,
    manager=Depends(get_agent_manager),
) -> StreamingResponse:
    """Execute a chat request and stream status updates via SSE.

    Emits ``session.started``, ``session.status`` (on every status change),
    and ``session.completed`` (with the final snapshot) events.
    """
    session = await manager.execute(
        AgentRequest(
            raw_input=body.message, session_id=body.session_id, user_id=body.user_id
        ),
        config=body.config,
        options=body.options,
    )

    async def _generator():
        yield sse_format("session.started", {"session_id": session.session_id})
        async for event, payload in poll_session(manager, session.session_id):
            yield sse_format(event, payload)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
