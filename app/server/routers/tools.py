"""Tool execution endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.server.dependencies import get_tool_executor, get_tool_registry
from app.server.schemas import ToolRunRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post("/run", summary="Execute a tool")
async def tool_run(
    body: ToolRunRequest,
    executor=Depends(get_tool_executor),
    registry=Depends(get_tool_registry),
) -> dict:
    """Run a registered tool synchronously and return the result."""
    if not registry.has(body.tool):
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{body.tool}' is not registered",
        )
    result = await executor.execute_async(
        body.tool,
        body.arguments or {},
        timeout_s=body.timeout_s,
        max_retries=body.max_retries,
    )
    return result
