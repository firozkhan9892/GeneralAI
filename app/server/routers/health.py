"""Health and metrics endpoints (public — no API key required)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.server.dependencies import (
    get_agent_manager,
    get_memory_engine,
    get_settings,
    get_tool_registry,
)
from app.server.schemas import HealthResponse, MetricsResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings=Depends(get_settings),
    manager=Depends(get_agent_manager),
) -> HealthResponse:
    """Return a lightweight liveness response.

    This endpoint is publicly accessible (no API key required).
    """
    return HealthResponse(
        status="ok",
        app_name=settings.title,
        version=settings.version,
        sessions_active=manager.active_count(),
        sessions_total=manager.registry.count(),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(
    request: Request,
    settings=Depends(get_settings),
    manager=Depends(get_agent_manager),
    memory=Depends(get_memory_engine),
    registry=Depends(get_tool_registry),
) -> MetricsResponse:
    """Return runtime metrics.

    This endpoint is publicly accessible (no API key required).
    """
    metrics_collector = request.app.state.metrics
    snapshot = metrics_collector.snapshot()
    return MetricsResponse(
        requests_total=snapshot["requests_total"],
        errors_total=snapshot["errors_total"],
        requests_by_path=snapshot["requests_by_path"],
        sessions_active=manager.active_count(),
        sessions_total=manager.registry.count(),
        memory_records=await memory.count(),
        tools_count=registry.count,
    )
