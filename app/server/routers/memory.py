"""Memory search endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.kernel.memory.models import MemoryQuery, MemoryTier
from app.server.dependencies import get_memory_engine
from app.server.schemas import MemorySearchResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/search", response_model=MemorySearchResponse, summary="Search memory")
async def memory_search(
    q: str = Query(..., min_length=1, description="Search keywords"),
    session_id: str | None = Query(default=None, description="Filter by session"),
    tier: str | None = Query(
        default=None, description="Filter by tier (short_term|long_term)"
    ),
    limit: int = Query(default=10, ge=1, le=100, description="Max results"),
    memory=Depends(get_memory_engine),
) -> MemorySearchResponse:
    """Search stored memory records by keywords, ranked by relevance."""
    tier_enum: MemoryTier | None = None
    if tier is not None:
        try:
            tier_enum = MemoryTier(tier)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier '{tier}'. Must be one of {[t.value for t in MemoryTier]}.",
            ) from exc
    keywords = tuple(q.split())
    query = MemoryQuery(
        keywords=keywords,
        session_id=session_id,
        tier=tier_enum,
        limit=limit,
    )
    hits = await memory.search(query)
    return MemorySearchResponse(query=q, total=len(hits), hits=hits)
