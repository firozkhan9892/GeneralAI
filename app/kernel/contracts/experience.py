"""Experience → Memory contract — stage 12 to storage.

Request carries the ``Experience`` to persist or a query to retrieve.
Response carries the persisted/retrieved experience data.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.experience.models import Experience, ExperienceQuery


class ExperienceToMemoryRequest(ContractRequest):
    """ExperienceEngine → MemoryStore request."""

    experience: Experience | None = Field(
        default=None, description="Experience to persist (None for query-only)"
    )
    query: ExperienceQuery | None = Field(
        default=None, description="Optional query to retrieve experiences"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.EXPERIENCE
        if "target_engine" not in data:
            data["target_engine"] = EngineType.MEMORY
        super().__init__(**data)


class ExperienceToMemoryResponse(ContractResponse):
    """ExperienceEngine ← MemoryStore response."""

    stored_id: str | None = Field(default=None, description="ID of stored experience")
    experiences: tuple[Experience, ...] = Field(
        default_factory=tuple, description="Retrieved experiences"
    )
    total_count: int = Field(default=0, ge=0, description="Total matching records")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.MEMORY
        if "target_engine" not in data:
            data["target_engine"] = EngineType.EXPERIENCE
        super().__init__(**data)
