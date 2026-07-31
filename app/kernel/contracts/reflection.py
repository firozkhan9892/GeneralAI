"""Reflection → Experience contract — stage 11 to stage 12.

Request carries the ``ReflectionReport`` and optional session data to persist.
Response carries recorded ``Experience`` id and status.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.experience.models import Experience
from app.kernel.reflection.models import ReflectionReport


class ReflectionToExperienceRequest(ContractRequest):
    """ReflectionEngine → ExperienceEngine request."""

    reflection_report: ReflectionReport = Field(
        ..., description="Reflection report to record"
    )
    session_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional session context"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.REFLECTION
        if "target_engine" not in data:
            data["target_engine"] = EngineType.EXPERIENCE
        super().__init__(**data)


class ReflectionToExperienceResponse(ContractResponse):
    """ReflectionEngine ← ExperienceEngine response."""

    experience: Experience | None = Field(
        default=None, description="Recorded experience"
    )
    experience_id: str = Field(
        default="", description="ID of the stored experience record"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.EXPERIENCE
        if "target_engine" not in data:
            data["target_engine"] = EngineType.REFLECTION
        super().__init__(**data)
