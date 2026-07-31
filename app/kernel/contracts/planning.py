"""Planner → Reasoning contract — stage 4 to stage 5.

Request carries the ``Plan`` to reason about.
Response carries the ``ReasoningTrace`` produced.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.planning.models import Plan
from app.kernel.reasoning.models import ReasoningTrace


class PlannerToReasoningRequest(ContractRequest):
    """PlanningEngine → ReasoningEngine request."""

    plan: Plan = Field(..., description="Plan to reason about")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.PLANNER
        if "target_engine" not in data:
            data["target_engine"] = EngineType.REASONING
        super().__init__(**data)


class PlannerToReasoningResponse(ContractResponse):
    """PlanningEngine ← ReasoningEngine response."""

    trace: ReasoningTrace | None = Field(
        default=None, description="Reasoning trace produced"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.REASONING
        if "target_engine" not in data:
            data["target_engine"] = EngineType.PLANNER
        super().__init__(**data)
