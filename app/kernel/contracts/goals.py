"""Goal → Planner contract — stage 3 to stage 4.

Request carries the root ``Goal`` to plan for.
Response carries the generated ``Plan``.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.goals.models import Goal
from app.kernel.planning.models import Plan


class GoalToPlannerRequest(ContractRequest):
    """GoalEngine → PlanningEngine request."""

    goal: Goal = Field(..., description="Goal to create a plan for")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.GOAL
        if "target_engine" not in data:
            data["target_engine"] = EngineType.PLANNER
        super().__init__(**data)


class GoalToPlannerResponse(ContractResponse):
    """GoalEngine ← PlanningEngine response."""

    plan: Plan | None = Field(default=None, description="Generated plan")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.PLANNER
        if "target_engine" not in data:
            data["target_engine"] = EngineType.GOAL
        super().__init__(**data)
