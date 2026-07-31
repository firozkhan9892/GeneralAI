"""Intent → Goal contract — stage 2 to stage 3.

Request carries the resolved ``Intent``.
Response carries the ``GoalHierarchy`` derived from that intent.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.goals.models import GoalHierarchy
from app.kernel.intent.models import Intent


class IntentToGoalRequest(ContractRequest):
    """IntentEngine → GoalEngine request."""

    intent: Intent = Field(..., description="Resolved user intent to decompose")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.INTENT
        if "target_engine" not in data:
            data["target_engine"] = EngineType.GOAL
        super().__init__(**data)


class IntentToGoalResponse(ContractResponse):
    """IntentEngine ← GoalEngine response."""

    goal_hierarchy: GoalHierarchy | None = Field(
        default=None, description="Goal hierarchy"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.GOAL
        if "target_engine" not in data:
            data["target_engine"] = EngineType.INTENT
        super().__init__(**data)
