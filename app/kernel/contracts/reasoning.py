"""Reasoning → Decision contract — stage 5 to stage 6.

Request carries the ``ReasoningTrace`` output.
Response carries the ``Decision`` (selected action).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.decision.models import Decision
from app.kernel.reasoning.models import ReasoningTrace


class ReasoningToDecisionRequest(ContractRequest):
    """ReasoningEngine → DecisionEngine request."""

    trace: ReasoningTrace = Field(..., description="Reasoning trace to decide on")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.REASONING
        if "target_engine" not in data:
            data["target_engine"] = EngineType.DECISION
        super().__init__(**data)


class ReasoningToDecisionResponse(ContractResponse):
    """ReasoningEngine ← DecisionEngine response."""

    decision: Decision | None = Field(default=None, description="Selected decision")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.DECISION
        if "target_engine" not in data:
            data["target_engine"] = EngineType.REASONING
        super().__init__(**data)
