"""Decision → Capability contract — stage 6 to stage 7.

Request carries the ``Decision`` (selected action).
Response carries capability resolution outcome.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.capability.models import CapabilityResult
from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.decision.models import Decision


class DecisionToCapabilityRequest(ContractRequest):
    """DecisionEngine → CapabilityManager request."""

    decision: Decision = Field(
        ..., description="Decision whose action needs capability resolution"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.DECISION
        if "target_engine" not in data:
            data["target_engine"] = EngineType.CAPABILITY
        super().__init__(**data)


class DecisionToCapabilityResponse(ContractResponse):
    """DecisionEngine ← CapabilityManager response."""

    capability_result: CapabilityResult | None = Field(
        default=None, description="Capability resolution outcome"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.CAPABILITY
        if "target_engine" not in data:
            data["target_engine"] = EngineType.DECISION
        super().__init__(**data)
