"""Capability → Policy contract — stage 7 to stage 8.

Request carries the capability resolution result.
Response carries the ``PolicyDecision`` (security verdict).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.capability.models import CapabilityResult
from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.policy.models import PolicyDecision


class CapabilityToPolicyRequest(ContractRequest):
    """CapabilityManager → PolicyEngine request."""

    capability_result: CapabilityResult = Field(
        ..., description="Capability resolution to evaluate against policy"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.CAPABILITY
        if "target_engine" not in data:
            data["target_engine"] = EngineType.POLICY
        super().__init__(**data)


class CapabilityToPolicyResponse(ContractResponse):
    """CapabilityManager ← PolicyEngine response."""

    policy_decision: PolicyDecision | None = Field(
        default=None, description="Policy evaluation verdict"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.POLICY
        if "target_engine" not in data:
            data["target_engine"] = EngineType.CAPABILITY
        super().__init__(**data)
