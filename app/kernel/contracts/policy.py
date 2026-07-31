"""Policy → Task contract — stage 8 to stage 9.

Request carries the approved ``PolicyDecision``.
Response carries the ``TaskResult`` from execution.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.policy.models import PolicyDecision
from app.kernel.tasks.models import TaskResult


class PolicyToTaskRequest(ContractRequest):
    """PolicyEngine → TaskEngine request."""

    policy_decision: PolicyDecision = Field(
        ..., description="Approved policy decision to execute"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.POLICY
        if "target_engine" not in data:
            data["target_engine"] = EngineType.TASK
        super().__init__(**data)


class PolicyToTaskResponse(ContractResponse):
    """PolicyEngine ← TaskEngine response."""

    task_result: TaskResult | None = Field(default=None, description="Execution result")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.TASK
        if "target_engine" not in data:
            data["target_engine"] = EngineType.POLICY
        super().__init__(**data)
