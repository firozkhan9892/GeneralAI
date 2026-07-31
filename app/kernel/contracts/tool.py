"""Tool → Reflection contract — stage 10 to stage 11.

Request carries the ``ToolResult`` to evaluate.
Response carries the ``ReflectionReport`` (quality assessment).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.reflection.models import ReflectionReport
from app.kernel.tools.models import ToolResult


class ToolToReflectionRequest(ContractRequest):
    """ToolEngine → ReflectionEngine request."""

    tool_result: ToolResult = Field(..., description="Tool result to reflect on")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.TOOL
        if "target_engine" not in data:
            data["target_engine"] = EngineType.REFLECTION
        super().__init__(**data)


class ToolToReflectionResponse(ContractResponse):
    """ToolEngine ← ReflectionEngine response."""

    reflection_report: ReflectionReport | None = Field(
        default=None, description="Reflection quality report"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.REFLECTION
        if "target_engine" not in data:
            data["target_engine"] = EngineType.TOOL
        super().__init__(**data)
