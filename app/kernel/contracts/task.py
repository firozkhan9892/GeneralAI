"""Task → Tool contract — stage 9 to stage 10.

Request carries the ``Task`` to be executed.
Response carries the ``ToolResult`` from tool invocation.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.tasks.models import Task
from app.kernel.tools.models import ToolResult


class TaskToToolRequest(ContractRequest):
    """TaskEngine → ToolEngine request."""

    task: Task = Field(..., description="Task to execute via tool")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.TASK
        if "target_engine" not in data:
            data["target_engine"] = EngineType.TOOL
        super().__init__(**data)


class TaskToToolResponse(ContractResponse):
    """TaskEngine ← ToolEngine response."""

    tool_result: ToolResult | None = Field(
        default=None, description="Tool execution result"
    )

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.TOOL
        if "target_engine" not in data:
            data["target_engine"] = EngineType.TASK
        super().__init__(**data)
