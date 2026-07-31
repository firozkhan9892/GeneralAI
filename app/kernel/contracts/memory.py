"""Memory → Response contract — stage 13 to stage 15.

Request carries final session data, context, and optional experience.
Response carries the ``OutputMessage`` for the caller.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.experience.models import Experience
from app.kernel.response.models import OutputMessage


class MemoryToResponseRequest(ContractRequest):
    """MemoryStore → ResponseBuilder request."""

    session_data: dict[str, Any] = Field(
        default_factory=dict, description="Final session state data"
    )
    experience: Experience | None = Field(
        default=None, description="Associated experience record"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.MEMORY
        if "target_engine" not in data:
            data["target_engine"] = EngineType.RESPONSE
        super().__init__(**data)


class MemoryToResponseResponse(ContractResponse):
    """MemoryStore ← ResponseBuilder response."""

    output: OutputMessage | None = Field(
        default=None, description="Final output message"
    )
    stream_id: str | None = Field(default=None, description="Stream ID if streaming")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.RESPONSE
        if "target_engine" not in data:
            data["target_engine"] = EngineType.MEMORY
        super().__init__(**data)
