"""Perception → Intent contract — stage 1 to stage 2.

Request carries the structured ``Percept`` from the perception engine.
Response carries the classified ``Intent``.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.kernel.contracts.base import ContractRequest, ContractResponse, EngineType
from app.kernel.intent.models import Intent
from app.kernel.perception.models import Percept


class PerceptionToIntentRequest(ContractRequest):
    """PerceptionEngine → IntentEngine request."""

    percept: Percept = Field(..., description="Structured percept to classify")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.PERCEPTION
        if "target_engine" not in data:
            data["target_engine"] = EngineType.INTENT
        super().__init__(**data)


class PerceptionToIntentResponse(ContractResponse):
    """PerceptionEngine ← IntentEngine response."""

    intent: Intent | None = Field(default=None, description="Classified intent")

    def __init__(self, **data: Any) -> None:
        if "source_engine" not in data:
            data["source_engine"] = EngineType.INTENT
        if "target_engine" not in data:
            data["target_engine"] = EngineType.PERCEPTION
        super().__init__(**data)
