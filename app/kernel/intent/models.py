"""Intent domain models — stage 2 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntentType(str, Enum):
    """Taxonomy of supported user intents.

    Each value represents a high-level category of what the user
    wants the system to do.
    """

    ASK_QUESTION = "ask_question"
    SOLVE_PROBLEM = "solve_problem"
    EXECUTE_TASK = "execute_task"
    PLAN_PROJECT = "plan_project"
    LEARN = "learn"
    CREATE_CONTENT = "create_content"
    EXPLORE = "explore"
    DEBUG = "debug"
    META = "meta"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


class IntentConfidence(BaseModel):
    """Confidence scores for an intent classification."""

    model_config = ConfigDict(frozen=True)

    primary: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the primary classification"
    )
    alternatives: tuple[tuple[IntentType, float], ...] = Field(
        default_factory=tuple, description="Alternative classifications with scores"
    )
    ambiguity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Below this → request clarification"
    )


class IntentClassification(BaseModel):
    """Raw result of intent classification before human-friendly enrichment."""

    model_config = ConfigDict(frozen=True)

    primary: IntentType = Field(..., description="Primary classified intent")
    confidence: IntentConfidence = Field(..., description="Confidence details")
    classifier_name: str = Field(default="", description="Name of the classifier used")


class ClarificationRequest(BaseModel):
    """A request for user clarification when intent is ambiguous."""

    model_config = ConfigDict(frozen=True)

    ambiguity_description: str = Field(..., description="What is unclear")
    options: tuple[str, ...] = Field(
        default_factory=tuple, description="Suggested clarification options"
    )
    freeform_prompt: str = Field(
        default="", description="Open-ended clarification prompt"
    )


class Intent(BaseModel):
    """Fully resolved, structured user intent after understanding."""

    model_config = ConfigDict(frozen=True)

    primary: IntentType = Field(..., description="Primary intent type")
    confidence: IntentConfidence = Field(
        ..., description="Confidence of the classification"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Intent-specific extracted parameters"
    )
    sub_intents: tuple[Intent, ...] = Field(
        default_factory=tuple, description="Hierarchical sub-intents"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="User-provided constraints"
    )
    clarification: ClarificationRequest | None = Field(
        default=None, description="Pending clarification if ambiguous"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the intent was resolved"
    )
