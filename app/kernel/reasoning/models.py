"""Reasoning domain models — stage 5 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReasoningStrategy(str, Enum):
    """Named reasoning strategy used by the engine."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    REFLEXION = "reflexion"
    STRAW_MAN = "straw_man"
    FIRST_PRINCIPLES = "first_principles"
    ANALOGICAL = "analogical"
    DECOMPOSITION = "decomposition"


class StepType(str, Enum):
    """Type or role of a single reasoning step."""

    THINK = "think"
    OBSERVE = "observe"
    ACT = "act"
    EVALUATE = "evaluate"
    SEARCH = "search"
    CALCULATE = "calculate"


class ReasoningStep(BaseModel):
    """A single atomic step within a reasoning trace."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique step identifier")
    type: StepType = Field(default=StepType.THINK, description="Step type")
    content: str = Field(default="", description="Step content / output")
    token_cost: int = Field(default=0, ge=0, description="Tokens consumed by this step")
    children: tuple[str, ...] = Field(
        default_factory=tuple, description="Child step IDs for tree structure"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Step-level metadata"
    )


class ReasoningRequest(BaseModel):
    """Input to the reasoning engine specifying the problem and constraints."""

    model_config = ConfigDict(frozen=True)

    problem: str = Field(..., description="The problem statement to reason about")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Supporting context"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Hard constraints on the solution"
    )
    strategy: ReasoningStrategy = Field(
        default=ReasoningStrategy.CHAIN_OF_THOUGHT, description="Reasoning strategy"
    )
    max_steps: int = Field(
        default=20, ge=1, le=100, description="Maximum allowed steps"
    )
    token_budget: int = Field(
        default=2000, ge=1, description="Maximum tokens for reasoning"
    )


class ReasoningTrace(BaseModel):
    """Complete output from the reasoning engine."""

    model_config = ConfigDict(frozen=True)

    steps: tuple[ReasoningStep, ...] = Field(
        default_factory=tuple, description="Ordered reasoning steps"
    )
    conclusion: str | None = Field(
        default=None, description="Final conclusion or answer"
    )
    strategy_used: ReasoningStrategy = Field(
        default=ReasoningStrategy.CHAIN_OF_THOUGHT, description="Strategy employed"
    )
    token_cost: int = Field(default=0, ge=0, description="Total tokens consumed")
    duration_ms: int = Field(
        default=0, ge=0, description="Execution duration in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary trace metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the trace was produced"
    )
