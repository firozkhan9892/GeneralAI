"""Planning domain models — stage 4 of the cognitive pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlanningStrategy(str, Enum):
    """High-level strategy used to construct a plan."""

    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"
    MEANS_END = "means_end"
    CASE_BASED = "case_based"
    REACTIVE = "reactive"


class SkillStep(BaseModel):
    """A single step in a plan representing a skill invocation."""

    model_config = ConfigDict(frozen=True)

    order: int = Field(..., ge=0, description="Step sequence number")
    skill_name: str = Field(..., description="Name of the skill to invoke")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters for the skill"
    )
    dependencies: tuple[int, ...] = Field(
        default_factory=tuple, description="Step orders this step depends on"
    )
    estimated_tokens: int = Field(default=0, ge=0, description="Estimated token cost")
    description: str = Field(default="", description="Human-readable step description")


class DependencyGraph(BaseModel):
    """A directed acyclic graph describing step dependencies."""

    model_config = ConfigDict(frozen=True)

    edges: tuple[tuple[int, int], ...] = Field(
        default_factory=tuple, description="(dependent, dependency) ordered pairs"
    )


class Plan(BaseModel):
    """An executable plan composed of ordered skill steps."""

    model_config = ConfigDict(frozen=True)

    goal_id: str = Field(default="", description="ID of the goal this plan serves")
    strategy: PlanningStrategy = Field(
        default=PlanningStrategy.TOP_DOWN, description="Strategy used to build the plan"
    )
    steps: tuple[SkillStep, ...] = Field(
        default_factory=tuple, description="Ordered execution steps"
    )
    dependencies: DependencyGraph = Field(
        default_factory=DependencyGraph, description="Step dependency DAG"
    )
    estimated_total_tokens: int = Field(
        default=0, ge=0, description="Sum of step token estimates"
    )
    revision: int = Field(
        default=0, ge=0, description="Revision counter (incremented on each revise)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extensible plan metadata"
    )
