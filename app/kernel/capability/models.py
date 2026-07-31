"""Capability domain models — stage 7 of the cognitive pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProviderType(str, Enum):
    """Category of a capability provider."""

    SKILL = "skill"
    TOOL = "tool"
    PLUGIN = "plugin"
    AGENT = "agent"
    SYSTEM = "system"


class CapabilityRequirement(BaseModel):
    """A declared requirement that some capability must exist."""

    model_config = ConfigDict(frozen=True)

    action_type: str = Field(..., description="The action that requires the capability")
    required_capability: str = Field(..., description="Name of the required capability")
    optional: bool = Field(
        default=False, description="If True, the system may proceed without it"
    )


class ResourceEstimate(BaseModel):
    """Estimated resource consumption for a capability invocation."""

    model_config = ConfigDict(frozen=True)

    tokens: int = Field(default=0, ge=0, description="Estimated token consumption")
    time_ms: int = Field(default=0, ge=0, description="Estimated execution time")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated monetary cost")


class CapabilityProvider(BaseModel):
    """Descriptor for a registered capability provider known to the system."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique provider name")
    provider_type: ProviderType = Field(..., description="Category of the provider")
    description: str = Field(default="", description="Human-readable description")
    health_status: str = Field(
        default="healthy", description="Current health (healthy, degraded, unavailable)"
    )
    required_permissions: tuple[str, ...] = Field(
        default_factory=tuple, description="Permissions needed to invoke"
    )
    requires_user_approval: bool = Field(
        default=False, description="Whether user must approve before invocation"
    )


class Capability(BaseModel):
    """A resolved capability that the system can execute."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Capability name")
    provider: CapabilityProvider = Field(
        ..., description="The provider that fulfills this capability"
    )
    available: bool = Field(
        default=True, description="Whether the capability is currently available"
    )
    estimated_cost: ResourceEstimate = Field(
        default_factory=ResourceEstimate, description="Cost estimate"
    )
    unavailability_reason: str | None = Field(
        default=None, description="Why the capability is unavailable"
    )


class CapabilityResult(BaseModel):
    """Outcome of a capability resolution / lookup operation."""

    model_config = ConfigDict(frozen=True)

    resolved: tuple[Capability, ...] = Field(
        default_factory=tuple, description="Successfully resolved capabilities"
    )
    missing: tuple[CapabilityRequirement, ...] = Field(
        default_factory=tuple, description="Unresolved requirements"
    )
    total_estimated_cost: ResourceEstimate = Field(
        default_factory=ResourceEstimate, description="Aggregate cost"
    )
