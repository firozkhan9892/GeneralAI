"""Policy domain models — stage 8 of the cognitive pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VerdictType(str, Enum):
    """Possible verdicts from policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"
    SANDBOX = "sandbox"


class PolicyRule(BaseModel):
    """A single policy rule — the atomic unit of the policy engine."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique rule name")
    description: str = Field(default="", description="Human-readable rule description")
    action_pattern: str = Field(
        ..., description="Glob/pattern matching action types this rule applies to"
    )
    verdict: VerdictType = Field(
        ..., description="Verdict to apply when this rule matches"
    )
    priority: int = Field(
        default=0, description="Rule priority (higher = evaluated first)"
    )
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="Additional matching conditions"
    )
    denial_reason: str | None = Field(
        default=None, description="Reason shown on denial"
    )


class PolicyAction(BaseModel):
    """The action being evaluated by the policy engine."""

    model_config = ConfigDict(frozen=True)

    action_type: str = Field(..., description="Type of action")
    tool_name: str | None = Field(
        default=None, description="Target tool name if applicable"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action parameters for inspection"
    )
    domain: str | None = Field(
        default=None, description="Operational domain (filesystem, network, etc.)"
    )
    session_id: str = Field(default="", description="Owning session identifier")


class AppliedPolicy(BaseModel):
    """Record of a single policy rule that was applied during evaluation."""

    model_config = ConfigDict(frozen=True)

    policy_name: str = Field(..., description="Name of the applied rule")
    verdict: VerdictType = Field(..., description="Verdict produced")
    rationale: str = Field(default="", description="Why this rule matched")


class SandboxConfig(BaseModel):
    """Constraints for sandboxed execution of an action."""

    model_config = ConfigDict(frozen=True)

    allow_network: bool = Field(
        default=False, description="Allow outbound network access"
    )
    allow_filesystem: bool = Field(
        default=False, description="Allow filesystem write access"
    )
    allow_subprocess: bool = Field(
        default=False, description="Allow subprocess creation"
    )
    memory_limit_mb: int = Field(
        default=256, ge=0, description="Memory limit in megabytes"
    )
    time_limit_s: int = Field(default=30, ge=0, description="Time limit in seconds")


class PolicyDecision(BaseModel):
    """The result of policy evaluation — the system's security verdict."""

    model_config = ConfigDict(frozen=True)

    action: PolicyAction = Field(..., description="The action that was evaluated")
    verdict: VerdictType = Field(..., description="Overall verdict")
    enforced_policies: tuple[AppliedPolicy, ...] = Field(
        default_factory=tuple, description="Rules that matched"
    )
    rationale: str = Field(default="", description="Combined rationale for the verdict")
    confirmation_id: str | None = Field(
        default=None, description="If CONFIRM, the ID for the user to confirm"
    )
    sandbox_config: SandboxConfig | None = Field(
        default=None, description="If SANDBOX, the sandbox constraints"
    )
    denial_reason: str | None = Field(
        default=None, description="If DENY, the reason for denial"
    )
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the evaluation occurred"
    )
