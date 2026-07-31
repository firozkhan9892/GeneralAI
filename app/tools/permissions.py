"""Permission system for tool invocations.

Rules are matched by tool name glob patterns and evaluated in
registration order; the first match wins.  When sandbox mode is enabled,
any tool that does not match an ``allow`` rule is denied by default.
"""

from __future__ import annotations

import fnmatch
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PermissionDecision(str, Enum):
    """Outcome of a permission check."""

    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


class PermissionResult(BaseModel):
    """Result of checking a tool invocation against the policy."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Tool name that was checked")
    decision: PermissionDecision = Field(
        ..., description="Whether the invocation is allowed"
    )
    reason: str = Field(default="", description="Human-readable explanation")


class PermissionSystem:
    """Declarative allow / deny / confirm policy for tools.

    Rules are added via :meth:`allow`, :meth:`deny`, and
    :meth:`confirm`, each taking a glob pattern for tool names.  Rules
    are evaluated in the order they were added and the first match
    decides the outcome.
    """

    def __init__(
        self,
        *,
        default: PermissionDecision = PermissionDecision.ALLOW,
        sandbox: bool = False,
    ) -> None:
        self._rules: list[tuple[str, PermissionDecision, str]] = []
        self._default = default
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Policy configuration
    # ------------------------------------------------------------------

    def allow(self, pattern: str, reason: str = "") -> None:
        """Allow tools whose names match *pattern*."""
        self._rules.append((pattern, PermissionDecision.ALLOW, reason))

    def deny(self, pattern: str, reason: str = "") -> None:
        """Deny tools whose names match *pattern*."""
        self._rules.append((pattern, PermissionDecision.DENY, reason))

    def confirm(self, pattern: str, reason: str = "") -> None:
        """Require confirmation for tools whose names match *pattern*."""
        self._rules.append((pattern, PermissionDecision.CONFIRM, reason))

    def set_default(self, decision: PermissionDecision) -> None:
        """Set the decision applied when no rule matches."""
        self._default = decision

    @property
    def sandbox(self) -> bool:
        """Return whether sandbox mode is enabled."""
        return self._sandbox

    def enable_sandbox(self) -> None:
        """Enable sandbox mode: unmatched tools are denied."""
        self._sandbox = True

    def disable_sandbox(self) -> None:
        """Disable sandbox mode."""
        self._sandbox = False

    def clear(self) -> None:
        """Remove all rules (keeps default and sandbox state)."""
        self._rules.clear()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def check(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> PermissionResult:
        """Evaluate the policy for a tool invocation.

        Args:
            tool_name: The tool being invoked.
            arguments: The invocation arguments (currently informational).

        Returns:
            The permission outcome.
        """
        del arguments
        for pattern, decision, reason in self._rules:
            if fnmatch.fnmatchcase(tool_name, pattern):
                return PermissionResult(
                    tool_name=tool_name,
                    decision=decision,
                    reason=reason,
                )
        if self._sandbox:
            return PermissionResult(
                tool_name=tool_name,
                decision=PermissionDecision.DENY,
                reason="Sandbox mode denies unmatched tools",
            )
        return PermissionResult(tool_name=tool_name, decision=self._default)
