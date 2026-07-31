"""Policy rule interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.kernel.policy.models import PolicyDecision, PolicyAction


class PolicyRule(ABC):
    """Evaluates a single security policy rule against an action."""

    @property
    def priority(self) -> int:
        return 0

    @abstractmethod
    async def evaluate(self, action: PolicyAction) -> PolicyDecision:
        """Evaluate *action* against this policy rule.

        Args:
            action: The action to evaluate.

        Returns:
            The policy verdict for this rule.
        """
