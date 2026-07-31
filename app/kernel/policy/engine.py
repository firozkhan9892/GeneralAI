"""Policy engine — stage 8 of the cognitive pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.kernel.decision.models import Decision
from app.kernel.policy.models import (
    AppliedPolicy,
    PolicyAction,
    PolicyDecision,
    VerdictType,
)
from app.kernel.policy.rules import PolicyRule as PolicyRuleABC
from app.kernel.policy.rules.base import AllowAllRule, BuiltinDenyRules, ModelPolicyRule

if TYPE_CHECKING:
    from app.kernel.policy.models import PolicyRule as PolicyRuleModel

log = logging.getLogger(__name__)


class PolicyEngine:
    """Security and governance layer.

    Every action passes through policy validation before execution.
    Supports ALLOW, DENY, CONFIRM, and SANDBOX verdicts.
    """

    def __init__(self) -> None:
        self._rules: list[PolicyRuleABC] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        for rule in BuiltinDenyRules.create_all():
            self._rules.append(rule)
        self._rules.append(AllowAllRule())

    async def evaluate(self, decision: Decision) -> PolicyDecision:
        """Evaluate a decision against all registered policies.

        Args:
            decision: The decision whose selected action to evaluate.

        Returns:
            Policy verdict for the action.
        """
        action = PolicyAction(
            action_type=decision.selected_action.action_type,
            parameters=decision.selected_action.parameters,
            session_id=decision.session_id,
        )

        return await self._evaluate_action(action)

    async def _evaluate_action(self, action: PolicyAction) -> PolicyDecision:
        applied: list[AppliedPolicy] = []

        for rule in sorted(self._rules, key=lambda r: r.priority, reverse=True):
            result = await rule.evaluate(action)
            if result.verdict != VerdictType.ALLOW:
                applied.extend(result.enforced_policies)
                log.info(
                    "Policy → %s for action=%s",
                    result.verdict.value,
                    action.action_type,
                )
                return PolicyDecision(
                    action=action,
                    verdict=result.verdict,
                    enforced_policies=tuple(applied),
                    rationale=result.rationale,
                    denial_reason=result.denial_reason,
                )
            # Rule returned ALLOW — if it applied (enforced_policies non-empty),
            # this is a definitive allow and we stop here.
            if result.enforced_policies:
                applied.extend(result.enforced_policies)
                log.info(
                    "Policy → ALLOW for action=%s (rule %s)",
                    action.action_type,
                    result.enforced_policies[0].policy_name,
                )
                return PolicyDecision(
                    action=action,
                    verdict=VerdictType.ALLOW,
                    enforced_policies=tuple(applied),
                    rationale="Allowed by policy rule",
                )
            # Rule did not apply — continue to next rule
            applied.extend(result.enforced_policies)

        log.info(
            "Policy → ALLOW for action=%s (default, %d rule(s) evaluated)",
            action.action_type,
            len(applied),
        )

        return PolicyDecision(
            action=action,
            verdict=VerdictType.ALLOW,
            enforced_policies=tuple(applied),
            rationale="All policies allowed this action",
        )

    def register_rule(self, rule: PolicyRuleABC) -> None:
        """Register a policy rule for evaluation.

        Args:
            rule: Policy rule implementation.
        """
        self._rules.insert(0, rule)
        log.debug("Registered policy rule: %s", getattr(rule, "name", str(rule)))

    def register_policy(self, rule: PolicyRuleModel) -> None:
        """Register a policy rule from a Pydantic model.

        Args:
            rule: Policy rule model to register.
        """
        wrapped = ModelPolicyRule(rule)
        self.register_rule(wrapped)
