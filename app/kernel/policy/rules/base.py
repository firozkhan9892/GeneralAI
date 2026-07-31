"""Policy rule implementations."""

from __future__ import annotations

import fnmatch

from app.kernel.policy.models import (
    AppliedPolicy,
    PolicyAction,
    PolicyDecision,
    PolicyRule as PolicyRuleModel,
    VerdictType,
)
from app.kernel.policy.rules import PolicyRule as PolicyRuleABC


class ModelPolicyRule(PolicyRuleABC):
    """Wraps a Pydantic PolicyRule model as an ABC PolicyRule."""

    def __init__(self, rule: PolicyRuleModel) -> None:
        self._rule = rule
        self._name = rule.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._rule.priority

    async def evaluate(self, action: PolicyAction) -> PolicyDecision:
        if not fnmatch.fnmatch(action.action_type, self._rule.action_pattern):
            return PolicyDecision(
                action=action,
                verdict=VerdictType.ALLOW,
                enforced_policies=(),
                rationale="Rule does not apply",
            )

        verdict = self._rule.verdict

        return PolicyDecision(
            action=action,
            verdict=verdict,
            enforced_policies=(
                AppliedPolicy(
                    policy_name=self._rule.name,
                    verdict=verdict,
                    rationale=self._rule.description
                    or f"Matched pattern '{self._rule.action_pattern}'",
                ),
            ),
            rationale=self._rule.denial_reason or f"Verdict: {verdict.value}",
            denial_reason=self._rule.denial_reason,
        )


class AllowAllRule(PolicyRuleABC):
    """Fallback rule that allows all actions when no other rule matches."""

    NAME = "allow_all"

    def __init__(self) -> None:
        self._name = self.NAME

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return -1

    async def evaluate(self, action: PolicyAction) -> PolicyDecision:
        return PolicyDecision(
            action=action,
            verdict=VerdictType.ALLOW,
            enforced_policies=(
                AppliedPolicy(
                    policy_name=self._name,
                    verdict=VerdictType.ALLOW,
                    rationale="No explicit policy matched — allowed by default",
                ),
            ),
            rationale="Default allow",
        )


_KNOWN_DANGEROUS: dict[str, str] = {
    "shell_exec": "Shell execution is not permitted",
    "delete_file": "File deletion is not permitted",
    "modify_system": "System modification is not permitted",
    "network_exec": "Network execution is not permitted",
}


class BuiltinDenyRules:
    """Factory for built-in deny rules."""

    @staticmethod
    def create_all() -> list[ModelPolicyRule]:
        rules: list[ModelPolicyRule] = []
        for action_type, reason in _KNOWN_DANGEROUS.items():
            rules.append(
                ModelPolicyRule(
                    PolicyRuleModel(
                        name=f"deny_{action_type}",
                        description=f"Deny dangerous action: {action_type}",
                        action_pattern=action_type,
                        verdict=VerdictType.DENY,
                        priority=100,
                        denial_reason=reason,
                    )
                )
            )
        return rules
