"""Policy engine for LLM provider routing decisions.

Sits between the router and the provider layer, allowing future
enterprise policies (cost limits, provider preferences, compliance
rules) to be enforced without modifying router internals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.llm.router_exceptions import PolicyViolationError
from app.llm.router_models import ProviderCapabilities, ProviderHealthSnapshot

log = logging.getLogger(__name__)


class PolicyRule(ABC):
    """Abstract base for a routing policy rule.

    Subclasses implement :meth:`evaluate` which returns ``True`` if
    the rule passes (i.e., the provider is allowed), or raises
    :class:`PolicyViolationError` with a reason.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the rule's name."""

    @abstractmethod
    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None,
        health: ProviderHealthSnapshot | None,
        context: dict[str, Any],
    ) -> bool:
        """Evaluate whether a provider is allowed.

        Args:
            provider_id: Provider name.
            capabilities: Provider capabilities.
            health: Provider health snapshot.
            context: Request context (budget, user preferences, etc.).

        Returns:
            ``True`` if the provider is allowed.

        Raises:
            PolicyViolationError: If the provider is blocked.
        """


class MaxCostPolicy(PolicyRule):
    """Reject providers with estimated cost exceeding a threshold."""

    def __init__(self, max_cost: float) -> None:
        self._max_cost = max_cost

    @property
    def name(self) -> str:
        return "max_cost"

    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None,
        health: ProviderHealthSnapshot | None,
        context: dict[str, Any],
    ) -> bool:
        cost = context.get("estimated_cost", 0.0)
        if cost > self._max_cost:
            raise PolicyViolationError(
                f"Provider '{provider_id}' cost {cost:.4f} exceeds "
                f"maximum {self._max_cost:.4f}",
                module="llm.policy_engine",
                context={
                    "provider": provider_id,
                    "cost": cost,
                    "max": self._max_cost,
                },
            )
        return True


class HealthThresholdPolicy(PolicyRule):
    """Reject providers below a health threshold."""

    def __init__(self, min_success_rate: float = 0.8) -> None:
        self._min_rate = min_success_rate

    @property
    def name(self) -> str:
        return "health_threshold"

    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None,
        health: ProviderHealthSnapshot | None,
        context: dict[str, Any],
    ) -> bool:
        if health is None:
            return True
        if health.success_rate < self._min_rate:
            raise PolicyViolationError(
                f"Provider '{provider_id}' health {health.success_rate:.2f} "
                f"below threshold {self._min_rate:.2f}",
                module="llm.policy_engine",
                context={
                    "provider": provider_id,
                    "success_rate": health.success_rate,
                    "threshold": self._min_rate,
                },
            )
        return True


class CapabilityRequirementPolicy(PolicyRule):
    """Reject providers missing required capabilities."""

    def __init__(self, required_flags: set[str]) -> None:
        self._required_flags = required_flags

    @property
    def name(self) -> str:
        return "capability_requirement"

    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None,
        health: ProviderHealthSnapshot | None,
        context: dict[str, Any],
    ) -> bool:
        if capabilities is None:
            return True
        for flag in self._required_flags:
            if not getattr(capabilities, flag, False):
                raise PolicyViolationError(
                    f"Provider '{provider_id}' lacks required capability '{flag}'",
                    module="llm.policy_engine",
                    context={
                        "provider": provider_id,
                        "missing_capability": flag,
                    },
                )
        return True


class ProviderAllowListPolicy(PolicyRule):
    """Only allow providers in an explicit allow-list."""

    def __init__(self, allowed: set[str]) -> None:
        self._allowed = allowed

    @property
    def name(self) -> str:
        return "allow_list"

    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None,
        health: ProviderHealthSnapshot | None,
        context: dict[str, Any],
    ) -> bool:
        if self._allowed and provider_id not in self._allowed:
            raise PolicyViolationError(
                f"Provider '{provider_id}' is not in the allow-list",
                module="llm.policy_engine",
                context={"provider": provider_id},
            )
        return True


class ProviderBlockListPolicy(PolicyRule):
    """Block providers in an explicit block-list."""

    def __init__(self, blocked: set[str]) -> None:
        self._blocked = blocked

    @property
    def name(self) -> str:
        return "block_list"

    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None,
        health: ProviderHealthSnapshot | None,
        context: dict[str, Any],
    ) -> bool:
        if provider_id in self._blocked:
            raise PolicyViolationError(
                f"Provider '{provider_id}' is block-listed",
                module="llm.policy_engine",
                context={"provider": provider_id},
            )
        return True


class PolicyEngine:
    """Chain-of-responsibility policy evaluator for provider selection.

    Holds an ordered list of :class:`PolicyRule` instances and
    evaluates each in turn.  If any rule raises
    :class:`PolicyViolationError`, that provider is rejected.

    Attributes:
        _rules: Ordered list of policy rules.
    """

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule, position: int | None = None) -> None:
        """Add a rule to the policy chain.

        Args:
            rule: The policy rule to add.
            position: Insert position (defaults to end).
        """
        if position is None:
            self._rules.append(rule)
        else:
            self._rules.insert(position, rule)
        log.debug("Added policy rule '%s'", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Returns ``True`` if a rule was removed.
        """
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def clear_rules(self) -> None:
        """Remove all rules."""
        self._rules.clear()

    def evaluate(
        self,
        provider_id: str,
        capabilities: ProviderCapabilities | None = None,
        health: ProviderHealthSnapshot | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate all rules for a provider.

        Returns ``True`` if the provider passes all rules.

        Raises:
            PolicyViolationError: If any rule rejects the provider.
        """
        ctx = context or {}
        for rule in self._rules:
            if not rule.evaluate(provider_id, capabilities, health, ctx):
                raise PolicyViolationError(
                    f"Rule '{rule.name}' rejected provider '{provider_id}'",
                    module="llm.policy_engine",
                )
        return True

    def filter_providers(
        self,
        provider_ids: list[str],
        capabilities_map: dict[str, ProviderCapabilities] | None = None,
        health_map: dict[str, ProviderHealthSnapshot] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Filter a list of providers through all policy rules.

        Returns only provider IDs that pass every rule.
        """
        caps = capabilities_map or {}
        health = health_map or {}
        allowed = []
        for pid in provider_ids:
            try:
                self.evaluate(
                    pid,
                    caps.get(pid),
                    health.get(pid),
                    context,
                )
                allowed.append(pid)
            except PolicyViolationError:
                log.debug("Provider '%s' filtered by policy", pid)
        return allowed

    @property
    def rule_names(self) -> list[str]:
        """Return names of all registered rules."""
        return [r.name for r in self._rules]

    def __iter__(self):
        """Allow iteration over rules."""
        return iter(self._rules)
