"""Tests for PolicyEngine and policy domain models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from app.kernel.decision.models import ActionCandidate, Decision
from app.kernel.policy import (
    AppliedPolicy,
    PolicyAction,
    PolicyDecision,
    PolicyEngine,
    VerdictType,
)
from app.kernel.policy.models import PolicyRule as PolicyRuleModel, SandboxConfig
from app.kernel.policy.rules.base import AllowAllRule, BuiltinDenyRules, ModelPolicyRule


def _decision(
    action_type: str = "respond",
    session_id: str = "",
) -> Decision:
    return Decision(
        selected_action=ActionCandidate(action_type=action_type),
        session_id=session_id,
    )


def _rule(
    name: str = "test_rule",
    pattern: str = "test_action",
    verdict: VerdictType = VerdictType.DENY,
    priority: int = 50,
    denial_reason: str | None = None,
) -> PolicyRuleModel:
    return PolicyRuleModel(
        name=name,
        action_pattern=pattern,
        verdict=verdict,
        priority=priority,
        denial_reason=denial_reason,
    )


# ── Domain model tests ───────────────────────────────────────────────────


class TestPolicyModels:
    """Tests for policy domain models."""

    def test_verdict_type_values(self) -> None:
        assert VerdictType.ALLOW.value == "allow"
        assert VerdictType.DENY.value == "deny"
        assert VerdictType.CONFIRM.value == "confirm"
        assert VerdictType.SANDBOX.value == "sandbox"

    def test_policy_action_create(self) -> None:
        action = PolicyAction(action_type="tool_call")
        assert action.action_type == "tool_call"
        assert action.tool_name is None
        assert action.parameters == {}
        assert action.domain is None
        assert action.session_id == ""

    def test_policy_action_frozen(self) -> None:
        action = PolicyAction(action_type="test")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            action.action_type = "changed"  # type: ignore[misc]

    def test_applied_policy_create(self) -> None:
        ap = AppliedPolicy(policy_name="test", verdict=VerdictType.ALLOW)
        assert ap.policy_name == "test"
        assert ap.verdict == VerdictType.ALLOW
        assert ap.rationale == ""

    def test_applied_policy_frozen(self) -> None:
        ap = AppliedPolicy(policy_name="test", verdict=VerdictType.ALLOW)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            ap.verdict = VerdictType.DENY  # type: ignore[misc]

    def test_policy_decision_create(self) -> None:
        action = PolicyAction(action_type="test")
        decision = PolicyDecision(action=action, verdict=VerdictType.ALLOW)
        assert decision.verdict == VerdictType.ALLOW
        assert decision.enforced_policies == ()
        assert decision.evaluated_at is not None

    def test_policy_decision_frozen(self) -> None:
        action = PolicyAction(action_type="test")
        d = PolicyDecision(action=action, verdict=VerdictType.ALLOW)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            d.verdict = VerdictType.DENY  # type: ignore[misc]

    def test_sandbox_config_defaults(self) -> None:
        sc = SandboxConfig()
        assert sc.allow_network is False
        assert sc.allow_filesystem is False
        assert sc.allow_subprocess is False
        assert sc.memory_limit_mb == 256
        assert sc.time_limit_s == 30

    def test_policy_rule_model_create(self) -> None:
        pr = PolicyRuleModel(name="test", action_pattern="*", verdict=VerdictType.ALLOW)
        assert pr.name == "test"
        assert pr.description == ""
        assert pr.priority == 0
        assert pr.conditions == {}
        assert pr.denial_reason is None

    def test_policy_rule_model_frozen(self) -> None:
        pr = PolicyRuleModel(name="test", action_pattern="*", verdict=VerdictType.ALLOW)
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            pr.name = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        action = PolicyAction(action_type="test")
        original = PolicyDecision(
            action=action, verdict=VerdictType.DENY, denial_reason="No"
        )
        data = original.model_dump()
        restored = PolicyDecision.model_validate(data)
        assert restored == original

    def test_applied_policy_equality(self) -> None:
        a = AppliedPolicy(policy_name="r1", verdict=VerdictType.ALLOW)
        b = AppliedPolicy(policy_name="r1", verdict=VerdictType.ALLOW)
        assert a == b

    def test_applied_policy_inequality(self) -> None:
        a = AppliedPolicy(policy_name="r1", verdict=VerdictType.ALLOW)
        b = AppliedPolicy(policy_name="r2", verdict=VerdictType.ALLOW)
        assert a != b


# ── ModelPolicyRule unit tests ───────────────────────────────────────────


class TestModelPolicyRule:
    """Tests for ModelPolicyRule wrapper."""

    @pytest.mark.asyncio
    async def test_rule_matches_action_type(self) -> None:
        rule = ModelPolicyRule(_rule("deny_exec", "shell_exec", VerdictType.DENY))
        action = PolicyAction(action_type="shell_exec")
        result = await rule.evaluate(action)
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_rule_does_not_match(self) -> None:
        rule = ModelPolicyRule(_rule("deny_exec", "shell_exec", VerdictType.DENY))
        action = PolicyAction(action_type="respond")
        result = await rule.evaluate(action)
        assert result.verdict == VerdictType.ALLOW
        assert result.enforced_policies == ()

    @pytest.mark.asyncio
    async def test_rule_pattern_glob(self) -> None:
        rule = ModelPolicyRule(_rule("deny_all_exec", "exec_*", VerdictType.DENY))
        action = PolicyAction(action_type="exec_script")
        result = await rule.evaluate(action)
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_rule_priority_property(self) -> None:
        rule = ModelPolicyRule(_rule("test", "a", VerdictType.DENY, priority=99))
        assert rule.priority == 99

    @pytest.mark.asyncio
    async def test_rule_denial_reason_included(self) -> None:
        rule = ModelPolicyRule(
            _rule("deny", "bad", VerdictType.DENY, denial_reason="Not allowed")
        )
        action = PolicyAction(action_type="bad")
        result = await rule.evaluate(action)
        assert result.denial_reason == "Not allowed"
        assert "Not allowed" in result.rationale


# ── Builtin deny rules ──────────────────────────────────────────────────


class TestBuiltinDenyRules:
    """Tests for built-in deny rules."""

    def test_creates_known_rules(self) -> None:
        rules = BuiltinDenyRules.create_all()
        assert len(rules) == 4

    def test_rule_names(self) -> None:
        rules = BuiltinDenyRules.create_all()
        names = {r._rule.name for r in rules}
        assert "deny_shell_exec" in names
        assert "deny_delete_file" in names
        assert "deny_modify_system" in names
        assert "deny_network_exec" in names

    @pytest.mark.asyncio
    async def test_shell_exec_denied(self) -> None:
        rules = BuiltinDenyRules.create_all()
        action = PolicyAction(action_type="shell_exec")
        for r in rules:
            result = await r.evaluate(action)
            if result.verdict != VerdictType.ALLOW:
                assert result.verdict == VerdictType.DENY
                return
        pytest.fail("No rule denied shell_exec")

    @pytest.mark.asyncio
    async def test_respond_allowed(self) -> None:
        rules = BuiltinDenyRules.create_all()
        action = PolicyAction(action_type="respond")
        for r in rules:
            result = await r.evaluate(action)
            assert result.verdict == VerdictType.ALLOW


# ── AllowAllRule ─────────────────────────────────────────────────────────


class TestAllowAllRule:
    """Tests for AllowAllRule fallback."""

    @pytest.mark.asyncio
    async def test_always_allows(self) -> None:
        rule = AllowAllRule()
        result = await rule.evaluate(PolicyAction(action_type="anything"))
        assert result.verdict == VerdictType.ALLOW

    @pytest.mark.asyncio
    async def test_name_is_allow_all(self) -> None:
        rule = AllowAllRule()
        assert rule.name == "allow_all"

    @pytest.mark.asyncio
    async def test_priority_lowest(self) -> None:
        rule = AllowAllRule()
        assert rule.priority == -1


# ── PolicyEngine evaluate ────────────────────────────────────────────────


class TestPolicyEngineEvaluate:
    """Tests for PolicyEngine.evaluate()."""

    @pytest.fixture
    def engine(self) -> PolicyEngine:
        return PolicyEngine()

    @pytest.mark.asyncio
    async def test_evaluate_returns_policy_decision(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("respond"))
        assert isinstance(result, PolicyDecision)

    @pytest.mark.asyncio
    async def test_evaluate_allow_safe_action(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("respond"))
        assert result.verdict == VerdictType.ALLOW

    @pytest.mark.asyncio
    async def test_evaluate_deny_dangerous_action(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("shell_exec"))
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_evaluate_deny_delete_file(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("delete_file"))
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_evaluate_deny_modify_system(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("modify_system"))
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_evaluate_deny_network_exec(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("network_exec"))
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_evaluate_applied_policies_populated(
        self, engine: PolicyEngine
    ) -> None:
        result = await engine.evaluate(_decision("respond"))
        assert len(result.enforced_policies) >= 1

    @pytest.mark.asyncio
    async def test_evaluate_denial_reason_included(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("shell_exec"))
        assert result.denial_reason is not None
        assert "not permitted" in result.denial_reason.lower()

    @pytest.mark.asyncio
    async def test_evaluate_deterministic(self, engine: PolicyEngine) -> None:
        d1 = await engine.evaluate(_decision("respond"))
        d2 = await engine.evaluate(_decision("respond"))
        assert d1.verdict == d2.verdict
        assert d1.rationale == d2.rationale
        assert len(d1.enforced_policies) == len(d2.enforced_policies)

    @pytest.mark.asyncio
    async def test_evaluate_passes_session_id(self, engine: PolicyEngine) -> None:
        decision = _decision("respond", session_id="sess_1")
        result = await engine.evaluate(decision)
        assert result.action.session_id == "sess_1"

    @pytest.mark.asyncio
    async def test_evaluate_action_type_stored(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("custom_action"))
        assert result.action.action_type == "custom_action"


# ── Rule registration ───────────────────────────────────────────────────


class TestRuleRegistration:
    """Tests for register_rule and register_policy."""

    @pytest.fixture
    def engine(self) -> PolicyEngine:
        return PolicyEngine()

    @pytest.mark.asyncio
    async def test_register_rule_takes_effect(self, engine: PolicyEngine) -> None:
        rule = ModelPolicyRule(
            _rule("deny_custom", "custom_action", VerdictType.DENY, priority=200)
        )
        engine.register_rule(rule)
        result = await engine.evaluate(_decision("custom_action"))
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_custom_rule_higher_priority_wins(self, engine: PolicyEngine) -> None:
        deny = ModelPolicyRule(
            _rule("deny_respond", "respond", VerdictType.DENY, priority=200)
        )
        engine.register_rule(deny)
        result = await engine.evaluate(_decision("respond"))
        assert result.verdict == VerdictType.DENY  # Custom rule overrides default allow

    @pytest.mark.asyncio
    async def test_register_policy_model(self, engine: PolicyEngine) -> None:
        rule = _rule("deny_test", "test_action", VerdictType.DENY, priority=150)
        engine.register_policy(rule)
        result = await engine.evaluate(_decision("test_action"))
        assert result.verdict == VerdictType.DENY

    @pytest.mark.asyncio
    async def test_latest_rule_evaluated_first(self, engine: PolicyEngine) -> None:
        allow = ModelPolicyRule(
            _rule("allow_all_custom", "*", VerdictType.ALLOW, priority=200)
        )
        engine.register_rule(allow)
        result = await engine.evaluate(_decision("shell_exec"))
        # The custom allow-all with priority 200 runs first and allows,
        # so the builtin deny rules are never reached
        assert result.verdict == VerdictType.ALLOW

    @pytest.mark.asyncio
    async def test_register_rule_after_evaluate(self, engine: PolicyEngine) -> None:
        result_before = await engine.evaluate(_decision("new_action"))
        assert result_before.verdict == VerdictType.ALLOW

        rule = ModelPolicyRule(
            _rule("deny_new", "new_action", VerdictType.DENY, priority=200)
        )
        engine.register_rule(rule)
        result_after = await engine.evaluate(_decision("new_action"))
        assert result_after.verdict == VerdictType.DENY


# ── Empty rules / edge cases ────────────────────────────────────────────


class TestPolicyEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def engine(self) -> PolicyEngine:
        return PolicyEngine()

    @pytest.mark.asyncio
    async def test_frozen_output(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("respond"))
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            result.verdict = VerdictType.DENY  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_evaluate_empty_rules(self) -> None:
        engine = PolicyEngine()
        # Engine has built-in rules by default; create one without
        engine._rules.clear()
        assert len(engine._rules) == 0
        result = await engine.evaluate(_decision("anything"))
        # Should still produce a valid decision
        assert isinstance(result, PolicyDecision)
        assert result.verdict == VerdictType.ALLOW

    @pytest.mark.asyncio
    async def test_unknown_action_type_allowed(self, engine: PolicyEngine) -> None:
        result = await engine.evaluate(_decision("unknown_operation"))
        assert result.verdict == VerdictType.ALLOW

    @pytest.mark.asyncio
    async def test_evaluate_confirm_verdict(self) -> None:
        engine = PolicyEngine()
        confirm_rule = ModelPolicyRule(
            _rule("confirm_test", "risky", VerdictType.CONFIRM, priority=200)
        )
        engine._rules.clear()
        engine.register_rule(confirm_rule)
        result = await engine.evaluate(_decision("risky"))
        assert result.verdict == VerdictType.CONFIRM

    @pytest.mark.asyncio
    async def test_evaluate_sandbox_verdict(self) -> None:
        engine = PolicyEngine()
        sandbox_rule = ModelPolicyRule(
            _rule("sandbox_test", "sandboxed_exec", VerdictType.SANDBOX, priority=200)
        )
        engine._rules.clear()
        engine.register_rule(sandbox_rule)
        result = await engine.evaluate(_decision("sandboxed_exec"))
        assert result.verdict == VerdictType.SANDBOX

    @pytest.mark.asyncio
    async def test_evaluate_with_disabled_default_rules(self) -> None:
        engine = PolicyEngine()
        engine._rules.clear()
        allow = AllowAllRule()
        engine.register_rule(allow)
        result = await engine.evaluate(_decision("shell_exec"))
        assert result.verdict == VerdictType.ALLOW  # Only AllowAllRule registered
