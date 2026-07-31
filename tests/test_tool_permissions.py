"""Tests for the permission system."""

from __future__ import annotations

from app.tools.permissions import (
    PermissionDecision,
    PermissionResult,
    PermissionSystem,
)


class TestPermissionDefaults:
    def test_default_allow(self) -> None:
        system = PermissionSystem()
        result = system.check("anything")
        assert result.decision == PermissionDecision.ALLOW

    def test_default_custom(self) -> None:
        system = PermissionSystem(default=PermissionDecision.DENY)
        assert system.check("anything").decision == PermissionDecision.DENY


class TestPermissionRules:
    def test_allow(self) -> None:
        system = PermissionSystem()
        system.allow("calc*")
        assert system.check("calculator").decision == PermissionDecision.ALLOW

    def test_deny(self) -> None:
        system = PermissionSystem()
        system.deny("shell*")
        assert system.check("shell_run").decision == PermissionDecision.DENY

    def test_confirm(self) -> None:
        system = PermissionSystem()
        system.confirm("web_*")
        assert system.check("web_fetch").decision == PermissionDecision.CONFIRM

    def test_first_match_wins(self) -> None:
        system = PermissionSystem()
        system.deny("*")
        system.allow("safe")
        assert system.check("safe").decision == PermissionDecision.DENY

    def test_result_fields(self) -> None:
        system = PermissionSystem()
        system.deny("shell_run", reason="dangerous")
        result = system.check("shell_run", {"command": "rm"})
        assert isinstance(result, PermissionResult)
        assert result.tool_name == "shell_run"
        assert result.reason == "dangerous"
        assert result.decision == PermissionDecision.DENY

    def test_clear(self) -> None:
        system = PermissionSystem()
        system.deny("*")
        system.clear()
        assert system.check("anything").decision == PermissionDecision.ALLOW


class TestPermissionSandbox:
    def test_sandbox_denies_unmatched(self) -> None:
        system = PermissionSystem()
        system.allow("safe")
        system.enable_sandbox()
        assert system.sandbox is True
        assert system.check("safe").decision == PermissionDecision.ALLOW
        assert system.check("unsafe").decision == PermissionDecision.DENY

    def test_sandbox_toggle(self) -> None:
        system = PermissionSystem()
        system.enable_sandbox()
        assert system.check("anything").decision == PermissionDecision.DENY
        system.disable_sandbox()
        assert system.sandbox is False
        assert system.check("anything").decision == PermissionDecision.ALLOW
