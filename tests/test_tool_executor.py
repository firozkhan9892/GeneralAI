"""Tests for the tool executor."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.mock import MockTool
from app.tools.permissions import PermissionSystem
from app.tools.registry import ToolRegistry


def _mock(name: str = "mock", **kwargs: Any) -> MockTool:
    return MockTool(name=name, **kwargs)


def _registry(*tools: MockTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


class TestSyncExecution:
    def test_success(self) -> None:
        executor = ToolExecutor()
        result = executor.execute(MockTool(), {"input": "hi"})
        assert result.success is True
        assert result.output == "Echo: hi"
        assert result.error is None
        assert result.tool_name == "mock"
        assert result.execution_time >= 0.0

    def test_success_by_name(self) -> None:
        executor = ToolExecutor(registry=_registry(_mock("greet")))
        result = executor.execute("greet", {"input": "hi"})
        assert result.success is True
        assert result.output == "Echo: hi"

    def test_fixed_result(self) -> None:
        executor = ToolExecutor()
        result = executor.execute(MockTool(echo_input=False, result={"ok": 1}))
        assert result.success is True
        assert result.output == {"ok": 1}

    def test_not_found(self) -> None:
        executor = ToolExecutor(registry=ToolRegistry())
        result = executor.execute("missing")
        assert result.success is False
        assert result.error is not None

    def test_no_registry_for_name(self) -> None:
        executor = ToolExecutor()
        result = executor.execute("missing")
        assert result.success is False
        assert "not registered" in (result.error or "")

    def test_validation_error_captured(self) -> None:
        executor = ToolExecutor()
        tool = MockTool()
        result = executor.execute(tool, {"nope": 1})
        assert result.success is False
        assert "Unknown parameter" in (result.error or "")

    def test_execution_error_captured(self) -> None:
        executor = ToolExecutor()
        result = executor.execute(_mock(fail="boom"))
        assert result.success is False
        assert result.error == "boom"

    def test_context_passed(self) -> None:
        context = ToolContext()
        received: dict[str, Any] = {}

        def handler(arguments, ctx):
            received["session"] = ctx.session_id
            return "ok"

        executor = ToolExecutor()
        tool = _mock(on_run=handler)
        result = executor.execute(tool, context=context)
        assert result.output == "ok"
        assert received["session"] is None


class TestAsyncExecution:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        executor = ToolExecutor()
        result = await executor.execute_async(MockTool(), {"input": "hi"})
        assert result.success is True
        assert result.output == "Echo: hi"

    @pytest.mark.asyncio
    async def test_success_by_name(self) -> None:
        executor = ToolExecutor(registry=_registry(_mock("greet")))
        result = await executor.execute_async("greet", {"input": "hi"})
        assert result.output == "Echo: hi"

    @pytest.mark.asyncio
    async def test_failure_captured(self) -> None:
        executor = ToolExecutor()
        result = await executor.execute_async(_mock(fail="boom"))
        assert result.success is False
        assert result.error == "boom"


class TestTimeout:
    def test_sync_timeout(self) -> None:
        executor = ToolExecutor()
        result = executor.execute(
            _mock(delay_s=2.0),
            timeout_s=0.1,
        )
        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error.lower()
        assert result.metadata.get("timed_out") is True

    @pytest.mark.asyncio
    async def test_async_timeout(self) -> None:
        executor = ToolExecutor()
        result = await executor.execute_async(
            _mock(delay_s=2.0),
            timeout_s=0.1,
        )
        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    def test_timeout_override_applies(self) -> None:
        executor = ToolExecutor()
        result = executor.execute(_mock(delay_s=2.0), timeout_s=0.05)
        assert result.success is False


class TestCancellation:
    def test_cancelled_before_run(self) -> None:
        context = ToolContext()
        context.cancel()
        executor = ToolExecutor()
        result = executor.execute(MockTool(), context=context)
        assert result.success is False
        assert "cancelled" in (result.error or "").lower()

    def test_cancelled_before_retry(self) -> None:
        context = ToolContext()
        executor = ToolExecutor()
        result = executor.execute(
            _mock(fail="boom", delay_s=0.0),
            context=context,
            max_retries=3,
            retry_delay_s=0.01,
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_async_cancelled(self) -> None:
        context = ToolContext()
        context.cancel()
        executor = ToolExecutor()
        result = await executor.execute_async(MockTool(), context=context)
        assert result.success is False
        assert "cancelled" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_async_cancel_mid_retry(self) -> None:
        executor = ToolExecutor()
        context = ToolContext()
        result = await executor.execute_async(
            _mock(fail="boom"),
            context=context,
            max_retries=5,
            retry_delay_s=0.001,
        )
        assert result.success is False


class TestRetry:
    def test_retry_until_success(self) -> None:
        executor = ToolExecutor()
        tool = _mock(fail_first_n=2)
        result = executor.execute(tool, max_retries=3, retry_delay_s=0.0)
        assert result.success is True
        assert tool.call_count == 3
        assert result.metadata.get("attempts") == 3

    def test_retry_exhausted(self) -> None:
        executor = ToolExecutor()
        tool = _mock(fail="always")
        result = executor.execute(tool, max_retries=2, retry_delay_s=0.0)
        assert result.success is False
        assert tool.call_count == 3

    def test_no_retry_by_default(self) -> None:
        executor = ToolExecutor()
        tool = _mock(fail="always")
        result = executor.execute(tool)
        assert result.success is False
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry(self) -> None:
        executor = ToolExecutor()
        tool = _mock(fail_first_n=1)
        result = await executor.execute_async(tool, max_retries=2)
        assert result.success is True
        assert tool.call_count == 2


class TestPermissions:
    def test_denied(self) -> None:
        permissions = PermissionSystem()
        permissions.deny("*")
        executor = ToolExecutor(permission_system=permissions)
        result = executor.execute(MockTool())
        assert result.success is False
        assert "not allowed" in (result.error or "")

    def test_allowed(self) -> None:
        permissions = PermissionSystem()
        permissions.allow("mock")
        executor = ToolExecutor(permission_system=permissions)
        result = executor.execute(MockTool(), {"input": "hi"})
        assert result.success is True

    def test_confirm_with_handler(self) -> None:
        permissions = PermissionSystem()
        permissions.confirm("mock")
        executor = ToolExecutor(
            permission_system=permissions,
            confirmation_handler=lambda tool, args: True,
        )
        result = executor.execute(MockTool())
        assert result.success is True

    def test_confirm_denied(self) -> None:
        permissions = PermissionSystem()
        permissions.confirm("mock")
        executor = ToolExecutor(
            permission_system=permissions,
            confirmation_handler=lambda tool, args: False,
        )
        result = executor.execute(MockTool())
        assert result.success is False
        assert "declined" in (result.error or "")

    def test_confirm_no_handler(self) -> None:
        permissions = PermissionSystem()
        permissions.confirm("mock")
        executor = ToolExecutor(permission_system=permissions)
        result = executor.execute(MockTool())
        assert result.success is False
        assert "requires confirmation" in (result.error or "")


class TestContextIntegration:
    def test_sync_runs_in_event_loop(self) -> None:
        executor = ToolExecutor()

        async def nested() -> Any:
            return await executor.execute_async(MockTool(), {"input": "x"})

        result = asyncio.run(nested())
        assert result.success is True
