"""Tests for SkillExecutor."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.kernel.skills.builtins import register_builtin_skills
from app.kernel.skills.executor import SkillExecutor, SkillSelector
from app.kernel.skills.models import Skill, SkillDescriptor, SkillResult
from app.kernel.tools.builtins import register_builtin_tools
from app.kernel.tools.executor import ToolExecutor, ToolResolver


class TestSkillExecutorSetup:
    """Tests for SkillExecutor setup."""

    def test_set_selector(self) -> None:
        executor = SkillExecutor()
        selector = SkillSelector()
        executor.set_selector(selector)

    def test_set_tool_resolver(self) -> None:
        executor = SkillExecutor()
        resolver = ToolResolver()
        executor.set_tool_resolver(resolver)

    def test_set_tool_executor(self) -> None:
        executor = SkillExecutor()
        tool_executor = ToolExecutor()
        executor.set_tool_executor(tool_executor)

    def test_execute_without_selector(self) -> None:
        executor = SkillExecutor()
        skill = Skill(
            name="test",
            descriptor=SkillDescriptor(name="test"),
            parameters={},
        )

        async def run() -> SkillResult:
            return await executor.execute(skill)

        with pytest.raises(RuntimeError, match="no selector configured"):
            asyncio.run(run())


class TestSkillExecutorExecute:
    """Tests for SkillExecutor.execute."""

    @pytest.mark.asyncio
    async def test_execute_echo_skill(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("echo", {"message": "hello"})
        result = await executor.execute(skill)
        assert result.success is True
        assert result.output == "hello"
        assert result.skill_name == "echo"

    @pytest.mark.asyncio
    async def test_execute_calculator_skill(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("calculator", {"expression": "2 + 3"})
        result = await executor.execute(skill)
        assert result.success is True
        assert result.output == 5.0

    @pytest.mark.asyncio
    async def test_execute_summarize_skill(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select(
            "summarize",
            {"text": "First. Second. Third.", "max_sentences": 2},
        )
        result = await executor.execute(skill)
        assert result.success is True
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_execute_respond_skill(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("respond", {"content": "Hello"})
        result = await executor.execute(skill)
        assert result.success is True
        assert result.output["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_execute_unregistered_skill(self) -> None:
        selector = SkillSelector()
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = Skill(
            name="nonexistent",
            descriptor=SkillDescriptor(name="nonexistent"),
            parameters={},
        )
        result = await executor.execute(skill)
        assert result.success is False
        assert result.error is not None
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        from unittest.mock import AsyncMock, patch

        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("echo", {"message": "hello"})

        with patch(
            "app.kernel.skills.builtins.echo.execute",
            new_callable=AsyncMock,
        ) as mock_execute:
            import asyncio as _asyncio

            async def slow_handler(params: dict) -> str:
                await _asyncio.sleep(1)
                return "slow"

            mock_execute.side_effect = slow_handler
            result = await executor.execute(skill, timeout_s=0.01)
            assert result.success is False
            assert result.error is not None
            assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_cancellation(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("echo", {"message": "hello"})
        token = MagicMock()
        token.is_cancelled = True
        result = await executor.execute(skill, cancellation_token=token)
        assert result.success is False
        assert result.error is not None
        assert "cancelled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_returns_duration(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("echo", {"message": "hello"})
        result = await executor.execute(skill)
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_returns_token_cost(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("echo", {"message": "hello"})
        result = await executor.execute(skill)
        assert result.token_cost >= 0

    @pytest.mark.asyncio
    async def test_execute_with_tool_dependencies(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        tool_resolver = ToolResolver()
        register_builtin_tools(tool_resolver)
        tool_executor = ToolExecutor()
        tool_executor.set_resolver(tool_resolver)
        executor = SkillExecutor()
        executor.set_selector(selector)
        executor.set_tool_resolver(tool_resolver)
        executor.set_tool_executor(tool_executor)

        skill = await selector.select("echo", {"message": "hello"})
        result = await executor.execute(skill)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_max_retries(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill = await selector.select("calculator", {"expression": "1/0"})
        result = await executor.execute(skill, max_retries=2)
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_deterministic(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        executor = SkillExecutor()
        executor.set_selector(selector)

        skill1 = await selector.select("echo", {"message": "hello"})
        r1 = await executor.execute(skill1)
        skill2 = await selector.select("echo", {"message": "hello"})
        r2 = await executor.execute(skill2)
        assert r1.output == r2.output
        assert r1.success == r2.success
