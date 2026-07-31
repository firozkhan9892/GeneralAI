"""Tests for built-in skills."""

from __future__ import annotations

import pytest

from app.kernel.skills.builtins import (
    calculator,
    echo,
    get_skill_handler,
    respond,
    search_memory,
    summarize,
)


class TestEchoSkill:
    """Tests for the echo skill."""

    @pytest.mark.asyncio
    async def test_echo_simple(self) -> None:
        result = await echo.execute({"message": "hello"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_echo_empty(self) -> None:
        result = await echo.execute({"message": ""})
        assert result == ""

    @pytest.mark.asyncio
    async def test_echo_missing_message(self) -> None:
        result = await echo.execute({})
        assert result == ""

    @pytest.mark.asyncio
    async def test_echo_long_message(self) -> None:
        msg = "x" * 1000
        result = await echo.execute({"message": msg})
        assert result == msg

    @pytest.mark.asyncio
    async def test_echo_special_chars(self) -> None:
        msg = "hello\nworld\ttab"
        result = await echo.execute({"message": msg})
        assert result == msg


class TestCalculatorSkill:
    """Tests for the calculator skill."""

    @pytest.mark.asyncio
    async def test_addition(self) -> None:
        result = await calculator.execute({"expression": "2 + 3"})
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_multiplication(self) -> None:
        result = await calculator.execute({"expression": "6 * 7"})
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_complex_expression(self) -> None:
        result = await calculator.execute({"expression": "(2 + 3) * 4"})
        assert result == 20.0

    @pytest.mark.asyncio
    async def test_missing_expression(self) -> None:
        with pytest.raises(ValueError, match="expression parameter is required"):
            await calculator.execute({})

    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        r1 = await calculator.execute({"expression": "2 + 2"})
        r2 = await calculator.execute({"expression": "2 + 2"})
        assert r1 == r2


class TestSummarizeSkill:
    """Tests for the summarize skill."""

    @pytest.mark.asyncio
    async def test_summarize_basic(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        result = await summarize.execute({"text": text})
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_empty_text(self) -> None:
        result = await summarize.execute({"text": ""})
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_single_sentence(self) -> None:
        text = "Only one sentence here."
        result = await summarize.execute({"text": text})
        assert result == text

    @pytest.mark.asyncio
    async def test_summarize_max_sentences(self) -> None:
        text = " ".join(f"Sentence {i}. " for i in range(10))
        result = await summarize.execute({"text": text, "max_sentences": 3})
        sentences = result.split(". ")
        assert len([s for s in sentences if s]) <= 3

    @pytest.mark.asyncio
    async def test_summarize_preserves_key_content(self) -> None:
        text = (
            "Important first sentence about AI. "
            "Second sentence with details about machine learning. "
            "Third sentence about deep learning."
        )
        result = await summarize.execute({"text": text, "max_sentences": 2})
        assert "Important" in result or "first" in result.lower()

    @pytest.mark.asyncio
    async def test_summarize_deterministic(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        r1 = await summarize.execute({"text": text, "max_sentences": 2})
        r2 = await summarize.execute({"text": text, "max_sentences": 2})
        assert r1 == r2


class TestSearchMemorySkill:
    """Tests for the search_memory skill."""

    @pytest.mark.asyncio
    async def test_search_returns_dict(self) -> None:
        result = await search_memory.execute({"query": "test"})
        assert isinstance(result, dict)
        assert result["query"] == "test"
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_with_limit(self) -> None:
        result = await search_memory.execute({"query": "test", "limit": 10})
        assert result["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_default_limit(self) -> None:
        result = await search_memory.execute({"query": "test"})
        assert result["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_empty_query(self) -> None:
        result = await search_memory.execute({"query": ""})
        assert result["query"] == ""
        assert result["total"] == 0


class TestRespondSkill:
    """Tests for the respond skill."""

    @pytest.mark.asyncio
    async def test_basic_response(self) -> None:
        result = await respond.execute({"content": "Hello there"})
        assert isinstance(result, dict)
        assert result["content"] == "Hello there"
        assert result["format"] == "text"

    @pytest.mark.asyncio
    async def test_response_with_format(self) -> None:
        result = await respond.execute({"content": "Hello", "format": "markdown"})
        assert result["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_response_metadata(self) -> None:
        result = await respond.execute({"content": "Hello"})
        assert "skill" in result["metadata"]
        assert result["metadata"]["skill"] == "respond"

    @pytest.mark.asyncio
    async def test_response_empty_content(self) -> None:
        result = await respond.execute({"content": ""})
        assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_response_content_length(self) -> None:
        result = await respond.execute({"content": "test"})
        assert result["metadata"]["content_length"] == 4


class TestGetSkillHandler:
    """Tests for the built-in skill handler registry."""

    @pytest.mark.asyncio
    async def test_get_echo_handler(self) -> None:
        handler = get_skill_handler("echo")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_calculator_handler(self) -> None:
        handler = get_skill_handler("calculator")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_summarize_handler(self) -> None:
        handler = get_skill_handler("summarize")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_search_memory_handler(self) -> None:
        handler = get_skill_handler("search_memory")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_respond_handler(self) -> None:
        handler = get_skill_handler("respond")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_unknown_handler(self) -> None:
        with pytest.raises(KeyError, match="Unknown built-in skill"):
            get_skill_handler("nonexistent")
