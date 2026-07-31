"""Tests for built-in tools."""

from __future__ import annotations


import pytest

from app.kernel.tools.builtins import (
    calculator,
    clock,
    get_tool_handler,
    json,
    text_utils,
    uuid,
)


class TestCalculatorTool:
    """Tests for the calculator tool."""

    @pytest.mark.asyncio
    async def test_addition(self) -> None:
        result = await calculator.execute({"expression": "2 + 3"})
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_subtraction(self) -> None:
        result = await calculator.execute({"expression": "10 - 4"})
        assert result == 6.0

    @pytest.mark.asyncio
    async def test_multiplication(self) -> None:
        result = await calculator.execute({"expression": "6 * 7"})
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_division(self) -> None:
        result = await calculator.execute({"expression": "15 / 3"})
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_floordiv(self) -> None:
        result = await calculator.execute({"expression": "17 // 5"})
        assert result == 3.0

    @pytest.mark.asyncio
    async def test_modulo(self) -> None:
        result = await calculator.execute({"expression": "17 % 5"})
        assert result == 2.0

    @pytest.mark.asyncio
    async def test_power(self) -> None:
        result = await calculator.execute({"expression": "2 ** 8"})
        assert result == 256.0

    @pytest.mark.asyncio
    async def test_unary_negative(self) -> None:
        result = await calculator.execute({"expression": "-42"})
        assert result == -42.0

    @pytest.mark.asyncio
    async def test_unary_positive(self) -> None:
        result = await calculator.execute({"expression": "+42"})
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_abs_function(self) -> None:
        result = await calculator.execute({"expression": "abs(-5)"})
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_min_function(self) -> None:
        result = await calculator.execute({"expression": "min(3, 1, 4, 1, 5)"})
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_max_function(self) -> None:
        result = await calculator.execute({"expression": "max(3, 1, 4, 1, 5)"})
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_round_function(self) -> None:
        result = await calculator.execute({"expression": "round(3.7)"})
        assert result == 4.0

    @pytest.mark.asyncio
    async def test_sum_function(self) -> None:
        result = await calculator.execute({"expression": "sum([1, 2, 3, 4, 5])"})
        assert result == 15.0

    @pytest.mark.asyncio
    async def test_complex_expression(self) -> None:
        result = await calculator.execute({"expression": "2 + 3 * 4 - 1"})
        assert result == 13.0

    @pytest.mark.asyncio
    async def test_nested_expression(self) -> None:
        result = await calculator.execute({"expression": "(2 + 3) * (4 - 1)"})
        assert result == 15.0

    @pytest.mark.asyncio
    async def test_missing_expression(self) -> None:
        with pytest.raises(ValueError, match="expression parameter is required"):
            await calculator.execute({})

    @pytest.mark.asyncio
    async def test_empty_expression(self) -> None:
        with pytest.raises(ValueError, match="expression parameter is required"):
            await calculator.execute({"expression": ""})

    @pytest.mark.asyncio
    async def test_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            await calculator.execute({"expression": "1 / 0"})

    @pytest.mark.asyncio
    async def test_unsupported_operator(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            await calculator.execute({"expression": "2 & 3"})

    @pytest.mark.asyncio
    async def test_float_input(self) -> None:
        result = await calculator.execute({"expression": "3.14 + 2.86"})
        assert result == pytest.approx(6.0)

    @pytest.mark.asyncio
    async def test_deterministic_results(self) -> None:
        r1 = await calculator.execute({"expression": "2 + 2"})
        r2 = await calculator.execute({"expression": "2 + 2"})
        assert r1 == r2


class TestClockTool:
    """Tests for the clock tool."""

    @pytest.mark.asyncio
    async def test_default_iso_format(self) -> None:
        result = await clock.execute({})
        assert isinstance(result, str)
        assert "T" in result

    @pytest.mark.asyncio
    async def test_unix_format(self) -> None:
        result = await clock.execute({"format": "unix"})
        assert isinstance(result, float)
        assert result > 0

    @pytest.mark.asyncio
    async def test_human_format(self) -> None:
        result = await clock.execute({"format": "human"})
        assert isinstance(result, str)
        assert "UTC" in result

    @pytest.mark.asyncio
    async def test_iso_format_explicit(self) -> None:
        result = await clock.execute({"format": "iso"})
        assert isinstance(result, str)
        assert "T" in result

    @pytest.mark.asyncio
    async def test_unknown_format_falls_back_to_iso(self) -> None:
        result = await clock.execute({"format": "unknown"})
        assert isinstance(result, str)
        assert "T" in result


class TestUuidTool:
    """Tests for the UUID tool."""

    @pytest.mark.asyncio
    async def test_default_uuid4(self) -> None:
        result = await uuid.execute({})
        assert isinstance(result, str)
        import uuid as uuid_mod

        parsed = uuid_mod.UUID(result)
        assert parsed.version == 4

    @pytest.mark.asyncio
    async def test_explicit_uuid4(self) -> None:
        result = await uuid.execute({"version": 4})
        assert isinstance(result, str)
        import uuid as uuid_mod

        parsed = uuid_mod.UUID(result)
        assert parsed.version == 4

    @pytest.mark.asyncio
    async def test_uuid1(self) -> None:
        result = await uuid.execute({"version": 1})
        assert isinstance(result, str)
        import uuid as uuid_mod

        parsed = uuid_mod.UUID(result)
        assert parsed.version == 1

    @pytest.mark.asyncio
    async def test_unique_uuids(self) -> None:
        r1 = await uuid.execute({})
        r2 = await uuid.execute({})
        assert r1 != r2


class TestJsonTool:
    """Tests for the JSON tool."""

    @pytest.mark.asyncio
    async def test_parse_string(self) -> None:
        result = await json.execute({"operation": "parse", "data": '{"key": "value"}'})
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_list(self) -> None:
        result = await json.execute({"operation": "parse", "data": "[1, 2, 3]"})
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_stringify_object(self) -> None:
        result = await json.execute(
            {"operation": "stringify", "data": {"key": "value"}}
        )
        assert isinstance(result, str)
        import json as json_mod

        assert json_mod.loads(result) == {"key": "value"}

    @pytest.mark.asyncio
    async def test_stringify_list(self) -> None:
        result = await json.execute({"operation": "stringify", "data": [1, 2, 3]})
        assert isinstance(result, str)
        import json as json_mod

        assert json_mod.loads(result) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_validate_valid(self) -> None:
        result = await json.execute({"operation": "validate", "data": '{"a": 1}'})
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_invalid(self) -> None:
        result = await json.execute({"operation": "validate", "data": "{invalid}"})
        assert result is False

    @pytest.mark.asyncio
    async def test_parse_already_object(self) -> None:
        result = await json.execute({"operation": "parse", "data": {"key": "value"}})
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_unknown_operation(self) -> None:
        with pytest.raises(ValueError, match="Unknown operation"):
            await json.execute({"operation": "unknown", "data": "{}"})

    @pytest.mark.asyncio
    async def test_parse_nested(self) -> None:
        result = await json.execute(
            {"operation": "parse", "data": '{"a": {"b": [1, 2]}}'}
        )
        assert result == {"a": {"b": [1, 2]}}


class TestTextUtilsTool:
    """Tests for the text utils tool."""

    @pytest.mark.asyncio
    async def test_uppercase(self) -> None:
        result = await text_utils.execute({"operation": "uppercase", "text": "hello"})
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_lowercase(self) -> None:
        result = await text_utils.execute({"operation": "lowercase", "text": "HELLO"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_strip(self) -> None:
        result = await text_utils.execute({"operation": "strip", "text": "  hello  "})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_truncate_short(self) -> None:
        result = await text_utils.execute(
            {"operation": "truncate", "text": "short", "max_length": 100}
        )
        assert result == "short"

    @pytest.mark.asyncio
    async def test_truncate_long(self) -> None:
        long_text = "a" * 200
        result = await text_utils.execute(
            {"operation": "truncate", "text": long_text, "max_length": 50}
        )
        assert len(result) == 53
        assert result.endswith("...")

    @pytest.mark.asyncio
    async def test_word_count(self) -> None:
        result = await text_utils.execute(
            {"operation": "word_count", "text": "hello world foo bar"}
        )
        assert result == 4

    @pytest.mark.asyncio
    async def test_char_count(self) -> None:
        result = await text_utils.execute({"operation": "char_count", "text": "hello"})
        assert result == 5

    @pytest.mark.asyncio
    async def test_replace(self) -> None:
        result = await text_utils.execute(
            {
                "operation": "replace",
                "text": "hello world",
                "old": "world",
                "new": "there",
            }
        )
        assert result == "hello there"

    @pytest.mark.asyncio
    async def test_split_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        result = await text_utils.execute(
            {"operation": "split_paragraphs", "text": text}
        )
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."

    @pytest.mark.asyncio
    async def test_unknown_operation(self) -> None:
        with pytest.raises(ValueError, match="Unknown operation"):
            await text_utils.execute({"operation": "unknown", "text": "test"})

    @pytest.mark.asyncio
    async def test_default_operation_strip(self) -> None:
        result = await text_utils.execute({"text": "  hello  "})
        assert result == "hello"


class TestGetToolHandler:
    """Tests for the built-in tool handler registry."""

    @pytest.mark.asyncio
    async def test_get_calculator_handler(self) -> None:
        handler = get_tool_handler("calculator")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_clock_handler(self) -> None:
        handler = get_tool_handler("clock")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_uuid_handler(self) -> None:
        handler = get_tool_handler("uuid")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_json_handler(self) -> None:
        handler = get_tool_handler("json")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_text_utils_handler(self) -> None:
        handler = get_tool_handler("text_utils")
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_get_unknown_handler(self) -> None:
        with pytest.raises(KeyError, match="Unknown built-in tool"):
            get_tool_handler("nonexistent")
