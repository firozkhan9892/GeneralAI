"""Tests for built-in category tools."""

from __future__ import annotations

from typing import Any

import pytest

from app.tools.categories.builtin import (
    CalculatorTool,
    ClockTool,
    EchoTool,
    TextUtilsTool,
)
from app.tools.categories.file import FileListTool, FileReadTool, FileWriteTool
from app.tools.categories.http import HttpRequestTool
from app.tools.categories.python import PythonEvalTool
from app.tools.categories.shell import ShellRunTool
from app.tools.categories.web import WebFetchTool
from app.tools.exceptions import (
    ToolExecutionError,
    ToolValidationError,
)
from app.tools.models import ToolCategory
from app.tools.network import HttpClient, HttpResponse


class TestCalculatorTool:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("6 * 7", 42),
            ("15 / 3", 5.0),
            ("2 ** 8", 256),
            ("17 % 5", 2),
            ("-42", -42),
            ("(2 + 3) * (4 - 1)", 15),
            ("abs(-5)", 5),
            ("min(3, 1, 4)", 1),
            ("max(3, 1, 4)", 4),
            ("round(3.7)", 4),
            ("sum([1, 2, 3])", 6),
        ],
    )
    def test_expression(self, expression: str, expected: object) -> None:
        assert CalculatorTool().run({"expression": expression}) == expected

    def test_missing_expression(self) -> None:
        with pytest.raises(ToolValidationError, match="expression"):
            CalculatorTool().run({"expression": "  "})

    def test_invalid_expression(self) -> None:
        with pytest.raises(ToolValidationError):
            CalculatorTool().run({"expression": "1 +"})

    def test_unsupported_syntax(self) -> None:
        with pytest.raises(ToolValidationError):
            CalculatorTool().run({"expression": "__import__('os')"})

    def test_category(self) -> None:
        assert CalculatorTool().category == ToolCategory.BUILTIN


class TestEchoTool:
    def test_echo(self) -> None:
        assert EchoTool().run({"text": "hello"}) == "hello"

    def test_required_text(self) -> None:
        with pytest.raises(ToolValidationError, match="text"):
            EchoTool().validate({})


class TestClockTool:
    def test_iso(self) -> None:
        output = ClockTool().run({"format": "iso"})
        assert isinstance(output, str)
        assert "T" in output

    def test_unix(self) -> None:
        output = ClockTool().run({"format": "unix"})
        assert isinstance(output, float)
        assert output > 0

    def test_human(self) -> None:
        output = ClockTool().run({"format": "human"})
        assert isinstance(output, str)
        assert "UTC" in output

    def test_default_iso(self) -> None:
        output = ClockTool().run({})
        assert "T" in output


class TestTextUtilsTool:
    def test_uppercase(self) -> None:
        assert TextUtilsTool().run({"text": "hi", "operation": "uppercase"}) == "HI"

    def test_lowercase(self) -> None:
        assert TextUtilsTool().run({"text": "HI", "operation": "lowercase"}) == "hi"

    def test_strip(self) -> None:
        assert TextUtilsTool().run({"text": "  hi  ", "operation": "strip"}) == "hi"

    def test_title(self) -> None:
        assert (
            TextUtilsTool().run({"text": "hello world", "operation": "title"})
            == "Hello World"
        )

    def test_unknown_operation(self) -> None:
        with pytest.raises(ToolValidationError, match="Unknown operation"):
            TextUtilsTool().run({"text": "hi", "operation": "nope"})


class TestFileTools:
    def test_write_then_read(self, tmp_path) -> None:
        path = tmp_path / "sub" / "note.txt"
        write = FileWriteTool().run({"path": str(path), "content": "hello"})
        assert write["bytes"] == 5
        assert path.exists()
        assert FileReadTool().run({"path": str(path)}) == "hello"

    def test_read_missing(self, tmp_path) -> None:
        with pytest.raises(ToolExecutionError):
            FileReadTool().run({"path": str(tmp_path / "nope.txt")})

    def test_write_nested_dirs(self, tmp_path) -> None:
        path = tmp_path / "a" / "b" / "c.txt"
        FileWriteTool().run({"path": str(path), "content": "x"})
        assert path.exists()

    def test_list_directory(self, tmp_path) -> None:
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "a.txt").write_text("a")
        entries = FileListTool().run({"path": str(tmp_path)})
        assert entries == ["a.txt", "b.txt"]

    def test_list_missing(self, tmp_path) -> None:
        with pytest.raises(ToolValidationError, match="does not exist"):
            FileListTool().run({"path": str(tmp_path / "missing")})

    def test_category_file(self) -> None:
        assert FileReadTool().category == ToolCategory.FILE


class FakeHttpClient(HttpClient):
    def __init__(
        self, response: HttpResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response or HttpResponse(
            status_code=200, body="body", headers={"x": "y"}
        )
        self._error = error
        self.requests: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: Any = None,
        timeout_s: float = 10.0,
    ) -> HttpResponse:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_s": timeout_s,
            }
        )
        if self._error is not None:
            raise self._error
        return self._response


class TestWebFetchTool:
    def test_fetch(self) -> None:
        client = FakeHttpClient(response=HttpResponse(status_code=200, body="content"))
        tool = WebFetchTool(client=client)
        result = tool.run({"url": "https://example.com"})
        assert result == {"status": 200, "body": "content"}
        assert client.requests[0]["method"] == "GET"

    def test_invalid_url(self) -> None:
        tool = WebFetchTool()
        with pytest.raises(ToolValidationError, match="Invalid URL"):
            tool.run({"url": "not a url"})

    def test_request_error(self) -> None:
        client = FakeHttpClient(error=ToolExecutionError("boom", module="network"))
        tool = WebFetchTool(client=client)
        with pytest.raises(ToolExecutionError):
            tool.run({"url": "https://example.com"})

    def test_category(self) -> None:
        assert WebFetchTool().category == ToolCategory.WEB
        assert WebFetchTool().requires_confirmation is True


class TestHttpRequestTool:
    def test_get(self) -> None:
        client = FakeHttpClient(
            response=HttpResponse(status_code=200, body="ok", headers={"a": "b"})
        )
        tool = HttpRequestTool(client=client)
        result = tool.run({"url": "https://example.com"})
        assert result == {"status": 200, "body": "ok", "headers": {"a": "b"}}

    def test_post_with_payload(self) -> None:
        client = FakeHttpClient()
        tool = HttpRequestTool(client=client)
        tool.run(
            {
                "url": "https://example.com/api",
                "method": "POST",
                "headers": {"Authorization": "Bearer x"},
                "payload": {"key": "value"},
            }
        )
        request = client.requests[0]
        assert request["method"] == "POST"
        assert request["payload"] == {"key": "value"}
        assert request["headers"] == {"Authorization": "Bearer x"}

    def test_invalid_url(self) -> None:
        tool = HttpRequestTool()
        with pytest.raises(ToolValidationError, match="Invalid URL"):
            tool.run({"url": "ftp://x"})

    def test_category(self) -> None:
        assert HttpRequestTool().category == ToolCategory.HTTP


class TestShellRunTool:
    def test_run_command(self) -> None:
        tool = ShellRunTool()
        result = tool.run({"command": "echo hello"})
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_empty_command(self) -> None:
        with pytest.raises(ToolValidationError, match="command"):
            ShellRunTool().run({"command": "  "})

    def test_category(self) -> None:
        assert ShellRunTool().category == ToolCategory.SHELL
        assert ShellRunTool().sandboxable is True


class TestPythonEvalTool:
    def test_arithmetic(self) -> None:
        assert PythonEvalTool().run({"expression": "2 + 3 * 4"}) == 14

    def test_safe_builtins(self) -> None:
        assert PythonEvalTool().run({"expression": "abs(-9)"}) == 9

    def test_restricted(self) -> None:
        with pytest.raises(ToolExecutionError):
            PythonEvalTool().run({"expression": "open('x')"})

    def test_category(self) -> None:
        assert PythonEvalTool().category == ToolCategory.PYTHON


class TestPlanTools:
    def test_all_plan_skills_present(self) -> None:
        from app.tools.categories.planning import PLAN_SKILL_NAMES, plan_tools

        tools = plan_tools()
        assert len(tools) == len(PLAN_SKILL_NAMES)
        assert {tool.name for tool in tools} == set(PLAN_SKILL_NAMES)

    def test_pass_through_run(self) -> None:
        from app.tools.categories.planning import plan_tools

        tool = next(t for t in plan_tools() if t.name == "analyze_question")
        assert tool.run({"text": "hello"}) == "hello"

    def test_default_completion_notice(self) -> None:
        from app.tools.categories.planning import plan_tools

        tool = next(t for t in plan_tools() if t.name == "analyze_question")
        assert tool.run({}) == "Completed analyze_question"

    def test_optional_text_parameter(self) -> None:
        from app.tools.categories.planning import plan_tools

        tool = next(t for t in plan_tools() if t.name == "analyze_question")
        assert tool.validate({}) == {"text": ""}

    def test_executes_through_registry(self) -> None:
        from app.tools.categories.planning import plan_tools
        from app.tools.executor import ToolExecutor
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry()
        for tool in plan_tools():
            registry.register(tool)
        executor = ToolExecutor(registry=registry)
        result = executor.execute("formulate_answer")
        assert result.success is True
