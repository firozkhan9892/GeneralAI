"""Tests for unified LLM provider models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm.models import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    ResponseFormat,
    Role,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
)


class TestMessage:
    def test_defaults(self) -> None:
        message = Message(role=Role.USER)
        assert message.content == ""
        assert message.name is None
        assert message.tool_call_id is None

    def test_full(self) -> None:
        message = Message(role=Role.TOOL, content="42", tool_call_id="call_1")
        assert message.role == Role.TOOL
        assert message.content == "42"
        assert message.tool_call_id == "call_1"

    def test_frozen(self) -> None:
        message = Message(role=Role.USER, content="hi")
        with pytest.raises(ValidationError):
            message.content = "changed"  # type: ignore[misc]

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="admin")  # type: ignore[arg-type]

    def test_role_coerces_from_string(self) -> None:
        message = Message(role="user")  # type: ignore[arg-type]
        assert message.role == Role.USER


class TestToolCall:
    def test_defaults(self) -> None:
        call = ToolCall(name="calculator")
        assert call.id == ""
        assert call.arguments == {}

    def test_arguments(self) -> None:
        call = ToolCall(id="c1", name="calc", arguments={"expression": "1+1"})
        assert call.arguments == {"expression": "1+1"}


class TestToolDefinition:
    def test_defaults(self) -> None:
        tool = ToolDefinition(name="calc")
        assert tool.description == ""
        assert tool.parameters == {}

    def test_full(self) -> None:
        schema = {"type": "object", "properties": {}}
        tool = ToolDefinition(name="calc", description="Add", parameters=schema)
        assert tool.description == "Add"
        assert tool.parameters == schema


class TestUsage:
    def test_defaults(self) -> None:
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Usage(prompt_tokens=-1)


class TestResponseFormat:
    def test_default(self) -> None:
        fmt = ResponseFormat()
        assert fmt.type == "text"

    def test_json(self) -> None:
        assert ResponseFormat(type="json").type == "json"

    def test_invalid_type(self) -> None:
        with pytest.raises(ValidationError):
            ResponseFormat(type="yaml")


class TestModelInfo:
    def test_defaults(self) -> None:
        info = ModelInfo(name="gpt-4o-mini", provider="openai")
        assert info.supports_streaming is False
        assert info.max_context_tokens == 4096

    def test_full(self) -> None:
        info = ModelInfo(
            name="gpt-4o-mini",
            provider="openai",
            supports_streaming=True,
            supports_tools=True,
            max_context_tokens=128000,
        )
        assert info.supports_tools is True


class TestChatRequest:
    def test_minimal(self) -> None:
        request = ChatRequest(
            messages=(Message(role=Role.USER, content="hi"),),
            model="gpt-4o-mini",
        )
        assert request.temperature == 0.7
        assert request.tools is None
        assert request.stream is False

    def test_defaults(self) -> None:
        request = ChatRequest(messages=(), model="m")
        assert request.stop == ()
        assert request.response_format is None
        assert request.timeout_s == 60.0

    def test_with_tools(self) -> None:
        request = ChatRequest(
            messages=(Message(role=Role.USER, content="hi"),),
            model="m",
            tools=(ToolDefinition(name="calc"),),
            response_format=ResponseFormat(type="json"),
        )
        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.response_format is not None
        assert request.response_format.type == "json"

    def test_frozen(self) -> None:
        request = ChatRequest(messages=(), model="m")
        with pytest.raises(ValidationError):
            request.model = "other"  # type: ignore[misc]

    def test_requires_messages(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(model="m")  # type: ignore[call-arg]

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(messages=(), model="m", temperature=3.0)


class TestChatResponse:
    def test_defaults(self) -> None:
        response = ChatResponse(model="gpt-4o-mini", provider="openai")
        assert response.content == ""
        assert response.tool_calls == ()
        assert response.finish_reason == "stop"
        assert response.usage.total_tokens == 0

    def test_full(self) -> None:
        response = ChatResponse(
            model="m",
            provider="openai",
            content="Hello",
            tool_calls=(ToolCall(name="calc"),),
            usage=Usage(total_tokens=10),
        )
        assert response.content == "Hello"
        assert response.tool_calls[0].name == "calc"


class TestStreamChunk:
    def test_defaults(self) -> None:
        chunk = StreamChunk(model="m", provider="openai")
        assert chunk.content == ""
        assert chunk.tool_calls == ()
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_content_chunk(self) -> None:
        chunk = StreamChunk(model="m", provider="openai", content="Hi")
        assert chunk.content == "Hi"
