"""Tests for the deterministic MockProvider."""

from __future__ import annotations

import pytest

from app.llm.models import (
    ChatRequest,
    Message,
    ModelInfo,
    ResponseFormat,
    Role,
    ToolCall,
    ToolDefinition,
)
from app.llm.providers.mock import MockProvider


def _request(
    text: str = "hello world",
    *,
    tools: bool = False,
    json_mode: bool = False,
    model: str = "",
) -> ChatRequest:
    return ChatRequest(
        messages=(Message(role=Role.USER, content=text),),
        model=model,
        tools=(ToolDefinition(name="calculator", description="Math"),)
        if tools
        else None,
        response_format=ResponseFormat(type="json") if json_mode else None,
    )


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider()


class TestMockMetadata:
    def test_name(self, provider: MockProvider) -> None:
        assert provider.name == "mock"

    def test_default_model(self, provider: MockProvider) -> None:
        assert provider.default_model == "mock-1"

    def test_model_info(self, provider: MockProvider) -> None:
        info = provider.model_info()
        assert isinstance(info, ModelInfo)
        assert info.provider == "mock"
        assert info.supports_streaming is True
        assert info.supports_tools is True
        assert info.supports_json is True


class TestMockGenerate:
    def test_echoes_last_user_message(self, provider: MockProvider) -> None:
        response = provider.generate(_request("ping"))
        assert response.content == "Echo: ping"
        assert response.finish_reason == "stop"

    def test_deterministic(self, provider: MockProvider) -> None:
        first = provider.generate(_request("same input"))
        second = provider.generate(_request("same input"))
        assert first.content == second.content
        assert first.tool_calls == second.tool_calls
        assert first.finish_reason == second.finish_reason
        assert first.usage == second.usage

    def test_model_from_request(self, provider: MockProvider) -> None:
        response = provider.generate(_request("hi", model="custom"))
        assert response.model == "custom"

    def test_model_default_fallback(self, provider: MockProvider) -> None:
        response = provider.generate(_request("hi"))
        assert response.model == "mock-1"

    def test_usage_positive(self, provider: MockProvider) -> None:
        response = provider.generate(_request("hello"))
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_echo_input_disabled(self) -> None:
        provider = MockProvider(echo_input=False)
        response = provider.generate(_request("hello"))
        assert response.content != "Echo: hello"
        assert response.content.startswith("Mock response")

    def test_no_user_message(self, provider: MockProvider) -> None:
        request = ChatRequest(
            messages=(Message(role=Role.SYSTEM, content="sys"),),
            model="mock-1",
        )
        response = provider.generate(request)
        assert response.content == "Echo: "

    def test_provider_is_in_response(self, provider: MockProvider) -> None:
        response = provider.generate(_request("hi"))
        assert response.provider == "mock"


class TestMockGenerateTools:
    def test_returns_tool_call(self, provider: MockProvider) -> None:
        response = provider.generate(_request("what is 1+1", tools=True))
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calculator"
        assert isinstance(response.tool_calls[0], ToolCall)
        assert response.finish_reason == "tool_calls"
        assert response.content == ""

    def test_no_tools_no_tool_call(self, provider: MockProvider) -> None:
        response = provider.generate(_request("hi"))
        assert response.tool_calls == ()

    def test_tool_calls_disabled(self) -> None:
        provider = MockProvider(tool_calls_on_tools=False)
        response = provider.generate(_request("hi", tools=True))
        assert response.tool_calls == ()
        assert response.finish_reason == "stop"


class TestMockGenerateJson:
    def test_json_mode(self, provider: MockProvider) -> None:
        response = provider.generate(_request("hello", json_mode=True))
        assert response.content.startswith("{")
        assert '"model"' in response.content
        assert '"reply"' in response.content

    def test_json_disabled(self) -> None:
        provider = MockProvider(json_mode_enabled=False)
        response = provider.generate(_request("hello", json_mode=True))
        assert not response.content.startswith("{")


class TestMockStream:
    def test_stream_concatenates(self, provider: MockProvider) -> None:
        request = _request("ping")
        chunks = list(provider.stream(request))
        assert "".join(c.content for c in chunks) == "Echo: ping"
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].usage is not None

    def test_stream_tool_calls(self, provider: MockProvider) -> None:
        chunks = list(provider.stream(_request("hi", tools=True)))
        assert chunks[0].tool_calls
        assert chunks[0].finish_reason == "tool_calls"
        assert len(chunks) == 1

    def test_stream_chunks_are_fixed_size(self, provider: MockProvider) -> None:
        request = _request("a" * 50)
        chunks = [c for c in provider.stream(request) if c.content]
        for chunk in chunks[:-1]:
            assert len(chunk.content) == 4
        assert len(chunks[-1].content) <= 4

    def test_stream_matches_generate(self, provider: MockProvider) -> None:
        request = _request("same input")
        streamed = "".join(c.content for c in provider.stream(request))
        assert streamed == provider.generate(request).content
