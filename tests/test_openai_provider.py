"""Tests for OpenAIProvider and OpenRouterProvider."""

from __future__ import annotations

import json

import pytest

from app.llm.models import (
    ChatRequest,
    Message,
    ResponseFormat,
    Role,
    ToolCall,
    ToolDefinition,
)
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.transport import HttpResponse


def _request(
    text: str = "hello",
    *,
    tools: bool = False,
    json_mode: bool = False,
    model: str = "",
    max_tokens: int | None = None,
    stop: tuple[str, ...] = (),
) -> ChatRequest:
    return ChatRequest(
        messages=(Message(role=Role.USER, content=text),),
        model=model,
        tools=(ToolDefinition(name="calculator", description="Math"),)
        if tools
        else None,
        response_format=ResponseFormat(type="json") if json_mode else None,
        max_tokens=max_tokens,
        stop=stop,
    )


def _provider(fake_transport, api_key: str = "test-key") -> OpenAIProvider:
    return OpenAIProvider(api_key=api_key, transport=fake_transport)


class TestOpenAIProviderMetadata:
    def test_name(self) -> None:
        assert OpenAIProvider().name == "openai"

    def test_default_model(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        assert provider.default_model == "gpt-4o-mini"

    def test_custom_model(self, fake_transport) -> None:
        provider = OpenAIProvider(model="gpt-4o", transport=fake_transport)
        assert provider.default_model == "gpt-4o"

    def test_model_info(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        info = provider.model_info()
        assert info.provider == "openai"
        assert info.supports_tools is True
        assert info.supports_json is True
        assert info.supports_streaming is True


class TestOpenAIPayload:
    def test_messages_mapped(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi"))
        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "hi"

    def test_tools_mapped(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", tools=True))
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "calculator"

    def test_json_format(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", json_mode=True))
        assert payload["response_format"] == {"type": "json_object"}

    def test_temperature_and_stop(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", stop=("END",)))
        assert payload["temperature"] == 0.7
        assert payload["stop"] == ["END"]

    def test_max_tokens(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", max_tokens=500))
        assert payload["max_tokens"] == 500

    def test_model_override(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", model="gpt-4-turbo"))
        assert payload["model"] == "gpt-4-turbo"

    def test_url_and_headers(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {},
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        provider.generate(_request("hi"))
        request = fake_transport.requests[0]
        assert request["url"] == "https://api.openai.com/v1/chat/completions"
        assert request["headers"]["Authorization"] == "Bearer test-key"

    def test_no_api_key_no_auth_header(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {},
                    }
                ).encode(),
            )
        )
        provider = OpenAIProvider(transport=fake_transport)
        provider.generate(_request("hi"))
        assert "Authorization" not in fake_transport.requests[0]["headers"]


class TestOpenAIGenerate:
    def test_parses_content(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": "Hello there"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                        "model": "gpt-4o-mini",
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        response = provider.generate(_request("hi"))
        assert response.content == "Hello there"
        assert response.provider == "openai"
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "stop"

    def test_parses_tool_calls(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "calculator",
                                                "arguments": '{"expression": "1+1"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {},
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        response = provider.generate(_request("hi", tools=True))
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert isinstance(call, ToolCall)
        assert call.name == "calculator"
        assert call.arguments == {"expression": "1+1"}
        assert response.finish_reason == "tool_calls"

    def test_missing_choices_raises(self, fake_transport) -> None:
        fake_transport.responses.append(HttpResponse(status_code=200, body=b"{}"))
        provider = _provider(fake_transport)
        with pytest.raises(Exception):
            provider.generate(_request("hi"))


class TestOpenAIStream:
    def test_stream_parses_events(self, fake_transport) -> None:
        fake_transport.stream_responses.append(
            [
                b'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}\n\n',
                b'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}\n\n',
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
        provider = _provider(fake_transport)
        chunks = list(provider.stream(_request("hi")))
        assert "".join(c.content for c in chunks) == "Hello world"
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].usage is not None

    def test_stream_sets_stream_flag(self, fake_transport) -> None:
        fake_transport.stream_responses.append([b"data: [DONE]\n\n"])
        provider = _provider(fake_transport)
        list(provider.stream(_request("hi")))
        assert fake_transport.requests[0]["payload"]["stream"] is True


class TestOpenRouterProvider:
    def test_name(self) -> None:
        assert OpenRouterProvider().name == "openrouter"

    def test_is_openai_compatible(self) -> None:
        assert isinstance(OpenRouterProvider(), OpenAIProvider)

    def test_default_model(self, fake_transport) -> None:
        provider = OpenRouterProvider(transport=fake_transport)
        assert provider.default_model == "openrouter/auto"

    def test_url(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {},
                    }
                ).encode(),
            )
        )
        provider = OpenRouterProvider(transport=fake_transport)
        provider.generate(_request("hi"))
        assert (
            fake_transport.requests[0]["url"]
            == "https://openrouter.ai/api/v1/chat/completions"
        )
