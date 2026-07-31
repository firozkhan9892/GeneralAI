"""Tests for OllamaProvider."""

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
from app.llm.providers.ollama import OllamaProvider
from app.llm.transport import HttpResponse


def _request(
    text: str = "hello",
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


def _provider(fake_transport) -> OllamaProvider:
    return OllamaProvider(transport=fake_transport)


class TestOllamaMetadata:
    def test_name(self) -> None:
        assert OllamaProvider().name == "ollama"

    def test_default_model(self, fake_transport) -> None:
        assert _provider(fake_transport).default_model == "llama3.2"

    def test_model_info(self, fake_transport) -> None:
        info = _provider(fake_transport).model_info()
        assert info.supports_tools is True
        assert info.supports_json is True

    def test_no_api_key_needed(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        assert provider._api_key is None


class TestOllamaPayload:
    def test_messages_mapped(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi"))
        assert payload["model"] == "llama3.2"
        assert payload["messages"][0] == {"role": "user", "content": "hi"}

    def test_options(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", json_mode=True))
        assert payload["options"]["temperature"] == 0.7
        assert payload["format"] == "json"

    def test_tools_mapped(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", tools=True))
        assert payload["tools"][0]["function"]["name"] == "calculator"

    def test_url(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "done": True,
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        provider.generate(_request("hi"))
        assert fake_transport.requests[0]["url"] == "http://localhost:11434/api/chat"


class TestOllamaGenerate:
    def test_parses_content(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "model": "llama3.2",
                        "message": {"role": "assistant", "content": "Hello"},
                        "done": True,
                        "prompt_eval_count": 7,
                        "eval_count": 3,
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        response = provider.generate(_request("hi"))
        assert response.content == "Hello"
        assert response.usage.total_tokens == 10

    def test_parses_tool_calls(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculator",
                                        "arguments": {"expression": "1+1"},
                                    }
                                }
                            ],
                        },
                        "done": True,
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

    def test_missing_message_raises(self, fake_transport) -> None:
        fake_transport.responses.append(HttpResponse(status_code=200, body=b"{}"))
        provider = _provider(fake_transport)
        with pytest.raises(Exception):
            provider.generate(_request("hi"))


class TestOllamaStream:
    def test_stream_parses_ndjson(self, fake_transport) -> None:
        fake_transport.stream_responses.append(
            [
                b'{"model": "llama3.2", "message": {"content": "Hel"}}\n',
                b'{"model": "llama3.2", "message": {"content": "lo"}, "done": true}\n',
            ]
        )
        provider = _provider(fake_transport)
        chunks = list(provider.stream(_request("hi")))
        assert "".join(c.content for c in chunks) == "Hello"
        assert chunks[-1].finish_reason == "stop"
