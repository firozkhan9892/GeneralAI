"""Tests for GeminiProvider."""

from __future__ import annotations

import json

import pytest

from app.llm.models import (
    ChatRequest,
    Message,
    ResponseFormat,
    Role,
    ToolDefinition,
)
from app.llm.providers.gemini import GeminiProvider
from app.llm.transport import HttpResponse


def _request(
    text: str = "hello",
    *,
    tools: bool = False,
    json_mode: bool = False,
    model: str = "",
) -> ChatRequest:
    return ChatRequest(
        messages=(
            Message(role=Role.SYSTEM, content="Be concise"),
            Message(role=Role.USER, content=text),
        ),
        model=model,
        tools=(ToolDefinition(name="calculator", description="Math"),)
        if tools
        else None,
        response_format=ResponseFormat(type="json") if json_mode else None,
    )


def _provider(fake_transport, api_key: str = "test-key") -> GeminiProvider:
    return GeminiProvider(api_key=api_key, transport=fake_transport)


class TestGeminiMetadata:
    def test_name(self) -> None:
        assert GeminiProvider().name == "gemini"

    def test_default_model(self, fake_transport) -> None:
        assert _provider(fake_transport).default_model == "gemini-1.5-pro"

    def test_model_info(self, fake_transport) -> None:
        info = _provider(fake_transport).model_info()
        assert info.supports_tools is True
        assert info.supports_json is True


class TestGeminiPayload:
    def test_contents_mapped(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi"))
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"] == [{"text": "hi"}]
        assert payload["systemInstruction"] == {"parts": [{"text": "Be concise"}]}

    def test_tools_mapped(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", tools=True))
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "calculator"

    def test_json_mode(self, fake_transport) -> None:
        provider = _provider(fake_transport)
        payload = provider._build_payload(_request("hi", json_mode=True))
        assert payload["generationConfig"]["responseMimeType"] == "application/json"

    def test_url_and_headers(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                        "usageMetadata": {},
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        provider.generate(_request("hi"))
        request = fake_transport.requests[0]
        assert (
            request["url"]
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        )
        assert request["headers"]["x-goog-api-key"] == "test-key"


class TestGeminiGenerate:
    def test_parses_content(self, fake_transport) -> None:
        fake_transport.responses.append(
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "Hello"}, {"text": " there"}]
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 7,
                            "candidatesTokenCount": 3,
                            "totalTokenCount": 10,
                        },
                    }
                ).encode(),
            )
        )
        provider = _provider(fake_transport)
        response = provider.generate(_request("hi"))
        assert response.content == "Hello there"
        assert response.usage.total_tokens == 10
        assert response.finish_reason == "STOP"

    def test_missing_candidates_raises(self, fake_transport) -> None:
        fake_transport.responses.append(HttpResponse(status_code=200, body=b"{}"))
        provider = _provider(fake_transport)
        with pytest.raises(Exception):
            provider.generate(_request("hi"))


class TestGeminiStream:
    def test_stream_parses_events(self, fake_transport) -> None:
        fake_transport.stream_responses.append(
            [
                b'data: {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}\n\n',
                b'data: {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]}\n\n',
            ]
        )
        provider = _provider(fake_transport)
        chunks = list(provider.stream(_request("hi")))
        assert "".join(c.content for c in chunks) == "Hello"
